"""Train SeqPTCGNet on game-grouped shards from seq_collect.py.
See docs/next-session-plan.md Phase 2. Local CPU smoke (small --limit-games,
1 epoch) before any Kaggle GPU run — Phase 2 mandates this.

Usage:
  python training/nn/train_seq.py --data training/seq_data_v29d*.pkl.gz \
      --out training/ptcg_seq.pth --epochs 1 --limit-games 20
"""
import argparse
import glob
import gzip
import os
import pickle
import random
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from encode import MAX_ACTIONS, CARD_VOCAB, NUM_FEATS  # noqa: E402
from encode_seq import encode_sample_seq, PHI4_DIM  # noqa: E402
from model_seq import SeqPTCGNet  # noqa: E402


def _encode_game(g, max_len=160):
    """Encodes + discards the raw obs dicts immediately (NOT kept resident) —
    raw obs dicts (full board/hand/discard/log state for both players) are
    5-10x+ bulkier in memory than the compact encoded form once unpickled,
    and a 4000-game corpus held raw was observed to balloon to ~26GB/39.6GB
    system RAM with zero training progress (same OOM pattern documented
    elsewhere in this project — see dataset.py's load_shards docstring).
    Encoding once at load time (not lazily per-epoch in __getitem__) is also
    strictly faster: encode_sample_seq runs once per decision, not once per
    decision per epoch."""
    decs = g["decisions"][:max_len]
    outcome = float(g.get("outcome", 0))
    steps = []
    for d in decs:
        sel = d["obs"].get("select")
        enc = encode_sample_seq(d["obs"], sel)
        n = enc["n_actions"]
        label = d["action"][0] if d["action"] else 0
        label = min(label, n - 1) if n else 0
        steps.append((enc, label))
    return steps, outcome


def load_seq_shards(pattern, limit_games=None, max_len=160):
    paths = []
    for part in pattern.split(","):
        paths.extend(glob.glob(part.strip(), recursive=True))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"no shards matched {pattern}")
    random.Random(0).shuffle(paths)
    encoded_games = []
    for p in paths:
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rb") as f:
            raw_games = pickle.load(f)
        for g in raw_games:
            encoded_games.append(_encode_game(g, max_len=max_len))
            if limit_games and len(encoded_games) >= limit_games:
                break
        del raw_games
        if limit_games and len(encoded_games) >= limit_games:
            break
    if limit_games:
        encoded_games = encoded_games[:limit_games]
    return encoded_games


class SeqDataset(Dataset):
    def __init__(self, encoded_games):
        self.games = encoded_games

    def __len__(self):
        return len(self.games)

    def __getitem__(self, i):
        return self.games[i]


def collate_seq(batch):
    B = len(batch)
    T = max(len(steps) for steps, _ in batch)
    T = max(T, 1)
    board_ids = torch.zeros(B, T, 13, dtype=torch.long)
    hand_ids = torch.zeros(B, T, 20, dtype=torch.long)
    discard_ids = torch.zeros(B, T, 20, dtype=torch.long)
    numeric = torch.zeros(B, T, NUM_FEATS, dtype=torch.float)
    phi4 = torch.zeros(B, T, PHI4_DIM, dtype=torch.float)
    action_type = torch.zeros(B, T, MAX_ACTIONS, dtype=torch.long)
    action_card = torch.zeros(B, T, MAX_ACTIONS, dtype=torch.long)
    action_attack = torch.zeros(B, T, MAX_ACTIONS, dtype=torch.long)
    action_numeric = torch.zeros(B, T, MAX_ACTIONS, 4, dtype=torch.float)
    action_mask = torch.zeros(B, T, MAX_ACTIONS, dtype=torch.float)
    labels = torch.zeros(B, T, dtype=torch.long)
    step_mask = torch.zeros(B, T, dtype=torch.float)
    values = torch.zeros(B, dtype=torch.float)

    for i, (steps, outcome) in enumerate(batch):
        values[i] = outcome
        for t, (enc, label) in enumerate(steps):
            board_ids[i, t] = torch.tensor(enc["board_ids"], dtype=torch.long)
            h = enc["hand_ids"][:20]
            hand_ids[i, t, :len(h)] = torch.tensor(h, dtype=torch.long)
            dcd = enc["discard_ids"][:20]
            discard_ids[i, t, :len(dcd)] = torch.tensor(dcd, dtype=torch.long)
            numeric[i, t] = torch.tensor(enc["numeric"], dtype=torch.float)
            phi4[i, t] = torch.tensor(enc["phi4"], dtype=torch.float)
            for j, a in enumerate(enc["actions"][:MAX_ACTIONS]):
                action_type[i, t, j] = a["type"]
                action_card[i, t, j] = a["card_id"]
                action_attack[i, t, j] = a["attack_id"]
                action_numeric[i, t, j] = torch.tensor(a["numeric"], dtype=torch.float)
                action_mask[i, t, j] = 1.0
            labels[i, t] = label
            step_mask[i, t] = 1.0

    return {
        "board_ids": board_ids, "hand_ids": hand_ids, "discard_ids": discard_ids,
        "numeric": numeric, "phi4": phi4, "action_type": action_type,
        "action_card": action_card, "action_attack": action_attack,
        "action_numeric": action_numeric, "action_mask": action_mask,
        "labels": labels, "step_mask": step_mask, "values": values,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit-games", type=int, default=None)
    ap.add_argument("--no-history", action="store_true", help="ablation control (a): drop causal transformer")
    ap.add_argument("--no-phi4", action="store_true", help="ablation control (b): drop Φ v4 features")
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--resume", default=None, help="checkpoint to resume from (weights + optimizer state + epoch count)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    games = load_seq_shards(args.data, limit_games=args.limit_games)
    print(f"loaded {len(games)} game-sequences from {args.data}, device={device}")
    ds = SeqDataset(games)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_seq)

    net = SeqPTCGNet(n_layers=args.n_layers, use_history=not args.no_history,
                      use_phi4=not args.no_phi4).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"model params: {n_params/1e6:.2f}M use_history={not args.no_history} use_phi4={not args.no_phi4}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt["state_dict"])
        if "opt_state_dict" in ckpt:
            opt.load_state_dict(ckpt["opt_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        net.train()
        total_loss = total_correct = total_steps = 0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, value = net(
                batch["board_ids"], batch["hand_ids"], batch["discard_ids"],
                batch["numeric"], batch["action_type"], batch["action_card"],
                batch["action_attack"], batch["action_numeric"], batch["action_mask"],
                phi4=batch["phi4"], step_mask=batch["step_mask"])
            B, T, A = logits.shape
            logits_flat = logits.reshape(B * T, A)
            labels_flat = batch["labels"].reshape(B * T)
            mask_flat = batch["step_mask"].reshape(B * T)
            ce = nn.functional.cross_entropy(logits_flat, labels_flat, reduction="none")
            loss = (ce * mask_flat).sum() / mask_flat.sum().clamp(min=1)

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred = logits_flat.argmax(dim=-1)
                correct = ((pred == labels_flat).float() * mask_flat).sum().item()
            total_loss += loss.item() * mask_flat.sum().item()
            total_correct += correct
            total_steps += mask_flat.sum().item()
        print(f"epoch {epoch}: loss={total_loss/max(total_steps,1):.4f} "
              f"train_acc={total_correct/max(total_steps,1):.4f} steps={int(total_steps)}")
        ckpt = {
            "state_dict": net.state_dict(),
            "opt_state_dict": opt.state_dict(),
            "use_history": not args.no_history,
            "use_phi4": not args.no_phi4,
            "n_layers": args.n_layers,
            "epoch": epoch,
        }
        torch.save(ckpt, args.out)
        print(f"checkpointed epoch {epoch} -> {args.out}")

    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
