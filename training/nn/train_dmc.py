"""DMC (Deep Monte Carlo, DouZero-style) training: treats the model's
per-action logit as a Q-value and regresses it directly to the game's
Monte Carlo return (+1/-1/0), NOT a softmax policy target. No BC/teacher
data mixing -- pure self-play outcome, the one operator in the 2026-07-04
literature review never tried (see docs/report-log.md 2026-07-05 design
decision entry). Warm-starts from an existing checkpoint's representations
(shape-filtered) but the training OBJECTIVE is regression, not imitation.

Usage:
  python training/nn/train_dmc.py --data "../exploiter_r1*.pkl.gz" \
      --init ../ptcg_dagger_r2.pth --out ../ptcg_dmc_r1.pth --epochs 2
"""
import argparse
import os
import random
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO_ROOT)

from dataset import load_shards  # noqa: E402
from model import PTCGNet  # noqa: E402
from net_common import encode_batch  # noqa: E402

DEVICE = torch.device("cpu")


def to_rows(samples):
    """Each sample has obs/action/outcome. Encode individually (fixed-size
    tensors per encode_batch), keep the taken action's index, drop samples
    where the taken action isn't a clean single pick or encoding fails."""
    rows = []
    for s in samples:
        obs, act, outcome = s.get("obs"), s.get("action"), s.get("outcome", 0)
        sel = obs.get("select") if obs else None
        if not sel or not sel.get("option") or not act:
            continue
        taken = act[0] if isinstance(act, list) else act
        try:
            batch, n = encode_batch(obs, sel)
        except Exception:
            continue
        if not isinstance(taken, int) or taken < 0 or taken >= n:
            continue
        rows.append((batch, taken, float(outcome)))
    return rows


def batches(rows, batch_size, seed):
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = [rows[j] for j in idx[i:i + batch_size]]
        keys = ["board_ids", "hand_ids", "discard_ids", "numeric", "action_type",
                "action_card", "action_attack", "action_numeric", "action_mask"]
        tensors = {k: torch.cat([c[0][ki] for c in chunk], dim=0) for ki, k in enumerate(keys)}
        taken = torch.tensor([c[1] for c in chunk], dtype=torch.long)
        target = torch.tensor([c[2] for c in chunk], dtype=torch.float)
        yield tensors, taken, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--init", default="../ptcg_dagger_r2.pth")
    ap.add_argument("--out", default="../ptcg_dmc_r1.pth")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.05)
    args = ap.parse_args()

    raw = load_shards(args.data, limit=args.limit)
    print(f"loaded {len(raw)} raw samples")
    rows = to_rows(raw)
    print(f"encoded {len(rows)} usable (state, taken_action, return) rows")
    random.Random(7).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    model = PTCGNet().to(DEVICE)
    state = torch.load(args.init, map_location=DEVICE)
    own = model.state_dict()

    # 2026-07-06 fix: a plain shape-mismatch filter fully discards
    # numeric_proj.0.weight (random reinit) whenever NUM_FEATS changes (e.g.
    # ENCODE_FEATURE_SET adding new features) -- confirmed via an ablation
    # (docs/report-log.md "AlphaZero-style push, Phase 1 ablation") that ALL
    # THREE independently-tested feature additions landed at the SAME
    # regressed accuracy regardless of which features were added, pointing
    # at this warm-start disruption rather than the features themselves.
    # Fix: when the mismatch is purely "more input columns" (same output
    # dim, more input dim), copy the old weights into the first N_old
    # columns and leave only the genuinely new columns at their fresh init,
    # instead of discarding everything already learned about the original
    # 13 features.
    key = "numeric_proj.0.weight"
    if key in state and key in own and state[key].shape != own[key].shape:
        old_w, new_w = state[key], own[key].clone()
        if old_w.shape[0] == new_w.shape[0] and old_w.shape[1] < new_w.shape[1]:
            new_w[:, : old_w.shape[1]] = old_w
            state[key] = new_w
            print(f"partial warm-start: copied {key} old {tuple(old_w.shape)} "
                  f"into new {tuple(new_w.shape)}, new columns kept fresh-init")

    state = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
    model.load_state_dict(state, strict=False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.HuberLoss(delta=1.0)

    def sign_acc(rs):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for tensors, taken, target in batches(rs, 128, 1):
                logits, _ = model(tensors["board_ids"], tensors["hand_ids"], tensors["discard_ids"],
                                   tensors["numeric"], tensors["action_type"], tensors["action_card"],
                                   tensors["action_attack"], tensors["action_numeric"], tensors["action_mask"])
                q = logits.gather(1, taken.unsqueeze(1)).squeeze(1)
                correct += ((q >= 0) == (target >= 0)).sum().item()
                total += len(target)
        model.train()
        return correct / max(total, 1)

    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for tensors, taken, target in batches(train_rows, args.batch_size, epoch):
            logits, _ = model(tensors["board_ids"], tensors["hand_ids"], tensors["discard_ids"],
                               tensors["numeric"], tensors["action_type"], tensors["action_card"],
                               tensors["action_attack"], tensors["action_numeric"], tensors["action_mask"])
            q = logits.gather(1, taken.unsqueeze(1)).squeeze(1)
            loss = loss_fn(q, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        acc = sign_acc(val_rows)
        print(f"epoch {epoch}: avg_loss={total_loss/max(n_batches,1):.4f} val_sign_acc={acc:.4f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), args.out)
    print(f"saved best (val_sign_acc={best:.4f}) to {args.out}")


if __name__ == "__main__":
    main()
