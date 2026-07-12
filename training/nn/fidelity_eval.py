"""Canonical fresh-state teacher-agreement (fidelity) protocol, formalized
2026-07-09 — no prior run of this measurement (BC 74.9%/73.1%, DAgger r1
79.7%/81.1%, r2 81.9%, per docs/nn-training.md) had a saved reusable script;
this reimplements the SAME protocol from its written description so the new
number is comparable: collect fresh temp≈0 deployment-realistic self-play
games from the CURRENT teacher, sample ~3000 decision points from them, and
measure how often the checkpoint's argmax matches the teacher's actual
recorded action at that point. Works for both the plain PTCGNet checkpoint
(training/nn/model.py) and SeqPTCGNet (model_seq.py, full-game causal
forward pass so each position sees exactly the history it would at
inference — see model_seq.py's single-forward-per-game design note).

Usage:
  python training/nn/fidelity_eval.py --games 60 --n-states 3000 \
      --seq-ckpt training/ptcg_seq.pth
  python training/nn/fidelity_eval.py --games 60 --n-states 3000 \
      --mlp-ckpt training/ptcg_dagger_r2.pth
"""
import argparse
import os
import random
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAINING_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_TRAINING_DIR)
for p in (_HERE, _TRAINING_DIR, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from harness import run_matches  # noqa: E402
from seq_collect import extract_decisions  # noqa: E402
from encode import encode_sample, MAX_ACTIONS  # noqa: E402
from encode_seq import encode_sample_seq, PHI4_DIM  # noqa: E402
from model import PTCGNet  # noqa: E402
from model_seq import SeqPTCGNet  # noqa: E402

MAIN = os.path.join(_REPO_ROOT, "main.py")


def collect_fresh_games(n_games, opponent=None, workers=None):
    opponent = opponent or MAIN
    results = run_matches(MAIN, opponent, n_games, workers=workers,
                           keep_steps=True, progress=False)
    games = []
    for r in results:
        if "error" in r or "steps" not in r:
            continue
        dec0 = extract_decisions(r["steps"], seat=0)
        if dec0:
            games.append(dec0)
        if os.path.abspath(opponent) == os.path.abspath(MAIN):
            dec1 = extract_decisions(r["steps"], seat=1)
            if dec1:
                games.append(dec1)
    return games


def _batch_state_tensors(enc):
    board_ids = torch.tensor(enc["board_ids"], dtype=torch.long).unsqueeze(0)
    h = enc["hand_ids"][:20]
    hand_ids = torch.zeros(1, 20, dtype=torch.long)
    hand_ids[0, :len(h)] = torch.tensor(h, dtype=torch.long)
    dcd = enc["discard_ids"][:20]
    discard_ids = torch.zeros(1, 20, dtype=torch.long)
    discard_ids[0, :len(dcd)] = torch.tensor(dcd, dtype=torch.long)
    numeric = torch.tensor(enc["numeric"], dtype=torch.float).unsqueeze(0)
    n = enc["n_actions"]
    action_type = torch.zeros(1, MAX_ACTIONS, dtype=torch.long)
    action_card = torch.zeros(1, MAX_ACTIONS, dtype=torch.long)
    action_attack = torch.zeros(1, MAX_ACTIONS, dtype=torch.long)
    action_numeric = torch.zeros(1, MAX_ACTIONS, 4, dtype=torch.float)
    action_mask = torch.zeros(1, MAX_ACTIONS, dtype=torch.float)
    for j, a in enumerate(enc["actions"][:MAX_ACTIONS]):
        action_type[0, j] = a["type"]
        action_card[0, j] = a["card_id"]
        action_attack[0, j] = a["attack_id"]
        action_numeric[0, j] = torch.tensor(a["numeric"], dtype=torch.float)
        action_mask[0, j] = 1.0
    return board_ids, hand_ids, discard_ids, numeric, action_type, action_card, action_attack, action_numeric, action_mask, n


def eval_mlp(ckpt_path, games, sample_idx):
    net = PTCGNet()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    net.load_state_dict(sd)
    net.eval()

    correct = total = 0
    with torch.no_grad():
        for gi, pos in sample_idx:
            d = games[gi][pos]
            sel = d["obs"].get("select")
            enc = encode_sample(d["obs"], sel)
            label = d["action"][0] if d["action"] else 0
            label = min(label, enc["n_actions"] - 1) if enc["n_actions"] else 0
            tensors = _batch_state_tensors(enc)
            logits, _ = net(*tensors[:-1])
            pred = logits[0].argmax().item()
            correct += int(pred == label)
            total += 1
    return correct, total


def eval_seq(ckpt_path, games, sample_idx):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net = SeqPTCGNet(n_layers=ckpt.get("n_layers", 2),
                      use_history=ckpt.get("use_history", True),
                      use_phi4=ckpt.get("use_phi4", True))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    # group requested (game, pos) samples by game so each game gets ONE
    # full-sequence causal forward pass (matches inference-time accumulation)
    by_game = {}
    for gi, pos in sample_idx:
        by_game.setdefault(gi, []).append(pos)

    correct = total = 0
    with torch.no_grad():
        for gi, positions in by_game.items():
            decs = games[gi]
            T = len(decs)
            board_ids = torch.zeros(1, T, 13, dtype=torch.long)
            hand_ids = torch.zeros(1, T, 20, dtype=torch.long)
            discard_ids = torch.zeros(1, T, 20, dtype=torch.long)
            numeric = torch.zeros(1, T, len(encode_sample_seq(decs[0]["obs"], decs[0]["obs"].get("select"))["numeric"]), dtype=torch.float)
            phi4 = torch.zeros(1, T, PHI4_DIM, dtype=torch.float)
            action_type = torch.zeros(1, T, MAX_ACTIONS, dtype=torch.long)
            action_card = torch.zeros(1, T, MAX_ACTIONS, dtype=torch.long)
            action_attack = torch.zeros(1, T, MAX_ACTIONS, dtype=torch.long)
            action_numeric = torch.zeros(1, T, MAX_ACTIONS, 4, dtype=torch.float)
            action_mask = torch.zeros(1, T, MAX_ACTIONS, dtype=torch.float)
            step_mask = torch.ones(1, T, dtype=torch.float)
            labels = []
            for t, d in enumerate(decs):
                sel = d["obs"].get("select")
                enc = encode_sample_seq(d["obs"], sel)
                n = enc["n_actions"]
                label = d["action"][0] if d["action"] else 0
                label = min(label, n - 1) if n else 0
                labels.append(label)
                board_ids[0, t] = torch.tensor(enc["board_ids"], dtype=torch.long)
                h = enc["hand_ids"][:20]
                hand_ids[0, t, :len(h)] = torch.tensor(h, dtype=torch.long)
                dcd = enc["discard_ids"][:20]
                discard_ids[0, t, :len(dcd)] = torch.tensor(dcd, dtype=torch.long)
                numeric[0, t] = torch.tensor(enc["numeric"], dtype=torch.float)
                phi4[0, t] = torch.tensor(enc["phi4"], dtype=torch.float)
                for j, a in enumerate(enc["actions"][:MAX_ACTIONS]):
                    action_type[0, t, j] = a["type"]
                    action_card[0, t, j] = a["card_id"]
                    action_attack[0, t, j] = a["attack_id"]
                    action_numeric[0, t, j] = torch.tensor(a["numeric"], dtype=torch.float)
                    action_mask[0, t, j] = 1.0

            logits, _ = net(board_ids, hand_ids, discard_ids, numeric,
                             action_type, action_card, action_attack, action_numeric,
                             action_mask, phi4=phi4, step_mask=step_mask)
            for pos in positions:
                pred = logits[0, pos].argmax().item()
                correct += int(pred == labels[pos])
                total += 1
    return correct, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--n-states", type=int, default=3000)
    ap.add_argument("--opponent", default=None, help="default: mirror (MAIN vs MAIN)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--mlp-ckpt", default=None)
    ap.add_argument("--seq-ckpt", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.mlp_ckpt and not args.seq_ckpt:
        raise SystemExit("pass --mlp-ckpt or --seq-ckpt")

    print(f"collecting {args.games} fresh deployment-realistic games from current teacher...")
    games = collect_fresh_games(args.games, opponent=args.opponent, workers=args.workers)
    print(f"collected {len(games)} sequences, "
          f"{sum(len(g) for g in games)} total decisions")

    rng = random.Random(args.seed)
    pool = [(gi, pos) for gi, g in enumerate(games) for pos in range(len(g))]
    n = min(args.n_states, len(pool))
    sample_idx = rng.sample(pool, n)
    print(f"sampled {n} states")

    if args.mlp_ckpt:
        correct, total = eval_mlp(args.mlp_ckpt, games, sample_idx)
        print(f"[MLP {args.mlp_ckpt}] fidelity = {correct}/{total} = {correct/max(total,1):.4f}")
    if args.seq_ckpt:
        correct, total = eval_seq(args.seq_ckpt, games, sample_idx)
        print(f"[SEQ {args.seq_ckpt}] fidelity = {correct}/{total} = {correct/max(total,1):.4f}")


if __name__ == "__main__":
    main()
