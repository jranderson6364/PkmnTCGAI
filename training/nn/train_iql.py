"""Offline RL on the existing logged corpora via Implicit Q-Learning
(Kostrikov et al. 2021) — 2026-07-07 training-methods plan item 3.

Why this family (untried here): every prior training run is imitation
(BC/DAgger, plateaus at teacher parity by construction) or Monte-Carlo
value regression (DMC/AWR, whose scalar targets saturate). IQL learns
Q(s,a) by bootstrapping through an expectile-regressed V(s) — TD stitching
across trajectories can in principle exceed the behavior policy, and the
expectile loss is asymmetric (not a saturating MSE-to-±1 fit). In-domain
precedent: Metamon (arXiv 2504.04395) reached human level in competitive
Pokémon from replay corpora via exactly this BC → offline-RL progression.

Design choices, kept minimal:
  - Q(s,a) = the existing per-action logits head (DMC convention), warm-
    started from --init-ckpt. Deployment therefore needs NOTHING new:
    export is a plain PTCGNet state_dict, played by dmc_agent.py argmax.
  - V(s) = a new linear head on the trunk, trained with expectile loss
    tau (default 0.7) against a slow EMA target copy of Q.
  - Rewards: sparse terminal outcome (+1/-1/0), gamma=1 — episodes are
    short (docs/game-nature.md §8), no discounting needed.
  - Transitions: corpora store per-game contiguous decision lists (see
    bc_collect/selfplay_collect); game boundaries are re-derived by turn
    reset. Games whose samples lack a raw `outcome` field are skipped
    (shaped value_targets are NOT valid rewards here).

Usage:
  python training/nn/train_iql.py --data "training/bc_data_v25c*.pkl*" \
      --init-ckpt training/ptcg_bc_v2.pth --out training/ptcg_iql_r1.pth
Gate afterwards (pre-registered):
  dmc_replay_gate.py --ckpt <out> (qmax convention — logits ARE Q here)
  + 400-game A/B: dmc_agent.py argmax vs training/baselines/v25c.py.
"""
import argparse
import copy
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from dataset import load_shards, BCDataset, collate  # noqa: E402
from model import PTCGNet  # noqa: E402


def build_transitions(samples):
    """Group game-contiguous samples into (i, next_i_or_None, reward, done).
    Boundary = turn number strictly decreasing vs the previous sample (turn
    is monotone within a game; a fresh game restarts at 0/1)."""
    transitions, game = [], []
    prev_turn = None

    def flush(game):
        if not game:
            return
        outcome = samples[game[-1]].get("outcome")
        if outcome is None:  # shaped-target-only corpus: not usable as reward
            return
        for j, idx in enumerate(game):
            last = j == len(game) - 1
            transitions.append((idx, None if last else game[j + 1],
                                float(outcome) if last else 0.0, last))

    for i, d in enumerate(samples):
        turn = ((d.get("obs") or {}).get("current") or {}).get("turn") or 0
        if prev_turn is not None and turn < prev_turn:
            flush(game)
            game = []
        game.append(i)
        prev_turn = turn
    flush(game)
    return transitions


class IQL(nn.Module):
    def __init__(self, base: PTCGNet):
        super().__init__()
        self.base = base
        self.v_head = nn.Linear(256, 1)

    def _trunk(self, b):
        base = self.base
        board_ctx = base.board_transformer(base.card_embed(b["board_ids"]))
        trunk_in = torch.cat([
            board_ctx.mean(dim=1),
            base.hand_bag(b["hand_ids"]),
            base.discard_bag(b["discard_ids"]),
            base.numeric_proj(b["numeric"]),
        ], dim=-1)
        return base.trunk(trunk_in)

    def q(self, b):
        logits, _ = self.base(
            b["board_ids"], b["hand_ids"], b["discard_ids"], b["numeric"],
            b["action_type"], b["action_card"], b["action_attack"],
            b["action_numeric"], b["action_mask"])
        return logits

    def v(self, b):
        return self.v_head(self._trunk(b)).squeeze(-1)


def expectile_loss(diff, tau):
    weight = torch.where(diff > 0, tau, 1 - tau)
    return (weight * diff.pow(2)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="shard glob(s), comma-separated")
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap samples read (RAM)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.7, help="expectile")
    ap.add_argument("--ema", type=float, default=0.995, help="target-net decay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seat", type=int, default=None, choices=(0, 1),
                    help="diagnostic: keep only games our side played from this "
                         "seat (obs current.yourIndex). Two independent trainers "
                         "(train_sp.py 2026-07-07, this one) both went "
                         "anti-predictive on the full mcts_p2_r3 corpus; if a "
                         "single-seat slice trains clean, the defect is "
                         "seat-dependent in the corpus/encode path, not in "
                         "either trainer.")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    samples = load_shards(args.data, limit=args.limit)
    if args.seat is not None:
        samples = [d for d in samples
                   if (((d.get("obs") or {}).get("current") or {})
                       .get("yourIndex")) == args.seat]
        print(f"seat filter {args.seat}: {len(samples)} samples kept")
    transitions = build_transitions(samples)
    n_terminal = sum(1 for t in transitions if t[3])
    print(f"samples={len(samples)} transitions={len(transitions)} "
          f"games(with outcome)={n_terminal}")
    if not transitions:
        sys.exit("no usable transitions (corpus lacks raw `outcome` fields?)")

    ds = BCDataset(samples)
    base = PTCGNet()
    state = torch.load(args.init_ckpt, map_location="cpu")
    base.load_state_dict(state.get("model", state), strict=False)
    model = IQL(base)
    target = copy.deepcopy(model)
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def batch_of(indices):
        return collate([ds[i] for i in indices])

    steps_per_epoch = max(1, len(transitions) // args.batch)
    for epoch in range(args.epochs):
        random.shuffle(transitions)
        tot_q, tot_v, seen = 0.0, 0.0, 0
        for step in range(steps_per_epoch):
            chunk = transitions[step * args.batch:(step + 1) * args.batch]
            if not chunk:
                break
            idx_t = [c[0] for c in chunk]
            rewards = torch.tensor([c[2] for c in chunk], dtype=torch.float)
            done = torch.tensor([c[3] for c in chunk], dtype=torch.float)
            b_t = batch_of(idx_t)
            labels = b_t["labels"]

            # next-state V (terminal rows get a dummy state, masked by done)
            idx_t1 = [c[1] if c[1] is not None else c[0] for c in chunk]
            with torch.no_grad():
                b_t1 = batch_of(idx_t1)
                v_next = model.v(b_t1)
                q_tgt = target.q(b_t).gather(1, labels.unsqueeze(1)).squeeze(1)

            # V <- expectile of target-Q  (the "implicit" max)
            v_t = model.v(b_t)
            v_loss = expectile_loss(q_tgt - v_t, args.tau)
            # Q <- r + gamma * V(s')   (gamma = 1, sparse terminal reward)
            q_pred = model.q(b_t).gather(1, labels.unsqueeze(1)).squeeze(1)
            q_loss = F.mse_loss(q_pred, rewards + (1 - done) * v_next)

            loss = q_loss + v_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                for pt, p in zip(target.parameters(), model.parameters()):
                    pt.mul_(args.ema).add_(p, alpha=1 - args.ema)
            tot_q += float(q_loss.detach())
            tot_v += float(v_loss.detach())
            seen += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{steps_per_epoch} "
                      f"q_loss={tot_q/seen:.4f} v_loss={tot_v/seen:.4f}", flush=True)
        print(f"epoch {epoch} done: q_loss={tot_q/max(seen,1):.4f} "
              f"v_loss={tot_v/max(seen,1):.4f}", flush=True)
        # deployment export: plain PTCGNet weights, dmc_agent.py-compatible
        torch.save({"model": model.base.state_dict()}, args.out)
        torch.save({"model": model.state_dict()}, args.out + ".iql_full")
        print(f"saved {args.out} (+ .iql_full with V head/target)", flush=True)


if __name__ == "__main__":
    main()
