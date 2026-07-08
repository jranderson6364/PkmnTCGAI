"""Replay-trained value model (2026-07-07 training-methods session, item
"replay value" — see docs/report-log.md).

Train a win-probability head DIRECTLY on real ladder replays instead of any
self-play corpus. Motivation: the day's converged finding is that the leaf
VALUE signal is the binding constraint on both the search family and the
value-training family, and every self-play corpus so far has been the weak
link (295W/5L imbalance). The replay pool is naturally ~47% win-balanced,
is exactly the distribution the gate measures, and needs no collection
compute.

Anti-saturation head (plan item 2, minimal correct form): outcomes here are
pure ±1, so the categorical/two-hot fix reduces to a LOGIT head trained
with binary cross-entropy — no tanh+MSE saturation (the AWR closure's named
root cause). Backbone: PTCGNet warm-started from --init-ckpt; the V head is
train_iql.py's IQL wrapper head (linear on trunk, no squash).

Methodology guard: the replay pool IS the project's standard gate corpus,
so games are split 70/30 by FILE (seeded); all reported numbers are
holdout-only, and Φ v2 is evaluated on the SAME holdout files for an
apples-to-apples bar. Running dmc_replay_gate.py on the full corpus with
this checkpoint would be train-set-contaminated — don't quote it.

RV-2 residual mode (2026-07-07 pre-registration, docs/report-log.md):
`--target resid` trains on `outcome − Φv2(s)` with a two-hot categorical
head (25 atoms on [−6, 6]) and deploys `V(s) = Φv2(s) + resid(s)` — the
control-variate fix for the Φ-shaping failure (the shaped quantity is only
the TARGET; adding Φ back at deployment restores cross-state sign
comparability, which folding Φ into a plain regression target destroyed —
see the autopsy entry). Per-epoch checkpoints are saved to `<out>.epN` so
no epoch's weights get overwritten (the incumbent run lost epoch 0's).

Usage:
  python training/nn/replay_value.py --init-ckpt training/ptcg_dmc_p0_v2_n1_richenc_v2.pth \
      --out training/ptcg_rv_r1.pth [--epochs 3] [--both-seats] [--target resid]
"""
import argparse
import glob
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

from dataset import collate  # noqa: E402
from encode import encode_sample  # noqa: E402
from model import PTCGNet  # noqa: E402
from train_iql import IQL  # noqa: E402
from phi_baseline import our_seat, bootstrap_ci, report, extract_rows, phi_v2  # noqa: E402


def extract_encoded(path, both_seats=True, need_phi=False):
    """Per-file: list of (enc, outcome, turn, phi) rows, pre-encoded so raw
    obs dicts don't have to stay in RAM across epochs. Uses BOTH players'
    perspectives when available (each side's observation + its own reward) —
    legit augmentation: the value question is seat-symmetric. phi is
    Φv2(cur, seat) when need_phi (rows where Φv2 fails are skipped), else
    None."""
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return []
    you = our_seat(d.get("info", {}))
    if you is None:
        return []
    rewards = d.get("rewards")
    if not rewards or len(rewards) != 2:
        return []
    seats = (you, 1 - you) if both_seats else (you,)
    rows = []
    for step in d.get("steps", []):
        for seat in seats:
            if rewards[seat] not in (1, -1):
                continue
            rec = step[seat] if len(step) > seat else None
            obs = rec.get("observation") if rec else None
            if not obs:
                continue
            cur = obs.get("current") or {}
            if cur.get("yourIndex") != seat:
                continue
            sel = obs.get("select")
            if not sel or not sel.get("option"):
                continue
            try:
                enc = encode_sample(obs, sel)
            except Exception:
                continue
            pv = None
            if need_phi:
                try:
                    pv = phi_v2(cur, seat)
                except Exception:
                    pv = None
                if pv is None:
                    continue
            rows.append((enc, float(rewards[seat]), cur.get("turn") or 0, pv))
    return rows


def batchify(rows):
    return collate([(enc, 0, outcome, None, None, []) for enc, outcome, _, _ in rows])


class TwoHotIQL(IQL):
    """IQL trunk with a two-hot categorical value head (anti-saturation,
    MuZero-family standard). v() returns the distribution's expectation."""

    def __init__(self, base, n_atoms=25, vmin=-6.0, vmax=6.0):
        super().__init__(base)
        self.register_buffer("support", torch.linspace(vmin, vmax, n_atoms))
        self.v_head2 = torch.nn.Linear(256, n_atoms)

    def v_logits(self, b):
        return self.v_head2(self._trunk(b))

    def v(self, b):
        probs = torch.softmax(self.v_logits(b), dim=-1)
        return (probs * self.support).sum(-1)


def two_hot(y, support):
    """Soft two-hot encoding of scalar targets onto the support atoms."""
    y = y.clamp(float(support[0]), float(support[-1]))
    idx = torch.searchsorted(support, y, right=True).clamp(1, len(support) - 1)
    lo, hi = support[idx - 1], support[idx]
    w_hi = ((y - lo) / (hi - lo)).clamp(0.0, 1.0)
    t = torch.zeros(len(y), len(support))
    t.scatter_(1, (idx - 1).unsqueeze(1), (1.0 - w_hi).unsqueeze(1))
    t.scatter_(1, idx.unsqueeze(1), w_hi.unsqueeze(1))
    return t


def eval_holdout(model, holdout_games, add_phi=False):
    """(v, outcome, turn) rows per game — same shape phi_baseline.report and
    bootstrap_ci consume. add_phi=True deploys V = Φv2(s) + resid(s)."""
    model.eval()
    game_rows = []
    with torch.no_grad():
        for rows in holdout_games:
            out = []
            for i in range(0, len(rows), 512):
                chunk = rows[i:i + 512]
                b = batchify(chunk)
                v = model.v(b)
                out.extend(
                    (float(vi) + (r[3] if add_phi else 0.0), r[1], r[2])
                    for vi, r in zip(v, chunk))
            game_rows.append(out)
    model.train()
    return game_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays-dir", default=os.path.join(_REPO_ROOT, "replays", "bulk"))
    ap.add_argument("--extra-dirs", default="replays/v28,replays/v26remake",
                    help="comma-separated extra replay dirs, train-only (not "
                         "part of the standard gate corpus)")
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--both-seats", action="store_true", default=True)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--target", default="outcome", choices=["outcome", "resid"],
                    help="outcome = BCE logit head on win/loss (original); "
                         "resid = two-hot head on outcome − Φv2(s), deployed "
                         "as V = Φv2(s) + resid(s)")
    args = ap.parse_args()
    resid = args.target == "resid"
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    bulk = sorted(glob.glob(os.path.join(args.replays_dir, "*.json")))
    if args.max_games:
        bulk = bulk[: args.max_games]
    rng.shuffle(bulk)
    n_hold = int(len(bulk) * args.holdout_frac)
    holdout_paths, train_paths = bulk[:n_hold], bulk[n_hold:]
    for extra in (args.extra_dirs or "").split(","):
        extra = extra.strip()
        if extra:
            train_paths.extend(glob.glob(os.path.join(_REPO_ROOT, extra, "*.json")))

    print(f"train files={len(train_paths)} holdout files={len(holdout_paths)}")
    train_rows = []
    for p in train_paths:
        train_rows.extend(extract_encoded(p, both_seats=args.both_seats, need_phi=resid))
    holdout_games = []
    for p in holdout_paths:
        rows = extract_encoded(p, both_seats=False,  # holdout = our seat only,
                               need_phi=resid)       # matching the standard gate
        if rows:
            holdout_games.append(rows)
    n_pos = sum(1 for _, o, _, _ in train_rows if o > 0)
    print(f"train rows={len(train_rows)} (win-labeled frac={n_pos/max(len(train_rows),1):.3f}) "
          f"holdout games={len(holdout_games)}")

    base = PTCGNet()
    state = torch.load(args.init_ckpt, map_location="cpu")
    base.load_state_dict(state.get("model", state), strict=False)
    if resid:
        model = TwoHotIQL(base)
    else:
        model = IQL(base)  # reuse its un-squashed v_head on the trunk
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    order = list(range(len(train_rows)))
    for epoch in range(args.epochs):
        rng.shuffle(order)
        tot, seen = 0.0, 0
        for i in range(0, len(order), args.batch):
            chunk = [train_rows[j] for j in order[i:i + args.batch]]
            b = batchify(chunk)
            if resid:
                y = torch.tensor([o - pv for _, o, _, pv in chunk], dtype=torch.float)
                logits = model.v_logits(b)
                t = two_hot(y, model.support)
                loss = -(t * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            else:
                target = torch.tensor([(o + 1) / 2 for _, o, _, _ in chunk], dtype=torch.float)
                logit = model.v(b)
                loss = F.binary_cross_entropy_with_logits(logit, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach())
            seen += 1
        print(f"epoch {epoch}: loss={tot/max(seen,1):.4f}", flush=True)

        game_rows = eval_holdout(model, holdout_games, add_phi=resid)
        report(f"HOLDOUT ALL (epoch {epoch})", game_rows)
        for label, lo_t, hi_t in (("EARLY", 0, 4), ("MID", 5, 10), ("LATE", 11, 10**9)):
            seg = [[r for r in g if lo_t <= r[2] <= hi_t] for g in game_rows]
            seg = [g for g in seg if g]
            report(f"HOLDOUT {label} (epoch {epoch})", seg)
        torch.save({"model": model.state_dict(), "target": args.target},
                   f"{args.out}.ep{epoch}")
        torch.save({"model": model.state_dict(), "target": args.target}, args.out)
        torch.save({"model": model.base.state_dict()}, args.out + ".base")

    # apples-to-apples bar: Phi v2 on the SAME holdout files
    phi_games = [g for g in (extract_rows(p, phi_fn=phi_v2) for p in holdout_paths) if g]
    report("PHI-V2 on same holdout (the bar)", phi_games)
    print(f"saved {args.out} (full, with v_head) and {args.out}.base (PTCGNet-only)")


if __name__ == "__main__":
    main()
