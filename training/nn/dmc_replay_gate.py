"""Phase 0 gate (docs/nn-training.md Phase 0 amendment (a)/(e)): sign-accuracy
of a trained DMC checkpoint's Q-head (max_a Q(s,a), the DMC convention -- see
dmc_nstep.py) against the same 1356-game real ladder replay corpus
phi_baseline.py was gated on, using the same game-level bootstrapped CI
methodology so the two numbers are directly comparable. A candidate passes
Phase 0's gate only if this beats phi_baseline's Φ-only numbers (ALL 0.563
[0.543, 0.583], LATE 0.606 [0.576, 0.635]) by a statistically meaningful
margin -- not the flat, differently-measured 62.5% oracle-critic figure.

Usage:
  python training/nn/dmc_replay_gate.py --ckpt training/ptcg_dmc_r2.pth [--replays-dir replays/bulk] [--max-games N]
"""
import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

from net_common import load_model, value_estimate  # noqa: E402
from dmc_nstep import q_max_value  # noqa: E402
from phi_baseline import our_seat, bootstrap_ci  # noqa: E402


def extract_rows(path, model, value_source="qmax"):
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return []
    info = d.get("info", {})
    you = our_seat(info)
    if you is None:
        return []
    rewards = d.get("rewards")
    if not rewards or len(rewards) != 2 or rewards[you] not in (1, -1):
        return []
    outcome = rewards[you]

    rows = []
    for step in d.get("steps", []):
        if len(step) <= you:
            continue
        rec = step[you]
        obs = rec.get("observation") if rec else None
        if not obs:
            continue
        cur = obs.get("current") or {}
        if cur.get("yourIndex") != you:
            continue
        sel = obs.get("select")
        if not sel or not sel.get("option"):
            continue
        turn = cur.get("turn") or 0
        try:
            v = value_estimate(model, obs, sel) if value_source == "head" else q_max_value(model, obs, sel)
        except Exception:
            continue
        rows.append((v, outcome, turn))
    return rows


def report(label, game_rows):
    flat = [r for g in game_rows for r in g]
    if not flat:
        print(f"{label}: no samples")
        return
    n = len(flat)
    acc = sum(1 for v, o, _ in flat if (v >= 0) == (o >= 0)) / n
    lo, hi = bootstrap_ci(game_rows) if game_rows else (float("nan"), float("nan"))
    print(f"{label}: n_decisions={n} n_games={len(game_rows)} sign_acc={acc:.3f} "
          f"game_level_95%CI=[{lo:.3f}, {hi:.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--replays-dir", default=os.path.join(_REPO_ROOT, "replays", "bulk"))
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--value-source", choices=["qmax", "head"], default="qmax",
                     help="qmax = DMC convention (max_a logits, default); "
                          "head = the model's own value_head output "
                          "(use for train_sp.py-trained checkpoints, whose "
                          "logits are policy preferences, not Q-values)")
    args = ap.parse_args()

    model = load_model(args.ckpt)
    paths = sorted(glob.glob(os.path.join(args.replays_dir, "*.json")))
    if args.max_games:
        paths = paths[: args.max_games]

    all_games, early_games, mid_games, late_games = [], [], [], []
    skipped = 0
    for p in paths:
        rows = extract_rows(p, model, value_source=args.value_source)
        if not rows:
            skipped += 1
            continue
        all_games.append(rows)
        early = [r for r in rows if r[2] <= 4]
        mid = [r for r in rows if 5 <= r[2] <= 10]
        late = [r for r in rows if r[2] >= 11]
        if early:
            early_games.append(early)
        if mid:
            mid_games.append(mid)
        if late:
            late_games.append(late)

    print(f"ckpt={args.ckpt} value_source={args.value_source} "
          f"replay files scanned={len(paths)} usable_games={len(all_games)} skipped={skipped}")
    report("ALL", all_games)
    report("EARLY (turn<=4)", early_games)
    report("MID (5<=turn<=10)", mid_games)
    report("LATE (turn>=11)", late_games)


if __name__ == "__main__":
    main()
