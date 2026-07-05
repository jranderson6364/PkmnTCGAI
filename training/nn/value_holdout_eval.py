"""Evaluator gate (step 4 of the diverse-data value pre-registration,
docs/report-log.md 2026-07-04): held-out outcome prediction on REAL ladder
games the net never trained on. Reports sign-accuracy (does sign(v) predict
the game's final outcome), saturation fraction, and a by-game-phase
breakdown (early states should be near-coinflip — honest — while late
states should approach high accuracy if the head carries real signal).

Usage:
  python training/nn/value_holdout_eval.py --ckpt ../ptcg_value_v2.pth [--max-samples 4000]
"""
import argparse
import gzip
import os
import pickle
import random
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO_ROOT)

from net_common import load_model, value_estimate  # noqa: E402

HOLDOUT = os.path.join(_REPO_ROOT, "training", "vd_ladder_holdout.pkl.gz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-samples", type=int, default=4000)
    args = ap.parse_args()

    with gzip.open(HOLDOUT, "rb") as f:
        samples = pickle.load(f)
    random.seed(11)
    if len(samples) > args.max_samples:
        samples = random.sample(samples, args.max_samples)

    model = load_model(args.ckpt)
    rows = []  # (v, outcome, turn)
    errors = 0
    for s in samples:
        obs, sel, outcome = s["obs"], s["obs"].get("select"), s["outcome"]
        if not sel or not sel.get("option"):
            continue
        turn = ((obs.get("current") or {}).get("turn")) or 0
        try:
            v = value_estimate(model, obs, sel)
        except Exception:
            errors += 1
            continue
        rows.append((v, outcome, turn))

    def report(label, rs):
        if not rs:
            print(f"{label}: no samples")
            return
        n = len(rs)
        acc = sum(1 for v, o, _ in rs if (v >= 0) == (o >= 0)) / n
        vals = [v for v, _, _ in rs]
        extreme = sum(1 for v in vals if abs(v) > 0.95) / n
        print(f"{label}: n={n} sign_acc={acc:.3f} mean_v={statistics.mean(vals):.3f} "
              f"std_v={statistics.pstdev(vals):.3f} frac_extreme={extreme:.1%}")

    print(f"ckpt={args.ckpt} holdout_samples={len(rows)} errors={errors}")
    report("ALL", rows)
    report("EARLY (turn<=4)", [r for r in rows if r[2] <= 4])
    report("MID (5<=turn<=10)", [r for r in rows if 5 <= r[2] <= 10])
    report("LATE (turn>=11)", [r for r in rows if r[2] >= 11])


if __name__ == "__main__":
    main()
