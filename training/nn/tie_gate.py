"""Pre-registered gate for the learned tie-breaker (tie_agent.py).

Bars (report-log 2026-07-16 pre-registration):
  1. Mirror A/B vs plain v29d, n=400, seats alternated: ADOPT only if the
     tie-breaker's win-rate 95% CI lower bound > 0.50 (CI-separable
     improvement over array-order tie-breaking).
  2. Anchor non-regression: n=200/anchor vs lucario+abomasnow, same-day
     plain-v29d reads, not CI-separably below on any anchor.

Usage: python training/nn/tie_gate.py [--mirror 400] [--anchor 200]
"""
import argparse
import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_REPO_ROOT, "training"), _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from regime_gate import _run_block  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", type=int, default=400)
    ap.add_argument("--anchor", type=int, default=200)
    ap.add_argument("--csv", default=os.path.join(_REPO_ROOT, "training",
                                                  "tie_gate_games.csv"))
    args = ap.parse_args()

    tie = os.path.join(_HERE, "tie_agent.py")
    v29d = os.path.join(_REPO_ROOT, "main.py")
    anchors = {"lucario": os.path.join(_REPO_ROOT, "opponents", "lucario_agent.py"),
               "abomasnow": os.path.join(_REPO_ROOT, "opponents", "abomasnow_agent.py")}

    with open(args.csv, "w", newline="") as f:
        csv_w = csv.writer(f)
        csv_w.writerow(["gate", "label", "seat", "outcome", ""])

        p, n = _run_block(tie, v29d, args.mirror, "tie vs v29d (mirror)", csv_w)
        se = math.sqrt(p * (1 - p) / n)
        lo = p - 1.96 * se
        g1 = lo > 0.5
        print(f"GATE A (mirror): {p:.3f} [{lo:.3f}, {p + 1.96*se:.3f}] n={n} "
              f"-> {'PASS' if g1 else 'FAIL'} (bar: CI lower bound > 0.50)")

        g2 = True
        for anch, opath in anchors.items():
            ph, nh = _run_block(tie, opath, args.anchor, f"tie vs {anch}", csv_w)
            pc, nc = _run_block(v29d, opath, args.anchor, f"v29d vs {anch}", csv_w)
            diff = ph - pc
            se = math.sqrt(ph * (1 - ph) / nh + pc * (1 - pc) / nc)
            hi = diff + 1.96 * se
            ok = hi >= 0
            g2 = g2 and ok
            print(f"GATE B {anch}: tie {ph:.3f} vs v29d {pc:.3f}, diff "
                  f"{diff:+.3f} [{diff - 1.96*se:+.3f}, {hi:+.3f}] -> "
                  f"{'OK' if ok else 'REGRESSION'}")
        print(f"GATE B {'PASS' if g2 else 'FAIL'}")
        print(f"SUMMARY: gateA={g1} gateB={g2} "
              f"adopt={'YES' if g1 and g2 else 'NO'}")


if __name__ == "__main__":
    main()
