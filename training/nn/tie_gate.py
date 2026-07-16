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
    ap.add_argument("--mirror", type=int, default=400,
                    help="0 skips the mirror block (anchor-centric follow-up)")
    ap.add_argument("--anchor", type=int, default=200)
    ap.add_argument("--anchors", default="lucario,abomasnow")
    ap.add_argument("--csv", default=os.path.join(_REPO_ROOT, "training",
                                                  "tie_gate_games.csv"))
    args = ap.parse_args()

    tie = os.path.join(_HERE, "tie_agent.py")
    v29d = os.path.join(_REPO_ROOT, "main.py")
    all_anchors = {"lucario": os.path.join(_REPO_ROOT, "opponents", "lucario_agent.py"),
                   "abomasnow": os.path.join(_REPO_ROOT, "opponents", "abomasnow_agent.py"),
                   "starmie": os.path.join(_REPO_ROOT, "opponents", "starmie_agent.py")}
    anchors = {k: all_anchors[k] for k in args.anchors.split(",") if k}

    with open(args.csv, "w", newline="") as f:
        csv_w = csv.writer(f)
        csv_w.writerow(["gate", "label", "seat", "outcome", ""])

        g1 = None
        if args.mirror:
            p, n = _run_block(tie, v29d, args.mirror, "tie vs v29d (mirror)", csv_w)
            se = math.sqrt(p * (1 - p) / n)
            lo = p - 1.96 * se
            g1 = lo > 0.5
            print(f"GATE A (mirror): {p:.3f} [{lo:.3f}, {p + 1.96*se:.3f}] n={n} "
                  f"-> {'PASS' if g1 else 'FAIL'} (bar: CI lower bound > 0.50)")

        g2 = True
        tw = tn = cw = cn = 0.0
        for anch, opath in anchors.items():
            ph, nh = _run_block(tie, opath, args.anchor, f"tie vs {anch}", csv_w)
            pc, nc = _run_block(v29d, opath, args.anchor, f"v29d vs {anch}", csv_w)
            tw += ph * nh
            tn += nh
            cw += pc * nc
            cn += nc
            diff = ph - pc
            se = math.sqrt(ph * (1 - ph) / nh + pc * (1 - pc) / nc)
            hi = diff + 1.96 * se
            ok = hi >= 0
            g2 = g2 and ok
            print(f"GATE B {anch}: tie {ph:.3f} vs v29d {pc:.3f}, diff "
                  f"{diff:+.3f} [{diff - 1.96*se:+.3f}, {hi:+.3f}] -> "
                  f"{'OK' if ok else 'REGRESSION'}")
        pt, pc_ = tw / max(1, tn), cw / max(1, cn)
        pooled = pt - pc_
        se = math.sqrt(pt * (1 - pt) / max(1, tn) + pc_ * (1 - pc_) / max(1, cn))
        plo, phi = pooled - 1.96 * se, pooled + 1.96 * se
        pooled_pass = plo > 0
        print(f"GATE B pooled: tie {pt:.3f} vs v29d {pc_:.3f}, diff "
              f"{pooled:+.3f} [{plo:+.3f}, {phi:+.3f}] "
              f"-> {'SEPARABLE' if pooled_pass else 'not separable'}")
        print(f"GATE B {'PASS' if g2 else 'FAIL'}")
        print(f"SUMMARY: gateA={g1} gateB={g2} pooled_positive={pooled_pass} "
              f"adopt={'YES' if (g1 if g1 is not None else pooled_pass and g2) else 'NO'}")


if __name__ == "__main__":
    main()
