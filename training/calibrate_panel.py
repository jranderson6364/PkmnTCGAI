"""Fit offline strength -> live publicScore, using public agents of known rating.

This is the thing Design Principle #1 said we could never have. Every reference
anchor this project owns reads <=6% against our champion, so offline results were
both undiscriminating AND uncalibrated: we knew offline overrates, but not by how
much or how consistently. Four public competition agents with published ladder
scores (739.7 - 933.8) plus our own submissions with known scores give a panel
that spans ~420 points of real rating and can be played locally.

Method: every agent is measured head-to-head against a common reference (v29d,
publicScore 673.5), the win rate is converted to an Elo difference, and
publicScore is regressed on that. Reports R^2, the fitted slope in publicScore
per offline Elo point, and a residual scale -- which is the honest width of any
"this offline gain is worth N ladder points" claim.

Run:  python training/calibrate_panel.py
"""
from __future__ import annotations

import math

REFERENCE = "v29d (main.py)"
REF_SCORE = 673.5

# name -> (win rate vs the reference, n, publicScore or None if unknown)
# All measured 2026-07-23, seats alternated, 0 errors, via training/ab_test.py.
PANEL = {
    "probability_v2 (933.8)":      (0.610, 100, 933.8),
    "advanced_heuristic (796.8)":  (0.640, 100, 796.8),
    "alakazam_v9 (778.2)":         (0.550, 100, 778.2),
    "alakazam_v8 (739.7)":         (0.370, 100, 739.7),
    REFERENCE:                     (0.500, 0,   673.5),
    "v30-exp (637.8)":             (0.517,  60, 637.8),
    "lucario sample (600)":        (0.040, 100, 600.0),
    "abomasnow sample (509.6)":    (0.100, 100, 509.6),
    # candidates with no ladder read yet -> predicted by the fit
    "main_lucario (ours, new)":    (0.130, 100, None),
}


def elo_from_p(p: float) -> float:
    """Elo difference implied by a win rate against the reference."""
    p = min(max(p, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / p - 1.0)


def wilson(p, n, z=1.96):
    if n <= 0:
        return p, p
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    yhat = [a + b * x for x in xs]
    ss_res = sum((y - h) ** 2 for y, h in zip(ys, yhat))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    resid_sd = math.sqrt(ss_res / max(1, n - 2))
    return a, b, r2, resid_sd


def main():
    print(f"Reference: {REFERENCE}, publicScore {REF_SCORE}\n")
    print(f"{'agent':30s} {'wr vs ref':>10s} {'95% CI':>16s} {'offline Elo':>12s} {'publicScore':>12s}")
    print("-" * 86)

    fit_x, fit_y, unknown = [], [], []
    for name, (p, n, score) in PANEL.items():
        e = elo_from_p(p)
        lo, hi = wilson(p, n)
        ci = f"[{lo:.2f},{hi:.2f}]" if n else "     ---"
        print(f"{name:30s} {p:10.3f} {ci:>16s} {e:+12.1f} "
              f"{(f'{score:.1f}' if score is not None else '   ?'):>12s}")
        if score is not None:
            fit_x.append(e)
            fit_y.append(score)
        else:
            unknown.append((name, e, p, n))

    a, b, r2, sd = linfit(fit_x, fit_y)
    print("\n" + "=" * 86)
    print(f"FIT  publicScore = {a:.1f} + {b:.3f} x offlineElo     "
          f"(n={len(fit_x)}, R^2={r2:.3f}, residual SD={sd:.1f})")
    print(f"     100 offline Elo  ~  {100*b:.0f} publicScore points")
    print(f"     +10pp win rate vs the reference near 50%  ~  "
          f"{b*(elo_from_p(0.60)-elo_from_p(0.50)):.0f} publicScore points")

    print("\nPREDICTIONS for agents with no ladder read:")
    for name, e, p, n in unknown:
        pred = a + b * e
        print(f"  {name:30s} offline Elo {e:+8.1f}  ->  predicted publicScore "
              f"{pred:.0f}  (+/- ~{2*sd:.0f} at 2 residual SD)")

    print("\nRESIDUALS (fit minus actual -- large = offline misranks this agent):")
    for name, (p, n, score) in PANEL.items():
        if score is None:
            continue
        r = (a + b * elo_from_p(p)) - score
        flag = "  <-- offline OVERRATES" if r > sd else ("  <-- offline UNDERRATES" if r < -sd else "")
        print(f"  {name:30s} {r:+8.1f}{flag}")


if __name__ == "__main__":
    main()
