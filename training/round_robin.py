"""Full round-robin among the scored panel agents -> Bradley-Terry gElo ->
recalibrated gElo->publicScore fit, free of the star-topology bias in P1.

P1's calibration measured everything against ONE reference (v29d), which biases
any agent whose matchup vs v29d is unrepresentative -- probability_v2 (933.8)
came out offline-underrated by 153 because our Psychic Alakazam is its worst
matchup. A round-robin fixes that: every agent's strength is estimated from its
record against the whole field, not one opponent.

Agents (all have a published ladder score, so all anchor the calibration):
  main.py                          v29d Alakazam        673.5
  opponents/public/probability_v2  Mega Lucario         933.8
  opponents/public/advanced_heuristic                   796.8
  opponents/public/alakazam_v9                          778.2
  opponents/public/alakazam_v8                          739.7

Run:  python training/round_robin.py --games 100
Writes the pairwise matrix to training/round_robin_matrix.csv so a re-fit does
not need to replay games.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "local_cg"))

from harness import load_agent, play_game

AGENTS = [
    ("v29d",               "main.py",                              673.5),
    ("probability_v2",     "opponents/public/probability_v2.py",   933.8),
    ("advanced_heuristic", "opponents/public/advanced_heuristic.py", 796.8),
    ("alakazam_v9",        "opponents/public/alakazam_v9.py",      778.2),
    ("alakazam_v8",        "opponents/public/alakazam_v8.py",      739.7),
]

MATRIX_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "round_robin_matrix.csv")


def play_pair(a_fn, a_deck, b_fn, b_deck, n):
    """A vs B over n games, seats alternated. Returns (a_wins, b_wins, ties)."""
    aw = bw = t = 0
    for g in range(n):
        if g % 2 == 0:
            r = play_game(a_fn, a_deck, b_fn, b_deck)
            ra, rb = r["rewards"][0], r["rewards"][1]
        else:
            r = play_game(b_fn, b_deck, a_fn, a_deck)
            ra, rb = r["rewards"][1], r["rewards"][0]
        if ra is None or rb is None or ra == rb:
            t += 1
        elif ra > rb:
            aw += 1
        else:
            bw += 1
    return aw, bw, t


def bradley_terry(names, wins, games, iters=1000, tol=1e-10):
    """MM-algorithm BT fit. wins[i][j] = wins of i over j (ties = 0.5 each).
    Returns gElo per name, anchored to mean 0 then scaled to Elo (400/ln10)."""
    idx = {n: k for k, n in enumerate(names)}
    m = len(names)
    p = [1.0] * m
    for _ in range(iters):
        newp = [0.0] * m
        for i in range(m):
            num = sum(wins[i][j] for j in range(m) if j != i)
            den = 0.0
            for j in range(m):
                if j == i:
                    continue
                nij = games[i][j]
                if nij > 0:
                    den += nij / (p[i] + p[j])
            newp[i] = num / den if den > 0 else p[i]
        s = sum(newp) / m
        newp = [x / s for x in newp]
        if max(abs(newp[k] - p[k]) for k in range(m)) < tol:
            p = newp
            break
        p = newp
    # log-strength -> Elo, centered
    theta = [math.log(x) for x in p]
    mu = sum(theta) / m
    scale = 400.0 / math.log(10.0)
    return {names[k]: (theta[k] - mu) * scale for k in range(m)}


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
    sd = math.sqrt(ss_res / max(1, n - 2))
    return a, b, r2, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--refit-only", action="store_true",
                    help="skip games; re-fit from round_robin_matrix.csv")
    args = ap.parse_args()

    names = [a[0] for a in AGENTS]
    score = {a[0]: a[2] for a in AGENTS}
    m = len(names)
    wins = [[0.0] * m for _ in range(m)]
    games = [[0] * m for _ in range(m)]

    if args.refit_only and os.path.exists(MATRIX_CSV):
        with open(MATRIX_CSV) as f:
            for row in csv.DictReader(f):
                i, j = names.index(row["a"]), names.index(row["b"])
                aw, bw, t = int(row["a_wins"]), int(row["b_wins"]), int(row["ties"])
                wins[i][j] += aw + 0.5 * t
                wins[j][i] += bw + 0.5 * t
                games[i][j] += aw + bw + t
                games[j][i] += aw + bw + t
    else:
        loaded = {a[0]: load_agent(a[1]) for a in AGENTS}
        rows = []
        for (na, nb) in itertools.combinations(names, 2):
            a_fn, a_deck, _ = loaded[na]
            b_fn, b_deck, _ = loaded[nb]
            aw, bw, t = play_pair(a_fn, a_deck, b_fn, b_deck, args.games)
            i, j = names.index(na), names.index(nb)
            wins[i][j] += aw + 0.5 * t
            wins[j][i] += bw + 0.5 * t
            games[i][j] += aw + bw + t
            games[j][i] += aw + bw + t
            rows.append(dict(a=na, b=nb, a_wins=aw, b_wins=bw, ties=t, n=args.games))
            print(f"  {na:20s} vs {nb:20s}  {aw:3d}-{bw:3d}-{t:2d}  "
                  f"({aw/max(1,aw+bw):.1%})", flush=True)
        with open(MATRIX_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["a", "b", "a_wins", "b_wins", "ties", "n"])
            w.writeheader()
            w.writerows(rows)

    gelo = bradley_terry(names, wins, games)

    print("\n=== round-robin Bradley-Terry gElo ===")
    order = sorted(names, key=lambda n: -gelo[n])
    for n in order:
        rec = sum(wins[names.index(n)]); tot = sum(games[names.index(n)])
        print(f"  {n:20s} gElo {gelo[n]:+7.1f}   overall {rec/max(1,tot):.1%}   publicScore {score[n]:.1f}")

    xs = [gelo[n] for n in names]
    ys = [score[n] for n in names]
    a, b, r2, sd = linfit(xs, ys)
    print("\n=== recalibrated fit (round-robin gElo -> publicScore) ===")
    print(f"  publicScore = {a:.1f} + {b:.3f} x gElo    (n={m}, R^2={r2:.3f}, residual SD={sd:.1f})")
    print(f"  +100 gElo ~ {100*b:.0f} publicScore points")
    print("\n  residuals (fit - actual):")
    for n in order:
        r = (a + b * gelo[n]) - score[n]
        print(f"    {n:20s} {r:+7.1f}")
    print("\n  (compare P1 star-topology fit: R^2=0.537, residual SD=97.0, "
          "probability_v2 residual -152.8)")


if __name__ == "__main__":
    main()
