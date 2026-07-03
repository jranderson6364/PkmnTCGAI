#!/usr/bin/env python3
"""Analytic consistency panel for a 60-card deck (Measurement standard #2).

Computes the hypergeometric numbers pros use to compare deck consistency
(JustInBasil Appendix IV / SixPrizes "Stats on Starts"):

  - deck composition by category (Basic / Stage 1 / Stage 2 / Item / ...)
  - mulligan probability: P(0 Basic Pokemon in the opening 7)
  - P(>=1 copy of a card group among the cards seen by turn N)
  - arbitrary combo odds via --want "label=id:count,id:count[@turn]"

"Cards seen by turn N" = 7 (opening hand) + 1 per draw step. By
exchangeability of a shuffled deck, the seen cards are a uniform random
subset of the 60, so plain (multivariate) hypergeometric is exact.
Mulligan redraws condition the hand on >=1 Basic; per standard practice
that small bias is ignored (noted in output).

Usage:
  python3 tools/deck_math.py deck.csv
  python3 tools/deck_math.py deck.csv --carddata docs/EN_Card_Data.csv \
      --want "candy-line=1079:1,743:1,741:1@3" --turns 1 2 3
"""
import argparse
import csv
import sys
from itertools import product
from math import comb
from pathlib import Path


def read_deck(path):
    ids = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            ids.append(int(line))
    if len(ids) != 60:
        print(f"warning: deck has {len(ids)} cards, expected 60", file=sys.stderr)
    return ids


def read_card_data(path):
    """cardId -> (name, stage/type string). Handles quoted multi-line fields."""
    info = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            if not row or not row[0].isdigit():
                continue
            cid = int(row[0])
            if cid not in info:  # one row per attack; first row suffices
                info[cid] = (row[1], row[4])
    return info


def p_at_least_one(copies, sample, total=60):
    """P(>=1 success) drawing `sample` from `total` with `copies` successes."""
    if copies <= 0:
        return 0.0
    if sample >= total - copies + 1:
        return 1.0
    return 1.0 - comb(total - copies, sample) / comb(total, sample)


def p_zero(copies, sample, total=60):
    return 1.0 - p_at_least_one(copies, sample, total)


def p_combo(groups, sample, total=60):
    """P(>=need_i drawn from every group i) — multivariate hypergeometric.

    groups: list of (copies, need). Sums over the joint counts of each
    group in the sample; the remainder comes from the rest of the deck.
    """
    rest = total - sum(c for c, _ in groups)
    denom = comb(total, sample)
    prob = 0.0
    ranges = [range(need, copies + 1) for copies, need in groups]
    for counts in product(*ranges):
        k_rest = sample - sum(counts)
        if k_rest < 0 or k_rest > rest:
            continue
        ways = comb(rest, k_rest)
        for (copies, _), k in zip(groups, counts):
            ways *= comb(copies, k)
        prob += ways / denom
    return prob


def parse_want(spec):
    """'label=1079:1,743:1@3' -> (label, [(id,need),...], turn or None)"""
    label, _, body = spec.partition('=')
    if not body:
        label, body = spec, spec
    turn = None
    if '@' in body:
        body, t = body.rsplit('@', 1)
        turn = int(t)
    parts = []
    for chunk in body.split(','):
        cid, _, need = chunk.partition(':')
        parts.append((int(cid), int(need) if need else 1))
    return label, parts, turn


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('deck', help='deck file, one card ID per line')
    ap.add_argument('--carddata', default=str(Path(__file__).resolve().parent.parent
                                              / 'docs' / 'EN_Card_Data.csv'))
    ap.add_argument('--turns', type=int, nargs='*', default=[1, 2, 3],
                    help='turn numbers for by-turn-N odds (cards seen = 7+N)')
    ap.add_argument('--want', action='append', default=[],
                    metavar='label=id:count,...[@turn]',
                    help='combo query, e.g. "candy-line=1079:1,743:1,741:1@3"')
    args = ap.parse_args()

    deck = read_deck(args.deck)
    info = read_card_data(args.carddata)
    counts = {}
    for cid in deck:
        counts[cid] = counts.get(cid, 0) + 1

    by_cat = {}
    for cid, n in counts.items():
        _, cat = info.get(cid, (f'#{cid}', 'UNKNOWN'))
        by_cat[cat] = by_cat.get(cat, 0) + n

    basics = sum(n for cid, n in counts.items()
                 if info.get(cid, ('', ''))[1] == 'Basic Pokémon')

    print(f"deck: {args.deck}  ({len(deck)} cards)")
    print("\ncomposition:")
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {n:3d}  {cat}")

    mull = p_zero(basics, 7)
    print(f"\nbasics: {basics}   P(mulligan) = {mull:.1%}")
    print("(by-turn odds treat seen cards as a uniform subset; mulligan-redraw"
          " bias ignored per standard practice)")

    print("\nopening-hand / by-turn odds, P(>=1 copy):")
    header = "  card (copies)".ljust(38) + "open-7" + "".join(
        f"  turn {t}" for t in args.turns)
    print(header)
    for cid, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        name, cat = info.get(cid, (f'#{cid}', '?'))
        row = f"  {name} #{cid} (x{n})".ljust(38) + f"{p_at_least_one(n, 7):6.1%}"
        for t in args.turns:
            row += f"  {p_at_least_one(n, 7 + t):6.1%}"
        print(row)

    if args.want:
        print("\ncombo odds:")
        for spec in args.want:
            label, parts, turn = parse_want(spec)
            groups = []
            ok = True
            for cid, need in parts:
                have = counts.get(cid, 0)
                if have < need:
                    print(f"  {label}: needs {need}x #{cid}, deck has {have} — impossible")
                    ok = False
                    break
                groups.append((have, need))
            if not ok:
                continue
            turns = [turn] if turn is not None else args.turns
            for t in turns:
                sample = 7 + t
                print(f"  {label} by turn {t} (seen {sample}): "
                      f"{p_combo(groups, sample):.1%}")


if __name__ == '__main__':
    main()
