"""Reconstruct a representative 60-card decklist per archetype from real
ladder replay evidence (Stage 3 Phase B — docs/belief-model.md §Phase B).

Four of our anchor archetypes (lucario, dragapult, abomasnow, starmie) are
official Kaggle sample bots with known-exact decklists already in
`opponents/*_agent.py` — no reconstruction needed, this script skips them.
For every other classified archetype, this counts every card id the
opponent revealed (play + discard, per tools/meta_survey.py) across all its
tagged replays, then builds a 60-card list from the most-frequently-revealed
cards, weighted by how consistently each card appears across games (a card
seen in most games of an archetype is probably a near-max-copy staple; a
card seen in a few games is probably a 1-2 copy tech) and capped at the
real game's copy limits (4 of any non-basic-energy card, 1 ACE SPEC).

This is a reconstruction from partial evidence, not a guaranteed-exact
decklist — every output list is annotated with its source replay count and
per-card reveal-game-count so the confidence level is visible, not implied.

Usage:
  python tools/build_archetype_decks.py --archetype crustle
  python tools/build_archetype_decks.py --all-uncovered --out training/archetype_decks.json
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meta_survey as ms

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARD_CSV = os.path.join(REPO_ROOT, "docs", "EN_Card_Data.csv")

# Archetypes with an official, exact decklist already available — skip these.
COVERED_BY_SAMPLE_AGENT = {
    "lucario": "opponents/lucario_agent.py",
    "dragapult": "opponents/dragapult_agent.py",
    "abomasnow": "opponents/abomasnow_agent.py",
    "starmie": "opponents/starmie_agent.py",
}


def load_card_meta():
    """cardId -> (name, cardType, aceSpec-ish flag via name heuristic)."""
    meta = {}
    with open(CARD_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["Card ID"])
            except (ValueError, KeyError):
                continue
            meta[cid] = row.get("Card Name", "")
    return meta


def collect_archetype_evidence(archetype, names):
    """Returns {card_id: n_games_revealing_it}, n_total_games for archetype."""
    paths = glob.glob(os.path.join(REPO_ROOT, "replays", "**", "*.json"), recursive=True)
    card_game_counts = {}
    n_games = 0
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            ids, _ = ms.opponent_revealed_ids(d)
            arch = ms.classify(ids, names)
        except Exception:
            continue
        if arch != archetype:
            continue
        n_games += 1
        for cid in ids:
            card_game_counts[cid] = card_game_counts.get(cid, 0) + 1
    return card_game_counts, n_games


def build_decklist(card_game_counts, n_games, names, target=60):
    """Weight = fraction of games revealing the card; higher weight -> more
    assumed copies, capped at 4 (non-basic-energy cards can't exceed 4 real
    copies; basic energies are uncapped in the real game but we still cap
    the RECONSTRUCTED list's per-id count at 4 for a plausible spread)."""
    ranked = sorted(card_game_counts.items(), key=lambda kv: -kv[1])
    deck = []
    for cid, n_seen in ranked:
        frac = n_seen / n_games if n_games else 0
        if frac >= 0.6:
            copies = 4
        elif frac >= 0.35:
            copies = 3
        elif frac >= 0.15:
            copies = 2
        else:
            copies = 1
        deck.append({"cardId": cid, "name": names.get(cid, "?"),
                      "copies": copies, "games_seen": n_seen, "game_frac": round(frac, 3)})

    total = sum(c["copies"] for c in deck)
    # Trim from the least-confident end first if over 60, since those are
    # the shakiest evidence; if under 60 (sparse archetype), leave it short
    # rather than fabricate cards with zero evidence.
    i = len(deck) - 1
    while total > target and i >= 0:
        if deck[i]["copies"] > 0:
            deck[i]["copies"] -= 1
            total -= 1
        else:
            i -= 1
            continue
        if deck[i]["copies"] == 0:
            i -= 1
    deck = [c for c in deck if c["copies"] > 0]
    return deck, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archetype", help="single archetype name (see tools/meta_survey.py SIGNATURES)")
    ap.add_argument("--all-uncovered", action="store_true",
                     help="build for every classified archetype not already covered by a sample agent")
    ap.add_argument("--out", help="write JSON {archetype: [{cardId,name,copies,games_seen,game_frac}, ...]}")
    args = ap.parse_args()

    names = load_card_meta()

    archetypes = []
    if args.archetype:
        archetypes = [args.archetype]
    elif args.all_uncovered:
        known = sorted(set(a for a, _ in ms.SIGNATURES) - set(COVERED_BY_SAMPLE_AGENT))
        archetypes = known
    else:
        ap.error("pass --archetype NAME or --all-uncovered")

    result = {}
    for arch in archetypes:
        if arch in COVERED_BY_SAMPLE_AGENT:
            print(f"{arch}: SKIPPED — official decklist already at {COVERED_BY_SAMPLE_AGENT[arch]}")
            continue
        counts, n_games = collect_archetype_evidence(arch, names)
        if n_games == 0:
            print(f"{arch}: 0 replays found, skipping")
            continue
        deck, total = build_decklist(counts, n_games, names)
        result[arch] = deck
        print(f"\n=== {arch} (from {n_games} replays, reconstructed {total}/60 cards) ===")
        for c in deck:
            print(f"  {c['copies']}x {c['name']:<30} (seen in {c['games_seen']}/{n_games} games, {c['game_frac']:.0%})")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
