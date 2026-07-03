"""Ladder meta survey: classify the opponent's archetype in each replay JSON.

Feeds three things (pre-registered, docs/report-log.md 2026-07-03):
  1. the deck bake-off decision rule's meta weights (archetype share of the
     observed ladder field),
  2. Stage 3 belief-model training labels,
  3. a report figure (meta share table/chart).

Classification: collect every Pokémon card id the opponent revealed in play or
discard across the whole game, map ids -> names via docs/EN_Card_Data.csv, and
match names against archetype signatures (ace attackers first, then support
lines). Unmatched games are listed as 'other/unknown' rather than guessed.

Usage:
  python tools/meta_survey.py replays/v25.2/*.json
  python tools/meta_survey.py --all            # every replays/**/*.json
  python tools/meta_survey.py --all --csv training/meta_survey.csv
"""
import argparse
import csv
import glob
import json
import os
import sys

# Windows cp1252 console chokes on unicode opponent names
sys.stdout.reconfigure(errors="replace")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARD_CSV = os.path.join(REPO_ROOT, "docs", "EN_Card_Data.csv")
OUR_TEAM = "Jason Anderson"

# (archetype, [name substrings, case-insensitive]) — first match wins, so aces
# precede generic support. Extend as new ladder archetypes appear.
SIGNATURES = [
    ("alakazam",   ["alakazam"]),
    ("dragapult",  ["dragapult"]),
    ("lucario",    ["lucario"]),
    ("abomasnow",  ["abomasnow"]),
    ("starmie",    ["starmie"]),
    ("crustle",    ["crustle"]),
    ("bellibolt",  ["bellibolt"]),
    ("gholdengo",  ["gholdengo"]),
    ("charizard",  ["charizard"]),
    ("gardevoir",  ["gardevoir"]),
    ("raging-bolt", ["raging bolt"]),
    ("terapagos",  ["terapagos"]),
    ("pidgeot-control", ["pidgeot"]),
    # observed on-ladder 2026-07 (see report-log meta survey entry)
    ("archaludon", ["archaludon"]),
    ("grimmsnarl", ["grimmsnarl"]),
    ("rockets-mewtwo", ["team rocket's mewtwo"]),
    ("snorlax-stall", ["hop's snorlax"]),
    ("lucario",    ["riolu"]),  # fighting box revealed without the ace yet
]


def load_names():
    names = {}
    with open(CARD_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["Card ID"])
            except (ValueError, KeyError):
                continue
            names.setdefault(cid, row.get("Card Name", ""))
    return names


def opponent_revealed_ids(d):
    """All card ids the opponent revealed in play or discard, plus their index."""
    info = d.get("info", {})
    team_names = info.get("TeamNames", [])
    you = team_names.index(OUR_TEAM) if OUR_TEAM in team_names else 1
    opp = 1 - you
    ids = set()
    for step in d.get("steps", []):
        for rec in step:
            obs = rec.get("observation") or {}
            cur = obs.get("current") or {}
            pl = cur.get("players") or []
            yidx = cur.get("yourIndex")
            if len(pl) != 2 or yidx is None:
                continue
            # map engine-relative index to absolute: players[yidx] is the
            # observing seat, so the opponent-of-record is players[opp] only
            # when this record belongs to seat `you`
            rec_opp = pl[1 - yidx] if rec is step[you] else pl[yidx]
            for zone in ("active", "bench", "discard"):
                for card in (rec_opp.get(zone) or []):
                    if isinstance(card, dict) and card.get("id") is not None:
                        ids.add(card["id"])
    opp_name = team_names[opp] if len(team_names) > 1 else "?"
    return ids, opp_name


def classify(ids, names):
    revealed = [names.get(i, "").lower() for i in ids]
    for archetype, sigs in SIGNATURES:
        for sig in sigs:
            if any(sig in n for n in revealed):
                return archetype
    return "other/unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replays", nargs="*", help="replay .json paths")
    ap.add_argument("--all", action="store_true", help="scan replays/**/*.json")
    ap.add_argument("--csv", help="also append per-replay rows to this CSV")
    args = ap.parse_args()

    paths = list(args.replays)
    if args.all:
        paths += glob.glob(os.path.join(REPO_ROOT, "replays", "**", "*.json"),
                           recursive=True)
    paths = sorted(set(paths))
    if not paths:
        ap.error("no replays given (use paths or --all)")

    names = load_names()
    rows = []
    counts = {}
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            ids, opp_name = opponent_revealed_ids(d)
            arch = classify(ids, names)
        except Exception as e:
            arch, opp_name = f"PARSE_ERROR({e})", "?"
        counts[arch] = counts.get(arch, 0) + 1
        rows.append({"replay": os.path.relpath(p, REPO_ROOT), "opponent": opp_name,
                     "archetype": arch})
        print(f"{os.path.basename(p):<28}{arch:<16}({opp_name})")

    n = len(rows)
    print(f"\nmeta share over {n} replays:")
    for arch, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {arch:<18}{c:>4}  {100.0 * c / n:5.1f}%")

    if args.csv:
        exists = os.path.exists(args.csv)
        with open(args.csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["replay", "opponent", "archetype"])
            if not exists:
                w.writeheader()
            w.writerows(rows)
        print(f"appended {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
