"""Tight-situation curriculum: harvest HARD positions from self-play traces.

The user's idea: train/evaluate the model disproportionately on tight spots
instead of uniformly sampling full games. Two mechanisms:

1. Bad-hand game filter (start-of-game curriculum): keep only games whose
   opening hand had no Abra and no search card. Cheap now that games run
   locally — no early-exit / battle_finish concern: we just discard the
   uninteresting games after the fact.

2. Hard-position mining (mid-game curriculum): scan every decision point in a
   trace and tag positions that match "tight" predicates (mist-walled, deck
   danger, behind on prizes, active immobile, opponent one KO from winning).
   Tagged positions are saved with their full obs dict; the NN track uses them
   as evaluation suites and (via cg.api search_begin on Kaggle) as alternate
   game starts for targeted self-play.

Usage:
  python training/curriculum.py --games 500 --out training/hard_positions.pkl.gz
"""
import argparse
import gzip
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_matches
from bc_collect import extract_decisions

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(REPO_ROOT, "main.py")

ABRA = 741
SEARCH_IDS = {1086, 1152, 1231}  # Poffin, Poke Pad, Dawn
MIST, ROCK = 11, 20


def _me(obs):
    cur = obs.get("current") or {}
    yi = cur.get("yourIndex", 0)
    pl = cur.get("players") or []
    me = pl[yi] if len(pl) > yi else {}
    opp = pl[1 - yi] if len(pl) == 2 else {}
    return cur, me, opp


def bad_opening_hand(decs):
    """Evaluate at the first decision where the hand is actually dealt (the very
    first select is the coin flip, before any cards exist)."""
    for d in decs[:6]:
        _, me, _ = _me(d["obs"])
        hand = me.get("hand") or []
        if hand:
            ids = {(c or {}).get("id") for c in hand}
            return ABRA not in ids and not (ids & SEARCH_IDS)
    return False


def tight_tags(obs):
    """Return list of tags describing why this position is hard (empty = easy)."""
    tags = []
    cur, me, opp = _me(obs)
    deck = me.get("deckCount") or 0
    if 0 < deck < 8:
        tags.append("deck_danger")
    my_p = len(me.get("prize") or [])
    opp_p = len(opp.get("prize") or [])
    if opp_p == 1:
        tags.append("opp_one_prize_from_win")
    if my_p - opp_p >= 3:
        tags.append("behind_3_prizes")
    oa = (opp.get("active") or [None])[0]
    if oa and any(ec.get("id") in (MIST, ROCK) for ec in (oa.get("energyCards") or [])):
        tags.append("mist_walled")
    ma = (me.get("active") or [None])[0]
    if ma and not (ma.get("energies") or []) and (ma.get("id") not in (743,)):
        tags.append("weak_active")
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "training", "hard_positions.pkl.gz"))
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    results = run_matches(MAIN, MAIN, args.games, workers=args.workers, keep_steps=True)
    hard_games, positions = 0, []
    for r in results:
        if "steps" not in r:
            continue
        decs = extract_decisions(r["steps"], seat=0)
        if decs and bad_opening_hand(decs):
            hard_games += 1
            for d in decs:
                d["tags"] = ["bad_opening"] + tight_tags(d["obs"])
                d["outcome"] = r["rewards"][0]
                positions.append(d)
            continue
        for d in decs:
            tags = tight_tags(d["obs"])
            if tags:
                d["tags"] = tags
                d["outcome"] = r["rewards"][0]
                positions.append(d)

    print(f"games={len(results)} bad-hand games={hard_games} tagged positions={len(positions)}")
    with gzip.open(args.out, "wb") as f:
        pickle.dump(positions, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
