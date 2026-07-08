"""Smoke test for determinize.py: run real games vs a known-archetype bot,
sample a determinization at every one of our decisions, and assert:
  1. every zone's length matches the observation's counts exactly,
  2. no None/invalid ids anywhere,
  3. our own sampled zones never contain a card we can see elsewhere beyond
     its real copy count (multiset consistency vs main.DECK),
  4. by mid-game vs lucario_agent, the sampler labels the opponent lucario
     and the sampled hidden cards come from the lucario decklist.
Usage: python training/belief/test_determinize.py [--games 3]
"""
import argparse
import collections
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "training"))
sys.path.insert(0, _HERE)

from harness import run_matches  # noqa: E402
from determinize import BeliefDeterminizer, FILLER_ID  # noqa: E402
import main as heuristic  # noqa: E402

MAIN = os.path.join(REPO_ROOT, "main.py")
LUCARIO = os.path.join(REPO_ROOT, "opponents", "lucario_agent.py")


def check_game(steps, our_seat, det, rng, stats):
    lucario_set = set(det.decks["lucario"])
    for step in steps:
        obs = step[our_seat].get("observation") if len(step) > our_seat else None
        if not obs or not obs.get("select") or not obs.get("current"):
            continue
        cur = obs["current"]
        if cur.get("yourIndex") != our_seat:
            continue
        me, opp = cur["players"][our_seat], cur["players"][1 - our_seat]
        s = det.sample(obs, our_seat, rng)
        stats["decisions"] += 1

        # 1. exact counts
        assert len(s["your_deck"]) == (me.get("deckCount") or 0), "your_deck count"
        assert len(s["your_prize"]) == len(me.get("prize") or []), "your_prize count"
        assert len(s["opponent_deck"]) == (opp.get("deckCount") or 0), "opp_deck count"
        assert len(s["opponent_prize"]) == len(opp.get("prize") or []), "opp_prize count"
        assert len(s["opponent_hand"]) == (opp.get("handCount") or 0), "opp_hand count"
        facedown = bool(opp.get("active")) and opp["active"][0] is None
        assert len(s["opponent_active"]) == (1 if facedown else 0), "opp_active count"

        # 2. all ids are positive ints
        for zone in ("your_deck", "your_prize", "opponent_deck",
                     "opponent_prize", "opponent_hand", "opponent_active"):
            assert all(isinstance(c, int) and c > 0 for c in s[zone]), f"bad id in {zone}"

        # 3. our sampled zones + visible cards never exceed main.DECK copy counts
        full = collections.Counter(heuristic.DECK)
        sampled = collections.Counter(s["your_deck"])
        for slot, sid in zip(me.get("prize") or [], s["your_prize"]):
            if slot is None:
                sampled[sid] += 1  # only face-down slots were sampled by us
        visible = collections.Counter()
        for c in (me.get("hand") or []) + (me.get("discard") or []):
            if (c or {}).get("id"):
                visible[c["id"]] += 1
        for cid, n in sampled.items():
            if cid == FILLER_ID:
                continue
            assert n + visible[cid] <= full[cid], (
                f"card {cid}: sampled {n} + visible {visible[cid]} > deck {full[cid]}")

        # 4. archetype + source-list checks (mid-game onward)
        if (cur.get("turn") or 0) >= 3:
            stats["labels"][s["archetype"]] += 1
            if s["archetype"] == "lucario":
                hidden = s["opponent_deck"] + s["opponent_hand"] + s["opponent_active"] + [
                    sid for slot, sid in zip(opp.get("prize") or [], s["opponent_prize"])
                    if slot is None]
                bad = [c for c in hidden if c not in lucario_set and c != FILLER_ID]
                assert not bad, f"non-lucario ids in belief-sampled zones: {bad[:5]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3)
    args = ap.parse_args()
    rng = random.Random(42)
    det = BeliefDeterminizer()
    stats = {"decisions": 0, "labels": collections.Counter()}

    for our_seat in (0, 1):
        paths = (MAIN, LUCARIO) if our_seat == 0 else (LUCARIO, MAIN)
        results = run_matches(paths[0], paths[1], max(1, args.games // 2),
                              workers=1, keep_steps=True, progress=False)
        for r in results:
            if "steps" not in r:
                print(f"game error: {r.get('error')}", file=sys.stderr)
                continue
            check_game(r["steps"], our_seat, det, rng, stats)

    print(f"OK: {stats['decisions']} decisions sampled and verified; "
          f"turn>=3 archetype labels: {dict(stats['labels'])}")
    assert stats["decisions"] > 0, "no decisions checked"
    assert stats["labels"].get("lucario", 0) > 0, "sampler never identified lucario mid-game"


if __name__ == "__main__":
    main()
