"""Mine per-decision disagreements against a STRONGER pilot of the SAME deck.

`opponents/public/alakazam_v9.py` (publicScore 778.2) runs a byte-identical
60-card list to ours (v29d, 673.5) and beats us 55% head-to-head. Because the
decks match exactly, the entire ~105-point gap is piloting. That makes it the
cleanest improvement signal this project has ever had: play our agent, and at
every one of our decisions ask the stronger pilot what it would have done.

This is the loss-mining workflow -- the only reliably positive one in the
project's history -- but pointed at a known-better opponent instead of at our own
losses, so it finds systematic policy differences rather than one-off blunders.

CAVEAT, stated up front: the advisor is queried on states from a game it is not
actually playing, so any per-turn state it caches internally (e.g. "have I used
this ability yet this turn") drifts. Disagreements in stateless-ish contexts
(deck search, target choice, attack selection) are trustworthy; ones that hinge
on the advisor's own turn memory are not. Frequency ranking is the output; each
candidate still needs its own gate.

Run:  python training/disagree_mine.py --games 40
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "local_cg"))

from harness import load_agent, play_game

OPT = {0:"NUMBER",1:"YES",2:"NO",3:"CARD",4:"TOOL_CARD",5:"ENERGY_CARD",6:"ENERGY",
       7:"PLAY",8:"ATTACH",9:"EVOLVE",10:"ABILITY",11:"DISCARD",12:"RETREAT",
       13:"ATTACK",14:"END",15:"SKILL",16:"SPECIAL_CONDITION"}
CTX = {0:"MAIN",1:"SETUP_ACTIVE",2:"SETUP_BENCH",3:"SWITCH",4:"TO_ACTIVE",5:"TO_BENCH",
       7:"TO_HAND",8:"DISCARD",9:"TO_DECK",21:"ATTACH_FROM",22:"ATTACH_TO",
       35:"ATTACK",37:"EVOLVE",41:"IS_FIRST",42:"MULLIGAN"}

AREA_HAND, AREA_BENCH, AREA_ACTIVE, AREA_DISCARD = 2, 5, 4, 3


def card_name(names, cid):
    return names.get(cid, str(cid)) if cid is not None else "-"


def opt_card_id(obs, o):
    """Best-effort card id behind an option, for human-readable labels."""
    cur = obs.get("current") or {}
    me = cur.get("yourIndex", 0)
    players = cur.get("players") or []
    p = players[o.get("playerIndex", me)] if len(players) > o.get("playerIndex", me) else {}
    area, idx = o.get("area"), o.get("index", 0)
    # PLAY / ATTACH / EVOLVE / TOOL / ENERGY_CARD index straight into the HAND and
    # carry no `area`, so the area lookup below misses them entirely -- which is
    # why the first mining pass labelled 128 of its top disagreements as a bare
    # "PLAY" with no card name.
    if o.get("type") in (7, 8, 9, 4, 5):
        try:
            return ((players[me] if len(players) > me else {}).get("hand") or [])[idx].get("id")
        except Exception:
            return None
    try:
        if area == AREA_HAND:    return (p.get("hand") or [])[idx].get("id")
        if area == AREA_BENCH:   return (p.get("bench") or [])[idx].get("id")
        if area == AREA_ACTIVE:  return (p.get("active") or [])[idx].get("id")
        if area == AREA_DISCARD: return (p.get("discard") or [])[idx].get("id")
        if area == 1:            return ((obs.get("select") or {}).get("deck") or [])[idx].get("id")
    except Exception:
        pass
    return None


def label(obs, o, names):
    t = OPT.get(o.get("type"), str(o.get("type")))
    if o.get("type") == 13:
        return f"ATTACK:{o.get('attackId')}"
    cid = opt_card_id(obs, o)
    return f"{t}:{card_name(names, cid)}" if cid is not None else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="main.py")
    ap.add_argument("--advisor", default="opponents/public/alakazam_v9.py")
    ap.add_argument("--opponent", default="opponents/public/alakazam_v9.py")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--top", type=int, default=18)
    args = ap.parse_args()

    try:
        from cg.api import all_card_data
        names = {c.cardId: c.name for c in all_card_data()}
    except Exception:
        names = {}

    ours_fn, ours_deck, _ = load_agent(args.ours)
    adv_fn, _, _ = load_agent(args.advisor)
    opp_fn, opp_deck, _ = load_agent(args.opponent)

    pairs = Counter()          # (ctx, ours_label, adv_label) -> count
    ctx_totals = Counter()
    ctx_disagree = Counter()
    examples = defaultdict(list)
    n_dec = 0
    wins = 0

    def probe(obs_dict):
        nonlocal n_dec
        out = ours_fn(obs_dict)
        try:
            sel = obs_dict.get("select")
            if not sel or not out:
                return out
            opts = sel.get("option") or []
            if len(opts) < 2:
                return out
            ctx = CTX.get(sel.get("context"), str(sel.get("context")))
            n_dec += 1
            ctx_totals[ctx] += 1
            adv = adv_fn(obs_dict)
            if not adv:
                return out
            i, j = out[0], adv[0]
            if i == j or not (0 <= i < len(opts)) or not (0 <= j < len(opts)):
                return out
            ctx_disagree[ctx] += 1
            lo, la = label(obs_dict, opts[i], names), label(obs_dict, opts[j], names)
            if lo == la:
                return out          # same card via a different index: not a real disagreement
            key = (ctx, lo, la)
            pairs[key] += 1
            if len(examples[key]) < 2:
                cur = obs_dict.get("current") or {}
                me = cur.get("yourIndex", 0)
                pl = cur.get("players") or []
                mine = pl[me] if len(pl) > me else {}
                opp = pl[1 - me] if len(pl) == 2 else {}
                examples[key].append(dict(
                    turn=cur.get("turn"), deck=mine.get("deckCount"),
                    hand=mine.get("handCount"),
                    prizes=(len(mine.get("prize") or []), len(opp.get("prize") or []))))
        except Exception:
            pass
        return out

    for g in range(args.games):
        if g % 2 == 0:
            r = play_game(probe, ours_deck, opp_fn, opp_deck)
            me = 0
        else:
            r = play_game(opp_fn, opp_deck, probe, ours_deck)
            me = 1
        rw = r["rewards"]
        if rw[me] is not None and rw[1 - me] is not None and rw[me] > rw[1 - me]:
            wins += 1
        if (g + 1) % 10 == 0:
            print(f"  {g+1}/{args.games} games, {n_dec} decisions, "
                  f"{sum(pairs.values())} disagreements", file=sys.stderr)

    total_dis = sum(pairs.values())
    print(f"\n=== {args.ours}  vs advisor {args.advisor} ===")
    print(f"games {args.games} | our win rate {wins}/{args.games} = {wins/args.games:.1%}")
    print(f"multi-option decisions {n_dec} | disagreements {total_dis} "
          f"({100*total_dis/max(1,n_dec):.1f}%)\n")

    print("disagreement rate by context:")
    for ctx, tot in ctx_totals.most_common():
        d = ctx_disagree.get(ctx, 0)
        print(f"  {ctx:14s} {d:5d}/{tot:5d} = {100*d/max(1,tot):5.1f}%")

    print(f"\ntop {args.top} disagreement classes (ours -> advisor):")
    for (ctx, lo, la), c in pairs.most_common(args.top):
        ex = examples[(ctx, lo, la)]
        exs = "; ".join(f"t{e['turn']} deck{e['deck']} hand{e['hand']} pz{e['prizes']}" for e in ex)
        print(f"  {c:4d}  [{ctx}] {lo:34s} -> {la:34s}   {exs}")


if __name__ == "__main__":
    main()
