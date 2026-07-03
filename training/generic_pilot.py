"""Deck-agnostic greedy pilot for the Stage 0c tier-2 (controlled) bake-off.

One fixed, deliberately simple policy piloting EVERY deck, so tier-2 rankings
reflect deck strength rather than pilot quality (pre-registered protocol,
docs/report-log.md 2026-07-03). Uses only deck-independent signals: the option
type codes of the current select (no card IDs, no archetype knowledge).

Greedy priority on the main action select: evolve > use abilities > play cards
> attach energy (active first) > attack (highest-index attack = the later,
usually stronger one) > end turn. Never retreats. Sub-selects take the first
minCount..maxCount options; yes/no prefers YES. A small stall guard falls back
to END / last option if the identical select repeats (mirrors the concern
main.py handles with _resolve_stalled_or).

DECK here is a placeholder so harness.load_agent() can import the module; the
bake-off passes each deck explicitly via bakeoff.py's orthogonal (agent, deck)
form.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import DECK  # noqa: E402  (placeholder only — see docstring)

# option type codes (see tools/analyze_replay.py OT_NAMES)
YES, NO = 1, 2
PLAY, ATTACH, EVOLVE, ABILITY, RETREAT, ATTACK, END = 7, 8, 9, 10, 12, 13, 14

# greedy order for the main-phase select; anything unlisted ranks between
# ATTACK and END (neutral), RETREAT ranks below END (never chosen voluntarily)
_PRIORITY = {EVOLVE: 0, ABILITY: 1, PLAY: 2, ATTACH: 3, ATTACK: 5, END: 8, RETREAT: 9}
_DEFAULT_RANK = 7

_stall = {"fp": None, "count": 0}


def _fingerprint(sel):
    opts = sel.get("option", [])
    return (sel.get("type"), len(opts),
            tuple((o.get("type"), o.get("index"), o.get("attackId")) for o in opts[:12]))


def agent(obs_dict: dict) -> list:
    sel = obs_dict.get("select")
    if sel is None:
        return DECK
    opts = sel.get("option", [])
    n = len(opts)
    if n == 0:
        return []
    mx = sel.get("maxCount", 1) or 1
    mn = sel.get("minCount", 0) or 0
    k = max(mn, min(mx, n))

    fp = _fingerprint(sel)
    if fp == _stall["fp"]:
        _stall["count"] += 1
    else:
        _stall["fp"], _stall["count"] = fp, 0
    if _stall["count"] >= 4:
        # stuck repeating the same select: end turn if we can, else last option
        for i, o in enumerate(opts):
            if o.get("type") == END:
                return [i]
        return [n - 1]

    types = [o.get("type") for o in opts]

    # yes/no gates: take YES (proceed with the effect)
    if YES in types or NO in types:
        return [types.index(YES) if YES in types else types.index(NO)]

    if k == 1:
        def rank(i):
            o = opts[i]
            r = _PRIORITY.get(o.get("type"), _DEFAULT_RANK)
            tie = 0
            if o.get("type") == ATTACK:
                tie = -i          # prefer the later-listed (usually stronger) attack
            elif o.get("type") == ATTACH:
                tie = 0 if o.get("inPlayArea") == 4 else 1  # active before bench
            else:
                tie = i           # otherwise stable: first listed
            return (r, tie)
        return [min(range(n), key=rank)]

    # multi-pick sub-selects (choose k cards/targets): first k options
    return list(range(k))
