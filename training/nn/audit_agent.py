"""Deck-search targeting audit wrapper (Phase E follow-up, 2026-07-09):
plays exactly like main.agent but logs every stype==1 deck-area select with
REAL card ids (visible live, stripped to None in stored replays) plus board
context — to find where the attacker-pipeline (Abra fetch) breaks in
board-thinning games.

Env: AUDIT_LOG (JSONL path, PID-suffixed).
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import main as heuristic  # noqa: E402
from main import DECK  # noqa: E402

_LOG = os.environ.get("AUDIT_LOG")
LINE = {heuristic.ABRA, heuristic.KADABRA, heuristic.ALAKAZAM}


def _line_in_play(me):
    n = 0
    a = (me.get("active") or [None])[0]
    for pk in [a] + list(me.get("bench") or []):
        if pk and pk.get("id") in LINE:
            n += 1
    return n


def agent(obs_dict: dict) -> list:
    out = heuristic.agent(obs_dict)
    try:
        sel = obs_dict.get("select") or {}
        opts = sel.get("option") or []
        if _LOG and sel.get("type") == 1 and opts and any(o.get("area") == 1 for o in opts):
            cur = obs_dict.get("current") or {}
            me_idx = cur.get("yourIndex", 0)
            me = (cur.get("players") or [{}, {}])[me_idx]
            deck = sel.get("deck") or []

            def opt_card(o):
                idx = o.get("index")
                if deck and idx is not None and 0 <= idx < len(deck):
                    return (deck[idx] or {}).get("id")
                return o.get("id")

            hand_ids = [c.get("id") for c in (me.get("hand") or []) if c]
            rec = {
                "turn": cur.get("turn"),
                "effect": (sel.get("effect") or {}).get("id"),
                "has_deck_list": bool(deck),
                "opt_ids": [opt_card(o) for o in opts],
                "chosen": [opt_card(opts[i]) for i in out if 0 <= i < len(opts)],
                "line_in_play": _line_in_play(me),
                "kadabra_alak_in_hand": sum(1 for c in hand_ids
                                            if c in (heuristic.KADABRA, heuristic.ALAKAZAM)),
                "abra_in_hand": hand_ids.count(heuristic.ABRA),
                "deck_count": me.get("deckCount"),
            }
            with open(f"{_LOG}.{os.getpid()}", "a") as f:
                f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
    return out
