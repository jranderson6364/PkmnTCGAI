"""Fast, no-engine regression tests for main.py's already-fixed heuristic bugs.

Each case replays a single saved decision state (either extracted verbatim from a
real ladder replay JSON, or a small hand-built synthetic state isolating the same
bug pattern) and asserts a *property* of main.agent()'s choice -- not a raw option
index, since option lists can reorder run to run. Run directly (python
training/regression/regression_states.py) or via pytest; exits nonzero on any
failure.

Real-replay cases load the exact obs_dict main.py's agent() receives from the
cited replay file/step -- see each case's `reason` for the replay ID and the
main.py comment it corresponds to.
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import main

REPLAYS = os.path.join(ROOT, "replays")


def _load_obs(rel_path, step_idx):
    with open(os.path.join(REPLAYS, rel_path), encoding="utf-8") as f:
        data = json.load(f)
    return data["steps"][step_idx][0]["observation"]


def _base_current(active, bench, hand=None, deck_count=40, opp_active=None,
                   opp_bench=None, my_prize_left=3, opp_prize_left=3, turn=5):
    """Minimal synthetic 'current' state -- only the fields main.py actually reads."""
    return {
        "turn": turn, "turnActionCount": 1, "yourIndex": 0, "firstPlayer": 0,
        "supporterPlayed": False, "stadiumPlayed": False, "energyAttached": False,
        "retreated": False, "stadium": [],
        "players": [
            {"active": [active] if active else [], "bench": bench, "benchMax": 5,
             "deckCount": deck_count, "discard": [], "hand": hand or [],
             "handCount": len(hand or []), "prize": [None] * my_prize_left},
            {"active": [opp_active] if opp_active else [], "bench": opp_bench or [],
             "benchMax": 5, "deckCount": 40, "discard": [], "hand": None,
             "handCount": 4, "prize": [None] * opp_prize_left},
        ],
    }


def _synthetic_obs(current, select):
    return {"current": current, "select": select, "logs": [], "step": 0,
            "remainingOverageTime": 500, "search_begin_input": None}


# Card ids (mirrors main.py's constants -- kept independent so a typo here fails
# loudly as a wrong test, not a silently-matching copy/paste).
ABRA, KADABRA, ALAKAZAM = 741, 742, 743
PSYDUCK = 858
ENRICHING = 13
WONDROUS_PATCH_EFFECT = 1146
POFFIN, POKE_PAD, HILDA, DAWN = 1086, 1152, 1225, 1231
_DRAW_SOURCE_IDS = {POFFIN, POKE_PAD, HILDA, DAWN}


def _hand_card_id(obs, opt):
    """Hand card id targeted by a type==7 (play from hand) option, else None."""
    if opt.get("type") != 7 or opt.get("index") is None:
        return None
    cur = obs["current"]; me = cur["yourIndex"]
    hand = cur["players"][me].get("hand") or []
    idx = opt["index"]
    return hand[idx].get("id") if 0 <= idx < len(hand) else None


def option_card_id(opts, sel, idx):
    """Card id targeted by option `idx`, for deck-search (area=1) selects."""
    deck = sel.get("deck") or []
    di = opts[idx].get("index")
    return deck[di].get("id") if di is not None and 0 <= di < len(deck) else None


def bench_card_id(obs, opts, idx):
    """Bench card id targeted by option `idx`, for area=5 (bench-target) selects."""
    cur = obs["current"]; me = cur["yourIndex"]
    bench = cur["players"][me].get("bench") or []
    bi = opts[idx].get("index")
    return bench[bi].get("id") if bi is not None and 0 <= bi < len(bench) else None


CASES = []


def case(name, reason, obs_fn, check_fn):
    CASES.append({"name": name, "reason": reason, "obs_fn": obs_fn, "check_fn": check_fn})


# ---------------------------------------------------------------------------
# REAL replay-derived cases
# ---------------------------------------------------------------------------

case(
    name="deck_search_no_alakazam_when_no_abra",
    reason=(
        "Replay 83461698 step 24: Hilda's search offered Alakazam x3 / Kadabra x2 / "
        "Dudunsparce with zero Abra anywhere (play or hand) -- main.py comment near "
        "line 439 documents fetching Alakazam here as dead weight. Regression would "
        "be re-introducing have_line_piece counting kadabra_in_hand as a line piece, "
        "which would spike Alakazam's score to 95 and make the agent fetch it again."
    ),
    obs_fn=lambda: _load_obs("v25.2/83461698.json", 24),
    check_fn=lambda obs, picks: option_card_id(obs["select"]["option"], obs["select"], picks[0]) != ALAKAZAM,
)

case(
    name="boss_not_played_when_active_cannot_attack",
    reason=(
        "Replay 83458785 step 93: main.py comment near line 1026 documents Boss's "
        "Orders (card id 1182) firing at a flat 199 whenever phase==PHASE_CLOSING "
        "with no target-quality check, yanking a Mega Starmie ex off active "
        "(already the best target) in favor of a fresh 70-HP Staryu. At this exact "
        "state our Alakazam has 0 energy (can't attack), so Boss should not be the "
        "chosen play."
    ),
    obs_fn=lambda: _load_obs("v25.2/83458785.json", 93),
    check_fn=lambda obs, picks: not any(
        _hand_card_id(obs, obs["select"]["option"][i]) == 1182 for i in picks  # 1182 == Boss's Orders
    ),
)

case(
    name="mulligan_redraws_dead_hand",
    reason=(
        "Replay 83358041 step 2: MULLIGAN prompt (select.context==42) with an "
        "opening hand containing zero Basic Pokemon -- main.py comment near line "
        "1315 documents this exact replay as the motivating case for redrawing a "
        "dead hand (YES) instead of the old hand-content-blind draw-prompt logic."
    ),
    obs_fn=lambda: _load_obs("v23/83358041.json", 2),
    check_fn=lambda obs, picks: obs["select"]["option"][picks[0]].get("type") == 1,  # YES
)

case(
    name="board_thinning_stops_drawing_when_hand_grossly_over",
    reason=(
        "replays/exploiter_wins/win_001_b1g5.json step 110: stuck on a non-attacker "
        "Kadabra (0 energy) active with hand_n=20 vs cards_needed=7 (opp active "
        "Alakazam mirror at 140 HP). main.py comment near line 876 documents this "
        "exact replay as root cause of the board-thinning bug -- the hand_surplus "
        "gate previously required ready_attacker_exists, so it never engaged here, "
        "letting Poffin/Dawn/Hilda/Poke Pad draw the deck to 0. Fixed behavior: do "
        "not play any of those draw sources with a hand already this far over "
        "threshold (agent may still end turn or take another non-draw action)."
    ),
    obs_fn=lambda: _load_obs("exploiter_wins/win_001_b1g5.json", 110),
    check_fn=lambda obs, picks: not any(
        _hand_card_id(obs, o) in _DRAW_SOURCE_IDS for o in [obs["select"]["option"][i] for i in picks]
    ),
)

# ---------------------------------------------------------------------------
# SYNTHETIC cases (hand-built, not from a replay -- isolate the same bug pattern
# referenced in main.py's comments where no matching real replay was found)
# ---------------------------------------------------------------------------

def _promote_kadabra_over_psyduck_obs():
    psyduck = {"id": PSYDUCK, "hp": 60, "maxHp": 60, "energies": [], "energyCards": [],
               "tools": [], "preEvolution": [], "appearThisTurn": False}
    kadabra = {"id": KADABRA, "hp": 80, "maxHp": 80, "energies": [5],
               "energyCards": [{"id": 19}], "tools": [], "preEvolution": [], "appearThisTurn": False}
    opts = [{"area": 5, "index": 0, "playerIndex": 0, "type": 3},
            {"area": 5, "index": 1, "playerIndex": 0, "type": 3}]
    sel = {"type": 1, "context": 0, "option": opts, "minCount": 1, "maxCount": 1,
           "deck": None, "effect": None, "remainDamageCounter": 0, "remainEnergyCost": 0}
    return _synthetic_obs(_base_current(None, [psyduck, kadabra]), sel)


case(
    name="promote_fueled_kadabra_over_pure_support_psyduck",
    reason=(
        "SYNTHETIC (isolates the pattern in main.py's comment near line 559: "
        "promoting a pure-support mon like Psyduck over an already-energized "
        "Kadabra used to be a coinflip on array order since both shared a -10 "
        "fallback score). Forced promotion select with bench = [Psyduck, fueled "
        "Kadabra] -- expect the Kadabra, not Psyduck."
    ),
    obs_fn=_promote_kadabra_over_psyduck_obs,
    check_fn=lambda obs, picks: bench_card_id(obs, obs["select"]["option"], picks[0]) == KADABRA,
)


def _wondrous_patch_targets_unfueled_obs():
    kadabra_fueled = {"id": KADABRA, "hp": 80, "maxHp": 80, "energies": [5],
                       "energyCards": [{"id": 19}], "tools": [], "preEvolution": [], "appearThisTurn": False}
    abra_unfueled = {"id": ABRA, "hp": 50, "maxHp": 50, "energies": [], "energyCards": [],
                      "tools": [], "preEvolution": [], "appearThisTurn": False}
    active = {"id": ALAKAZAM, "hp": 90, "maxHp": 140, "energies": [], "energyCards": [],
              "tools": [], "preEvolution": [], "appearThisTurn": False}
    opts = [{"area": 5, "index": 0, "playerIndex": 0, "type": 3},
            {"area": 5, "index": 1, "playerIndex": 0, "type": 3}]
    sel = {"type": 1, "context": 0, "option": opts, "minCount": 1, "maxCount": 1,
           "deck": None, "effect": {"id": WONDROUS_PATCH_EFFECT},
           "remainDamageCounter": 0, "remainEnergyCost": 0}
    return _synthetic_obs(_base_current(active, [kadabra_fueled, abra_unfueled]), sel)


case(
    name="wondrous_patch_targets_unfueled_not_already_fueled",
    reason=(
        "SYNTHETIC (isolates the pattern in main.py's docstring near line 573-581: "
        "Wondrous Patch attaches to whoever NEEDS the energy, the opposite tiebreak "
        "from retreat/promotion targeting which prefers whoever can already attack). "
        "Bench = [fueled Kadabra, unfueled Abra] with select.effect.id==WONDROUS_PATCH "
        "-- expect the unfueled Abra, not the already-fueled Kadabra."
    ),
    obs_fn=_wondrous_patch_targets_unfueled_obs,
    check_fn=lambda obs, picks: bench_card_id(obs, obs["select"]["option"], picks[0]) == ABRA,
)


def _enriching_deck_safety_gate_obs():
    active = {"id": ALAKAZAM, "hp": 100, "maxHp": 140, "energies": [5],
              "energyCards": [{"id": 19}], "tools": [], "preEvolution": [{"id": ABRA}, {"id": KADABRA}],
              "appearThisTurn": False}
    opp_active = {"id": ALAKAZAM, "hp": 140, "maxHp": 140, "energies": [5],
                  "energyCards": [{"id": 19}], "tools": [], "preEvolution": [], "appearThisTurn": False}
    hand = [{"id": ENRICHING, "playerIndex": 0, "serial": 1}] + [
        {"id": 999, "playerIndex": 0, "serial": i} for i in range(2, 19)]
    opts = [{"type": 8, "area": 2, "index": 0, "inPlayArea": 4, "inPlayIndex": 0},  # attach Enriching to active
            {"type": 14}]  # end turn
    sel = {"type": 0, "context": 0, "option": opts, "minCount": 1, "maxCount": 1,
           "deck": None, "effect": None, "remainDamageCounter": 0, "remainEnergyCost": 0}
    cur = _base_current(active, [], hand, deck_count=4, opp_active=opp_active)
    return _synthetic_obs(cur, sel)


case(
    name="enriching_respects_deck_safety_gate",
    reason=(
        "SYNTHETIC (isolates the pattern in main.py's comment near line 1191-1199: "
        "Enriching's unconditional draw-4 had no deck-safety gate, contributing to "
        "a real deck-out in replay 83156504 at deck=5/hand=18). deck_count=4 "
        "(deck_critical), hand already 18 cards, opp active at full HP (cards_needed "
        "not yet at threshold) -- expect end-turn, not attaching Enriching for "
        "another unconditional draw-4."
    ),
    obs_fn=_enriching_deck_safety_gate_obs,
    check_fn=lambda obs, picks: obs["select"]["option"][picks[0]].get("type") == 14,  # END
)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run():
    failures = 0
    for c in CASES:
        try:
            obs = c["obs_fn"]()
            picks = main.agent(obs)
            ok = bool(picks) and c["check_fn"](obs, picks)
        except Exception as e:  # noqa: BLE001 -- want a FAIL line, not a crash, per case
            ok = False
            picks = f"<exception: {e!r}>"
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {c['name']}  (agent returned {picks})")
        if not ok:
            print(f"         reason: {c['reason']}")
            failures += 1
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if run() else 0)


def test_regression_states():
    """pytest entry point."""
    assert run() == 0
