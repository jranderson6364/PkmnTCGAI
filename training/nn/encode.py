"""Observation -> tensor encoding for the BC/self-play net.

v1 rebuilt design (all prior encoding code was lost with the reset — see
docs/nn-training.md). Deliberately simpler than the original 22000-vocab
transformer-decoder sketch: a 13-slot board sequence + hand/discard bag
embeddings + a small numeric feature vector, and a per-candidate-action
feature vector for the policy head. No cg-lib dependency for the base
features — everything there is read directly off the raw obs_dict,
identical to what `main.py` consumes.

CARD_VOCAB / ATTACK_VOCAB are hardcoded upper bounds (real max observed card
ID is 1267 as of 2026-07-01; enums may grow during the competition per the
official docs, hence the safety margin) rather than calling all_card_data() —
this keeps the base features usable with or without cg-lib attached.

2026-07-05 (user-directed): added situational-belief features to
numeric_feats -- the net previously saw only 13 raw counters (HP ratios,
hand/deck/prize counts, one hardcoded Mist flag, turn) with no archetype
belief, no opponent-threat estimate, and no evolution-line progress, despite
all three already existing elsewhere in the project (main.py's own belief
model, this session's threat.py). This IS a new dependency on `main` (repo
root) and `threat` (needs cg.api's local shim, training-side only, never
used by the ladder-submitted main.py itself) -- acceptable here since
encode.py is training-pipeline-only, unlike main.py's dependency-free
constraint.
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import main as _heuristic  # noqa: E402
from threat import net_threat_diff as _net_threat_diff  # noqa: E402
from tempo_features import tempo_features as _tempo_features, TEMPO_DIM  # noqa: E402

CARD_VOCAB = 2000
ATTACK_VOCAB = 2000
OPTION_TYPE_VOCAB = 17  # OptionType enum, 0..16
N_BOARD_SLOTS = 13      # my_active, my_bench x5, opp_active, opp_bench x5, stadium
MAX_ACTIONS = 64

# 2026-07-05 ablation (docs/report-log.md "AlphaZero-style push, Phase 1"):
# the combined 25-feature encoding scored WORSE than the plain 13-feature
# baseline on the real-replay gate, despite each addition being individually
# well-motivated. ENCODE_FEATURE_SET isolates which addition (if any) is
# responsible, matching the isolated-component-before-trusting-the-
# combination discipline already used for Φ. "full" (default) preserves the
# already-tested combined behavior unchanged.
_FEATURE_SET = os.environ.get("ENCODE_FEATURE_SET", "full")
_FEATURE_SET_SIZES = {
    "base": 13,
    "base+threat": 14,
    "base+census": 16,   # +line_progress, +has_alakazam, +hand_advantage
    "base+belief": 21,   # +5 posterior probs, +wall_revealed, +crustle_seen, +unavailable flag
    "full": 25,
    "full+tempo": 25 + TEMPO_DIM,  # +prize-race pace, +hand-growth pace, +setup pace (2026-07-12)
}
NUM_FEATS = _FEATURE_SET_SIZES[_FEATURE_SET]


def _pk_id(pk):
    return (pk or {}).get("id", 0) or 0


def _active(p):
    a = (p or {}).get("active")
    return a[0] if a and len(a) > 0 and a[0] else None


def board_slot_ids(obs):
    """13 card-id tokens: [my_active, my_bench(5), opp_active, opp_bench(5), stadium]."""
    cur = obs.get("current") or {}
    me_idx = cur.get("yourIndex", 0)
    pl = cur.get("players") or []
    me = pl[me_idx] if len(pl) > me_idx else {}
    opp = pl[1 - me_idx] if len(pl) == 2 else {}
    my_bench = (me.get("bench") or [])[:5]
    opp_bench = (opp.get("bench") or [])[:5]
    ids = [_pk_id(_active(me))]
    ids += [_pk_id(b) for b in my_bench] + [0] * (5 - len(my_bench))
    ids.append(_pk_id(_active(opp)))
    ids += [_pk_id(b) for b in opp_bench] + [0] * (5 - len(opp_bench))
    stadium = (cur.get("stadium") or [None])
    ids.append(_pk_id(stadium[0]) if stadium else 0)
    return [min(i, CARD_VOCAB - 1) for i in ids]


def hand_ids(obs, cap=20):
    cur = obs.get("current") or {}
    me_idx = cur.get("yourIndex", 0)
    pl = cur.get("players") or []
    me = pl[me_idx] if len(pl) > me_idx else {}
    hand = me.get("hand") or []
    ids = [min(_pk_id(c), CARD_VOCAB - 1) for c in hand if _pk_id(c)]
    return ids[:cap] or [0]


def discard_ids(obs, cap=20):
    cur = obs.get("current") or {}
    me_idx = cur.get("yourIndex", 0)
    pl = cur.get("players") or []
    me = pl[me_idx] if len(pl) > me_idx else {}
    disc = me.get("discard") or []
    ids = [min(_pk_id(c), CARD_VOCAB - 1) for c in disc if _pk_id(c)]
    return ids[:cap] or [0]


_BELIEF_CLASS_ORDER = _heuristic._BELIEF_CLASSES  # fixed 5-class order, reused for a stable feature layout


def _belief_feats(opp, turn):
    """5 archetype posterior probabilities + wall_revealed + crustle_seen,
    via main.py's own embedded belief model (the same function main.py
    itself calls at inference) -- zero-filled + a bit set on failure so a
    belief-model exception never breaks the whole feature vector."""
    try:
        post, wall_revealed, crustle_seen = _heuristic._belief_posterior(opp, turn)
    except Exception:
        post, wall_revealed, crustle_seen = None, False, False
    if post is None:
        return [0.0] * 5 + [0.0, 0.0, 1.0]  # last 1.0 flags "belief unavailable"
    probs = [post.get(c, 0.0) for c in _BELIEF_CLASS_ORDER]
    return probs + [1.0 if wall_revealed else 0.0, 1.0 if crustle_seen else 0.0, 0.0]


def numeric_feats(obs):
    """Fixed-size float feature vector, roughly matching main.py's _census inputs,
    plus situational-belief features added 2026-07-05 (archetype posterior,
    zero-sum threat estimate, evolution-line progress, hand-vs-KO-threshold
    advantage) -- see module docstring."""
    cur = obs.get("current") or {}
    me_idx = cur.get("yourIndex", 0)
    pl = cur.get("players") or []
    me = pl[me_idx] if len(pl) > me_idx else {}
    opp = pl[1 - me_idx] if len(pl) == 2 else {}
    my_active = _active(me)
    opp_active = _active(opp)
    my_hp = (my_active or {}).get("hp", 0) or 0
    my_maxhp = (my_active or {}).get("maxHp", 1) or 1
    opp_hp = (opp_active or {}).get("hp", 0) or 0
    opp_maxhp = (opp_active or {}).get("maxHp", 1) or 1
    my_hand_n = me.get("handCount") or len(me.get("hand") or [])
    opp_hand_n = opp.get("handCount", 0) or 0
    my_deck = me.get("deckCount", 0) or 0
    opp_deck = opp.get("deckCount", 0) or 0
    my_prizes = len(me.get("prize") or [])
    opp_prizes = len(opp.get("prize") or [])
    opp_energies = set()
    for ec in (opp_active or {}).get("energyCards") or []:
        opp_energies.add(ec.get("id"))
    opp_mist = 1.0 if (11 in opp_energies or 20 in opp_energies) else 0.0
    turn = cur.get("turn", 0) or 0

    try:
        cen = _heuristic._census(my_active, me.get("bench") or [])
        line_progress = cen["line_count"] / 2.0
        has_alakazam = 1.0 if cen["has_alakazam"] else 0.0
    except Exception:
        line_progress, has_alakazam = 0.0, 0.0

    try:
        opp_cen = _heuristic._census(opp_active, opp.get("bench") or [])
        opp_line_progress = opp_cen["line_count"] / 2.0
    except Exception:
        opp_line_progress = 0.0

    if opp_hp:
        cards_needed = math.ceil(opp_hp / _heuristic.PH_DMG_PER_CARD)
        hand_advantage = max(-1.0, min(1.0, (my_hand_n - cards_needed) / 10.0))
    else:
        hand_advantage = min(my_hand_n / 10.0, 1.0)

    try:
        threat_diff = _net_threat_diff(cur, me_idx)
    except Exception:
        threat_diff = 0.0

    base = [
        my_hp / max(my_maxhp, 1),
        opp_hp / max(opp_maxhp, 1),
        min(my_hand_n, 30) / 30.0,
        min(opp_hand_n, 30) / 30.0,
        min(my_deck, 60) / 60.0,
        min(opp_deck, 60) / 60.0,
        min(my_prizes, 6) / 6.0,
        min(opp_prizes, 6) / 6.0,
        opp_mist,
        1.0 if cur.get("supporterPlayed") else 0.0,
        1.0 if cur.get("energyAttached") else 0.0,
        1.0 if cur.get("retreated") else 0.0,
        min(turn, 60) / 60.0,
    ]
    census_group = [line_progress, has_alakazam, hand_advantage]
    threat_group = [threat_diff]
    belief_group = _belief_feats(opp, turn)

    try:
        tempo_group = _tempo_features(my_hand_n, opp_hand_n, my_prizes, opp_prizes,
                                       line_progress, opp_line_progress, turn).tolist()
    except Exception:
        tempo_group = [0.0] * TEMPO_DIM

    if _FEATURE_SET == "base":
        return base
    if _FEATURE_SET == "base+threat":
        return base + threat_group
    if _FEATURE_SET == "base+census":
        return base + census_group
    if _FEATURE_SET == "base+belief":
        return base + belief_group
    if _FEATURE_SET == "full+tempo":
        return base + census_group + threat_group + belief_group + tempo_group
    return base + census_group + threat_group + belief_group  # "full"


def _opt_card_id(o, hand, bench):
    """Mirrors main.py._opt_card_id — resolve an option's associated card id."""
    ot = o.get("type")
    idx = o.get("index")
    if ot in (3, 4, 5, 7, 8, 9):  # CARD/TOOL_CARD/ENERGY_CARD/PLAY/ATTACH/EVOLVE
        if idx is not None and 0 <= idx < len(hand):
            return _pk_id(hand[idx])
        return 0
    if ot == 10:  # ABILITY
        area = o.get("area")
        if area == 4:
            return 0  # active resolved separately by caller if needed
        if area == 5 and 0 <= idx < len(bench):
            return _pk_id(bench[idx])
    return 0


def encode_action(obs, o):
    """Per-candidate-option feature dict: type id, resolved card id, attack id, numerics."""
    cur = obs.get("current") or {}
    me_idx = cur.get("yourIndex", 0)
    pl = cur.get("players") or []
    me = pl[me_idx] if len(pl) > me_idx else {}
    hand = me.get("hand") or []
    bench = me.get("bench") or []
    ot = o.get("type") or 0
    cid = _opt_card_id(o, hand, bench)
    if ot == 10 and o.get("area") == 4:  # ABILITY on active
        cid = _pk_id(_active(me))
    attack_id = o.get("attackId") or 0
    area = o.get("area") or 0
    in_play_area = o.get("inPlayArea") or 0
    index = o.get("index") or 0
    in_play_index = o.get("inPlayIndex") or 0
    return {
        "type": min(ot, OPTION_TYPE_VOCAB - 1),
        "card_id": min(cid, CARD_VOCAB - 1),
        "attack_id": min(attack_id, ATTACK_VOCAB - 1),
        "numeric": [area / 12.0, in_play_area / 12.0, index / 20.0, in_play_index / 6.0],
    }


def encode_sample(obs, sel):
    """Full encoding for one decision point. Returns a dict of plain python
    lists/ints (torch-free) so this module works without a torch import —
    the Dataset class converts to tensors at collate time."""
    opts = (sel.get("option") or [])[:MAX_ACTIONS]
    return {
        "board_ids": board_slot_ids(obs),
        "hand_ids": hand_ids(obs),
        "discard_ids": discard_ids(obs),
        "numeric": numeric_feats(obs),
        "actions": [encode_action(obs, o) for o in opts],
        "n_actions": len(opts),
    }
