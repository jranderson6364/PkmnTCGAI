"""Observation -> tensor encoding for the BC/self-play net.

v1 rebuilt design (all prior encoding code was lost with the reset — see
docs/nn-training.md). Deliberately simpler than the original 22000-vocab
transformer-decoder sketch: a 13-slot board sequence + hand/discard bag
embeddings + a small numeric feature vector, and a per-candidate-action
feature vector for the policy head. No cg-lib dependency — everything is
read directly off the raw obs_dict, identical to what `main.py` consumes.

CARD_VOCAB / ATTACK_VOCAB are hardcoded upper bounds (real max observed card
ID is 1267 as of 2026-07-01; enums may grow during the competition per the
official docs, hence the safety margin) rather than calling all_card_data() —
this keeps encode.py usable with or without cg-lib attached.
"""
import math

CARD_VOCAB = 2000
ATTACK_VOCAB = 2000
OPTION_TYPE_VOCAB = 17  # OptionType enum, 0..16
N_BOARD_SLOTS = 13      # my_active, my_bench x5, opp_active, opp_bench x5, stadium
MAX_ACTIONS = 64
NUM_FEATS = 13


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


def numeric_feats(obs):
    """Fixed-size float feature vector, roughly matching main.py's _census inputs."""
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
    return [
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
        min(cur.get("turn", 0) or 0, 60) / 60.0,
    ]


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
