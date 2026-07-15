"""Zero-sum-consistent threat estimation, per a user design session
2026-07-05: Φ (docs/nn-training.md, phi_baseline.py) is NOT actually
zero-sum -- prize_diff is antisymmetric (flips sign from the opponent's
seat), but hand_advantage/wall_penalty/line_progress are one-sided "my own
progress" measures with no opposing term. This module builds a genuinely
antisymmetric term: `my_threat_against(opp) - opp_threat_against(me)`, where
threat_against is a WELL-DEFINED one-sided function of state (independent of
whose "turn" it is), so the difference is antisymmetric by construction --
evaluated from the other seat, the two terms swap and the sign flips
automatically, matching the real 2-player zero-sum game.

Threat estimate: for a side's Active Pokemon, look up its REAL attacks
(cardId -> attackIds -> (damage, energy cost) via the local cg.api shim,
same real card database used by main.py's own deck), and score each attack
by how much of the opponent's current Active HP it could deal, discounted by
how many more turns of energy attachment (assumed 1/turn, ignoring color
requirements) would be needed to afford it:

    attack_threat = min(1, damage / defender_hp) / (1 + turns_to_afford)
    threat(attacker, defender) = max over attacker's known attacks

KNOWN LIMITATION, found and accepted rather than hidden during a real check
against `all_attack()`: static `Attack.damage` is 0 for our own Powerful
Hand (its real damage is computed dynamically from hand size via the skill
text, not a static field) -- this generalizes to ANY attack with
conditional/scaling damage, not just ours. This will systematically
undercount such attacks. Energy cost also ignores per-type color
requirements (treats `len(attack.energies)` as a fungible count) -- both are
deliberate simplifications, not oversights; a color-aware bag-matching
implementation would be the natural next refinement if this proves useful.

Card database source: `cg.api` when importable (training/setup_local_search.py's
local shim), else the bundled `card_tables.json` dump of the same two tables.
The fallback exists because Kaggle's agent sandbox has NO `cg` module — the
2026-07-12 DMC live-read submission (54624481) died on this exact top-level
import chain (encode.py -> threat.py -> cg.api) at validation
(episode 85648727), so any submission shipping encode.py must bundle the JSON.
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_LOCAL_CG = os.path.join(_REPO_ROOT, "training", "local_cg")
if _LOCAL_CG not in sys.path:
    sys.path.insert(0, _LOCAL_CG)

_CARD_ATTACKS = None
_ATTACK_INFO = None


def _load_tables():
    global _CARD_ATTACKS, _ATTACK_INFO
    if _CARD_ATTACKS is None:
        try:
            from cg.api import all_card_data, all_attack
            _CARD_ATTACKS = {c.cardId: c.attacks for c in all_card_data()}
            _ATTACK_INFO = {a.attackId: (a.damage, len(a.energies))
                            for a in all_attack()}
        except ImportError:
            with open(os.path.join(_HERE, "card_tables.json"),
                      encoding="utf-8") as f:
                t = json.load(f)
            _CARD_ATTACKS = {int(k): v for k, v in t["card_attacks"].items()}
            _ATTACK_INFO = {int(k): tuple(v) for k, v in t["attack_info"].items()}


def _pk_id(pk):
    return (pk or {}).get("id", -1)


def _active(p):
    a = p.get("active")
    return a[0] if a and len(a) > 0 and a[0] else None


def threat_against(attacker_pk, defender_pk):
    """One-sided: how threatening is attacker_pk to defender_pk right now,
    in [0, 1] (0 if either side is missing/unknown or attacker has no
    attacks on record). Well-defined regardless of whose "turn" it is --
    same inputs always give the same output."""
    _load_tables()
    if not attacker_pk or not defender_pk:
        return 0.0
    defender_hp = defender_pk.get("hp") or 0
    if defender_hp <= 0:
        return 0.0
    attack_ids = _CARD_ATTACKS.get(_pk_id(attacker_pk)) or []
    if not attack_ids:
        return 0.0
    current_energy = len(attacker_pk.get("energies") or [])
    best = 0.0
    for aid in attack_ids:
        info = _ATTACK_INFO.get(aid)
        if not info:
            continue
        damage, needed = info
        if damage <= 0:
            continue  # 0-damage attacks (utility moves, or our own dynamic
            # Powerful Hand) carry no static threat signal here -- see
            # module docstring's known-limitation note
        turns_to_afford = max(0, needed - current_energy)
        score = min(1.0, damage / defender_hp) / (1 + turns_to_afford)
        best = max(best, score)
    return best


def net_threat_diff(cur, me_idx):
    """Antisymmetric by construction: evaluated with me_idx swapped to the
    opponent's seat, this returns the negation of this call's result, since
    it is literally my_threat - opp_threat computed from the SAME
    perspective-independent threat_against() calls either way."""
    players = cur.get("players") or []
    if len(players) != 2:
        return 0.0
    my_p, opp_p = players[me_idx], players[1 - me_idx]
    my_active, opp_active = _active(my_p), _active(opp_p)
    my_threat = threat_against(my_active, opp_active)
    opp_threat = threat_against(opp_active, my_active)
    return my_threat - opp_threat
