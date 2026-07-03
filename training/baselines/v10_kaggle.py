"""Alakazam heuristic agent v6.

v6 critical fix: REMOVED false-positive wall detection that was suppressing
attacks against ALL Fighting/Colorless decks. The agent now attacks aggressively
whenever Alakazam has energy. Carries forward v5 smart switch-in and attach.

Rule: if Alakazam has energy and ATTACK is available, ATTACK. Period.
"""
from collections import Counter
import math

NUMBER, YES, NO, CARD, TOOL_CARD, ENERGY_CARD, ENERGY, PLAY, ATTACH, EVOLVE, \
    ABILITY, DISCARD, RETREAT, ATTACK, END, SKILL, SPECIAL_CONDITION = range(17)

POWERFUL_HAND_ATTACK_IDS = {1072}
ALAKAZAM_ATTACKER_IDS    = {743}
ALAKAZAM_LINE_IDS        = {741, 742, 743}
DRAW_ABILITY_CARD_IDS    = {742, 743, 66}
SUPPRESS_ABILITY_IDS     = {109}
BATTLE_CAGE_IDS          = {1264}
ENHANCED_HAMMER_IDS      = {1081}
BOSS_IDS                 = {1182}
PH_DMG_PER_CARD          = 20
NON_ATTACKER_IDS = {305, 65, 66, 142, 343, 858, 140, 741}

DECK = (
    [741]*4 + [742]*4 + [743]*3 + [305]*3 + [66]*3 +
    [142] + [343] + [858] + [140] +
    [1086]*4 + [1152]*4 + [1161]*2 + [1182]*3 + [1184] +
    [1264]*4 + [1231]*4 + [1146] + [1129] + [1225]*3 + [1081]*2 + [1079]*3 +
    [5]*2 + [13] + [19]*4
)
assert len(DECK) == 60

_CARDS = None; _ATKMAP = None
def _meta():
    global _CARDS, _ATKMAP
    if _CARDS is None:
        _CARDS = {}; _ATKMAP = {}
        try:
            from cg.api import all_card_data, all_attack
            for a in all_attack():
                _ATKMAP[a.attackId] = (getattr(a, 'damage', 0) or 0,
                                       tuple(getattr(a, 'energies', ()) or ()))
            for c in all_card_data():
                _CARDS[c.cardId] = c
        except Exception:
            pass
    return _CARDS, _ATKMAP

def _can_pay(cost, have):
    need = Counter(e for e in cost if e != 0)
    colorless = sum(1 for e in cost if e == 0)
    pool = Counter(have)
    for col, k in need.items():
        if pool[col] < k: return False
        pool[col] -= k
    return sum(pool.values()) >= colorless

def _hand_size(state, me):
    try:
        p = state['players'][me]
        h = p.get('handCount')
        return h if h is not None else len(p.get('hand') or [])
    except Exception:
        return 0

def _active(p):
    a = p.get('active')
    return a[0] if a and len(a) > 0 and a[0] else None

def _pk_id(pk): return (pk or {}).get('id', -1)
def _has_energy(pk): return bool((pk or {}).get('energies'))
def _active_can_ph(pk):
    return pk is not None and _pk_id(pk) in ALAKAZAM_ATTACKER_IDS and _has_energy(pk)
def _is_non_attacker(pk):
    if not pk: return True
    return _pk_id(pk) in NON_ATTACKER_IDS

def _affordable_dmg(pk, hand_size=0):
    if not pk: return 0
    C, A = _meta()
    have = pk.get('energies') or []
    c = C.get(_pk_id(pk)); best = 0
    if c:
        for aid in (getattr(c, 'attacks', []) or []):
            dmg, cost = A.get(aid, (0, ()))
            if not _can_pay(cost, have): continue
            if aid in POWERFUL_HAND_ATTACK_IDS or (_pk_id(pk) in ALAKAZAM_ATTACKER_IDS and dmg == 0):
                dmg = PH_DMG_PER_CARD * hand_size
            best = max(best, dmg)
    return best

def _clamp(indices, sel):
    mn = sel.get('minCount', 0) or 0
    mx = sel.get('maxCount', 1) or 1
    n  = len(sel.get('option', []))
    out = []
    for i in indices:
        if 0 <= i < n and i not in out: out.append(i)
        if len(out) >= mx: break
    i = 0
    while len(out) < mn and i < n:
        if i not in out: out.append(i)
        i += 1
    return out

def _pick_bench_target(obs, opts):
    cur = obs.get('current') or {}
    me_idx = cur.get('yourIndex', 0)
    players = cur.get('players', [])
    me = players[me_idx] if players and len(players) > me_idx else {}
    bench = me.get('bench') or []
    best_i, best_score = 0, -1
    for i, o in enumerate(opts):
        if o.get('area') == 5:
            idx = o.get('index', 0)
            if idx < len(bench):
                pk = bench[idx]; pid = _pk_id(pk)
                if pid in ALAKAZAM_ATTACKER_IDS and _has_energy(pk): s = 100
                elif pid in ALAKAZAM_ATTACKER_IDS: s = 80
                elif pid == 742: s = 60
                elif pid in NON_ATTACKER_IDS: s = 5
                else: s = 20
                if s > best_score: best_score, best_i = s, i
    return [best_i]

def _score_attach(o, my_active, bench):
    ta = o.get('inPlayArea', -1); ti = o.get('inPlayIndex', 0)
    if ta == 4: tid = _pk_id(my_active)
    elif ta == 5 and bench and ti < len(bench): tid = _pk_id(bench[ti])
    else: tid = -1
    if tid in ALAKAZAM_ATTACKER_IDS:
        pk = my_active if ta == 4 else (bench[ti] if bench and ti < len(bench) else None)
        return 50 if (pk and not _has_energy(pk)) else 30
    if tid == 742: return 15
    return 5

def _main_phase(obs, sel):
    opts = sel['option']; n = len(opts)
    if n == 0: return []
    cur = obs.get('current') or {}
    me_idx = cur.get('yourIndex', 0)
    players = cur.get('players', [])
    my  = players[me_idx] if players and len(players) > me_idx else {}
    opp = players[1 - me_idx] if players and len(players) == 2 else {}
    my_active  = _active(my)
    opp_active = _active(opp)
    hand = _hand_size(cur, me_idx)
    bench = my.get('bench') or []

    in_play = [x for x in ([my_active] + bench) if x]
    lone = len(in_play) <= 1

    active_non_atk = _is_non_attacker(my_active)
    active_can_attack = _active_can_ph(my_active)
    bench_has_alak = any(_pk_id(b) in ALAKAZAM_ATTACKER_IDS for b in bench if b)
    bench_has_alak_ready = any(
        _pk_id(b) in ALAKAZAM_ATTACKER_IDS and _has_energy(b) for b in bench if b
    )

    opp_hp = (opp_active or {}).get('hp', 99999) or 99999
    my_dmg = _affordable_dmg(my_active, hand)
    can_ko = my_active is not None and opp_active is not None and my_dmg >= opp_hp

    cards_needed = math.ceil(opp_hp / PH_DMG_PER_CARD) if opp_hp < 99999 else 999
    at_threshold = hand >= cards_needed and active_can_attack

    def score(o):
        ot  = o.get('type')
        cid = o.get('cardId') or o.get('id', -1)

        # ======== ATTACK — v6: ALWAYS attack if Alakazam has energy ========
        if ot == ATTACK:
            if active_non_atk:
                return -5                       # non-attacker can't deal meaningful dmg
            if can_ko:
                return 200                      # LETHAL — take it immediately
            if active_can_attack and at_threshold:
                return 150                      # at threshold — swing
            if active_can_attack:
                return 8.0                      # v6: attack even below threshold
                                                # (was 1.5 in v5 — too passive!)
            return 1.0                          # non-PH attack

        # ======== RETREAT ========
        if ot == RETREAT:
            if active_non_atk and bench_has_alak_ready:
                return 20.0
            if active_non_atk and bench_has_alak:
                return 15.0
            if active_non_atk:
                return 10.0
            # v6: NEVER retreat Alakazam that has energy — it should attack!
            if active_can_attack:
                return -2.0
            return 0.5

        # ======== ABILITY ========
        if ot == ABILITY:
            if cid in SUPPRESS_ABILITY_IDS: return -10
            if lone: return -10
            if cid in DRAW_ABILITY_CARD_IDS:
                # v6: draw is great, but if we can already KO, just attack
                if can_ko: return 2.0
                return 9.0
            return 6.0

        # ======== EVOLVE ========
        if ot == EVOLVE:
            if active_non_atk: return 12.0
            if can_ko: return 2.0               # v6: don't evolve when lethal
            return 8.5

        # ======== PLAY ========
        if ot == PLAY:
            if cid in BATTLE_CAGE_IDS: return 7.5
            if cid in ENHANCED_HAMMER_IDS: return 5.0
            if cid in BOSS_IDS: return 5.0
            if can_ko: return 1.0               # v6: suppress items when lethal
            if at_threshold: return 1.0
            return 5.0

        if ot == SKILL: return 5.0

        # ======== ATTACH ========
        if ot == ATTACH:
            if can_ko: return 0.5
            base = _score_attach(o, my_active, bench)
            if active_non_atk:
                ta = o.get('inPlayArea', -1); ti = o.get('inPlayIndex', 0)
                if ta == 5 and bench and ti < len(bench):
                    if _pk_id(bench[ti]) in ALAKAZAM_ATTACKER_IDS:
                        return 11.0
                return 2.0
            return min(base / 10.0 + 3.0, 6.0)

        if ot == DISCARD: return 0.0
        if ot == END:     return 1.0
        return 2.0

    best_i, best_s = 0, -1e18
    for i, o in enumerate(opts):
        s = score(o)
        if s > best_s:
            best_s, best_i = s, i
    return [best_i]

def _choose(obs):
    sel = obs.get('select')
    if sel is None: return DECK
    opts = sel.get('option', []); n = len(opts)
    if n == 0: return []
    stype = sel.get('type')
    mn = sel.get('minCount', 0) or 0
    mx = sel.get('maxCount', 1) or 1

    if stype == 0: return _main_phase(obs, sel)

    if stype == 1:
        has_bench = any(o.get('area') == 5 for o in opts)
        if has_bench:
            picks = _pick_bench_target(obs, opts)
        else:
            # yes/no check
            yes_i = [i for i, o in enumerate(opts) if o.get('type') == YES]
            if yes_i:
                picks = yes_i
            else:
                picks = list(range(n))   # default: all indices, clamp trims it
        return _clamp(picks, sel)

    _, A = _meta()
    if stype == 6:
        return [max(range(n), key=lambda i: (A.get(opts[i].get('attackId'), (0,))[0] or 1))]
    if stype == 8:
        return [max(range(n), key=lambda i: opts[i].get('number', 0) or 0)]
    if stype == 9:
        for i, o in enumerate(opts):
            if o.get('type') == YES: return [i]
        return [0]
    if stype == 4:
        return _clamp(list(range(n)), sel)
    k = mn if mn > 0 else (1 if mx >= 1 else 0)
    return _clamp(list(range(n))[:k] if k else [], sel)

def _safe_return(result, sel):
    """Final safety: ensure result satisfies minCount/maxCount and all indices in range."""
    if not sel: return result
    n  = len(sel.get('option', []))
    mn = sel.get('minCount', 0) or 0
    mx = sel.get('maxCount', 1) or 1
    if not isinstance(result, list): result = [0]
    # filter out-of-range
    result = [i for i in result if 0 <= i < n]
    # trim to max
    result = result[:mx]
    # pad to min
    i = 0
    while len(result) < mn and i < n:
        if i not in result: result.append(i)
        i += 1
    return result if result else ([0] if n > 0 else [])

def agent(obs_dict):
    try:
        sel = obs_dict.get('select')
        out = _choose(obs_dict)
        if isinstance(out, list):
            return _safe_return(out, sel) if sel else out
    except Exception: pass
    try:
        sel = obs_dict.get('select')
        if not sel: return []
        n = len(sel.get('option', []))
        if n == 0: return []
        mn = sel.get('minCount', 1) or 0
        result = list(range(min(max(mn, 1), n)))
        return _safe_return(result, sel)
    except Exception: return [0]
