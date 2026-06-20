"""
================================================================================
POKÉMON TCG AI BATTLE CHALLENGE — MEGA LUCARIO ex AGENT v5 (NOTEBOOK)
================================================================================

Full strategy notebook combining:
  1. The complete strategy document
  2. Cell-by-cell Kaggle notebook structure (Cells 1–6)
  3. Self-contained agent with evaluator
  4. Local validation harness

To use on Kaggle:
  - Create a new notebook in the competition (auto-attaches competition data).
  - Copy each CELL (marked below) into a separate notebook cell, in order.
  - Paste as Code cells, run sequentially.
  - Save Version and submit from the competition's Submit tab.

================================================================================
SECTION 0: THE MEGA LUCARIO ex STRATEGY — COMPLETE REFERENCE
================================================================================

COMPETITION CONTEXT
  - Kaggle simulation competition on the cabt C engine (libcg.so).
  - Two tracks: Simulation (ladder) and Strategy (report).
  - Decision interface: agent(obs_dict) -> list[int] of legal option indices.
  - No rule reimplementation needed: search the engine and evaluate boards.

THE DECK — Mega Lucario ex (mono-Fighting aggressive)
  Primary attacker:  Mega Lucario ex (678, 340 HP, Stage 1 from Riolu 333)
    - Aura Jab (1F):   130 damage + accelerate 3 Fighting from discard to bench
    - Mega Brave (2F): 270 damage, cannot use next turn (cooldown)
  Secondary:         Bloodmoon Ursaluna ex (135, Basic, Fighting)
  Consistency:       4x Riolu, draw/search Trainers, 15x Fighting Energy
  Strategy: 340-HP wall survives and swings again (prize tempo win).

AGENT ARCHITECTURE — Determinized Forward Search
  1. On MAIN decisions with >=2 options:
     a. Determinize hidden info (deck order, prizes, opp deck/hand).
     b. For each legal option: search_step, rollout YOUR turn, evaluate leaf.
     c. Keep the option with highest evaluate value.
  2. Any error -> fall back to a rule-based fast policy.

THE EVALUATOR — Prize-Trade Anchored (weight scale = 1000 pts/prize)
  PRIZE (1000/prize)           opp_prizes_left - my_prizes_left
  KO_THEM (350 x pv)           can I KO the opponent active right now?
  KO_ME (480 x pv x 1.4×if_ex) can they KO me next turn?
  SURVIVE (up to 240)          my active hp survives their output
  ACTIVE_READY (120)           my active can afford an attack
  BACKUP (170 x <=2)           benched attackers ready to promote
  CONDITIONS (±220/170)        my locked? opp locked?
  BENCH_EXPOSURE (-110/ex)     benched ex beyond 1 is a liability
  DRAW_ACCESS (45 x <=3)       hold draw/search cards
  HAND_SIZE (15 x range)       baseline 3, cap 8
  DECKOUT (20)                 late-game deep-grind

================================================================================
"""

import glob
import os
import random
import dataclasses
from collections import Counter

try:
    import kaggle_environments
    KE = os.path.dirname(kaggle_environments.__file__)
except Exception:
    KE = None

DECK = [
    333, 333, 333, 333,           # Riolu
    678, 678, 678, 678,           # Mega Lucario ex
    135, 135, 135,                # Bloodmoon Ursaluna ex
    1224, 1224, 1224, 1224,       # Trainer
    1192, 1192, 1192, 1192,       # Trainer
    1213, 1213, 1213, 1213,       # Trainer
    1182, 1182, 1182,             # Trainer
    1121, 1121, 1121, 1121,       # Trainer
    1125,                          # Master Ball (ACE SPEC)
    1119, 1119, 1119, 1119,       # Trainer
    1116, 1116, 1116, 1116,       # Trainer
    1123, 1123, 1123,             # Trainer
    1252, 1252, 1252,             # Trainer
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6  # Fighting Energy
]
assert len(DECK) == 60

NUMBER, YES, NO, CARD, TOOL_CARD, ENERGY_CARD, ENERGY, PLAY, ATTACH, EVOLVE, \
    ABILITY, DISCARD, RETREAT, ATTACK, END, SKILL, SPECIAL_CONDITION = range(17)

STRONG_ATTACK = 110
ACTIONS_PER_TURN_CAP = 80
CHARGE_TO_BEST = True

W_PRIZE = 1000.0
W_KO_THEM = 350.0
W_KO_ME = 480.0
W_SURVIVE = 240.0
W_ACTIVE_READY = 120.0
W_BACKUP = 170.0
W_CONDITIONS_SELF = 220.0
W_CONDITIONS_OPP = 170.0
W_BENCH_EXPOSURE = 110.0
W_DRAW_ACCESS = 45.0
W_HAND = 15.0
W_DECKOUT = 20.0
BIG = 1e9

_CARDS = None
_ATTACKS = None

def _meta():
    """Load card metadata from the engine."""
    global _CARDS, _ATTACKS
    if _CARDS is None:
        _CARDS = {}
        _ATTACKS = {}
        try:
            from cg.api import all_card_data, all_attack
            for c in all_card_data():
                _CARDS[c.cardId] = c
            for a in all_attack():
                if a.attackId not in _ATTACKS:
                    _ATTACKS[a.attackId] = []
                _ATTACKS[a.attackId].append((a.damage or 0, tuple(a.energies or [])))
        except Exception:
            pass
    return _CARDS, _ATTACKS

def _prize_value(cid):
    """Return prize value: 3 if megaEx, 2 if ex, else 1."""
    C, _ = _meta()
    c = C.get(cid)
    if c:
        if getattr(c, 'megaEx', False):
            return 3
        if getattr(c, 'ex', False):
            return 2
    return 1

def _is_ex(cid):
    """Return True if card is ex or megaEx."""
    C, _ = _meta()
    c = C.get(cid)
    return bool(c and (getattr(c, 'ex', False) or getattr(c, 'megaEx', False)))

def _can_pay(cost, have):
    """Check if energy list `have` can pay `cost`."""
    need = Counter(e for e in cost if e != 0)
    colorless = sum(1 for e in cost if e == 0)
    pool = Counter(have)
    for col, k in need.items():
        if pool[col] < k:
            return False
        pool[col] -= k
    return sum(pool.values()) >= colorless

def _affordable_dmg(pk):
    """Return best damage this Pokémon can deal with current energy."""
    if not pk:
        return 0
    C, A = _meta()
    have = pk.get("energies") or []
    c = C.get(pk.get("id"))
    best = 0
    if c:
        for aid in getattr(c, 'attacks', []):
            if aid in A:
                for dmg, cost in A[aid]:
                    if _can_pay(cost, have):
                        best = max(best, dmg)
    return best

def _damage_to(att, dfn):
    """Compute damage from attacker to defender, including Weakness."""
    if not att or not dfn:
        return 0
    C, _ = _meta()
    d = _affordable_dmg(att)
    if d == 0:
        return 0
    ca = C.get(att.get("id"))
    cd = C.get(dfn.get("id"))
    if ca and cd:
        att_type = getattr(ca, 'energyType', None)
        weakness = getattr(cd, 'weakness', None)
        if weakness and att_type == weakness:
            d *= 2
    return d

def _attack_ceiling(pk):
    """Return highest-damage attack this Pokémon has."""
    C, A = _meta()
    if not pk:
        return 0
    c = C.get(pk.get("id"))
    if not c:
        return 0
    dmgs = []
    for aid in getattr(c, 'attacks', []):
        if aid in A:
            for dmg, _ in A[aid]:
                dmgs.append(dmg)
    return max(dmgs) if dmgs else 0

_state = {"turn": None, "actions": 0}

def _clamp(indices, sel):
    """Clamp indices to selection bounds."""
    mn, mx, n = sel["minCount"], sel["maxCount"], len(sel["option"])
    out = []
    for i in indices:
        if 0 <= i < n and i not in out:
            out.append(i)
        if len(out) >= mx:
            break
    i = 0
    while len(out) < mn and i < n:
        if i not in out:
            out.append(i)
        i += 1
    return out

def _quick_policy(obs_dict, deck):
    """Fast fallback rule policy."""
    sel = obs_dict.get("select")
    if sel is None:
        return deck
    opts = sel["option"]
    n = len(opts)
    if n == 0:
        return []
    
    mn, mx = sel["minCount"], sel["maxCount"]
    stype = sel["type"]
    cur = obs_dict.get("current") or {}
    t = cur.get("turn")
    
    if t != _state["turn"]:
        _state["turn"] = t
        _state["actions"] = 0
    _state["actions"] += 1
    panic = _state["actions"] > ACTIONS_PER_TURN_CAP
    
    if stype == 0:
        me = cur.get("yourIndex", 0)
        players = cur.get("players", [])
        active = (players[me]["active"][0] if players and players[me].get("active") else None)
        opp = players[1 - me] if players and len(players) == 2 else None
        oa = opp["active"][0] if opp and opp.get("active") else None
        opp_hp = oa.get("hp", 9999) if oa else 9999
        ba = _affordable_dmg(active)
        bm = _attack_ceiling(active)
        has_attach = any(o.get("type") == ATTACH for o in opts)
        can_ko = ba > 0 and ba >= opp_hp
        
        if CHARGE_TO_BEST:
            ready = can_ko or (ba >= STRONG_ATTACK and (ba >= bm or not has_attach))
        else:
            ready = ba >= STRONG_ATTACK or can_ko
        
        PRI = {ABILITY: 6, EVOLVE: 5, PLAY: 4, ATTACH: 3, SKILL: 4, END: 1, RETREAT: 0, DISCARD: 0}
        bi, bs = 0, -1e9
        for i, o in enumerate(opts):
            ot = o.get("type")
            if ot == ATTACK:
                s = 7 if ready else 1.5
            else:
                s = PRI.get(ot, 2)
                if ot == ATTACH and o.get("inPlayArea") == 4:
                    s += 0.5
            if s > bs:
                bs, bi = s, i
        
        if panic:
            for i, o in enumerate(opts):
                if o.get("type") == ATTACK:
                    return [i]
            for i, o in enumerate(opts):
                if o.get("type") == END:
                    return [i]
        
        return [bi]
    
    if stype == 6:
        _, A = _meta()
        return [max(range(n), key=lambda i: max((x[0] for x in A.get(opts[i].get("attackId"), [(0, ())])), default=0))]
    if stype == 8:
        return [max(range(n), key=lambda i: opts[i].get("number", 0) or 0)]
    if stype == 9:
        for i, o in enumerate(opts):
            if o.get("type") == YES:
                return [i]
        return [0]
    if stype == 4:
        return _clamp(list(range(n)), sel)
    
    k = mn if mn > 0 else (1 if mx >= 1 else 0)
    return _clamp(list(range(n))[:k] if k else [], sel)

def _active(p):
    """Get active Pokémon from PlayerState."""
    a = p.get("active")
    return a[0] if a and len(a) > 0 and a[0] else None

def _hp_tot(p):
    """Sum total current HP."""
    return sum(
        (x.get("hp", 0) or 0)
        for x in (p.get("active") or []) + (p.get("bench") or [])
        if x
    )

def evaluate(state, me):
    """Evaluate board state. Higher = better FOR YOU."""
    if state is None:
        return 0.0
    
    P = state.get("players", [])
    if len(P) < 2:
        return 0.0
    
    mp = P[me]
    op = P[1 - me]
    res = state.get("result", -1)
    
    if res == me:
        return BIG
    if res != -1:
        return -BIG
    
    mypz = len(mp.get("prize") or [])
    oppz = len(op.get("prize") or [])
    
    if mypz == 0:
        return BIG
    if oppz == 0:
        return -BIG
    
    ma = _active(mp)
    oa = _active(op)
    mb = [x for x in (mp.get("bench") or []) if x]
    ob = [x for x in (op.get("bench") or []) if x]
    
    if oa is None and not ob:
        return BIG
    if ma is None and not mb:
        return -BIG
    
    s = W_PRIZE * (oppz - mypz)
    s -= 0.20 * _hp_tot(op)
    s += 0.10 * _hp_tot(mp)
    
    if ma and oa:
        oah = oa.get("hp", 0) or 0
        mah = ma.get("hp", 0) or 0
        my_dmg = _damage_to(ma, oa)
        opp_dmg = _damage_to(oa, ma)
        
        if oah > 0 and my_dmg >= oah:
            s += W_KO_THEM * _prize_value(oa.get("id"))
        
        if mah > 0 and opp_dmg >= mah:
            ex_mult = 1.4 if _is_ex(ma.get("id")) else 1.0
            s -= W_KO_ME * ex_mult * _prize_value(ma.get("id"))
        
        if mah > 0 and opp_dmg < mah:
            buffer = mah - opp_dmg
            surv_score = W_SURVIVE * min(1.0, (buffer ** 0.5) / 16.0)
            s += surv_score
    
    if ma and _affordable_dmg(ma) > 0:
        s += W_ACTIVE_READY
    
    backup_count = 0
    for pk in mb:
        if pk.get("id") in [333, 678, 135]:
            if _affordable_dmg(pk) > 0:
                backup_count += 1
    s += W_BACKUP * min(2, backup_count)
    
    unneeded_ex = max(0, sum(1 for x in mb if _is_ex(x.get("id"))) - 1)
    s -= W_BENCH_EXPOSURE * unneeded_ex
    
    ma_locked = ma and (
        ma.get("asleep") or ma.get("paralyzed") or ma.get("confused")
    )
    oa_locked = oa and (
        oa.get("asleep") or oa.get("paralyzed") or oa.get("confused")
    )
    
    if ma_locked:
        s -= W_CONDITIONS_SELF
    if oa_locked:
        s += W_CONDITIONS_OPP
    
    hc = mp.get("handCount", 0) or 0
    s += W_HAND * (min(hc, 8) - 3)
    
    my_deck = mp.get("deckCount", 0) or 0
    opp_deck = op.get("deckCount", 0) or 0
    if min(my_deck, opp_deck) <= 6:
        s += W_DECKOUT * (my_deck - opp_deck)
    
    return s

def _vis(state, me):
    """Return visible card ids for player `me`."""
    seen = []
    p = state.get("players", [{}])[me]
    for c in (p.get("hand") or []):
        seen.append(c.get("id"))
    for c in (p.get("discard") or []):
        seen.append(c.get("id"))
    
    def add(pk):
        if not pk:
            return
        seen.append(pk.get("id"))
        for e in (pk.get("energyCards") or []):
            seen.append(e.get("id"))
        for tl in (pk.get("tools") or []):
            seen.append(tl.get("id"))
        for pe in (pk.get("preEvolution") or []):
            seen.append(pe.get("id"))
    
    for pk in (p.get("active") or []):
        add(pk)
    for pk in (p.get("bench") or []):
        add(pk)
    
    return seen

def _determinize(obs, my_deck):
    """Determinize hidden information for search."""
    state = obs.get("current", {})
    me = state.get("yourIndex", 0)
    opp = 1 - me
    mp = state.get("players", [{}])[me]
    op = state.get("players", [{}])[opp]
    
    rem = list((Counter(my_deck) - Counter(_vis(state, me))).elements())
    random.shuffle(rem)
    nd = mp.get("deckCount", 0)
    npz = len(mp.get("prize", []))
    while len(rem) < nd + npz:
        rem.append(my_deck[0])
    yd = rem[:nd]
    yp = rem[nd:nd + npz]
    
    pool = list(my_deck) * 3
    random.shuffle(pool)
    od = pool[:op.get("deckCount", 0)]
    opz = pool[:len(op.get("prize", []))]
    oh = pool[:op.get("handCount", 0)]
    oa = [678] if (op.get("active") and op["active"] and op["active"][0] is None) else []
    
    return yd, yp, od, opz, oh, oa

def _d(x):
    """Convert dataclass to dict."""
    return dataclasses.asdict(x) if (x is not None and dataclasses.is_dataclass(x)) else x

def _rollout(child, me, deck, max_steps=64):
    """Rollout from child state: finish YOUR turn, evaluate leaf."""
    try:
        from cg.api import search_step
    except Exception:
        return evaluate(_d(child.observation.current) if hasattr(child, 'observation') else None, me)
    
    so = child.observation
    steps = 0
    
    while steps < max_steps:
        cur = _d(so.current)
        sel = so.select
        
        if cur is not None and cur.get("result", -1) != -1:
            return BIG if cur["result"] == me else -BIG
        
        if sel is None:
            break
        
        if cur is not None and cur.get("yourIndex", me) != me:
            break
        
        try:
            child = search_step(child.searchId, _quick_policy(_d(so), deck))
        except Exception:
            break
        
        so = child.observation
        steps += 1
    
    return evaluate(_d(so.current), me)

def _search_choose(obs_dict, deck):
    """Determinized forward search on MAIN decisions."""
    sel = obs_dict.get("select")
    if sel is None:
        return deck
    
    if sel.get("type") != 0 or len(sel.get("option", [])) < 2:
        return _quick_policy(obs_dict, deck)
    
    try:
        from cg.api import to_observation_class, search_begin, search_step, search_end
        
        o = to_observation_class(obs_dict)
        yd, yp, od, op, oh, oa = _determinize(obs_dict, deck)
        root = search_begin(o, yd, yp, od, op, oh, oa, manual_coin=False)
        me = obs_dict["current"]["yourIndex"]
        best_i, best_v = 0, -1e18
        
        for i in range(len(sel["option"])):
            try:
                child = search_step(root.searchId, [i])
            except Exception:
                continue
            
            v = _rollout(child, me, deck)
            if v > best_v:
                best_v, best_i = v, i
        
        search_end()
        return [best_i]
    
    except Exception:
        return _quick_policy(obs_dict, deck)

def agent(obs_dict):
    """Main agent entry point."""
    return _search_choose(obs_dict, DECK)
