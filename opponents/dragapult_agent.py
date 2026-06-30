"""
Dragapult ex training opponent for self-play pool.
Archetype: Stage 2 spread. Phantom Dive: 200 to Active + 60 to ALL bench every turn.
  Wipes Alakazam's evolution line (Abra/Kadabra on bench take 60 each turn).
  Forces Battle Cage or line-wiped Alakazam setup.

TODO: Fill DECK with real card IDs.
  In a Kaggle notebook: {c.cardId: c.name for c in all_card_data()}

Strategy encoded here:
  - Stage 2 line: Dreepy → Drakloak → Dragapult ex. Speed is everything.
  - Rare Candy (skip Drakloak) gets Dragapult ex out fastest — score EVOLVE very high
  - Attack EVERY turn — 60 spread accumulates; Abra dies in 1 hit, Kadabra in 2-3
  - Energy on Dragapult ex (highest maxHp pokemon) — use maxHp as routing signal
  - Never retreat — spread resets opponent's decision math if Dragapult leaves
  - Phantom Dive selects a bench target: pick highest-value or most damaged
"""
import sys, glob

for _p in ['/kaggle/input/**/cg-lib', '/kaggle/input/cg-lib']:
    _m = glob.glob(_p, recursive=True)
    if _m: sys.path.insert(0, _m[0]); break

try:
    from cg.api import all_attack
    _ATK = {a.attackId: getattr(a, 'damage', 0) or 0 for a in all_attack()}
except Exception:
    _ATK = {}

NUMBER, YES, NO, CARD, TOOL_CARD, ENERGY_CARD, ENERGY, PLAY, ATTACH, EVOLVE, \
    ABILITY, DISCARD, RETREAT, ATTACK, END, SKILL, SPECIAL_CONDITION = range(17)
CTX_SETUP_ACTIVE = 1
CTX_SETUP_BENCH  = 2

# ── Card IDs ─────────────────────────────────────────────────────────────────
# TODO: replace 0 with real IDs
DREEPY       = 0   # Basic Psychic/Dragon, ~60 HP
DRAKLOAK     = 0   # Stage 1
DRAGAPULT_EX = 0   # Stage 2 ex (or megaEx), high HP

# Build a real 60-card Dragapult ex deck once IDs are known.
# Typical: 4 Dreepy + 3 Drakloak + 3 Dragapult ex + Rare Candy + Psychic energy
DECK = [0] * 60   # TODO: 60 real card IDs

# ── Helpers ──────────────────────────────────────────────────────────────────
def _pk_id(pk):  return (pk or {}).get('id', -1)
def _active(p):  a = p.get('active'); return a[0] if a and a[0] else None
def _bench(p):   return p.get('bench') or []
def _hand(p):    return p.get('hand') or []

def _prize_val(pk):
    if not pk: return 0
    if pk.get('megaEx'): return 3
    if pk.get('ex'):     return 2
    return 1

def _clamp(idxs, sel):
    mn = sel.get('minCount', 0) or 0
    mx = sel.get('maxCount', 1) or 1
    n  = len(sel.get('option', []))
    out = []
    for i in idxs:
        if 0 <= i < n and i not in out: out.append(i)
        if len(out) >= mx: break
    j = 0
    while len(out) < mn and j < n:
        if j not in out: out.append(j)
        j += 1
    return out

def _players(obs):
    cur = obs.get('current') or {}
    me  = cur.get('yourIndex', 0)
    pl  = cur.get('players', [])
    my  = pl[me]   if len(pl) > me  else {}
    opp = pl[1-me] if len(pl) == 2  else {}
    return cur, me, my, opp

# ── Phantom Dive bench target: pick highest prize value, then lowest HP ───────
def _pick_spread_target(obs, sel):
    _, me, _, opp = _players(obs)
    opp_bench = _bench(opp)
    opts = sel.get('option', [])
    best_i, best_pv, best_hp = 0, -1, 99999
    for i, o in enumerate(opts):
        bi = o.get('index', 0)
        pk = opp_bench[bi] if 0 <= bi < len(opp_bench) else None
        pv = _prize_val(pk)
        hp = (pk or {}).get('hp', 99999) or 99999
        if pv > best_pv or (pv == best_pv and hp < best_hp):
            best_pv, best_hp, best_i = pv, hp, i
    return [best_i]

# ── Main phase ────────────────────────────────────────────────────────────────
def _main_phase(obs, sel):
    opts = sel['option']
    n    = len(opts)
    _, me, my, opp = _players(obs)
    my_active = _active(my)
    my_bench  = _bench(my)

    active_is_ex = (my_active or {}).get('ex', False) or (my_active or {}).get('megaEx', False)

    def score(o):
        ot = o.get('type')
        if ot == ATTACK:  return 100.0
        if ot == EVOLVE:  return 30.0   # Stage 2 setup is the entire game plan
        if ot == ATTACH:
            ta     = o.get('inPlayArea', -1)
            ti     = o.get('inPlayIndex', 0)
            tgt    = my_active if ta == 4 else (my_bench[ti] if ta == 5 and 0 <= ti < len(my_bench) else None)
            pv     = _prize_val(tgt)
            max_hp = (tgt or {}).get('maxHp', 0) or 0
            # Route energy to the highest HP pokemon — that's Dragapult ex
            return 10.0 + pv * 3.0 + max_hp / 50.0
        if ot == PLAY:    return 5.0
        if ot == RETREAT:
            # Keep Dragapult ex active — spread needs consistent turns
            if active_is_ex: return -20.0
            return 5.0   # switch in if current active can't attack
        if ot == END:     return 0.5
        return 1.0

    return [max(range(n), key=lambda i: score(opts[i]))]

# ── Agent entry point ─────────────────────────────────────────────────────────
def agent(obs_dict: dict) -> list:
    try:
        sel  = obs_dict.get('select')
        if sel is None: return DECK
        opts = sel.get('option', [])
        n    = len(opts)
        if n == 0: return []
        stype = sel.get('type')
        ctx   = sel.get('context', 0)
        mn    = sel.get('minCount', 0) or 0
        mx    = sel.get('maxCount', 1) or 1

        if stype == 0: return _main_phase(obs_dict, sel)
        if stype == 1:
            if ctx == CTX_SETUP_ACTIVE:
                for i, o in enumerate(opts):
                    if o.get('cardId', o.get('id', -1)) == DREEPY: return [i]
                return [0]
            if ctx == CTX_SETUP_BENCH: return list(range(n))
            # Check for Phantom Dive bench target selection
            cur = obs_dict.get('current') or {}
            me  = cur.get('yourIndex', 0)
            if all(o.get('playerIndex') == (1-me) and o.get('area') == 5 for o in opts):
                return _pick_spread_target(obs_dict, sel)
            yes_i = [i for i, o in enumerate(opts) if o.get('type') == YES]
            return yes_i if yes_i else [0]
        if stype in (2, 3, 4, 7): return _clamp(list(range(n)), sel)
        if stype == 5:             return [0]
        if stype == 6:
            return [max(range(n), key=lambda i: _ATK.get(opts[i].get('attackId'), 0))]
        if stype == 9:
            for i, o in enumerate(opts):
                if o.get('type') == YES: return [i]
            return [0]
        k = mn if mn > 0 else (1 if mx >= 1 else 0)
        return _clamp(list(range(n))[:k] if k else [], sel)
    except Exception:
        sel = obs_dict.get('select')
        if not sel: return DECK
        n = len(sel.get('option', [])); mn = sel.get('minCount', 0) or 0
        return list(range(min(max(mn, 1), n))) if n else []
