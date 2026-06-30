"""
Mega Lucario ex training opponent for self-play pool.
Archetype: Fighting aggro + Rocky Energy lock. Stage 1 megaEx (340 HP, 3 prizes).
  Fighting Fury: ~190 vs ex targets with Korrina + Arena of Antiquity buffs.
  Rocky Fighting Energy (ID=20) on Lucario ex → blocks Powerful Hand effects → Alakazam deals 0.

TODO: Fill DECK with real card IDs.
  In a Kaggle notebook: {c.cardId: c.name for c in all_card_data()}
  Key known ID: ROCK_ENERGY = 20 (Rocky Fighting Energy, ACE SPEC).

Strategy encoded here:
  - Bench Riolu ASAP, evolve to Lucario ex
  - ATTACH Rocky Energy (20) to Lucario ex as top priority — this is the win condition
    (once attached, Alakazam's Powerful Hand places 0 damage counters)
  - Attack every turn
  - Never retreat Lucario ex — Rocky Energy stays attached, keep pressure on
  - Play Korrina + Arena of Antiquity before attacking for max damage buff
  - Do NOT bench Fezandipiti ex (2-prize target that hurts us vs single-prize Alakazam)
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
ROCK_ENERGY = 20   # Rocky Fighting Energy — CONFIRMED ID; blocks Powerful Hand

# TODO: replace 0 with real IDs
RIOLU      = 0   # Basic Fighting, ~60-80 HP
LUCARIO_EX = 0   # Stage 1 megaEx, 340 HP, 3 prizes

# Build a real 60-card Lucario ex deck once IDs are known.
# Typical: 4 Riolu + 3 Lucario ex + ROCK_ENERGY + fighting energy + Korrina + Arena
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

def _has_rocky(pk):
    return ROCK_ENERGY in ((pk or {}).get('energies') or [])

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

# ── Main phase ────────────────────────────────────────────────────────────────
def _main_phase(obs, sel):
    opts = sel['option']
    n    = len(opts)
    _, me, my, opp = _players(obs)
    my_active = _active(my)
    my_bench  = _bench(my)
    hand      = _hand(my)

    active_is_mega = (my_active or {}).get('megaEx', False)
    rocky_on_active = _has_rocky(my_active)

    def score(o):
        ot = o.get('type')
        if ot == ATTACK: return 100.0
        if ot == EVOLVE: return 25.0   # always evolve toward Lucario ex ASAP
        if ot == ATTACH:
            ta  = o.get('inPlayArea', -1)
            ti  = o.get('inPlayIndex', 0)
            tgt = my_active if ta == 4 else (my_bench[ti] if ta == 5 and 0 <= ti < len(my_bench) else None)
            idx = o.get('index', -1)
            cid = (hand[idx] or {}).get('id', -1) if 0 <= idx < len(hand) else -1
            pv  = _prize_val(tgt)
            # Rocky Energy on Lucario ex is the win condition — top priority
            if cid == ROCK_ENERGY and pv == 3:  return 35.0
            if cid == ROCK_ENERGY:               return 20.0
            return 10.0 + pv * 4.0
        if ot == PLAY: return 5.0
        if ot == RETREAT:
            # Never retreat Lucario ex — Rocky Energy stays with it, forcing the lock
            if active_is_mega: return -50.0
            return 3.0
        if ot == END: return 0.5
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
                    if o.get('cardId', o.get('id', -1)) == RIOLU: return [i]
                return [0]
            if ctx == CTX_SETUP_BENCH: return list(range(n))
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
