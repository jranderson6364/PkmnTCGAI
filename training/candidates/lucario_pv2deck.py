"""Mega Lucario ex pilot (v31L) — Fighting aggro, multi-attacker.

Deck switch from Alakazam single-prize control, 2026-07-23, on the user's call
after the public-notebook survey (docs/competitor-notebook-survey.md):
Mega Lucario ex reads 76.4% archetype win rate vs Alakazam's -238 aggregate
delta, and the two strongest public agents (933.8 and LB 1091) both pilot it.

Written from the CARD MECHANICS, not from any competitor's source. The 60-card
list is the official sample list (a game fact, published by the organizers);
every scoring decision below is ours. Reuses the deck-agnostic crash-safety and
stall-resolution scaffolding proven across v22-v30 of the Alakazam pilot.

Engine, in one paragraph: Riolu -> Mega Lucario ex is a STAGE 1 line (no Rare
Candy). Aura Jab (1 energy, 130) attaches up to 3 Basic F from the DISCARD to
the bench, so it is the accelerator, not just an attack. Lunatone's Lunar Cycle
discards a Basic F from hand to draw 3 — which FEEDS the discard that Aura Jab
harvests, so discarding energy is a tempo gain, not a loss. Mega Brave (2
energy, 270) is the closer but locks itself for a turn, so the loop alternates
Aura Jab -> Mega Brave. Hariyama's Heave-Ho Catcher gusts on evolution, giving
the deck four gust effects (2 Boss + 2 Hariyama). Solrock's Cosmic Beam ignores
Weakness AND Resistance, which is the answer to resistance walls. Gravity
Mountain (-30 HP to every Stage 2 in play) is asymmetric: we are Stage 1.
"""
import sys, glob

for _pat in ['/kaggle/input/**/cg-lib', '/kaggle/input/cg-lib']:
    _paths = glob.glob(_pat, recursive=True)
    if _paths: sys.path.insert(0, _paths[0]); break

try:
    from cg.api import all_card_data
except ImportError:
    all_card_data = lambda: []

NUMBER,YES,NO,CARD,TOOL_CARD,ENERGY_CARD,ENERGY,PLAY,ATTACH,EVOLVE,\
    ABILITY,DISCARD,RETREAT,ATTACK,END,SKILL,SPECIAL_CONDITION = range(17)

CTX_MAIN, CTX_SETUP_ACTIVE, CTX_SETUP_BENCH = 0, 1, 2
CTX_SWITCH, CTX_TO_ACTIVE, CTX_TO_BENCH     = 3, 4, 5
CTX_TO_HAND, CTX_DISCARD                    = 7, 8
CTX_ATTACH_FROM, CTX_ATTACH_TO              = 21, 22
CTX_IS_FIRST                                = 41

AREA_DECK,AREA_HAND,AREA_DISCARD,AREA_ACTIVE,AREA_BENCH = 1,2,3,4,5

CARDTYPE_POKEMON, CARDTYPE_ITEM, CARDTYPE_SUPPORTER = 0, 1, 3

# ---- deck ----
MAKUHITA,HARIYAMA = 673,674
LUNATONE,SOLROCK  = 675,676
RIOLU,MLUCARIO    = 677,678
DUSK_BALL         = 1102
SWITCH_ITEM       = 1123
PPP               = 1141   # Premium Power Pro: +30 dmg this turn (before W/R)
GONG              = 1142   # Fighting Gong: search Basic F energy or Basic F Pokemon
POKE_PAD          = 1152   # search a Pokemon without a Rule Box
HERO_CAPE         = 1159   # ACE SPEC tool: +100 HP
BOSS              = 1182
CARMINE           = 1192   # discard hand, draw 5
LILLIE            = 1227   # shuffle hand into deck, draw 6 (8 if exactly 6 prizes)
GRAVITY_MOUNTAIN  = 1252   # every Stage 2 in play gets -30 HP
F_ENERGY          = 6
LEGACY_ENERGY     = 12
LILLIES_PEARL     = 1172

FIGHTING_TYPE = 6
PSYCHIC_TYPE  = 5

# Deck ratios adopted from probability_v2 (publicScore 933.8, the strongest
# public Lucario pilot): +1 Riolu (line consistency), +1 Boss (gust), +1
# Fighting Energy, -2 Poke Pad, -1 Gravity Mountain vs the sample list. Deck
# lists are a game-fact layer, published openly; the pilot logic remains ours.
DECK = ([MAKUHITA]*2 + [HARIYAMA]*2 + [LUNATONE]*2 + [SOLROCK]*3 +
        [RIOLU]*4 + [MLUCARIO]*4 +
        [DUSK_BALL]*4 + [SWITCH_ITEM]*2 + [PPP]*4 + [GONG]*4 + [POKE_PAD]*2 +
        [HERO_CAPE] + [BOSS]*3 + [CARMINE]*4 + [LILLIE]*4 +
        [GRAVITY_MOUNTAIN]*1 + [F_ENERGY]*14)
assert len(DECK) == 60, f"deck is {len(DECK)}, expected 60"

# ---- attacks ----
AURA_JAB, MEGA_BRAVE = 982, 983
WILD_PRESS           = 978          # 210, 3 energy, 70 recoil
COSMIC_BEAM          = 980          # 70, 1 energy, needs Lunatone benched, ignores W/R
ACCEL_STAB           = 981          # 30, 1 energy, self-locks
POWER_GEM            = 979          # 50, 2 energy
CORKSCREW, CONFRONT  = 976, 977

ATK_BASE = {AURA_JAB:130, MEGA_BRAVE:270, WILD_PRESS:210, COSMIC_BEAM:70,
            ACCEL_STAB:30, POWER_GEM:50, CORKSCREW:10, CONFRONT:30}
ATK_COST = {AURA_JAB:1, MEGA_BRAVE:2, WILD_PRESS:3, COSMIC_BEAM:1,
            ACCEL_STAB:1, POWER_GEM:2, CORKSCREW:1, CONFRONT:2}
# Cosmic Beam's damage "isn't affected by Weakness or Resistance".
ATK_IGNORES_WR = {COSMIC_BEAM}

ATTACKERS      = {MLUCARIO, HARIYAMA, SOLROCK, RIOLU, MAKUHITA, LUNATONE}
NON_ATTACKERS  = {LUNATONE}            # never voluntarily attack with the engine
LINE_BASICS    = {RIOLU, MAKUHITA, SOLROCK, LUNATONE}
EVOLUTIONS     = {RIOLU: MLUCARIO, MAKUHITA: HARIYAMA}

_CARDS = None
_STALL_MEMO = {}

W = {
    'evolve_lucario':   9000,
    'evolve_hariyama':  8600,
    'play_basic':       7800,
    'ability_draw':     7000,
    'search_item':      4200,
    'draw_supporter':   4000,
    'stadium':          3600,
    'boss_for_ko':      6400,
    'ppp_for_ko':       6600,
    'attach':           5000,
    'tool':             4600,
    'retreat_to_atk':   6000,
    'attack':           1000,
    'end':              0.5,
}


def _meta():
    global _CARDS
    if _CARDS is None:
        try:
            _CARDS = {c.cardId: c for c in all_card_data()}
        except Exception:
            _CARDS = {}
    return _CARDS


# ---------------- basic accessors (raw dict, no cg objects on the hot path) ---
def _pk_id(pk):      return (pk or {}).get('id', -1)
def _energies(pk):   return (pk or {}).get('energies') or []
def _energy_cards(pk): return (pk or {}).get('energyCards') or []
def _tools(pk):      return (pk or {}).get('tools') or []
def _hand(p):        return p.get('hand') or []
def _bench(p):       return p.get('bench') or []
def _discard(p):     return p.get('discard') or []


def _active(p):
    a = p.get('active') or []
    return a[0] if a else None


def _n_fighting(pk):
    """Fighting energy actually available on this Pokemon."""
    return sum(1 for e in _energies(pk) if e in (FIGHTING_TYPE, 0) or e == FIGHTING_TYPE)


def _card_data(cid):
    return _meta().get(cid)


def _is_stage2(pk):
    d = _card_data(_pk_id(pk))
    return bool(d and getattr(d, 'stage2', False))


def _max_hp(pk):
    """Printed HP plus Hero's Cape, minus Gravity Mountain if Stage 2."""
    d = _card_data(_pk_id(pk))
    hp = getattr(d, 'hp', 0) if d else 0
    for t in _tools(pk):
        if (t or {}).get('id') == HERO_CAPE:
            hp += 100
    return hp or 0


def _prize_value(pk):
    """Prizes the opponent takes for KOing this Pokemon."""
    d = _card_data(_pk_id(pk))
    if not d: return 1
    n = 3 if getattr(d, 'megaEx', False) else 2 if getattr(d, 'ex', False) else 1
    for e in _energy_cards(pk):
        if (e or {}).get('id') == LEGACY_ENERGY:
            n -= 1
    for t in _tools(pk):
        if (t or {}).get('id') == LILLIES_PEARL:
            n -= 1
    return max(0, n)


def _weak_to_fighting(pk):
    d = _card_data(_pk_id(pk))
    return bool(d and getattr(d, 'weakness', None) == FIGHTING_TYPE)


def _resists_fighting(pk):
    d = _card_data(_pk_id(pk))
    return bool(d and getattr(d, 'resistance', None) == FIGHTING_TYPE)


def _damage(attack_id, target, ppp_active, is_active_target, ppp_count=None):
    """Damage this attack deals to `target`.

    Premium Power Pro adds +30 to the opponent's ACTIVE only, and explicitly
    'before applying Weakness and Resistance', so the bonus is doubled by
    weakness too. Cosmic Beam ignores both Weakness and Resistance.
    """
    dmg = ATK_BASE.get(attack_id, 0)
    if dmg <= 0: return 0
    if ppp_count is None:
        ppp_count = 1 if ppp_active else 0
    if ppp_count and is_active_target:
        # Premium Power Pro is an Item, and items are not once-per-turn, so
        # copies stack. Three of them put Mega Brave at 360, which is the only
        # way this deck one-shots a 350 HP Mega Abomasnow ex.
        dmg += 30 * ppp_count
    if attack_id in ATK_IGNORES_WR:
        return dmg
    if _weak_to_fighting(target):
        dmg *= 2
    elif _resists_fighting(target):
        dmg -= 30
    return max(0, dmg)


_EVO_INTO = None


def _evolves_into_big(cid):
    """True if this card is a pre-evolution of an ex / mega-ex.

    Killing the Basic denies the whole line for one prize. Against Mega
    Abomasnow ex this is the entire matchup: the Mega is 350 HP and weak to
    METAL, so Fighting gets no weakness bonus and Mega Brave (270, +30 per
    Premium Power Pro) cannot one-shot it -- while Mega Brave's own self-lock
    means we need three turns to what they answer in two. Snover is 90 HP.
    Independently confirmed as a real lever by a public writeup's H6
    ('Snover-first prevents Mega evolution', +WR).
    """
    global _EVO_INTO
    if _EVO_INTO is None:
        _EVO_INTO = {}
        try:
            cards = _meta()
            by_name = {}
            for c in cards.values():
                nm = getattr(c, 'name', None)
                if nm: by_name.setdefault(nm, []).append(c)
            for c in cards.values():
                src = getattr(c, 'evolvesFrom', None)
                if not src: continue
                big = getattr(c, 'megaEx', False) or getattr(c, 'ex', False)
                for pre in by_name.get(src, []):
                    if big:
                        _EVO_INTO[pre.cardId] = True
        except Exception:
            _EVO_INTO = {}
    return bool(_EVO_INTO.get(cid))


def _target_priority(pk, opp_prizes_left):
    """How much we want this Pokemon gone, independent of whether we can KO it."""
    if not pk: return 0
    score = _prize_value(pk) * 2200
    score += len(_energies(pk)) * 320       # deny invested energy
    score += len(_tools(pk)) * 220
    d = _card_data(_pk_id(pk))
    if d:
        if getattr(d, 'stage2', False): score += 520
        elif getattr(d, 'stage1', False): score += 260
    # A Basic that becomes an ex is worth far more dead than its 1 prize says.
    if _evolves_into_big(_pk_id(pk)) and getattr(d, 'basic', False):
        score += 2600
    score += (pk or {}).get('hp', 0) // 10
    return score


# ---------------- crash safety / stall handling (proven scaffolding) ----------
def _clamp(indices, sel):
    mn = sel.get('minCount', 0) or 0
    mx = sel.get('maxCount', 1) or 1
    n = len(sel.get('option', []))
    out = []
    for i in indices:
        if 0 <= i < n and i not in out: out.append(i)
        if len(out) >= mx: break
    i = 0
    while len(out) < mn and i < n:
        if i not in out: out.append(i)
        i += 1
    return out


def _select_fingerprint(obs, sel):
    opts = sel.get('option', [])
    opt_sig = tuple(sorted((o.get('area'), o.get('type'), o.get('playerIndex')) for o in opts))
    cur = obs.get('current') or {}
    me_idx = cur.get('yourIndex', 0)
    pl = cur.get('players') or []
    me = pl[me_idx] if len(pl) > me_idx else {}
    opp = pl[1 - me_idx] if len(pl) == 2 else {}
    state_sig = (me.get('deckCount'), me.get('handCount'), len(me.get('prize') or []),
                 opp.get('deckCount'), len(opp.get('prize') or []))
    return (sel.get('type'), sel.get('context'), sel.get('minCount'),
            sel.get('maxCount'), opt_sig, state_sig)


def _resolve_stalled_or(obs, sel, fallback):
    global _STALL_MEMO
    fp = _select_fingerprint(obs, sel)
    seen = _STALL_MEMO.get(fp, 0)
    _STALL_MEMO[fp] = seen + 1
    if seen == 0: return fallback
    n = len(sel.get('option', []))
    if n == 0: return fallback
    mx = sel.get('maxCount', 1) or 1
    mn = sel.get('minCount', 0) or 0
    k = max(mn, min(mx, n))
    offset = (seen * k) % n
    return [(offset + i) % n for i in range(k)]


def _safe_return(result, sel):
    if not sel: return result
    n = len(sel.get('option', []))
    mn = sel.get('minCount', 0) or 0
    mx = sel.get('maxCount', 1) or 1
    if not isinstance(result, list): result = [0]
    result = [i for i in result if 0 <= i < n][:mx]
    i = 0
    while len(result) < mn and i < n:
        if i not in result: result.append(i)
        i += 1
    return result if result else ([0] if n > 0 else [])


# ---------------- attack planning ---------------------------------------------
class Plan:
    """The turn's chosen line: which attack, on which target, and what it needs."""
    def __init__(self):
        self.attack_id = -1
        self.target_bench_idx = -1   # -1 = opponent's active; else bench index
        self.needs_boss = False
        self.needs_ppp = False
        self.ppp_needed = 0
        self.kos = False
        self.score = -1


def _plan_attack(obs, sel, my, opp, opts):
    """Pick the best attack line available to the CURRENT active this turn.

    Only the active can attack, so candidate attacks come straight from the
    legal ATTACK options. For each we consider the opponent's active plus, if a
    gust is available, each benched target.
    """
    plan = Plan()
    my_active = _active(my)
    opp_active = _active(opp)
    if not my_active: return plan

    opp_bench = _bench(opp)
    opp_prizes = len(opp.get('prize') or [])
    hand = _hand(my)
    hand_ids = [_pk_id(c) for c in hand]
    supporter_played = (obs.get('current') or {}).get('supporterPlayed', False)

    boss_available = (BOSS in hand_ids) and not supporter_played
    ppp_available = PPP in hand_ids
    n_ppp_hand = hand_ids.count(PPP)

    legal_attacks = [o.get('attackId') for o in opts if o.get('type') == ATTACK]
    if not legal_attacks: return plan

    # Cosmic Beam does literally nothing without a Lunatone on OUR bench
    # ("If you don't have Lunatone on your Bench, this attack does nothing").
    # The first draft fired it 8 times in 12 games regardless -- pure wasted
    # turns. Verified against card text 2026-07-23.
    lunatone_benched = any(_pk_id(b) == LUNATONE for b in _bench(my) if b)

    # Aura Jab's real payload: Basic F waiting in the discard, and benched
    # Pokemon that are still short of their attack threshold.
    f_in_discard = sum(1 for c in _discard(my) if _pk_id(c) == F_ENERGY)
    bench_energy_demand = 0
    for b in _bench(my):
        if not b or _pk_id(b) == LUNATONE: continue
        bench_energy_demand += max(0, _attack_threshold(_pk_id(b)) - len(_energies(b)))

    for aid in legal_attacks:
        if aid is None: continue
        if aid == COSMIC_BEAM and not lunatone_benched:
            continue
        # Wild Press hurts us 70; skip it if it would KO our own attacker for
        # nothing (Hariyama is 150 HP, so the second Wild Press is suicide).
        recoil_kills = (aid == WILD_PRESS and (my_active or {}).get('hp', 0) <= 70)

        targets = [(-1, opp_active, True)]
        if boss_available:
            targets += [(i, b, False) for i, b in enumerate(opp_bench) if b]

        for bidx, tgt, is_act in targets:
            if not tgt: continue
            for n_ppp in range(0, n_ppp_hand + 1):
                use_ppp = n_ppp > 0
                dmg = _damage(aid, tgt, use_ppp, is_act, ppp_count=n_ppp)
                tgt_hp = (tgt or {}).get('hp', 0)
                kos = dmg >= tgt_hp > 0
                # spending extra copies is only justified if they buy the KO
                if n_ppp > 0 and not kos:
                    continue
                score = _target_priority(tgt, opp_prizes)
                if kos:
                    score += 12000
                    # taking the last prize ends the game -- always the best line
                    if _prize_value(tgt) >= opp_prizes:
                        score += 500000
                else:
                    score = int(score * (dmg / max(1, tgt_hp)))
                if bidx >= 0: score -= 900        # gusting costs the supporter
                if use_ppp:   score -= 200 * n_ppp   # each PPP copy costs a card
                if recoil_kills: score -= 3000
                if aid == MEGA_BRAVE: score += 120   # prefer the closer on ties
                if aid == ACCEL_STAB: score -= 400   # self-locking chip damage
                # Aura Jab is the deck's accelerator, not just 130 damage: it
                # moves up to 3 Basic F from the discard onto the BENCH, which
                # is how the next Mega Brave gets paid for. Score that payload
                # explicitly or a marginally-better chip attack wins every tie
                # and the engine never spins up.
                if aid == AURA_JAB:
                    score += min(3, f_in_discard, bench_energy_demand) * 900
                if aid == COSMIC_BEAM:
                    score -= 300                 # cheapest attack; last resort
                if score > plan.score:
                    plan.score = score
                    plan.attack_id = aid
                    plan.target_bench_idx = bidx
                    plan.needs_boss = bidx >= 0
                    plan.needs_ppp = use_ppp
                    plan.ppp_needed = n_ppp
                    plan.kos = kos
    return plan


# ---------------- energy routing ----------------------------------------------
def _attack_threshold(pid):
    """Energy this Pokemon needs before it can do something worth a turn."""
    return {MLUCARIO: 2, HARIYAMA: 3, MAKUHITA: 2, SOLROCK: 1,
            RIOLU: 1, LUNATONE: 2}.get(pid, 2)


def _energy_priority(pk, my, plan, is_active=False):
    """How much this Pokemon wants the next Fighting energy.

    The manual attach is once per turn and only the ACTIVE can attack, so a
    Pokemon in the Active spot that is short of its threshold outranks anything
    on the bench. Diagnostic (training/lucario_diag.py, 2026-07-23) found the
    first draft attacking on only ~5 of 11 turns because a fully-fueled Mega
    Lucario sat on the bench behind an empty Solrock: attacks were simply not
    legal. Aura Jab already showers the bench with energy from the discard, so
    the hand attachment should almost always go to whoever is up front.
    """
    if not pk: return -1
    pid = _pk_id(pk)
    n = len(_energies(pk))
    need = _attack_threshold(pid)

    if pid == MLUCARIO:
        base = 980 if n < 1 else 960 if n < 2 else 300
    elif pid == RIOLU:
        # energy on Riolu carries through the evolution to Mega Lucario
        base = 900 if n < 2 else 250
    elif pid == HARIYAMA:
        base = 700 if n < 3 else 200
    elif pid == MAKUHITA:
        base = 640 if n < 3 else 180
    elif pid == SOLROCK:
        base = 520 if n < 1 else 100
    elif pid == LUNATONE:
        base = 40                     # engine, not an attacker
    else:
        base = 150

    if is_active and n < need:
        # turning "cannot attack" into "can attack" this turn dominates any
        # amount of bench preparation
        base += 1400
    return base


# ---------------- context handlers --------------------------------------------
def _pick_setup_active(obs, sel, opts):
    """Opening active. Riolu is the line piece; Solrock is a fine turn-1 attacker
    (70 for one energy) but only with Lunatone support, which we rarely have on
    turn 1. Never open on Lunatone -- it cannot attack."""
    order = {RIOLU: 5, SOLROCK: 4, MAKUHITA: 3, LUNATONE: 0}
    best, best_s = 0, -1
    for i, o in enumerate(opts):
        cid = _opt_pokemon_id(obs, o)
        s = order.get(cid, 1)
        if s > best_s: best_s, best = s, i
    return [best]


def _opt_pokemon_id(obs, o):
    """Card id behind a CARD-type option, from whichever area it points at."""
    cur = obs.get('current') or {}
    me_idx = cur.get('yourIndex', 0)
    players = cur.get('players') or []
    pidx = o.get('playerIndex', me_idx)
    p = players[pidx] if len(players) > pidx else {}
    area, idx = o.get('area'), o.get('index', 0)
    try:
        if area == AREA_HAND:    return _pk_id(_hand(p)[idx])
        if area == AREA_BENCH:   return _pk_id(_bench(p)[idx])
        if area == AREA_ACTIVE:  return _pk_id((p.get('active') or [])[idx])
        if area == AREA_DISCARD: return _pk_id(_discard(p)[idx])
        if area == AREA_DECK:    return _pk_id((sel_deck(obs) or [])[idx])
    except Exception:
        pass
    return -1


def sel_deck(obs):
    return ((obs.get('select') or {}).get('deck')) or []


def _score_deck_search(obs, sel, opts, my, plan):
    """Choosing a card out of the deck (Dusk Ball / Gong / Poke Pad).

    Priority is whatever unblocks the engine: a missing line piece first, then
    the draw engine, then energy.
    """
    bench = _bench(my)
    active = _active(my)
    board = [p for p in ([active] + bench) if p]
    board_ids = [_pk_id(p) for p in board]
    hand_ids = [_pk_id(c) for c in _hand(my)]
    have = lambda cid: cid in board_ids or cid in hand_ids

    n_lucario_line = sum(1 for i in board_ids + hand_ids if i in (RIOLU, MLUCARIO))
    scores = []
    for o in opts:
        cid = _opt_pokemon_id(obs, o)
        s = 0
        if cid == RIOLU:
            s = 900 if n_lucario_line < 2 else 300
        elif cid == MLUCARIO:
            # only worth fetching if a Riolu is actually in play to evolve
            s = 880 if any(_pk_id(p) == RIOLU for p in board) else 240
        elif cid == SOLROCK:
            s = 700 if not have(SOLROCK) else 120
        elif cid == LUNATONE:
            # the draw engine needs BOTH halves; Lunatone alone does nothing
            s = 680 if (have(SOLROCK) and not have(LUNATONE)) else 150
        elif cid == MAKUHITA:
            s = 420 if not have(MAKUHITA) else 100
        elif cid == HARIYAMA:
            s = 440 if any(_pk_id(p) == MAKUHITA for p in board) else 110
        elif cid == F_ENERGY:
            s = 500 if len(_energies(active)) < 2 else 260
        else:
            s = 60
        scores.append(s)
    if not scores: return [0]
    return [max(range(len(scores)), key=lambda i: scores[i])]


def _score_to_hand(obs, sel, opts, my):
    """Generic 'put a card into your hand' — same shape as deck search."""
    return _score_deck_search(obs, sel, opts, my, None)


def _score_discard(obs, sel, opts, my):
    """What to throw away. Aura Jab harvests Basic F from the discard, so
    energy is the CHEAPEST thing to pitch once a Mega Lucario is in play."""
    active = _active(my)
    bench = _bench(my)
    board_ids = [_pk_id(p) for p in ([active] + bench) if p]
    lucario_out = MLUCARIO in board_ids or RIOLU in board_ids
    scores = []
    for o in opts:
        cid = _opt_pokemon_id(obs, o)
        if cid == F_ENERGY:
            s = 900 if lucario_out else 500      # recoverable via Aura Jab
        elif cid in (LUNATONE, SOLROCK) and cid in board_ids:
            s = 700                              # duplicate engine piece
        elif cid in (RIOLU, MLUCARIO, MAKUHITA, HARIYAMA):
            s = 100                              # never pitch the attacker line
        elif cid in (BOSS, PPP):
            s = 200
        else:
            s = 450
        scores.append(s)
    k = max(1, sel.get('minCount', 1) or 1)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return order[:k]


def _score_active_choice(obs, sel, opts, my, opp, plan):
    """Promoting/switching a Pokemon into the Active spot."""
    scores = []
    for o in opts:
        cid = _opt_pokemon_id(obs, o)
        pk = None
        cur = obs.get('current') or {}
        me_idx = cur.get('yourIndex', 0)
        if o.get('playerIndex', me_idx) != me_idx:
            # choosing an OPPONENT Pokemon to drag up (Boss / Heave-Ho)
            opp_bench = _bench(opp)
            idx = o.get('index', 0)
            tgt = opp_bench[idx] if 0 <= idx < len(opp_bench) else None
            scores.append(_target_priority(tgt, len(opp.get('prize') or [])))
            continue
        bench = _bench(my)
        idx = o.get('index', 0)
        if 0 <= idx < len(bench): pk = bench[idx]
        s = len(_energies(pk)) * 60
        if cid == MLUCARIO:  s += 900 + (300 if len(_energies(pk)) >= 2 else 0)
        elif cid == HARIYAMA: s += 600 + (250 if len(_energies(pk)) >= 3 else 0)
        elif cid == SOLROCK:  s += 400
        elif cid == RIOLU:    s += 300
        elif cid == MAKUHITA: s += 250
        elif cid == LUNATONE: s += 40
        s += (pk or {}).get('hp', 0) // 20
        scores.append(s)
    if not scores: return [0]
    k = max(1, min(sel.get('maxCount', 1) or 1, len(scores)))
    return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]


# ---------------- MAIN phase ---------------------------------------------------
def _score_main(obs, sel, opts):
    cur = obs.get('current') or {}
    me_idx = cur.get('yourIndex', 0)
    players = cur.get('players') or []
    my = players[me_idx] if len(players) > me_idx else {}
    opp = players[1 - me_idx] if len(players) == 2 else {}

    my_active = _active(my)
    opp_active = _active(opp)
    bench = _bench(my)
    hand = _hand(my)
    hand_ids = [_pk_id(c) for c in hand]
    board = [p for p in ([my_active] + bench) if p]
    board_ids = [_pk_id(p) for p in board]

    deck_count = my.get('deckCount', 0) or 0
    deck_low = deck_count <= 8
    supporter_played = cur.get('supporterPlayed', False)
    energy_attached = cur.get('energyAttached', False)
    turn = cur.get('turn') or 0
    prizes = len(my.get('prize') or [])
    opp_prizes = len(opp.get('prize') or [])

    plan = _plan_attack(obs, sel, my, opp, opts)

    have_solrock = SOLROCK in board_ids
    have_lunatone = LUNATONE in board_ids
    engine_online = have_solrock and have_lunatone

    # --- can the CURRENT active actually do anything this turn? ---
    # Only the active attacks, so an unfuelled active in front of a loaded bench
    # is a wasted turn. This is what the first draft got wrong (attacked on only
    # ~5 of 11 turns despite 2.6 energy on Mega Lucario).
    active_can_attack = any(o.get('type') == ATTACK for o in opts)
    active_id = _pk_id(my_active)
    active_e = len(_energies(my_active))
    active_short = (not active_can_attack) or (
        active_id == MLUCARIO and active_e < 2)

    def _bench_readiness(b):
        """How ready this benched Pokemon is to attack if promoted."""
        if not b: return -1
        pid = _pk_id(b)
        if pid == LUNATONE: return -1
        e = len(_energies(b))
        if e < _attack_threshold(pid): return -1
        val = {MLUCARIO: 1000, HARIYAMA: 620, SOLROCK: 380,
               RIOLU: 240, MAKUHITA: 200}.get(pid, 120)
        return val + e * 30

    bench_best = max((_bench_readiness(b) for b in bench if b), default=-1)

    # Closing the engine loop: Aura Jab loads the BENCH from the discard, so
    # after a Jab the fuelled attacker is behind, not in front. If the active
    # can only Jab again while a benched Mega Lucario is already paid up for
    # Mega Brave, promoting is worth far more than another 130. Without this the
    # agent Jabs forever -- Mega Brave usage fell 11 -> 5 per 14 games when the
    # acceleration bonus landed (lucario_diag, 2026-07-23).
    active_can_mega = (active_id == MLUCARIO and active_e >= 2)
    bench_mega_ready = any(_pk_id(b) == MLUCARIO and len(_energies(b)) >= 2
                           for b in bench if b)
    bench_hariyama_ready = any(_pk_id(b) == HARIYAMA and len(_energies(b)) >= 3
                               for b in bench if b)
    upgrade_available = (not active_can_mega) and (bench_mega_ready or bench_hariyama_ready)

    want_switch = (active_short or upgrade_available) and bench_best > 0
    n_energy_hand = hand_ids.count(F_ENERGY)
    bench_space = max(0, 5 - len([b for b in bench if b]))
    opp_has_stage2 = any(_is_stage2(p) for p in ([opp_active] + _bench(opp)) if p)
    lucario_out = (MLUCARIO in board_ids) or (RIOLU in board_ids)
    stadium = (cur.get('stadium') or [])
    stadium_id = _pk_id(stadium[0]) if stadium else -1

    def score(o):
        ot = o.get('type')

        if ot == END:
            return W['end']

        if ot == ATTACK:
            aid = o.get('attackId')
            if aid != plan.attack_id or plan.attack_id < 0:
                return W['attack'] - 50
            # Do not fire the plan until its prerequisites are satisfied: if the
            # line needs a gust or PPP and we have not played it yet, those
            # options must go first (they score far higher anyway).
            if plan.needs_boss and not supporter_played:
                return W['attack'] - 60
            if plan.needs_ppp and PPP in hand_ids:
                return W['attack'] - 60
            return W['attack'] + (400 if plan.kos else 0)

        if ot == ABILITY:
            cid = _opt_pokemon_id(obs, o)
            if cid == LUNATONE:
                # Lunar Cycle: pitch a Basic F to draw 3. Needs Solrock in play.
                #
                # The pitched energy is NOT a cost once a Lucario line is out --
                # Aura Jab attaches up to 3 Basic F straight back out of the
                # discard. Lunar Cycle is the FEEDER for that engine, so the
                # first draft's conservative gate (only fire with 2+ energy
                # spare) starved it: the discard stayed empty, Aura Jab had
                # nothing to fetch, and Mega Lucario averaged 1.57 energy
                # against a Mega Brave threshold of 2 (lucario_diag, 2026-07-23).
                if not have_solrock or n_energy_hand == 0: return -50
                if deck_low: return -40
                keep = 0 if lucario_out else (1 if not energy_attached else 0)
                if n_energy_hand <= keep: return -30
                return W['ability_draw'] + (600 if lucario_out else 0)
            return 100

        if ot == EVOLVE:
            cid = _opt_pokemon_id(obs, o)
            if cid == MLUCARIO:
                return W['evolve_lucario']
            if cid == HARIYAMA:
                # Heave-Ho Catcher gusts on evolution. Hold it when there is a
                # juicy bench target and spend it immediately otherwise, since a
                # 150 HP Hariyama also just wants to be on the board.
                opp_bench = [b for b in _bench(opp) if b]
                if opp_bench:
                    best = max(_target_priority(b, opp_prizes) for b in opp_bench)
                    cur_pri = _target_priority(opp_active, opp_prizes)
                    if best > cur_pri + 800:
                        return W['evolve_hariyama'] + 400
                return W['evolve_hariyama']
            return 5000

        if ot == PLAY:
            cid = _opt_pokemon_id(obs, o)
            d = _card_data(cid)
            ctype = getattr(d, 'cardType', None) if d else None

            if ctype == CARDTYPE_POKEMON:
                if bench_space <= 0: return -10
                # duplicates of engine pieces are near-worthless
                if cid in (LUNATONE, SOLROCK) and cid in board_ids: return 900
                if cid == RIOLU and board_ids.count(RIOLU) + board_ids.count(MLUCARIO) >= 3:
                    return 900
                if cid == LUNATONE and not have_solrock:
                    return W['play_basic'] - 1500   # half an engine does nothing
                return W['play_basic']

            if cid == BOSS:
                if supporter_played: return -10
                return W['boss_for_ko'] if plan.needs_boss else -5

            if cid == PPP:
                return W['ppp_for_ko'] if plan.needs_ppp else -5

            if cid == SWITCH_ITEM:
                # Switch is free (no energy cost, unlike retreating a 2-cost
                # Mega Lucario), so it is the right way to trade an exhausted or
                # unfuelled active for a loaded bench attacker.
                if not my_active: return -5
                if want_switch: return W['retreat_to_atk'] + 400
                if active_id == LUNATONE and any(b for b in bench):
                    return W['retreat_to_atk']
                return -5

            if cid in (DUSK_BALL, GONG, POKE_PAD):
                if deck_count <= 1: return -5
                return W['search_item']

            if cid == GRAVITY_MOUNTAIN:
                if stadium_id == GRAVITY_MOUNTAIN: return -5
                # -30 to every Stage 2 in play. We are Stage 1, so this is
                # strictly asymmetric whenever they have a Stage 2 out.
                if opp_has_stage2: return W['stadium'] + 800
                if stadium_id > 0: return W['stadium'] - 1200   # deny theirs
                return -5

            if cid in (CARMINE, LILLIE):
                if supporter_played: return -10
                if deck_low: return -8
                if plan.needs_boss: return -6      # Boss wants the supporter slot
                hand_n = len(hand)
                if cid == LILLIE:
                    # shuffle-and-draw-6; strictly better the smaller our hand
                    return W['draw_supporter'] + (300 if hand_n <= 4 else 0)
                # Carmine discards the hand -- bad with a loaded hand
                return W['draw_supporter'] - (250 * max(0, hand_n - 3))
            return 3000

        if ot == ATTACH:
            cid = _opt_pokemon_id(obs, o)
            tgt = _attach_target(o, my_active, bench)
            if cid == HERO_CAPE:
                # +100 HP; best on the Mega Lucario that is actually tanking
                if _pk_id(tgt) == MLUCARIO: return W['tool'] + 600
                if _pk_id(tgt) == RIOLU:    return W['tool'] + 300
                return W['tool'] - 2000
            if cid == F_ENERGY:
                is_act = tgt is not None and tgt is my_active
                return W['attach'] + _energy_priority(tgt, my, plan, is_act)
            return W['attach']

        if ot == RETREAT:
            if not my_active: return -5
            # Retreating costs energy off the active, so it is strictly worse
            # than Switch -- but still far better than passing the turn with an
            # attacker that cannot attack.
            if want_switch and SWITCH_ITEM not in hand_ids:
                return W['retreat_to_atk'] - 200
            if active_id == LUNATONE and any(b for b in bench):
                return W['retreat_to_atk']
            return -3

        if ot in (CARD, TOOL_CARD, ENERGY_CARD, ENERGY):
            return 500
        return 10

    return [score(o) for o in opts]


def _attach_target(o, my_active, bench):
    ta = o.get('inPlayArea', -1)
    ti = o.get('inPlayIndex', 0)
    if ta == AREA_ACTIVE: return my_active
    if ta == AREA_BENCH and bench and 0 <= ti < len(bench): return bench[ti]
    return None


# ---------------- dispatch -----------------------------------------------------
def _choose(obs):
    sel = obs.get('select')
    if not sel:
        return DECK
    opts = sel.get('option') or []
    if not opts:
        return []

    ctx = sel.get('context')
    cur = obs.get('current') or {}
    me_idx = cur.get('yourIndex', 0)
    players = cur.get('players') or []
    my = players[me_idx] if len(players) > me_idx else {}
    opp = players[1 - me_idx] if len(players) == 2 else {}

    if ctx == CTX_IS_FIRST:
        # Going first is worth ~56/44 in this format and this deck wants the
        # extra turn of setup far more than it wants the extra card.
        for i, o in enumerate(opts):
            if o.get('type') == YES: return [i]
        return [0]

    if ctx == CTX_SETUP_ACTIVE:
        return _clamp(_pick_setup_active(obs, sel, opts), sel)

    if ctx == CTX_SETUP_BENCH:
        return _clamp(list(range(len(opts))), sel)

    if ctx == CTX_MAIN:
        scores = _score_main(obs, sel, opts)
        order = sorted(range(len(opts)), key=lambda i: -scores[i])
        return _clamp(order[:max(1, sel.get('maxCount', 1) or 1)], sel)

    if ctx in (CTX_SWITCH, CTX_TO_ACTIVE):
        plan = _plan_attack(obs, sel, my, opp, opts)
        return _clamp(_score_active_choice(obs, sel, opts, my, opp, plan), sel)

    if ctx == CTX_TO_HAND:
        return _clamp(_score_to_hand(obs, sel, opts, my), sel)

    if ctx in (CTX_DISCARD,):
        return _clamp(_score_discard(obs, sel, opts, my), sel)

    if ctx == CTX_TO_BENCH:
        return _clamp(list(range(len(opts))), sel)

    if ctx == CTX_ATTACH_FROM or ctx == CTX_ATTACH_TO:
        plan = Plan()
        scores = []
        for o in opts:
            tgt = _attach_target(o, _active(my), _bench(my))
            if tgt is None:
                cid = _opt_pokemon_id(obs, o)
                scores.append(400 if cid == F_ENERGY else 100)
            else:
                scores.append(_energy_priority(tgt, my, plan))
        order = sorted(range(len(opts)), key=lambda i: -scores[i])
        return _clamp(order, sel)

    # Anything unmodelled: take the first legal answer, but rotate if the engine
    # asks the identical question again (genuine stall).
    n = len(opts)
    mn = sel.get('minCount', 1) or 1
    mx = sel.get('maxCount', 1) or 1
    k = max(min(mn, n), min(mx, n), 1)
    return _clamp(_resolve_stalled_or(obs, sel, list(range(min(k, n)))), sel)


def agent(obs_dict: dict) -> list[int]:
    try:
        sel = obs_dict.get('select')
        out = _choose(obs_dict)
        if isinstance(out, list):
            return _safe_return(out, sel) if sel else out
    except Exception:
        pass
    try:
        sel = obs_dict.get('select')
        if not sel: return DECK
        n = len(sel.get('option', []))
        if n == 0: return []
        mn = sel.get('minCount', 1) or 0
        return _safe_return(list(range(min(max(mn, 1), n))), sel)
    except Exception:
        return [0]
