import sys, glob, random, math

for _pat in ['/kaggle/input/**/cg-lib', '/kaggle/input/cg-lib']:
    _paths = glob.glob(_pat, recursive=True)
    if _paths: sys.path.insert(0, _paths[0]); break

try:
    from cg.api import all_card_data, all_attack
except ImportError:
    all_card_data = lambda: []
    all_attack    = lambda: []

NUMBER,YES,NO,CARD,TOOL_CARD,ENERGY_CARD,ENERGY,PLAY,ATTACH,EVOLVE,\
    ABILITY,DISCARD,RETREAT,ATTACK,END,SKILL,SPECIAL_CONDITION = range(17)

CTX_SETUP_ACTIVE = 1
CTX_SETUP_BENCH  = 2
LOG_PLAY         = 10
LOG_HP_CHANGE    = 16
CARDTYPE_SUPPORTER = 3

ABRA,KADABRA,ALAKAZAM             = 741,742,743
DUNSPARCE,DUNSPARCE2,DUDUNSPARCE  = 305,65,66
GENESECT,SHAYMIN,PSYDUCK,FEZ      = 142,343,858,140
POFFIN,POKE_PAD,HANDHELD_FAN      = 1086,1152,1161
BOSS,LANA,BATTLE_CAGE,DAWN        = 1182,1184,1264,1231
WONDROUS_PATCH,SACRED_ASH,HILDA   = 1146,1129,1225
NIGHT_STRETCHER = 1097
ENHANCED_HAMMER,RARE_CANDY        = 1081,1079
BASIC_P,ENRICHING,TELEPATH_P      = 5,13,19

MIST_ENERGY = 11
ROCK_ENERGY = 20
PSYCHIC_TYPE = 5

ALAKAZAM_LINE_IDS      = {ABRA,KADABRA,ALAKAZAM}
DRAW_ABILITY_CARD_IDS  = {KADABRA,ALAKAZAM,DUDUNSPARCE}
SUPPRESS_ABILITY_IDS   = set()  # No own-pokemon abilities to suppress in this deck
PSYCHIC_ENERGY_IDS     = {BASIC_P,TELEPATH_P}
BENCHABLE_BASIC_IDS    = {ABRA,DUNSPARCE,DUNSPARCE2,PSYDUCK,SHAYMIN,GENESECT,FEZ}
LINE_SEARCH_HAND_IDS   = {DAWN,HILDA}
DRAW_ENGINE_IDS        = {DUNSPARCE,DUNSPARCE2,DUDUNSPARCE}
NON_ATTACKER_IDS       = {DUNSPARCE,DUNSPARCE2,DUDUNSPARCE,GENESECT,SHAYMIN,PSYDUCK,FEZ,ABRA,KADABRA}
# Kadabra has a real attack (Super Psy Bolt, {P}->30 flat dmg) but this deck's
# ATTACK scoring only ever rewards Alakazam's Powerful Hand (active_can_attack
# requires is_alak; any Kadabra attack scores -5 regardless of state) -- Kadabra
# is functionally a non-attacker here. It was missing from this set, so a stuck
# Kadabra active (no attack, e.g. no energy) with a ready, fully-fueled Alakazam
# waiting on bench fell through every retreat-priority tier and scored a flat
# 0.5 -- LOWER than just ending the turn (1.0). Verified via score_options_main
# on a synthetic Kadabra-active/fueled-bench-Alakazam state before this fix.
PIVOT_FREE_RETREAT_IDS = {SHAYMIN}
TOOL_IDS               = {HANDHELD_FAN}
PH_DMG_PER_CARD = 20

# v23 deck: reverted from v24 (Alakazam/Dunsparce 4th copies, Genesect/Psyduck cut)
# 2026-07-03 — user call to revert on v24's early ladder trend (680 at 7h, 780
# at 24h), ahead of the documented 48h decision-rule checkpoint.
DECK = ([ABRA]*4+[KADABRA]*4+[ALAKAZAM]*3+[DUNSPARCE]*3+[DUDUNSPARCE]*3+
        [GENESECT]+[SHAYMIN]+[PSYDUCK]+[FEZ]+
        [POFFIN]*4+[POKE_PAD]*2+[NIGHT_STRETCHER]*2+[HANDHELD_FAN]*2+[BOSS]*3+[LANA]+
        [BATTLE_CAGE]*4+[DAWN]*4+[WONDROUS_PATCH]+[SACRED_ASH]+[HILDA]*3+
        [ENHANCED_HAMMER]*2+[RARE_CANDY]*3+[BASIC_P]*2+[ENRICHING]+[TELEPATH_P]*4)

_CARDS = None; _ATKMAP = None
_STALL_MEMO = {}

# Tunable scoring weights (v22). Defaults ARE v21 behavior; tuning harnesses may
# mutate this dict in-process — the submission never reads env/files.
W = {
    'atk_threshold':150.0,'atk_default':7.0,
    'dudun_base':11.0,'fez_base':8.0,
    'evo_bench_establish':40.0,'evo_bench_convert':25.0,'evo_bench_late':12.0,
    'boss_target':16.0,'boss_mega_chip':18.0,'boss_mist_escape':12.0,
    'cage_base':6.0,'cage_reactive':22.0,'hammer_mist':45.0,
    'poffin_estab':25.0,'poffin_need':19.0,
    'dawn_estab':22.0,'dawn_need':11.0,'hilda_estab':24.0,'hilda_immobile':18.0,
    'pad_no_backup':13.0,'end_convert':4.0,
    'candy_ready':30.0,'candy_estab':25.0,'candy_active_abra':45.0,
    'attach_kadabra':9.0,'attach_abra':6.0,
    'retreat_nonatk_ready':30.0,'retreat_alak_stuck':22.0,'desperation_draw':30.0,
    # MCTS heuristic-prior blend temperatures (used by training/nn/mcts.py).
    # score_options_main's scale spans ~4 (default) to 600 (KO) — prior_T_h_main
    # is tuned large enough that a softmax doesn't fully saturate on the KO/
    # boss-snipe outliers alone. Non-MAIN select types (_score_deck_search,
    # _score_bench_target, etc.) score in a much flatter ~0-100 range, hence the
    # separate, smaller default temperature. prior_T_net is the net policy
    # logits' own softmax temperature (unrelated scale, tuned independently).
    'prior_T_h_main':40.0,'prior_T_h_default':3.0,'prior_T_net':1.0,
}

def _select_fingerprint(obs, sel):
    """Coarse signature of a select + game state, for detecting a genuinely stuck
    engine selection (same question, same state, asked again). Deliberately loose:
    false negatives (missing a real stall) are free; false positives just cost one
    extra, still-valid rotation of an equally-good blind pick."""
    opts = sel.get('option', [])
    opt_sig = tuple(sorted(
        (o.get('area'), o.get('type'), o.get('playerIndex')) for o in opts))
    cur = obs.get('current') or {}
    me_idx = cur.get('yourIndex', 0)
    pl = cur.get('players') or []
    me = pl[me_idx] if len(pl) > me_idx else {}
    opp = pl[1-me_idx] if len(pl) == 2 else {}
    state_sig = (
        me.get('deckCount'), me.get('handCount'), len(me.get('prize') or []),
        opp.get('deckCount'), len(opp.get('prize') or []))
    return (sel.get('type'), sel.get('context'), sel.get('minCount'),
            sel.get('maxCount'), opt_sig, state_sig)

def _resolve_stalled_or(obs, sel, fallback_indices):
    """If the exact same select+state was seen on a prior call with no progress,
    rotate to a different (still valid) combination instead of repeating the
    identical answer forever. First occurrence always uses fallback_indices."""
    global _STALL_MEMO
    fp = _select_fingerprint(obs, sel)
    seen = _STALL_MEMO.get(fp, 0)
    _STALL_MEMO[fp] = seen + 1
    if seen == 0:
        return fallback_indices
    n = len(sel.get('option', []))
    mx = sel.get('maxCount', 1) or 1
    mn = sel.get('minCount', 0) or 0
    if n == 0: return fallback_indices
    k = max(mn, min(mx, n))
    offset = (seen * k) % n
    idxs = [(offset + i) % n for i in range(k)]
    return idxs

def _meta():
    global _CARDS,_ATKMAP
    if _CARDS is None:
        _CARDS={}; _ATKMAP={}
        try:
            for a in all_attack():
                _ATKMAP[a.attackId]=(getattr(a,'damage',0) or 0,
                                     tuple(getattr(a,'energies',()) or ()))
            for c in all_card_data(): _CARDS[c.cardId]=c
        except: pass
    return _CARDS,_ATKMAP

def _prize_value_pk(pk):
    if not pk: return 1
    if pk.get('megaEx',False): return 3
    if pk.get('ex',False):     return 2
    return 1

def _analyze_logs(obs_dict, me_idx):
    logs = obs_dict.get('logs') or []
    opp_idx = 1 - me_idx
    opp_played = set()
    bench_damage_received = False
    we_were_kod_last_turn = False
    C, _ = _meta()
    for log in logs[-30:]:
        ltype = log.get('type')
        lplayer = log.get('playerIndex')
        if ltype == LOG_PLAY and lplayer == opp_idx:
            cid = log.get('cardId', 0)
            if cid: opp_played.add(cid)
        if ltype == LOG_HP_CHANGE and lplayer == me_idx:
            if log.get('putDamageCounter', False):
                bench_damage_received = True
        if log.get('type') == 6 and lplayer == me_idx:
            from_area = log.get('fromArea'); to_area = log.get('toArea')
            if from_area in (4, 5) and to_area == 3:
                we_were_kod_last_turn = True
    opp_has_ace_spec = any(
        C.get(cid) and getattr(C.get(cid), 'aceSpec', False)
        for cid in opp_played)
    return opp_played, bench_damage_received, we_were_kod_last_turn, opp_has_ace_spec

def _pk_id(pk): return (pk or {}).get('id',-1)
def _energies(pk): return (pk or {}).get('energies') or []
def _has_psychic(pk): return PSYCHIC_TYPE in _energies(pk)
def _can_ph(pk): return _pk_id(pk)==ALAKAZAM and _has_psychic(pk)
def _hand_list(p): return p.get('hand') or []
def _hand_size(state,me):
    try:
        p=state['players'][me]; h=p.get('handCount')
        return h if h is not None else len(_hand_list(p))
    except: return 0
def _active(p):
    a=p.get('active'); return a[0] if a and len(a)>0 and a[0] else None
def _is_non_attacker(pk): return not pk or _pk_id(pk) in NON_ATTACKER_IDS
def _opp_threatening(opp_active):
    """Cheap proxy for 'opponent's active can plausibly swing next turn':
    already carrying attached energy. Doesn't know their actual attack costs
    or damage -- good enough to gate a defensive-retreat override, not a real
    lookahead; false negatives just fall back to the existing target-priority
    scoring."""
    return bool(opp_active) and len(_energies(opp_active))>0

def _opp_has_blocking_energy(opp_active):
    if not opp_active: return False
    for ec in (opp_active.get('energyCards') or []):
        if ec.get('id') in (MIST_ENERGY, ROCK_ENERGY): return True
    return False

# ---- Stage 3 Phase C: archetype belief model (docs/belief-model.md) ----
# Phase A logistic-regression weights (training/belief/train.py), embedded so
# the submission stays a single dependency-free file. Inference is a sparse
# dot product + softmax over just the features present this decision.
_BELIEF_CLASSES=['abomasnow', 'alakazam', 'dragapult', 'lucario', 'starmie']
_BELIEF_INTERCEPT=[3.6972,-13.9598,3.4858,4.3845,2.3923]
_BELIEF_COEF={
 'card_1030':(-1.6093,-2.0067,-1.0127,-0.8535,5.4823),
 'card_1031':(-0.3760,-0.2426,-0.0696,-0.0163,0.7045),
 'card_1071':(-0.6380,-0.9967,3.3556,-0.6908,-1.0300),
 'card_1079':(-0.1592,0.1880,0.2210,-0.1622,-0.0876),
 'card_1080':(-0.0642,-0.0413,0.2111,-0.0847,-0.0208),
 'card_1081':(-0.2300,1.5441,-0.8033,-0.2518,-0.2590),
 'card_1086':(-0.7273,0.0711,0.3475,-0.3196,0.6283),
 'card_1097':(-0.2998,-0.1298,0.4002,-0.1516,0.1809),
 'card_1102':(-0.1612,-0.1813,-0.2128,0.5981,-0.0428),
 'card_1120':(-0.1754,-0.2313,0.6789,-0.1959,-0.0763),
 'card_1121':(0.1163,-0.2989,0.3812,-0.4025,0.2040),
 'card_1123':(-0.1153,-0.0986,-0.1569,0.3985,-0.0277),
 'card_1126':(0.4494,-0.1648,-0.1196,-0.1057,-0.0592),
 'card_1129':(-0.0463,0.1489,-0.0566,-0.0270,-0.0190),
 'card_1141':(-0.1681,-0.1361,-0.2254,0.5781,-0.0484),
 'card_1142':(-0.1599,-0.1826,-0.2086,0.5925,-0.0414),
 'card_1146':(-0.0277,0.0803,-0.0319,-0.0146,-0.0062),
 'card_1152':(-0.4481,0.0622,0.4874,0.1958,-0.2973),
 'card_1156':(-0.1114,-0.3169,0.6368,-0.1264,-0.0821),
 'card_1159':(-0.0780,-0.0770,-0.1020,0.2752,-0.0181),
 'card_1161':(-0.4762,2.2399,-1.0466,-0.3546,-0.3626),
 'card_1182':(-0.5774,1.4469,-0.2799,-0.2101,-0.3795),
 'card_1184':(-0.0167,0.0720,-0.0297,-0.0156,-0.0101),
 'card_119':(-0.9974,-1.3342,4.6081,-0.9459,-1.3307),
 'card_1192':(-0.1778,-0.2764,-0.4494,0.1865,0.7171),
 'card_1198':(-0.1603,-0.4678,0.8769,-0.1837,-0.0651),
 'card_120':(-0.1611,-0.3324,0.7406,-0.1959,-0.0511),
 'card_121':(-0.1332,-0.1489,0.4891,-0.1653,-0.0417),
 'card_1210':(-0.2257,-0.2067,0.4775,-0.1363,0.0912),
 'card_1225':(-0.1097,0.5067,-0.1838,-0.0922,-0.1210),
 'card_1227':(0.2629,-0.9058,0.4827,0.1064,0.0538),
 'card_1231':(-0.0729,0.3130,-0.1159,-0.0674,-0.0568),
 'card_1252':(-0.0569,-0.0346,-0.0701,0.1713,-0.0097),
 'card_1256':(-0.1103,-0.2322,0.5103,-0.1038,-0.0639),
 'card_1262':(0.0887,-0.1398,-0.1762,-0.1365,0.3638),
 'card_13':(-0.0207,0.0907,-0.0273,-0.0155,-0.0273),
 'card_140':(-1.4882,3.0688,2.0158,-1.5750,-2.0214),
 'card_142':(-0.8622,4.4454,-1.2781,-1.0592,-1.2459),
 'card_184':(-0.6768,-1.0155,3.4719,-0.7185,-1.0612),
 'card_19':(-0.1237,0.4879,-0.1747,-0.0861,-0.1034),
 'card_2':(-0.1585,-0.4050,0.8307,-0.1801,-0.0870),
 'card_235':(-0.8030,-1.1330,3.9113,-0.8302,-1.1451),
 'card_3':(0.8316,-1.3201,-0.6358,-0.3415,1.4658),
 'card_305':(-1.1519,5.1270,-1.3307,-1.2522,-1.3921),
 'card_343':(-0.8381,4.3288,-1.2361,-1.0143,-1.2403),
 'card_5':(-0.4774,0.9052,0.1570,-0.3684,-0.2164),
 'card_6':(-0.1851,-0.1843,-0.2404,0.6628,-0.0531),
 'card_66':(-0.0358,0.1355,-0.0405,-0.0291,-0.0301),
 'card_673':(-0.7785,-0.8822,-0.8680,3.6639,-1.1352),
 'card_674':(-0.1067,-0.0763,-0.1485,0.3568,-0.0252),
 'card_675':(-0.7794,-1.0258,-0.8766,3.8579,-1.1761),
 'card_676':(-0.9059,-0.9769,-0.9723,4.1075,-1.2524),
 'card_677':(-0.9320,-0.9377,-0.9911,4.1270,-1.2663),
 'card_678':(-0.1295,-0.1512,-0.1867,0.4979,-0.0306),
 'card_721':(4.5619,-0.8983,-1.0592,-0.9764,-1.6281),
 'card_722':(4.9754,-0.9643,-1.1832,-1.1094,-1.7185),
 'card_723':(0.6568,-0.1137,-0.1999,-0.1723,-0.1708),
 'card_741':(-1.4255,6.2033,-2.0198,-1.3126,-1.4454),
 'card_742':(-0.1017,0.5202,-0.2496,-0.0797,-0.0893),
 'card_743':(-0.0864,0.3866,-0.1405,-0.0776,-0.0821),
 'card_858':(-0.8104,4.1247,-1.1835,-0.9561,-1.1746),
 'energy_0':(-0.0021,0.0155,-0.0051,-0.0032,-0.0051),
 'energy_2':(-0.1628,-0.3699,0.8553,-0.2020,-0.1206),
 'energy_3':(1.4939,-1.2081,-0.6295,-0.5068,0.8505),
 'energy_5':(-0.3017,0.2192,0.7092,-0.3459,-0.2808),
 'energy_6':(-0.2714,-0.2535,-0.4088,1.0222,-0.0886),
 'opp_bench_n':(-0.8345,1.8342,-0.0563,-0.3744,-0.5690),
 'opp_discard_n':(0.2012,0.0345,0.0536,-0.0474,-0.2419),
 'opp_hand_n':(-0.6089,2.3041,-0.5815,-0.7374,-0.3762),
 'opp_prizes_taken':(0.0559,-0.2472,0.0926,0.1285,-0.0298),
 'turn':(0.0096,-0.5821,0.1320,0.0106,0.4299),
}
# Ace-spec reveal rate by archetype across 679 real ladder replays (tech
# survey 2026-07-04 -- see docs/report-log.md Phase C entry).
_ACE_RATE={'lucario':0.60,'alakazam':0.63,'dragapult':0.46,'abomasnow':0.48,'starmie':0.29}
# Dwebble/Crustle line: the one recognized archetype that meaningfully techs
# Mist/Rock (35.8% of its 53 tagged ladder replays; every other recognized
# archetype is 0-3%). The unrecognized long tail techs 29.0%.
_CRUSTLE_LINE_IDS={344,345,532,533}

def _belief_posterior(opp, turn):
    """P(archetype | opponent's public board+discard). Mirrors
    training/belief/collect.py's feature extraction: revealed card ids from
    active/bench (self + preEvolution chain + tools) + discard; energy TYPE
    counts read fresh from the board; scalar counts. Returns (posterior dict,
    wall_energy_revealed, crustle_line_seen); (None, False, False) on any
    failure so callers fall back to pre-Phase-C behavior. Pure -- safe under
    score_options_main's thousands-of-calls contract."""
    try:
        z=list(_BELIEF_INTERCEPT)
        def add(name,val):
            w=_BELIEF_COEF.get(name)
            if w:
                for k in range(5): z[k]+=w[k]*val
        seen=set(); energy={}; wall=False
        for pk in list(opp.get('active') or[])+list(opp.get('bench') or[]):
            if not pk: continue
            if pk.get('id'): seen.add(pk['id'])
            for pre in pk.get('preEvolution') or[]:
                if (pre or{}).get('id'): seen.add(pre['id'])
            for t in pk.get('tools') or[]:
                if (t or{}).get('id'): seen.add(t['id'])
            for et in pk.get('energies') or[]:
                energy[et]=energy.get(et,0)+1
            for ec in pk.get('energyCards') or[]:
                if (ec or{}).get('id') in(MIST_ENERGY,ROCK_ENERGY): wall=True
        for c in opp.get('discard') or[]:
            if (c or{}).get('id'): seen.add(c['id'])
        if MIST_ENERGY in seen or ROCK_ENERGY in seen: wall=True
        for cid in seen: add('card_%d'%cid,1.0)
        for et,cnt in energy.items(): add('energy_%d'%et,float(cnt))
        add('turn',float(turn))
        add('opp_bench_n',float(len(opp.get('bench') or[])))
        add('opp_discard_n',float(len(opp.get('discard') or[])))
        add('opp_hand_n',float(opp.get('handCount') or 0))
        add('opp_prizes_taken',float(6-len(opp.get('prize') or[])))
        m=max(z); exps=[math.exp(v-m) for v in z]; s=sum(exps)
        return ({c:e/s for c,e in zip(_BELIEF_CLASSES,exps)},
                wall, bool(seen&_CRUSTLE_LINE_IDS))
    except: return None,False,False

def _opt_card_id(o,hand,my_active,bench):
    ot=o.get('type')
    if ot in(PLAY,ATTACH,EVOLVE,TOOL_CARD,ENERGY_CARD):
        idx=o.get('index')
        if idx is not None and 0<=idx<len(hand): return _pk_id(hand[idx])
        return -1
    if ot==ABILITY:
        area=o.get('area'); idx=o.get('index',0)
        if area==4: return _pk_id(my_active)
        if area==5 and 0<=idx<len(bench): return _pk_id(bench[idx])
        return -1
    return -1
def _attach_target(o,my_active,bench):
    ta=o.get('inPlayArea',-1); ti=o.get('inPlayIndex',0)
    if ta==4: return my_active
    if ta==5 and bench and 0<=ti<len(bench): return bench[ti]
    return None
def _clamp(indices,sel):
    mn=sel.get('minCount',0) or 0; mx=sel.get('maxCount',1) or 1
    n=len(sel.get('option',[])); out=[]
    for i in indices:
        if 0<=i<n and i not in out: out.append(i)
        if len(out)>=mx: break
    i=0
    while len(out)<mn and i<n:
        if i not in out: out.append(i)
        i+=1
    return out

def _census(my_active, bench):
    all_mine=[my_active]+list(bench)
    abra_count=sum(1 for p in all_mine if _pk_id(p)==ABRA)
    genesect_with_tool = any(
        _pk_id(b)==GENESECT and (b or {}).get('tools')
        for b in bench if b)
    return {
        'abra_count':     abra_count,
        'backup_abra':    abra_count>=2,
        'line_count':     sum(1 for p in all_mine if _pk_id(p) in ALAKAZAM_LINE_IDS),
        'draw_count':     sum(1 for p in all_mine if _pk_id(p) in DRAW_ENGINE_IDS),
        'dudun_bench':    sum(1 for p in bench if _pk_id(p)==DUDUNSPARCE),
        'need_line':      sum(1 for p in all_mine if _pk_id(p) in ALAKAZAM_LINE_IDS)<2,
        'need_draw':      sum(1 for p in all_mine if _pk_id(p) in DRAW_ENGINE_IDS)<2,
        'has_alakazam':   any(_pk_id(p)==ALAKAZAM for p in all_mine),
        'kadabra_can_evolve': any(
            _pk_id(p)==KADABRA and not (p or {}).get('appearThisTurn',False)
            for p in all_mine if p),
        'dudun_no_energy': any(
            _pk_id(p)==DUDUNSPARCE and not _energies(p)
            for p in bench if p),
        'has_energy_plan': any(_has_psychic(p) for p in all_mine if p),
        'bench_count':    len([b for b in bench if b]),
        'has_psyduck':    any(_pk_id(b)==PSYDUCK for b in bench if b),
        'has_shaymin':    any(_pk_id(b)==SHAYMIN for b in bench if b),
        'has_genesect':   any(_pk_id(b)==GENESECT for b in bench if b),
        'has_fez':        any(_pk_id(b)==FEZ for b in bench if b),
        'genesect_active': genesect_with_tool,
        'active_is_kadabra': _pk_id(my_active)==KADABRA,
    }

PHASE_ESTABLISH=1; PHASE_CONVERT=2; PHASE_PRESSURE=3; PHASE_CLOSING=4

def _detect_phase(cen,can_ko,at_threshold,opp_prizes_left,hand_n):
    if opp_prizes_left<=2: return PHASE_CLOSING
    # NOTE: backup_abra (2+ Abra) and draw_count (Dunsparce/Dudunsparce currently in
    # play) are deliberately excluded here. Both are ephemeral, self-cycling resources
    # — Dudunsparce's own ability shuffles ITSELF back into the deck on use, so
    # draw_count routinely drops to 0 mid-game the instant you use the engine, and a
    # backup Abra is rarely available once it's been used climbing the evolution line.
    # Gating phase progression on either kept whole games stuck in ESTABLISH (and its
    # overdraw-permissive scoring) even with a fully-fueled, attacking Alakazam and a
    # huge hand. The only real gate for "established" is: is Alakazam up and fed.
    not_established=(not cen['has_alakazam'] or not cen['has_energy_plan'])
    if not_established: return PHASE_ESTABLISH
    if can_ko or at_threshold: return PHASE_PRESSURE
    return PHASE_CONVERT

def _score_setup_active(obs,opts):
    # v22 fix: options never carry cardId — they are {type:CARD, area:2(HAND), index}.
    # Resolve through our hand. (v21 read o['cardId'], which is always None, so every
    # option scored the default 20 and it silently picked opts[0] every game.)
    PREF={ABRA:90,DUNSPARCE:100,DUNSPARCE2:100,PSYDUCK:40,SHAYMIN:30,GENESECT:25,FEZ:5}
    cur=obs.get('current') or{}; me=cur.get('yourIndex',0)
    players=cur.get('players',[])
    hand=_hand_list(players[me]) if len(players)>me else []
    scores=[]
    for o in opts:
        idx=o.get('index')
        cid=_pk_id(hand[idx]) if idx is not None and 0<=idx<len(hand) else -1
        scores.append(float(PREF.get(cid,20)))
    return scores

def _pick_setup_active(obs,opts):
    if not opts: return[0]
    s=_score_setup_active(obs,opts)
    return[max(range(len(s)),key=lambda i:s[i])]

def _score_deck_search(obs,sel):
    """v22: deck searches are NOT blind — sel['deck'] lists the deck and each option's
    'index' points into it. Score candidates by what the board actually needs instead
    of taking the first N options (which is what the old generic fallback did)."""
    deck=sel.get('deck') or []
    opts=sel.get('option',[])
    cur=obs.get('current') or{}; me=cur.get('yourIndex',0)
    players=cur.get('players',[])
    my=players[me] if len(players)>me else {}
    bench=my.get('bench') or[]; my_active=_active(my)
    hand=_hand_list(my); hand_ids=[_pk_id(c) for c in hand]
    cen=_census(my_active,bench)
    candy_in_hand=RARE_CANDY in hand_ids
    kadabra_in_hand=KADABRA in hand_ids
    alak_in_hand=ALAKAZAM in hand_ids
    abra_in_hand=ABRA in hand_ids
    # cen['line_count'] here is entirely Abra/Kadabra (has_alakazam is false in this
    # branch) -- i.e. whether we have anything IN PLAY to promote into an Alakazam.
    # Deliberately excludes kadabra_in_hand: a Kadabra card sitting in hand with no
    # Abra in play (or hand) can't be played AT ALL -- it isn't a Basic, and Rare
    # Candy also needs a Basic already in play to Candy from. It's just as dead as
    # an Alakazam in that state, so it doesn't count as "having a line piece"
    # either (confirmed via replay 83461698: Hilda's search offered Alakazam/
    # Kadabra/Dudunsparce with zero Abra anywhere -- both evolution stages were
    # equally unplayable that turn).
    have_line_piece=cen['line_count']>0 or abra_in_hand
    def card_score(cid):
        if cid==ALAKAZAM:
            if not cen['has_alakazam'] and not alak_in_hand:
                # Fetching Alakazam with no Abra/Kadabra anywhere (play or hand) to
                # evolve it from is dead weight in hand -- grab the line piece
                # instead (Abra scores 90 below when abra_count==0). Confirmed via
                # replay: Poke Pad fetched Alakazam turn 1 with zero Abra in play
                # or hand, leaving it stuck unplayable.
                return 95.0 if have_line_piece else 20.0
            return 40.0
        if cid==KADABRA:
            if cen['abra_count']>0 and not kadabra_in_hand and not candy_in_hand: return 70.0
            return 30.0
        if cid==ABRA:
            if cen['abra_count']==0: return 90.0
            if not cen['backup_abra']: return 60.0
            return 25.0
        if cid in(DUNSPARCE,DUNSPARCE2):
            if cen['draw_count']==0: return 55.0
            return 20.0
        if cid==DUDUNSPARCE:
            if cen['draw_count']>0 and cen['dudun_bench']==0: return 50.0
            return 18.0
        if cid in PSYCHIC_ENERGY_IDS:
            if not cen['has_energy_plan']: return 85.0
            return 35.0
        if cid==ENRICHING: return 22.0
        if cid==PSYDUCK:  return 12.0
        if cid==SHAYMIN:  return 10.0
        if cid==GENESECT: return 11.0
        if cid==FEZ:      return 4.0
        if cid==RARE_CANDY: return 28.0
        if cid==BOSS:     return 26.0
        return 15.0
    scores=[]
    for o in opts:
        di=o.get('index')
        cid=_pk_id(deck[di]) if di is not None and 0<=di<len(deck) else -1
        scores.append(card_score(cid))
    return scores

def _deck_search_pick(obs,sel):
    opts=sel.get('option',[])
    scores=_score_deck_search(obs,sel)
    order=sorted(range(len(opts)),key=lambda i:-scores[i])
    mn=sel.get('minCount',0) or 0; mx=sel.get('maxCount',1) or 1
    picks=order[:mx]
    return _clamp(picks,sel) if len(picks)>=mn else _clamp(list(range(len(opts))),sel)

def _score_energy_discard(obs,sel):
    """v22: Enhanced Hammer / discard-energy selects (stype=4). Options carry
    area/index/energyIndex into the target pokemon's energyCards. Prefer discarding
    Mist/Rocky (the Powerful Hand blockers); otherwise keep the old first-pick."""
    cur=obs.get('current') or{}; me=cur.get('yourIndex',0)
    players=cur.get('players',[])
    opts=sel.get('option',[])
    scores=[]
    for o in opts:
        pidx=o.get('playerIndex'); area=o.get('area'); idx=o.get('index',0)
        ei=o.get('energyIndex',0)
        eid=None
        if pidx is not None and pidx<len(players):
            pl=players[pidx]
            pk=_active(pl) if area==4 else ((pl.get('bench') or[None]*5)[idx] if idx<len(pl.get('bench') or[]) else None)
            ecs=(pk or{}).get('energyCards') or []
            eid=ecs[ei].get('id') if 0<=ei<len(ecs) else None
        scores.append(100.0 if eid in (MIST_ENERGY,ROCK_ENERGY) else 1.0)
    return scores

def _pick_energy_discard(obs,sel):
    opts=sel.get('option',[])
    scores=_score_energy_discard(obs,sel)
    order=sorted(range(len(opts)),key=lambda i:-scores[i])
    return _clamp(order,sel)

def _score_evolve_target(obs,sel):
    """v22: Rare Candy target select (stype=7, ctx=EVOLVE). Options carry
    inPlayArea/inPlayIndex (which Abra). Prefer the ACTIVE Abra — it's already
    positioned to attack, so Candying it gets an attacker online immediately,
    whereas Candying a bench Abra still needs a promotion/retreat afterward.
    Psychic-fueled is a secondary tiebreak on top of that."""
    cur=obs.get('current') or{}; me=cur.get('yourIndex',0)
    players=cur.get('players',[])
    my=players[me] if len(players)>me else {}
    bench=my.get('bench') or[]; my_active=_active(my)
    opts=sel.get('option',[])
    scores=[]
    for o in opts:
        tgt=_attach_target(o,my_active,bench)
        s=0.0
        if o.get('inPlayArea')==4: s+=15.0
        if _has_psychic(tgt): s+=10.0
        scores.append(s)
    return scores

def _pick_evolve_target(obs,sel):
    if not sel.get('option'): return[0]
    s=_score_evolve_target(obs,sel)
    return[max(range(len(s)),key=lambda i:s[i])]

def _pick_setup_bench(opts): return list(range(len(opts)))

def _bench_target_priority(pk):
    """Shared 'how good a target is this bench Pokemon' ranking, used both for
    forced post-KO promotion (_score_bench_target) and as a same-tier tiebreak
    for voluntary retreats (score()'s RETREAT branch). An un-evolved line piece
    (Kadabra especially) is a much better bet than a pure-support mon (Psyduck/
    Fez/Genesect) that can never become the attacker; Dunsparce/Dunsparce2 rank
    just above those (a live path to Dudunsparce's draw engine) but below
    everything else. Confirmed via replay 84709203/84710776 (2026-07-07): a
    fully-fueled, ready Alakazam was voluntarily retreated INTO a 0-energy
    Psyduck purely because every RETREAT option scored identically regardless
    of target and Psyduck's happened to sort first -- it could never retreat
    back out (no energy to pay the cost) and the game ended in a deck-out loss."""
    pid=_pk_id(pk)
    if pid==ALAKAZAM and _has_psychic(pk): return 100
    if pid==ALAKAZAM: return 80
    if pid==KADABRA and _has_psychic(pk): return 70
    if pid==KADABRA: return 55
    if pid==DUDUNSPARCE: return 50
    if pid in PIVOT_FREE_RETREAT_IDS: return 40
    if pid==ABRA and _has_psychic(pk): return 30
    if pid==ABRA: return 20
    if pid in(DUNSPARCE,DUNSPARCE2): return 5
    return -10

def _score_bench_target(obs,opts):
    cur=obs.get('current') or{}; me_idx=cur.get('yourIndex',0)
    players=cur.get('players',[]); me=players[me_idx] if players and len(players)>me_idx else{}
    bench=me.get('bench') or[]
    area5=[(i,o) for i,o in enumerate(opts) if o.get('area')==5]
    scores=[-1e9]*len(opts)
    for order,(i,o) in enumerate(area5):
        idx=o.get('index',order)
        pk=bench[idx] if 0<=idx<len(bench) else(bench[order] if order<len(bench) else None)
        # These used to share the same -10 fallback for every non-line-piece
        # Pokemon, so ties broke on array order instead of board value
        # (confirmed losing this exact coinflip in a replay: promoted Psyduck
        # over an already-energized Kadabra sitting right next to it).
        scores[i]=_bench_target_priority(pk)
    return scores

def _score_wondrous_patch_target(obs,opts):
    """Wondrous Patch attaches a recovered Basic {P} Energy to the selected benched
    {P} Pokemon -- the OPPOSITE tiebreak from retreat/promotion targeting (which
    wants whoever can ALREADY attack): here we want whoever NEEDS the energy.
    Reusing _score_bench_target for this (both are stype==1, area==5 selects) would
    prefer an already-fueled Alakazam over an unfueled Kadabra/Abra, wasting the
    attach. Distinguished via sel['effect']['id']==WONDROUS_PATCH -- confirmed via
    replay that this is unambiguous (plain retreat/promotion selects have effect=
    None or a different card's id, e.g. Boss's Orders' 1182)."""
    cur=obs.get('current') or{}; me_idx=cur.get('yourIndex',0)
    players=cur.get('players',[]); me=players[me_idx] if players and len(players)>me_idx else{}
    bench=me.get('bench') or[]
    scores=[-1e9]*len(opts)
    for i,o in enumerate(opts):
        if o.get('area')!=5: continue
        idx=o.get('index',i)
        pk=bench[idx] if 0<=idx<len(bench) else None
        pid=_pk_id(pk)
        if _has_psychic(pk): s=-10.0  # already fueled -- this attach would be wasted
        elif pid==ALAKAZAM: s=100.0
        elif pid==KADABRA:  s=70.0
        elif pid==ABRA:     s=40.0
        else: s=0.0
        scores[i]=s
    return scores

def _pick_wondrous_patch_target(obs,opts):
    if not opts: return[0]
    s=_score_wondrous_patch_target(obs,opts)
    return[max(range(len(s)),key=lambda i:s[i])]

def _pick_discard_return(obs,opts):
    """Night Stretcher: choose which discard card to return to hand. Prefer a
    line piece (Abra plays to bench immediately > Kadabra > Alakazam), else a
    Psychic energy (still +1 hand = +20 Powerful Hand damage)."""
    cur=obs.get('current') or{}; me=cur.get('yourIndex',0)
    players=cur.get('players') or[]
    mine=players[me] if len(players)>me else{}
    discard=mine.get('discard') or[]
    def _rank(o):
        idx=o.get('index')
        cid=_pk_id(discard[idx]) if (idx is not None and 0<=idx<len(discard)) else -1
        if cid==ABRA: return 5
        if cid==KADABRA: return 4
        if cid==ALAKAZAM: return 3
        if cid in PSYCHIC_ENERGY_IDS: return 2
        return 0
    order=sorted(range(len(opts)),key=lambda i:-_rank(opts[i]))
    return [order[0]]

def _pick_bench_target(obs,opts):
    if not opts: return[0]
    s=_score_bench_target(obs,opts)
    return[max(range(len(s)),key=lambda i:s[i])]

def _score_boss_target(obs,sel):
    """Standalone scoring mirror of _pick_boss_target's tiered KO > damage >
    mist-KO > fallback logic, for use as an MCTS prior (score_options). Kept as
    a SEPARATE function rather than refactoring _pick_boss_target into a thin
    wrapper around it — Boss's Orders targeting is too game-critical to risk
    subtle drift from collapsing tiered tie-break logic into one scalar via
    lexicographic weight encoding. _pick_boss_target's own decision logic is
    untouched. Encoding: tiers are separated by 1e9 (KO=3e9, damage=2e9,
    mist-KO=1e9); within a tier, weights (1e5, 1e2, 1) are wide enough apart
    that pv<=3, hp<1000, and energy-count<100 can never bleed into the next
    weight's digit range, so tie-break ordering exactly matches the tuple-key
    max() used by _pick_boss_target."""
    cur=obs.get('current') or{}; me=cur.get('yourIndex',0)
    players=cur.get('players',[]); opp_idx=1-me
    opp=players[opp_idx] if len(players)>opp_idx else{}
    opp_bench=opp.get('bench') or[]
    hand_n=_hand_size(cur,me); boss_dmg=(hand_n-1)*PH_DMG_PER_CARD
    opts=sel.get('option',[])
    scores=[0.0]*len(opts)
    any_candidate=False
    for i,o in enumerate(opts):
        bi=o.get('index',0)
        pk=opp_bench[bi] if 0<=bi<len(opp_bench) else None
        if not pk:
            scores[i]=-1e9; continue
        any_candidate=True
        pk_hp=(pk.get('hp',99999) or 99999); pv=_prize_value_pk(pk)
        ec=len((pk or{}).get('energies') or [])
        pid=_pk_id(pk)
        pk_walled=_opp_has_blocking_energy(pk)
        if boss_dmg>=pk_hp and not pk_walled:
            scores[i]=3e9 + pv*1e5 + min(pk_hp,999)*1e2 + min(ec,99)
        elif boss_dmg>=pk_hp:
            scores[i]=1e9 + pv*1e5 + min(pk_hp,999)*1e2 + min(ec,99)
        elif not pk_walled:
            dmg_pct=min(boss_dmg/pk_hp, 1.0) if pk_hp>0 else 0
            threat_score=100 if pid==ALAKAZAM else(40 if pid==KADABRA else(30 if pid in PIVOT_FREE_RETREAT_IDS else 10))
            scores[i]=2e9 + pv*300 + dmg_pct*100 + threat_score
        else:
            scores[i]=-1e9
    if not any_candidate:
        scores=[0.0]+[-1e9]*(len(opts)-1) if opts else []
    return scores

def _pick_boss_target(obs,sel):
    cur=obs.get('current') or{}; me=cur.get('yourIndex',0)
    players=cur.get('players',[]); opp_idx=1-me
    opp=players[opp_idx] if len(players)>opp_idx else{}
    opp_bench=opp.get('bench') or[]
    hand_n=_hand_size(cur,me); boss_dmg=(hand_n-1)*PH_DMG_PER_CARD
    opts=sel.get('option',[]); ko_targets=[]; dmg_targets=[]; mist_ko_targets=[]
    for i,o in enumerate(opts):
        bi=o.get('index',0)
        pk=opp_bench[bi] if 0<=bi<len(opp_bench) else None
        if not pk: continue
        pk_hp=(pk.get('hp',99999) or 99999); pv=_prize_value_pk(pk)
        ec=len((pk or{}).get('energies') or [])
        pid=_pk_id(pk)
        # Gusting a Pokemon that ALSO has Mist/Rocky Energy just recreates the wall —
        # only counts as a real KO/damage option if the wall isn't there too.
        pk_walled=_opp_has_blocking_energy(pk)
        if boss_dmg>=pk_hp and not pk_walled:
            ko_targets.append((i,pv,pk_hp,ec))
        elif boss_dmg>=pk_hp:
            mist_ko_targets.append((i,pv,pk_hp,ec))
        elif not pk_walled:
            dmg_pct=min(boss_dmg/pk_hp, 1.0) if pk_hp>0 else 0
            dmg_targets.append((i,pv,pid,dmg_pct,pk_hp))
    if ko_targets:
        best=max(ko_targets,key=lambda x:(x[1],x[2],x[3]))
        return[best[0]]
    if dmg_targets:
        def dmg_value(t):
            i,pv,pid,dmg_pct,hp=t
            threat_score=0
            if pid==ALAKAZAM: threat_score=100
            elif pid==KADABRA: threat_score=40
            elif pid in PIVOT_FREE_RETREAT_IDS: threat_score=30
            else: threat_score=10
            return pv*300 + dmg_pct*100 + threat_score
        best=max(dmg_targets,key=dmg_value)
        return[best[0]]
    # Every bench option is also Mist/Rocky-walled — no choice makes progress, so fall
    # back to the highest-prize KO available (denies the biggest investment at least).
    if mist_ko_targets:
        best=max(mist_ko_targets,key=lambda x:(x[1],x[2],x[3]))
        return[best[0]]
    return[0]

def _main_phase_features(obs,sel):
    """Computes every board-state local the MAIN-phase scorer needs, then
    returns the `score(o)` closure itself (unchanged body, just relocated so
    it's reachable outside the argmax-and-return control flow that used to
    own it). This lets score_options_main(obs,sel) expose a per-option score
    vector — the heuristic side of the MCTS prior blend — via one call,
    instead of re-deriving this analysis inline. Pure: no I/O, no RNG, no
    mutation of _STALL_MEMO or any other global (search may call this
    thousands of times per game)."""
    opts=sel['option']
    cur=obs.get('current') or{}; me_idx=cur.get('yourIndex',0)
    players=cur.get('players',[])
    my=players[me_idx]   if players and len(players)>me_idx else{}
    opp=players[1-me_idx] if players and len(players)==2     else{}
    supporter_played=cur.get('supporterPlayed',False)
    energy_attached =cur.get('energyAttached',False)
    retreated       =cur.get('retreated',False)
    my_active=_active(my); opp_active=_active(opp)
    hand_n=_hand_size(cur,me_idx); hand=_hand_list(my); bench=my.get('bench') or[]
    lone=len([x for x in([my_active]+bench) if x])<=1
    cen=_census(my_active,bench)
    deck_count=my.get('deckCount',0) or 0
    discard=my.get('discard') or[]
    deck_critical=deck_count<10
    deck_danger  =deck_count<5
    prizes=len(my.get('prize') or[]); opp_prizes=len(opp.get('prize') or[])
    # Desperation: opponent can end the game on their next KO -- either they only
    # need 1 more prize, or they need 2 and we have an ex in play/bench (a single
    # KO on it hands over both at once). Confirmed via replay 83429870: at 1-1
    # prizes we lost trying to out-survive a lethal swing, when the actual winning
    # line was maxing hand size and Boss-sniping for the last prize instead (see
    # docs/report-log.md). Nothing here is about surviving longer -- if we can't
    # close it out this turn, next turn is likely a loss anyway, so deck-out risk
    # stops mattering relative to just building toward lethal.
    we_have_ex=any((p or{}).get('ex',False) for p in([my_active]+bench) if p)
    desperation=opp_prizes<=1 or(opp_prizes<=2 and we_have_ex)
    opp_played, bench_dmg_received, we_were_kod, opp_ace_observed = _analyze_logs(obs, me_idx)
    opp_mist=_opp_has_blocking_energy(opp_active)
    # ---- Stage 3 Phase C: belief-driven reads (docs/belief-model.md §Phase C) ----
    turn_n=cur.get('turn') or 0
    belief_post,opp_wall_revealed,opp_crustle_seen=_belief_posterior(opp,turn_n)
    if belief_post:
        b_top=max(belief_post,key=belief_post.get); b_conf=belief_post[b_top]
    else:
        b_top=None; b_conf=0.0
    # Ace-spec read: observed (logs) beats inferred; a low-confidence read keeps
    # the pre-Phase-C conservative True default; a confident read uses the
    # archetype's real ladder ace-spec reveal rate (679-replay tech survey).
    opp_likely_ace=opp_ace_observed or b_conf<0.97 or _ACE_RATE.get(b_top,1.0)>=0.35
    opp_hp=(opp_active or{}).get('hp',99999) or 99999
    attack_available=any(o.get('type')==ATTACK for o in opts)
    active_is_alak=_pk_id(my_active)==ALAKAZAM
    active_can_attack=active_is_alak and attack_available
    active_non_atk=_is_non_attacker(my_active)
    alak_stuck=active_is_alak and not attack_available
    bench_has_alak=any(_pk_id(b)==ALAKAZAM for b in bench if b)
    bench_has_alak_ready=any(_can_ph(b) for b in bench if b)
    bench_has_attacker=any(_pk_id(b) in{ALAKAZAM,KADABRA} for b in bench if b)
    my_dmg=(PH_DMG_PER_CARD*hand_n) if active_can_attack and not opp_mist else 0
    can_ko=active_can_attack and opp_active is not None and my_dmg>=opp_hp and not opp_mist
    opp_bench=opp.get('bench') or[]
    opp_bench_empty=len([b for b in opp_bench if b])==0
    opp_bench_wall=any(_opp_has_blocking_energy(b) for b in opp_bench if b)
    # Mist/Rocky ANTICIPATION (not just reaction, which opp_mist covers): real-
    # ladder walls live almost entirely in crustle decks and the unrecognized
    # long tail (tech survey 2026-07-04: crustle 35.8%, unknown 29.0%, the 5
    # classifier archetypes 0-3%). Threat = wall energy already revealed
    # anywhere on their side, a Crustle/Dwebble line, or a deck the classifier
    # can't confidently place by turn 2+. On belief failure (belief_post None)
    # this stays reactive-only -- pre-Phase-C behavior.
    mist_threat=(opp_wall_revealed or opp_bench_wall or opp_crustle_seen or
                 (belief_post is not None and b_conf<0.97 and turn_n>=2))
    # Cheap, rough lookahead (deliberately NOT a real turn simulation): if the
    # opponent has nothing but their Active in play, a big enough hand ends
    # their turn staring at an empty board, which is worth spending reserved
    # draw sources on NOW rather than banking them for a hypothetical future
    # hand-disruption effect (the piloting-guide's normal "bank surplus draw"
    # advice) -- there's no bigger prize than winning the board back outright.
    # Headroom estimate: current hand + each ready Dudunsparce's Run Away Draw
    # (+3 each, piloting-guide §3) + the best not-yet-played supporter in hand
    # (Hilda's fetch chain or Dawn) + an unattached Enriching Energy's draw 4.
    _untapped_draw=3*cen['dudun_bench']
    if not supporter_played:
        if any(_pk_id(c)==HILDA for c in hand): _untapped_draw+=4
        elif any(_pk_id(c)==DAWN for c in hand): _untapped_draw+=2
    if any(_pk_id(c)==ENRICHING for c in hand): _untapped_draw+=3
    max_hand_estimate=hand_n+_untapped_draw
    lone_active_opportunity=(
        opp_bench_empty and opp_active is not None and opp_hp<99999 and
        not opp_mist and max_hand_estimate*PH_DMG_PER_CARD>=opp_hp)
    opp_active_pv=_prize_value_pk(opp_active)
    # Anti-tank support farming (report-log 2026-07-18 grimmsnarl diagnosis):
    # when the opponent's active is beyond even the optimistic hand ceiling
    # (Marnie's Grimmsnarl ex 320 under Adrena-Brain healing was the found
    # case -- 57 mined losses, Boss played 0-1 times/game while KO-able
    # 70-110 HP support sat benched), its prize value is unreachable this
    # turn, so the "don't downgrade prize value" gates on Boss targeting
    # below compare against a prize we cannot actually take. State-keyed
    # (not archetype-keyed) on purpose: self-resolves the moment the hand
    # grows into real range of the active.
    opp_tank_unreachable=(opp_active is not None and not opp_mist and
                          opp_hp<99999 and
                          opp_hp>max_hand_estimate*PH_DMG_PER_CARD)
    boss_in_hand=any(_pk_id(c)==BOSS for c in hand)
    hammer_in_hand=any(_pk_id(c)==ENHANCED_HAMMER for c in hand)
    tool_in_hand=any(_pk_id(c) in TOOL_IDS for c in hand)
    # Hilda's own search pool is Stage-1/2 (Kadabra/Alakazam) + energy only --
    # docs/piloting-guide.md confirms Dawn alone grabs "Basic + Stage 1 + Stage
    # 2." With zero line pieces in play and no Abra in hand, Hilda cannot fetch
    # the one card that actually unblocks us (confirmed via replay 83461698:
    # Hilda's own deck-search options only ever offered Alakazam/Kadabra/
    # Dudunsparce -- no Abra was even a legal choice). Dawn's flat ESTABLISH
    # weight already narrowly beats Hilda's (22 vs 24) in the scoring below, so
    # this needs to actively suppress Hilda here rather than rely on the
    # existing weight gap.
    abra_in_hand=any(_pk_id(c)==ABRA for c in hand)
    need_basic_abra=cen['line_count']==0 and not abra_in_hand
    # Opponent's Active is permanently walling Powerful Hand (Mist/Rocky Energy) and we
    # have no way left to answer it this turn (no Hammer to strip the energy, no Boss
    # to gust a different, unwalled target) — more searching/drawing can't fix a wall
    # made of card TYPE, only of hand size, so it's pure deck-out risk for zero payoff.
    hopelessly_walled=opp_mist and not hammer_in_hand and not boss_in_hand
    boss_dmg=(hand_n-1)*PH_DMG_PER_CARD
    ready_alak_exists=active_can_attack or bench_has_alak_ready
    opp_bench_ko_gte=any(
        0<(b or{}).get('hp',99999)<=boss_dmg and _prize_value_pk(b)>=opp_active_pv
        for b in opp_bench if b)
    boss_snipe_plan=(boss_in_hand and ready_alak_exists and opp_bench_ko_gte and opp_hp>boss_dmg)
    cards_needed=math.ceil(opp_hp/PH_DMG_PER_CARD) if opp_hp<99999 else 999
    at_threshold=active_can_attack and hand_n>=cards_needed and not opp_mist
    hand_too_small=hand_n<max(3,math.ceil(opp_hp/PH_DMG_PER_CARD/3))
    emergency_draw=hand_n<=4
    phase=_detect_phase(cen,can_ko,at_threshold,opp_prizes,hand_n)
    in_late_phase=phase in(PHASE_CONVERT,PHASE_PRESSURE,PHASE_CLOSING)
    # v23: Boss's Orders is a one-shot resource (3 copies) that should be SAVORED for
    # high-value bench snipes once the hand is actually built up, not spent early to
    # pick off any killable small-fry bench mon — that trades a scarce out for a target
    # that often wasn't threatening anything anyway. Require BOTH: (a) hand already
    # developed (in_late_phase — ESTABLISH means Alakazam isn't even fed yet, far too
    # early to burn a Boss) and (b) the target is actually worth gusting: an ex/mega
    # (2-3 prizes) or a beefy (150+ HP) Pokemon — not just "kills whatever's back there".
    boss_target_exists=(
        # opp_mist doesn't block Boss — it blocks Powerful Hand against the CURRENT
        # active only. A Mist-walled active makes ANY killable bench target the
        # correct play (it's the only way to make progress at all), not a reason to
        # skip this check in favor of the un-informed opp_mist fallback below.
        active_can_attack and (opp_mist or opp_hp>my_dmg) and
        (in_late_phase or opp_tank_unreachable) and
        any(0<(b or{}).get('hp',99999)<=boss_dmg and
            (_prize_value_pk(b)>=2 or (b or{}).get('hp',99999)>=150 or
             # unreachable-tank state: ANY bench KO converts a dead turn
             # into a prize (grimmsnarl support farming, 2026-07-18)
             opp_tank_unreachable) and
            (_prize_value_pk(b)>=opp_active_pv or opp_tank_unreachable)
            for b in opp_bench if b))
    active_hp=(my_active or{}).get('hp',99999) or 99999
    active_max_hp=(my_active or{}).get('maxHp',999) or 999
    active_below_half=(active_max_hp-active_hp)>active_max_hp//2
    active_vulnerable=active_hp<60 or (active_below_half and opp_prizes<=3)
    boss_ex_snipe=(
        can_ko and boss_in_hand and not opp_mist and
        any(boss_dmg>=(pk.get('hp',99999) or 99999) and _prize_value_pk(pk)>opp_active_pv
            for pk in opp_bench if pk))
    boss_can_damage_mega=(
        boss_in_hand and not opp_mist and ready_alak_exists and
        any(boss_dmg>=50 and _prize_value_pk(pk)==3
            for pk in opp_bench if pk))
    alak_in_discard=any(_pk_id(c) in ALAKAZAM_LINE_IDS for c in discard)
    opp_bench_low_hp=any(0<(b or{}).get('hp',999)<=100 for b in opp_bench if b)
    enriching_on_dudun=any(_pk_id(b)==DUDUNSPARCE and _energies(b) for b in bench if b)
    active_kadabra_can_evolve=(
        _pk_id(my_active)==KADABRA and
        not (my_active or{}).get('appearThisTurn',False))
    active_abra_can_evolve=(
        _pk_id(my_active)==ABRA and
        not (my_active or{}).get('appearThisTurn',False))
    candy_playable=any(
        o.get('type')==PLAY and _opt_card_id(o,hand,my_active,bench)==RARE_CANDY
        for o in opts)
    retreat_available=any(o.get('type')==RETREAT for o in opts)
    active_free_retreat=_pk_id(my_active) in PIVOT_FREE_RETREAT_IDS
    # Energy only "frees" an immobile Active if it actually enables something:
    # Alakazam can attack with 1 Psychic regardless of bench; anyone else needs a
    # bench Alakazam that's ALREADY ready to attack to retreat INTO (retreating into
    # an un-fueled Kadabra/Abra/support mon still leaves you unable to attack this
    # turn, so it isn't a fix) — and a free-retreater (Shaymin) was never blocked by
    # energy in the first place, so attaching to it fixes nothing either way.
    active_immobile=(
        my_active is not None and not attack_available and not retreat_available and
        len(_energies(my_active))==0 and not active_free_retreat and
        (active_is_alak or bench_has_alak_ready))
    # Threshold discipline (§4/§10 piloting-guide): once a ready attacker exists and the
    # hand is already at the KO threshold, more draw is pure deck-out risk -> stop drawing.
    ready_attacker_exists=active_can_attack or bench_has_alak_ready
    # Board-thinning root cause (replays win_001/win_010): this gate previously required
    # ready_attacker_exists, so it never engaged mid-rebuild (attacker just KO'd, no bench
    # backup) -- exactly when hand was already 2-3x cards_needed from earlier banking.
    # Powerful Hand's damage is capped by opp_hp regardless of attacker readiness, so a
    # hand this far over cards_needed is already-wasted value no matter what; confirmed
    # both replays burn the deck to 0 in a single turn via Poffin/Dawn/Hilda/Poke Pad
    # spam at hand_n=16-23 vs cards_needed=7 while stuck on a non-attacker.
    hand_grossly_over=opp_hp<99999 and hand_n>=cards_needed+6
    hand_surplus=(
        (ready_attacker_exists or hand_grossly_over) and opp_hp<99999 and hand_n>=cards_needed and
        not boss_snipe_plan and not emergency_draw and not desperation and
        not lone_active_opportunity)
    # Down to a single Pokemon in play (no bench at all) is a distinct existential
    # risk regardless of prize lead or hand size: one KO on the Active with nothing
    # to promote is an instant loss. Confirmed losing exactly this way in a replay
    # (140 HP Alakazam active, empty bench, KO'd on the opponent's next turn with
    # a commanding prize lead otherwise) — surplus-hand and phase gating shouldn't
    # suppress rebuilding the bench once it's completely empty.
    bench_empty=cen['bench_count']==0
    # Manual evolve (Abra->Kadabra->Alakazam) banks Kadabra's own +2 Psychic Draw
    # that Rare Candy skips (Abra->Alakazam directly) -- +1 net card overall, per
    # docs/piloting-guide.md §3, but costs a turn's tempo. "Racing" = we have no
    # attacker in play at all AND either we're in danger, past the free-setup
    # phase, or already low on cards -- exactly when the guide says Candy is
    # worth its cost ("speed only... to land a turn-2 Alakazam, or to rebuild
    # after a KO"). Otherwise we have time to bank the extra card.
    # NOTE: deliberately uses active_below_half (relative), not active_vulnerable
    # (which has an active_hp<60 absolute clause -- always true for Abra, whose
    # max HP is 50, making it a false positive for exactly the Pokemon this
    # check most needs to evaluate correctly). Also deliberately excludes
    # emergency_draw (hand_n<=4) -- that's spuriously true turn 1-2 before the
    # draw engine has run at all, which is exactly when we DO have time to
    # climb manually, not an emergency. Both caught by direct testing before
    # this was trusted. phase in(PRESSURE,CLOSING) mostly reduces to "opponent
    # down to <=2 prizes" here, since _detect_phase forces ESTABLISH whenever
    # we have no Alakazam yet (its own not_established gate) except for that
    # opp_prizes_left<=2 short-circuit.
    # Offensive racing trigger: the two conditions above are purely defensive/phase-
    # based and blind to a simpler case -- Candying straight to Alakazam now sets up
    # a next-turn KO on the opponent's CURRENT active (Candy disables attacking the
    # turn it's played, so the actual attack happens next turn; current hand size is
    # the lethal-capacity proxy since it can only grow by then). Worth rushing even
    # turn 1-2, unlike the HP/phase triggers.
    candy_lethal_soon=(
        active_abra_can_evolve and not cen['has_alakazam'] and opp_active is not None and
        not opp_mist and hand_n*PH_DMG_PER_CARD>=opp_hp)
    racing_for_alakazam=(
        desperation or
        (not cen['has_alakazam'] and
         ((active_below_half and opp_prizes<=3) or phase in(PHASE_PRESSURE,PHASE_CLOSING) or
          candy_lethal_soon)))
    # Rescue mode (report-log 2026-07-16 rescue mining + same-day amendment):
    # in the board-thinning regime (issue-#3's pre-registered detector rule,
    # inlined: turn>=9 AND (no line piece in play OR deck<=6 with hand>=15)),
    # attacking IMMEDIATELY forfeits the turn's remaining development for
    # nothing -- the KO does not expire within the turn. One-step-deviation
    # data: from states where this scorer chose ATTACK, deviating into
    # ATTACH/PLAY/EVOLVE/ABILITY rescued ~2.5x baseline (19-22% vs 7.9%),
    # while deviating to END (2.7%) or RETREAT (0%) from the SAME states was
    # catastrophic -- the value is the development, not the delay. Amendment
    # (behavior check, same day, before any gate ran): hand>=15 makes every
    # available un-misted attack a KO, so the demotion must apply to LETHAL
    # attacks too, margin-gated -- develop first only while the hand keeps
    # >=3 cards of slack over the KO threshold (each development spends <=1
    # hand card = 20 damage; the per-select re-score restores ATTACK to 500
    # the moment slack thins). Never in desperation (opponent about to win:
    # take the prize NOW), and ATTACK always stays above END.
    rescue_mode=(turn_n>=9 and not desperation and
                 (cen['line_count']==0 or (deck_count<=6 and hand_n>=15)))
    rescue_hold_ko=rescue_mode and hand_n-cards_needed>=3

    def score(o):
        ot=o.get('type'); cid=_opt_card_id(o,hand,my_active,bench)
        if ot==ATTACK:
            if not active_can_attack: return-5
            if opp_mist: return-5
            if rescue_hold_ko: return 1.2
            if can_ko: return 500
            if at_threshold: return W['atk_threshold']
            if rescue_mode: return 1.2
            if hand_too_small: return 0.5
            return W['atk_default']
        if ot==RETREAT:
            if retreated: return-50
            tgt=_attach_target(o,my_active,bench); tgt_id=_pk_id(tgt)
            # Feeding an unevolved/mid-line piece (Abra, Kadabra) that can't
            # attack this turn anyway into an opponent who's already armed to
            # swing trades a piece we need for nothing -- it can't retreat
            # back out either. Confirmed via replay 84710513: a full-health
            # 210 HP Fez was pulled for a 50 HP unfueled Abra right as the
            # opponent evolved into a bigger attacker and one-shot it; the
            # right play was leaving Fez in (or sacrificing an actual
            # expendable body), not feeding the line piece. Deliberately
            # scores below every non-RETREAT fallback (END=1.0 etc.) so
            # holding still wins when nothing on the bench is both usable
            # and safe -- this overrides the "get an attacker established"
            # preference below, it doesn't just tiebreak within it.
            if tgt_id in(ABRA,KADABRA) and _opp_threatening(opp_active):
                return -4.0
            # Every RETREAT option below shares the same board-state
            # conditions (they don't depend on the target) and used to score
            # identically regardless of WHICH bench Pokemon we'd retreat
            # into -- ties then broke on array order, not board value (see
            # _bench_target_priority's docstring for the replay this caused).
            # This tiebreak is scaled small (max ~1.0) so it can only break
            # ties among RETREAT options themselves, never change whether
            # retreating beats another action.
            target_bonus=_bench_target_priority(tgt)/100.0
            if alak_stuck:
                if bench_has_alak_ready: return W['retreat_alak_stuck']+target_bonus
                return -5.0+target_bonus
            if active_non_atk and bench_has_alak_ready:
                return(W['retreat_nonatk_ready'] if active_below_half else W['retreat_alak_stuck'])+target_bonus
            if active_non_atk and bench_has_alak:
                return(20.0 if active_below_half else 16.0)+target_bonus
            if active_non_atk:                     return -3.0+target_bonus
            if active_can_attack:                  return-2.0+target_bonus
            return 0.5+target_bonus
        if ot==ABILITY:
            if lone: return-10
            if cid in SUPPRESS_ABILITY_IDS: return-10
            if cid in DRAW_ABILITY_CARD_IDS:
                # rescue_hold_ko: fall through to the normal deck-safety-gated
                # scoring -- the flat 2.0 would now fire BEFORE the deferred
                # KO attack and draw 3 at deck<=6 is a deck-out hazard
                if can_ko and not rescue_hold_ko: return 2.0
                if cid==DUDUNSPARCE:
                    if hand_surplus: return 0.5
                    if (desperation or lone_active_opportunity) and not emergency_draw: return W['desperation_draw']
                    if hopelessly_walled and not emergency_draw: return-6.0
                    if deck_danger and not emergency_draw:   return-8.0
                    if deck_critical and not emergency_draw: return-2.0
                    if hand_n>=14 and not emergency_draw:    return 1.0
                    if cen['dudun_bench']>1 and not emergency_draw: return 6.0
                    return W['dudun_base']
                if emergency_draw: return 15.0
                return 10.0
            if cid==FEZ:
                if deck_danger: return-5.0
                if hand_surplus: return-3.0
                if deck_critical and not emergency_draw: return-2.0
                return W['fez_base']
            return 5.0
        if ot==EVOLVE:
            evo_area=o.get('inPlayArea',4)
            if cid==ALAKAZAM:
                if evo_area==5:
                    if can_ko and not rescue_hold_ko: return 3.0
                    if not cen['has_alakazam']: return 50.0
                    if phase==PHASE_ESTABLISH: return W['evo_bench_establish']
                    if phase==PHASE_CONVERT: return W['evo_bench_convert']
                    return W['evo_bench_late']
                post_evo_dmg=(hand_n+3)*PH_DMG_PER_CARD
                if phase==PHASE_ESTABLISH and active_kadabra_can_evolve:
                    return 300
                if not can_ko and post_evo_dmg>=opp_hp and active_kadabra_can_evolve:
                    return 280
                if active_kadabra_can_evolve:
                    return 260
                if can_ko: return 5.0
                if not cen['has_alakazam']: return 16.0
                return 10.0
            if cid==KADABRA:
                if can_ko and not rescue_hold_ko: return 3.0
                if evo_area==4 and active_abra_can_evolve and candy_playable:
                    if racing_for_alakazam:
                        # Rare Candy can take THIS SAME active Abra straight to
                        # Alakazam this turn — normal-evolving to Kadabra first
                        # burns the turn's evolution on an intermediate stage and
                        # pushes Alakazam a full turn later. Worth it when we're
                        # racing (no attacker yet + in danger/past-establish/low
                        # on cards). Let Candy (scored below) win.
                        return 2.0
                    # Have time: bank Kadabra's own +2 Psychic Draw that Candy
                    # skips (net +1 card over the 2-turn climb, piloting-guide §3).
                    return 15.0
                return 13.0
            if can_ko and not rescue_hold_ko: return 2.0
            if active_non_atk: return 12.0
            return 8.5
        if ot==PLAY:
            if cid==ENHANCED_HAMMER:
                if opp_mist: return W['hammer_mist']
                # Phase C: a wall energy parked on their BENCH is a wall being
                # prepared -- strip it before it promotes (the select scorer
                # already prefers Mist/Rocky targets).
                if opp_bench_wall: return 36.0
                return 3.0
            if cid==BATTLE_CAGE:
                if bench_dmg_received: return W['cage_reactive']
                if in_late_phase and hand_n>=8: return 1.0
                return W['cage_base']
            if cid==BOSS:
                if boss_ex_snipe:        return 600.0
                if can_ko: return 1.0
                # NOTE: phase==PHASE_CLOSING used to return a flat 199 here with NO
                # target-quality check -- it fired even with an empty/useless bench,
                # or (worse) when the CURRENT active was already our best KO target.
                # Confirmed losing real value this way (replay 83458785, step94):
                # active Alakazam had 0 energy (attack_available False that turn, so
                # boss_target_exists below is also False), opponent's active was
                # Mega Starmie ex at 110/430 (already the best target on their
                # board) -- Boss's Orders still fired at 199, yanking it off active
                # in favor of a fresh 70-HP Staryu, undoing our own damage progress
                # for a 1-prize KO instead of the lined-up 2-3-prize one. Fold the
                # phase bump INTO boss_target_exists (which already verifies a real,
                # worthwhile bench target and that we can actually act this turn)
                # instead of firing unconditionally.
                if boss_target_exists:
                    return 199.0 if phase==PHASE_CLOSING else W['boss_target']
                if boss_can_damage_mega:
                    # Phase C Mist anticipation: mega chip damage is optional;
                    # vs wall-teching decks Boss is the escape of last resort,
                    # so hold it until the wall question is settled.
                    return 4.0 if (mist_threat and not opp_mist) else W['boss_mega_chip']
                if opp_mist and ready_alak_exists: return W['boss_mist_escape']
                return 4.0
            # rescue_hold_ko: KO deferred within the turn -- keep normal PLAY
            # routing (development plays were the 21.9%-win deviation class)
            if can_ko and not rescue_hold_ko: return 1.0
            if cid in BENCHABLE_BASIC_IDS:
                bc=cen['bench_count']
                if cid==FEZ:
                    if bc>=5: return-5.0
                    if active_vulnerable and not cen['has_fez']: return 14.0
                    if opp_bench_low_hp and not cen['has_fez'] and cen['has_alakazam']: return 10.0
                    if we_were_kod and not cen['has_fez']: return 16.0
                    if bc==0: return 3.0
                    return -1.0
                if cid==ABRA:
                    if not cen['backup_abra']:  return 20.0
                    if bc<3:                    return 10.0
                    return 4.0
                if cid in(DUNSPARCE,DUNSPARCE2):
                    if cen['draw_count']==0:    return 18.0
                    if bc<3:                    return 9.0
                    return 3.0
                if cid==PSYDUCK:
                    opp_ability_threat = bool(opp_played & {131, 132, 133})  # Duskull/Dusclops/Dusknoir (Cursed Blast)
                    if opp_ability_threat and not cen['has_psyduck']: return 18.0
                    if phase==PHASE_ESTABLISH and bc<3 and not cen['has_psyduck']: return 7.0
                    return 1.5
                if cid==SHAYMIN:
                    if bench_dmg_received and not cen['has_shaymin']: return 16.0
                    if phase==PHASE_ESTABLISH and bc<3 and not cen['has_shaymin']: return 5.0
                    return 1.0
                if cid==GENESECT:
                    if cen['has_genesect']: return 1.0
                    if tool_in_hand and opp_likely_ace: return 11.0
                    if bc<2: return 6.0
                    return 2.0
                if bc==0: return 10.0
                if bc==1: return 7.0
                return 3.0
            if cid==RARE_CANDY:
                if active_abra_can_evolve:
                    if racing_for_alakazam:
                        # Get the already-positioned Active online NOW. Worth the
                        # skipped Kadabra +2 draw when racing (no attacker yet +
                        # in danger/past-establish/low on cards) — it avoids the
                        # wasted-tempo pattern of normal-evolving to Kadabra this
                        # turn and only Candying a *different* (bench) Abra later,
                        # which leaves the active stuck needing its own extra turn.
                        return W['candy_active_abra']
                    # Have time: manual evolve (scored above, 15.0) banks the
                    # extra card instead — piloting-guide §3's "hold Candy when
                    # you have time" principle. Still usable, just not dominant.
                    return 10.0
                if cen['kadabra_can_evolve']:
                    if at_threshold or phase in(PHASE_PRESSURE,PHASE_CLOSING): return W['candy_ready']
                    if phase==PHASE_ESTABLISH: return W['candy_estab']
                    return 8.0
                if not cen['has_alakazam']: return 28.0
                if cen['need_line']:        return 14.0
                return 4.0
            if cid==POFFIN:
                if desperation or lone_active_opportunity: return W['desperation_draw']
                if deck_danger: return-8.0
                if hopelessly_walled: return-3.0
                if bench_empty: return W['poffin_estab']
                if hand_surplus: return 2.0
                if phase==PHASE_ESTABLISH and (not cen['backup_abra'] or cen['draw_count']==0):
                    if cen['bench_count']<5: return W['poffin_estab']
                if not cen['backup_abra'] or cen['draw_count']==0:
                    if cen['bench_count']<4: return W['poffin_need']
                if cen['bench_count']<2: return 14.0
                if phase==PHASE_ESTABLISH: return 12.0
                if in_late_phase: return 2.0
                return 6.0
            if cid==SACRED_ASH:
                if deck_danger:   return 35.0
                if deck_critical: return 25.0
                if alak_in_discard and not cen['has_alakazam']: return 12.0
                if phase==PHASE_CLOSING: return 5.0
                return 2.0
            if cid==LANA:
                if alak_in_discard and not cen['has_alakazam']: return 15.0
                if alak_in_discard: return 8.0
                if phase==PHASE_CLOSING: return 5.0
                return 2.0
            if cid==NIGHT_STRETCHER:
                # return a line piece (or Psychic energy) from discard to hand:
                # fixes board-thinning (10/27 fresh losses end with 0 line pieces
                # in play) and adds a card = 20 Powerful Hand damage.
                if alak_in_discard and not cen['has_alakazam']: return 16.0
                if alak_in_discard and cen['bench_count']<2:     return 11.0
                if alak_in_discard:                              return 6.0
                if phase==PHASE_CLOSING:                         return 4.0
                return 2.0
            if cid==DAWN:
                if supporter_played: return-5.0
                if desperation or lone_active_opportunity: return W['desperation_draw']
                if deck_danger: return-8.0
                if active_immobile: return-3.0
                if hopelessly_walled: return-3.0
                if hand_surplus: return 2.0
                if phase==PHASE_ESTABLISH and (cen['need_line'] or not cen['has_alakazam']): return W['dawn_estab']
                if boss_snipe_plan and not emergency_draw: return 1.0
                if deck_critical: return 1.0
                if hand_n>=12: return 2.0
                if emergency_draw: return 14.0
                if cen['need_line'] or not cen['has_alakazam']: return W['dawn_need']
                if phase==PHASE_CONVERT: return 8.0
                return 6.0
            if cid==HILDA:
                if supporter_played: return-5.0
                if desperation or lone_active_opportunity: return W['desperation_draw']
                if deck_danger: return-8.0
                if active_immobile: return W['hilda_immobile']
                if need_basic_abra: return 3.0
                if hopelessly_walled: return-3.0
                if hand_surplus: return 2.0
                if phase==PHASE_ESTABLISH and (cen['need_line'] or not cen['has_alakazam']): return W['hilda_estab']
                if boss_snipe_plan and not emergency_draw: return 1.0
                if deck_critical: return 1.0
                if not enriching_on_dudun and cen['draw_count']>0: return 11.0
                if not cen['has_alakazam']: return 13.0
                if emergency_draw: return 12.0
                if cen['need_line']: return 10.0
                if phase==PHASE_CONVERT: return 7.0
                return 5.0
            if cid==POKE_PAD:
                if desperation or lone_active_opportunity: return W['desperation_draw']
                if deck_danger: return-8.0
                if active_immobile: return-3.0
                if hopelessly_walled: return-3.0
                if bench_empty: return W['pad_no_backup']
                if hand_surplus: return 2.0
                if not cen['backup_abra']: return W['pad_no_backup']
                if cen['need_line']:       return 9.0
                if cen['need_draw']:       return 10.0
                if in_late_phase:          return 2.0
                return 5.0
            if cid==WONDROUS_PATCH:
                if not cen['has_energy_plan']: return 8.0
                if in_late_phase: return 2.0
                return 5.0
            if in_late_phase: return 1.5
            return 4.0
        if ot==SKILL: return 5.0
        if ot==ATTACH:
            # rescue_hold_ko: the KO is deliberately deferred within the turn,
            # so attaches keep their normal routing value (fueling the backup
            # line was the 19.4%-win deviation class) instead of the "game's
            # about to be won, don't waste the attach" guard.
            if can_ko and not rescue_hold_ko: return 0.5
            tgt=_attach_target(o,my_active,bench); tid=_pk_id(tgt)
            is_energy_card=cid in PSYCHIC_ENERGY_IDS or cid==ENRICHING
            if active_immobile and tgt is my_active and is_energy_card:
                # Free the stranded Active. Prefer real Psychic so a stuck Alakazam can
                # both retreat AND attack; Enriching still frees it but pays no {P}.
                # Tools (Handheld Fan etc.) provide zero Energy and do NOT belong here
                # — they can't pay a retreat cost or an attack cost.
                if cid in PSYCHIC_ENERGY_IDS: return 65.0
                # Enriching only buys retreat, never attack (see the ENRICHING block
                # below). If tgt is Alakazam and there's no ready bench Alakazam to
                # retreat INTO, unsticking it just swaps into an equally-unable-to-
                # attack Abra/Kadabra/support mon — no upside — while permanently
                # burning the turn's one attach on a card that can never power
                # Powerful Hand (confirmed replay 84136810: softlocked a 5-1-ahead
                # Alakazam attack-dead for the rest of a deck-out loss). Fall through
                # to the normal ENRICHING routing (Dudunsparce priority, Alakazam
                # vetoed) instead of the blanket rescue score.
                if tid==ALAKAZAM and cid==ENRICHING and not bench_has_alak_ready:
                    pass
                else:
                    return 55.0
            if cid==HANDHELD_FAN:
                if tid==GENESECT and not (tgt or{}).get('tools'): return 15.0
                return 1.5
            if cid==ENRICHING:
                # Enriching's "draw 4" fires unconditionally on attach (no may-use
                # prompt, unlike Kadabra/Alakazam's Psychic Draw) -- unlike every
                # other draw source in this file it had NO deck-safety gate at all.
                # Confirmed contributing to a real deck-out loss (replay 83156504):
                # attached at deck=5 with hand already at 18, dropping the deck to 1
                # in one action. Bigger single draw than Dawn/Hilda (2) or Poffin, so
                # gate on deck_critical (<10), not just deck_danger (<5).
                if deck_critical and not emergency_draw and not desperation and not lone_active_opportunity: return -6.0
                if tid==DUDUNSPARCE and cen['dudun_no_energy']: return 20.0
                if tid==DUDUNSPARCE:                             return 13.0
                # Enriching is Colorless — it can NEVER pay Powerful Hand's Psychic
                # cost. Attaching it to Alakazam (fueled or not) wastes the card:
                # fueled, it does nothing; unfueled, it still leaves Alakazam unable
                # to attack while consuming the energy-drop for the turn instead of
                # a real Psychic source. Never route it here (see CLAUDE.md).
                if tid==ALAKAZAM: return -8.0
                return 1.0
            if cid in PSYCHIC_ENERGY_IDS:
                # Powerful Hand costs exactly 1 Psychic — a 2nd energy on the SAME
                # physical card does nothing (damage scales with hand size, not
                # energy count), and unlike a bench pivot there's no cost this ever
                # pays down. Hard cap this at the target level, before the per-line
                # priority order below, so it can never lose to "nothing better
                # this turn" and get attached anyway — the card is strictly more
                # valuable left in hand as +20 future Powerful Hand damage.
                if _has_psychic(tgt): return -6.0
                # Route further Psychic to the next-best un-fueled pre-load target:
                # Alakazam (fuels the attacker itself) > Kadabra > Abra (so the
                # energy is already there the moment it evolves) > other bench
                # support (never preemptively fuel Dudunsparce/Genesect/Shaymin/
                # Psyduck/Fez — they don't attack, and the one legitimate case,
                # paying a real retreat cost into a waiting bench Alakazam, is
                # already handled above via the active_immobile rescue block).
                if tid==ALAKAZAM: return 16.0
                if tid==KADABRA:  return W['attach_kadabra']
                if tid==ABRA:     return W['attach_abra']
                return -2.0
            if active_non_atk:
                if tid==ALAKAZAM: return 11.0
                return 2.0
            if tid==ALAKAZAM: return 6.0
            return 3.0
        if ot==DISCARD: return 0.0
        if ot==END:
            # rescue_mode: END must stay strictly below the demoted non-lethal
            # ATTACK (1.2) in every phase branch -- ending without the free
            # chip damage was the 2.7%-win deviation class.
            if rescue_mode: return 0.8
            if phase==PHASE_CONVERT and hand_n>=8: return W['end_convert']
            if phase==PHASE_PRESSURE and at_threshold: return 3.0
            return 1.0
        return 2.0

    return score

def score_options_main(obs,sel):
    """Standalone, reusable per-option heuristic score vector for MAIN-phase
    (stype==0) decisions — the heuristic half of the MCTS prior blend."""
    opts=sel.get('option') or[]
    if not opts: return[]
    score=_main_phase_features(obs,sel)
    return[score(o) for o in opts]

def _main_phase(obs,sel):
    opts=sel.get('option') or[]
    if not opts: return[]
    s=score_options_main(obs,sel)
    return[max(range(len(opts)),key=lambda i:s[i])]

def _choose(obs):
    sel=obs.get('select')
    if sel is None: return DECK
    opts=sel.get('option',[]); n=len(opts)
    if n==0: return[]
    stype=sel.get('type'); ctx=sel.get('context',0)
    mn=sel.get('minCount',0) or 0; mx=sel.get('maxCount',1) or 1
    if stype==0: return _main_phase(obs,sel)
    if stype==1:
        if ctx==CTX_SETUP_ACTIVE: return _pick_setup_active(obs,opts)
        if ctx==CTX_SETUP_BENCH:  return _pick_setup_bench(opts)
        cur=obs.get('current') or{}; me=cur.get('yourIndex',0)
        # v22: deck searches expose sel['deck'] (they are NOT blind) — options index
        # into it. Route any deck-area select through need-based scoring.
        if sel.get('deck') and any(o.get('area')==1 for o in opts):
            return _deck_search_pick(obs,sel)
        is_boss_target=(len(opts)>0
            and all(o.get('playerIndex')==(1-me) and o.get('area')==5 for o in opts))
        if is_boss_target: return _pick_boss_target(obs,sel)
        if (sel.get('effect') or{}).get('id')==WONDROUS_PATCH and any(o.get('area')==5 for o in opts):
            return _clamp(_pick_wondrous_patch_target(obs,opts),sel)
        if any(o.get('area')==5 for o in opts):
            return _clamp(_pick_bench_target(obs,opts),sel)
        if any(o.get('area')==3 for o in opts):   # Night Stretcher: from discard
            return _clamp(_pick_discard_return(obs,opts),sel)
        yes_i=[i for i,o in enumerate(opts) if o.get('type')==YES]
        if yes_i: return _clamp(yes_i,sel)
        # Generic blind pick (this is where prize-card selection lands: same-shaped
        # option list, minCount==maxCount==KO'd Pokemon's prize value). If the engine
        # re-asks the identical question with no state change, rotate the pick instead
        # of resubmitting the same answer forever.
        return _clamp(_resolve_stalled_or(obs,sel,list(range(n))),sel)
    if stype==4: return _pick_energy_discard(obs,sel)
    if stype==7: return _clamp(_pick_evolve_target(obs,sel),sel)
    if stype in(2,3): return _clamp(list(range(n)),sel)
    if stype==5:           return _clamp(list(range(n))[:max(mn,1)],sel)
    if stype==6:
        _,A=_meta()
        return[max(range(n),key=lambda i:(A.get(opts[i].get('attackId'),(0,))[0] or 1))]
    if stype==8:
        return[max(range(n),key=lambda i:opts[i].get('number',0) or 0)]
    if stype==9:
        # "May use this Ability?" prompt (Psychic Draw, Run Away Draw, etc. when
        # asked this way rather than as a main-phase ABILITY option) -- every OTHER
        # draw source in this file (Dawn/Hilda/Poffin/Poke Pad/Dudunsparce-ability)
        # is gated on deck_danger, but this prompt always said yes unconditionally.
        # Confirmed root cause of a real ladder loss (replay 83348630): evolving a
        # bench Kadabra into Alakazam at deck=3 auto-triggered this prompt, and the
        # unconditional yes drew 3 more cards with a hand already at 17 (needing just
        # 7 for lethal) and a 5-2 prize lead -- decked out the same turn in a winning
        # game. Decline once the deck is already at the danger floor AND the hand is
        # already well past the KO threshold, since more cards can't help a banked
        # lethal and drawing is exactly how a winning position mills itself.
        cur=obs.get('current') or{}; me_idx=cur.get('yourIndex',0)
        players=cur.get('players',[])
        me=players[me_idx] if len(players)>me_idx else{}
        opp=players[1-me_idx] if len(players)==2 else{}
        if ctx==42:
            # MULLIGAN (ladder-only; local engine auto-resolves instead): cg-lib
            # api.py documents it as "Would you like to redraw the cards?". Both
            # ladder sightings (replays 83358041, 83455146) hit an opening hand
            # with zero Basic Pokemon. Redraw a dead hand (no Basic = cannot set
            # up), keep anything with a Basic -- the draw-prompt logic below
            # answers on deck count, not hand contents, and would say YES here.
            has_basic=any((c or{}).get('id') in BENCHABLE_BASIC_IDS
                          for c in me.get('hand') or[])
            want=NO if has_basic else YES
            pick=[i for i,o in enumerate(opts) if o.get('type')==want]
            if pick: return pick[:1]
            return[0]
        deck_count=me.get('deckCount',99) or 99
        hand_n=_hand_size(cur,me_idx)
        opp_hp=(_active(opp) or{}).get('hp',99999) or 99999
        cards_needed=math.ceil(opp_hp/PH_DMG_PER_CARD) if opp_hp<99999 else 999
        # Desperation (opponent one KO from winning, or two with an ex of ours in
        # play/bench) overrides the deck-out guard below: if we can't close it out
        # this turn, next turn is likely a loss anyway, so keep drawing toward the
        # biggest possible hand instead of preserving deck count. See the matching
        # `desperation` computation in _main_phase_features.
        opp_prizes=len(opp.get('prize') or[])
        bench=me.get('bench') or[]
        we_have_ex=any((p or{}).get('ex',False) for p in([_active(me)]+bench) if p)
        desperation=opp_prizes<=1 or(opp_prizes<=2 and we_have_ex)
        # Matching lone_active_opportunity from _main_phase_features: opponent has
        # nothing but their Active in play, so a big enough hand ends their turn
        # with an empty board -- worth drawing past the normal deck-out guard.
        opp_bench_empty=len([b for b in opp.get('bench') or[] if b])==0
        lone_active_opportunity=(
            opp_bench_empty and opp_hp<99999 and hand_n+3>=cards_needed)
        if deck_count<5 and hand_n>=cards_needed+3 and not desperation and not lone_active_opportunity:
            no_i=[i for i,o in enumerate(opts) if o.get('type')==NO]
            if no_i: return no_i
        for i,o in enumerate(opts):
            if o.get('type')==YES: return[i]
        return[0]
    if stype==10: return _clamp(list(range(n))[:max(mn,1)],sel)
    k=mn if mn>0 else(1 if mx>=1 else 0)
    return _clamp(list(range(n))[:k] if k else[],sel)

def score_options(obs,sel):
    """Per-option heuristic score vector for ANY select shape, mirroring
    _choose's dispatch. This is the heuristic half of the MCTS prior blend
    (docs/nn-training.md's heuristic-weighted-search plan) — pure, deterministic,
    side-effect-free (never touches _STALL_MEMO), safe to call repeatedly inside
    a search tree. Where no meaningful per-option ranking exists (bare YES/NO,
    forced picks, blind selections), returns a flat/uniform vector — softmax of
    a flat vector is a uniform prior, which is the honest answer there."""
    opts=sel.get('option') or[]
    n=len(opts)
    if n==0: return[]
    stype=sel.get('type'); ctx=sel.get('context',0)
    if stype==0: return score_options_main(obs,sel)
    if stype==1:
        if ctx==CTX_SETUP_ACTIVE: return _score_setup_active(obs,opts)
        if ctx==CTX_SETUP_BENCH:  return[0.0]*n
        cur=obs.get('current') or{}; me=cur.get('yourIndex',0)
        if sel.get('deck') and any(o.get('area')==1 for o in opts):
            return _score_deck_search(obs,sel)
        is_boss_target=(n>0 and all(o.get('playerIndex')==(1-me) and o.get('area')==5 for o in opts))
        if is_boss_target: return _score_boss_target(obs,sel)
        if (sel.get('effect') or{}).get('id')==WONDROUS_PATCH and any(o.get('area')==5 for o in opts):
            return _score_wondrous_patch_target(obs,opts)
        if any(o.get('area')==5 for o in opts):
            return _score_bench_target(obs,opts)
        return[0.0]*n
    if stype==4: return _score_energy_discard(obs,sel)
    if stype==7: return _score_evolve_target(obs,sel)
    return[0.0]*n

def _safe_return(result,sel):
    if not sel: return result
    n=len(sel.get('option',[])); mn=sel.get('minCount',0) or 0; mx=sel.get('maxCount',1) or 1
    if not isinstance(result,list): result=[0]
    result=[i for i in result if 0<=i<n][:mx]; i=0
    while len(result)<mn and i<n:
        if i not in result: result.append(i)
        i+=1
    return result if result else([0] if n>0 else[])

def agent(obs_dict: dict) -> list[int]:
    try:
        sel=obs_dict.get('select'); out=_choose(obs_dict)
        if isinstance(out,list): return _safe_return(out,sel) if sel else out
    except: pass
    try:
        sel=obs_dict.get('select')
        if not sel: return[]
        n=len(sel.get('option',[]))
        if n==0: return[]
        mn=sel.get('minCount',1) or 0
        return _safe_return(list(range(min(max(mn,1),n))),sel)
    except: return[0]