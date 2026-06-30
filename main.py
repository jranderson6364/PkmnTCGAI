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
ENHANCED_HAMMER,RARE_CANDY        = 1081,1079
BASIC_P,ENRICHING,TELEPATH_P      = 5,13,19

MIST_ENERGY = 11
ROCK_ENERGY = 20
PSYCHIC_TYPE = 5

ALAKAZAM_LINE_IDS      = {ABRA,KADABRA,ALAKAZAM}
DRAW_ABILITY_CARD_IDS  = {KADABRA,ALAKAZAM,DUDUNSPARCE}
SUPPRESS_ABILITY_IDS   = {109}
PSYCHIC_ENERGY_IDS     = {BASIC_P,TELEPATH_P}
BENCHABLE_BASIC_IDS    = {ABRA,DUNSPARCE,DUNSPARCE2,PSYDUCK,SHAYMIN,GENESECT,FEZ}
LINE_SEARCH_HAND_IDS   = {DAWN,HILDA}
DRAW_ENGINE_IDS        = {DUNSPARCE,DUNSPARCE2,DUDUNSPARCE}
NON_ATTACKER_IDS       = {DUNSPARCE,DUNSPARCE2,DUDUNSPARCE,GENESECT,SHAYMIN,PSYDUCK,FEZ,ABRA}
PIVOT_FREE_RETREAT_IDS = {SHAYMIN}
TOOL_IDS               = {HANDHELD_FAN}
PH_DMG_PER_CARD = 20

DECK = ([ABRA]*4+[KADABRA]*4+[ALAKAZAM]*3+[DUNSPARCE]*3+[DUDUNSPARCE]*3+
        [GENESECT]+[SHAYMIN]+[PSYDUCK]+[FEZ]+
        [POFFIN]*4+[POKE_PAD]*4+[HANDHELD_FAN]*2+[BOSS]*3+[LANA]+
        [BATTLE_CAGE]*4+[DAWN]*4+[WONDROUS_PATCH]+[SACRED_ASH]+[HILDA]*3+
        [ENHANCED_HAMMER]*2+[RARE_CANDY]*3+[BASIC_P]*2+[ENRICHING]+[TELEPATH_P]*4)

_CARDS = None; _ATKMAP = None
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
    opp_likely_ace_spec = True
    return opp_played, bench_damage_received, we_were_kod_last_turn, opp_likely_ace_spec

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

def _opp_has_blocking_energy(opp_active):
    if not opp_active: return False
    for ec in (opp_active.get('energyCards') or []):
        if ec.get('id') in (MIST_ENERGY, ROCK_ENERGY): return True
    return False

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
    not_established=(
        not cen['has_alakazam'] or not cen['backup_abra'] or
        cen['draw_count']==0 or not cen['has_energy_plan'])
    if not_established: return PHASE_ESTABLISH
    if can_ko or at_threshold: return PHASE_PRESSURE
    return PHASE_CONVERT

def _pick_setup_active(opts):
    PREF={ABRA:90,DUNSPARCE:100,DUNSPARCE2:100,PSYDUCK:40,SHAYMIN:30,GENESECT:25,FEZ:5}
    best_i,best_s=0,-1
    for i,o in enumerate(opts):
        cid=o.get('cardId') or o.get('id',-1)
        s=PREF.get(cid,20)
        if s>best_s: best_s,best_i=s,i
    return[best_i]

def _pick_setup_bench(opts): return list(range(len(opts)))

def _pick_bench_target(obs,opts):
    cur=obs.get('current') or{}; me_idx=cur.get('yourIndex',0)
    players=cur.get('players',[]); me=players[me_idx] if players and len(players)>me_idx else{}
    bench=me.get('bench') or[]
    area5=[(i,o) for i,o in enumerate(opts) if o.get('area')==5]
    best_i,best_score=(area5[0][0] if area5 else 0),-1
    for order,(i,o) in enumerate(area5):
        idx=o.get('index',order)
        pk=bench[idx] if 0<=idx<len(bench) else(bench[order] if order<len(bench) else None)
        pid=_pk_id(pk)
        if pid==ALAKAZAM and _has_psychic(pk): s=100
        elif pid==ALAKAZAM: s=80
        elif pid==DUDUNSPARCE: s=50
        elif pid in PIVOT_FREE_RETREAT_IDS: s=40
        else: s=-10
        if s>best_score: best_score,best_i=s,i
    return[best_i]

def _pick_boss_target(obs,sel):
    cur=obs.get('current') or{}; me=cur.get('yourIndex',0)
    players=cur.get('players',[]); opp_idx=1-me
    opp=players[opp_idx] if len(players)>opp_idx else{}
    opp_bench=opp.get('bench') or[]
    hand_n=_hand_size(cur,me); boss_dmg=(hand_n-1)*PH_DMG_PER_CARD
    opts=sel.get('option',[]); ko_targets=[]; dmg_targets=[]
    for i,o in enumerate(opts):
        bi=o.get('index',0)
        pk=opp_bench[bi] if 0<=bi<len(opp_bench) else None
        if not pk: continue
        pk_hp=(pk.get('hp',99999) or 99999); pv=_prize_value_pk(pk)
        ec=len((pk or{}).get('energies') or [])
        pid=_pk_id(pk)
        if boss_dmg>=pk_hp:
            ko_targets.append((i,pv,pk_hp,ec))
        else:
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
    return[0]

def _main_phase(obs,sel):
    opts=sel['option']; n=len(opts)
    if n==0: return[]
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
    opp_played, bench_dmg_received, we_were_kod, opp_likely_ace = _analyze_logs(obs, me_idx)
    opp_mist=_opp_has_blocking_energy(opp_active)
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
    opp_active_pv=_prize_value_pk(opp_active)
    boss_in_hand=any(_pk_id(c)==BOSS for c in hand)
    tool_in_hand=any(_pk_id(c) in TOOL_IDS for c in hand)
    boss_dmg=(hand_n-1)*PH_DMG_PER_CARD
    boss_target_exists=(
        active_can_attack and not opp_mist and opp_hp>my_dmg and
        any(0<(b or{}).get('hp',99999)<=boss_dmg and
            _prize_value_pk(b)>=opp_active_pv
            for b in opp_bench if b))
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
    retreat_available=any(o.get('type')==RETREAT for o in opts)
    active_immobile=(
        my_active is not None and not attack_available and not retreat_available and
        len(_energies(my_active))==0)
    # Threshold discipline (§4/§10 piloting-guide): once a ready attacker exists and the
    # hand is already at the KO threshold, more draw is pure deck-out risk -> stop drawing.
    ready_attacker_exists=active_can_attack or bench_has_alak_ready
    hand_surplus=(
        ready_attacker_exists and opp_hp<99999 and hand_n>=cards_needed and
        not boss_snipe_plan and not emergency_draw)

    def score(o):
        ot=o.get('type'); cid=_opt_card_id(o,hand,my_active,bench)
        if ot==ATTACK:
            if not active_can_attack: return-5
            if opp_mist: return-5
            if can_ko: return 500
            if at_threshold: return 150
            if hand_too_small: return 0.5
            return 7.0
        if ot==RETREAT:
            if retreated: return-50
            if alak_stuck:
                if bench_has_alak_ready: return 22.0
                return -5.0
            if active_non_atk and bench_has_alak_ready:
                return 30.0 if active_below_half else 22.0
            if active_non_atk and bench_has_alak:
                return 20.0 if active_below_half else 16.0
            if active_non_atk:                     return -3.0
            if opp_mist and active_is_alak:        return 9.0
            if active_can_attack:                  return-2.0
            return 0.5
        if ot==ABILITY:
            if lone: return-10
            if cid in SUPPRESS_ABILITY_IDS: return-10
            if cid in DRAW_ABILITY_CARD_IDS:
                if can_ko: return 2.0
                if cid==DUDUNSPARCE:
                    if hand_surplus: return 0.5
                    if deck_danger and not emergency_draw:   return-8.0
                    if deck_critical and not emergency_draw: return-2.0
                    if hand_n>=14 and not emergency_draw:    return 1.0
                    if cen['dudun_bench']>1 and not emergency_draw: return 6.0
                    return 11.0
                if emergency_draw: return 15.0
                return 10.0
            if cid==FEZ:
                if deck_danger: return-5.0
                if hand_surplus: return-3.0
                if deck_critical and not emergency_draw: return-2.0
                return 8.0
            return 5.0
        if ot==EVOLVE:
            evo_area=o.get('inPlayArea',4)
            if cid==ALAKAZAM:
                if evo_area==5:
                    if can_ko: return 3.0
                    if not cen['has_alakazam']: return 50.0
                    if phase==PHASE_ESTABLISH: return 40.0
                    if phase==PHASE_CONVERT: return 25.0
                    return 12.0
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
                if can_ko: return 3.0
                return 13.0
            if can_ko: return 2.0
            if active_non_atk: return 12.0
            return 8.5
        if ot==PLAY:
            if cid==ENHANCED_HAMMER:
                if opp_mist: return 45.0
                return 3.0
            if cid==BATTLE_CAGE:
                if bench_dmg_received: return 22.0
                if in_late_phase and hand_n>=8: return 1.0
                return 6.0
            if cid==BOSS:
                if boss_ex_snipe:        return 600.0
                if can_ko: return 1.0
                if phase==PHASE_CLOSING: return 199.0
                if boss_target_exists:   return 16.0
                if boss_can_damage_mega: return 18.0
                if opp_mist and ready_alak_exists: return 12.0
                return 4.0
            if can_ko: return 1.0
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
                    opp_ability_threat = bool(opp_played & {109, 110, 111})
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
                if cen['kadabra_can_evolve']:
                    if at_threshold or phase in(PHASE_PRESSURE,PHASE_CLOSING): return 30.0
                    if phase==PHASE_ESTABLISH: return 25.0
                    return 8.0
                if not cen['has_alakazam']: return 28.0
                if cen['need_line']:        return 14.0
                return 4.0
            if cid==POFFIN:
                if phase==PHASE_ESTABLISH and (not cen['backup_abra'] or cen['draw_count']==0):
                    if cen['bench_count']<5: return 25.0
                if not cen['backup_abra'] or cen['draw_count']==0:
                    if cen['bench_count']<4: return 19.0
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
            if cid==DAWN:
                if supporter_played: return-5.0
                if deck_danger: return-8.0
                if active_immobile: return-3.0
                if hand_surplus: return 2.0
                if phase==PHASE_ESTABLISH and (cen['need_line'] or not cen['has_alakazam']): return 22.0
                if boss_snipe_plan and not emergency_draw: return 1.0
                if deck_critical: return 1.0
                if hand_n>=12: return 2.0
                if emergency_draw: return 14.0
                if cen['need_line'] or not cen['has_alakazam']: return 11.0
                if phase==PHASE_CONVERT: return 8.0
                return 6.0
            if cid==HILDA:
                if supporter_played: return-5.0
                if deck_danger: return-8.0
                if active_immobile: return 18.0
                if hand_surplus: return 2.0
                if phase==PHASE_ESTABLISH and (cen['need_line'] or not cen['has_alakazam']): return 24.0
                if boss_snipe_plan and not emergency_draw: return 1.0
                if deck_critical: return 1.0
                if not enriching_on_dudun and cen['draw_count']>0: return 11.0
                if not cen['has_alakazam']: return 13.0
                if emergency_draw: return 12.0
                if cen['need_line']: return 10.0
                if phase==PHASE_CONVERT: return 7.0
                return 5.0
            if cid==POKE_PAD:
                if deck_danger: return-8.0
                if active_immobile: return-3.0
                if hand_surplus: return 2.0
                if not cen['backup_abra']: return 13.0
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
            if can_ko: return 0.5
            tgt=_attach_target(o,my_active,bench); tid=_pk_id(tgt)
            if active_immobile and tgt is my_active:
                # Free the stranded Active. Prefer real Psychic so a stuck Alakazam can
                # both retreat AND attack; any energy still beats leaving it locked.
                if cid in PSYCHIC_ENERGY_IDS: return 65.0
                return 55.0
            if cid==HANDHELD_FAN:
                if tid==GENESECT and not (tgt or{}).get('tools'): return 15.0
                return 1.5
            if cid==ENRICHING:
                if tid==DUDUNSPARCE and cen['dudun_no_energy']: return 20.0
                if tid==DUDUNSPARCE:                             return 13.0
                if tid==ALAKAZAM and not _has_psychic(tgt):     return 1.0
                return 6.0
            if cid in PSYCHIC_ENERGY_IDS:
                if tid==ALAKAZAM and not _has_psychic(tgt): return 16.0
                if tid==ALAKAZAM:                            return 8.0
                if tid==KADABRA:                             return 7.0
                return 3.0
            if active_non_atk:
                if tid==ALAKAZAM: return 11.0
                return 2.0
            if tid==ALAKAZAM: return 6.0
            return 3.0
        if ot==DISCARD: return 0.0
        if ot==END:
            if phase==PHASE_CONVERT and hand_n>=8: return 4.0
            if phase==PHASE_PRESSURE and at_threshold: return 3.0
            return 1.0
        return 2.0

    best_i,best_s=0,-1e18
    for i,o in enumerate(opts):
        s=score(o)
        if s>best_s: best_s,best_i=s,i
    return[best_i]

def _choose(obs):
    sel=obs.get('select')
    if sel is None: return DECK
    opts=sel.get('option',[]); n=len(opts)
    if n==0: return[]
    stype=sel.get('type'); ctx=sel.get('context',0)
    mn=sel.get('minCount',0) or 0; mx=sel.get('maxCount',1) or 1
    if stype==0: return _main_phase(obs,sel)
    if stype==1:
        if ctx==CTX_SETUP_ACTIVE: return _pick_setup_active(opts)
        if ctx==CTX_SETUP_BENCH:  return _pick_setup_bench(opts)
        cur=obs.get('current') or{}; me=cur.get('yourIndex',0)
        is_boss_target=(len(opts)>0
            and all(o.get('playerIndex')==(1-me) and o.get('area')==5 for o in opts))
        if is_boss_target: return _pick_boss_target(obs,sel)
        if any(o.get('area')==5 for o in opts):
            return _clamp(_pick_bench_target(obs,opts),sel)
        yes_i=[i for i,o in enumerate(opts) if o.get('type')==YES]
        return _clamp(yes_i if yes_i else list(range(n)),sel)
    if stype in(2,3,4,7): return _clamp(list(range(n)),sel)
    if stype==5:           return _clamp(list(range(n))[:max(mn,1)],sel)
    if stype==6:
        _,A=_meta()
        return[max(range(n),key=lambda i:(A.get(opts[i].get('attackId'),(0,))[0] or 1))]
    if stype==8:
        return[max(range(n),key=lambda i:opts[i].get('number',0) or 0)]
    if stype==9:
        for i,o in enumerate(opts):
            if o.get('type')==YES: return[i]
        return[0]
    if stype==10: return _clamp(list(range(n))[:max(mn,1)],sel)
    k=mn if mn>0 else(1 if mx>=1 else 0)
    return _clamp(list(range(n))[:k] if k else[],sel)

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