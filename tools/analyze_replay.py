#!/usr/bin/env python3
"""
Replay analyzer for PTCG AI Battle Challenge kaggle-env JSON logs.

KEY INSIGHT (reverse-engineered): steps[i]['action'] is the action that was
CHOSEN using steps[i-1]'s observation.select.option list. steps[i]['observation']
is the RESULTING state after that action was applied. So to know "what did the
agent pick at decision point i-1", read steps[i]['action'] and index into
steps[i-1]'s option list.

Usage: python3 analyze_replay.py <replay.json> [--out out.txt]
"""
import json, sys, csv, math

ABRA,KADABRA,ALAKAZAM = 741,742,743
DUNSPARCE,DUNSPARCE2,DUDUNSPARCE = 305,65,66
GENESECT,SHAYMIN,PSYDUCK,FEZ = 142,343,858,140
POFFIN,POKE_PAD,HANDHELD_FAN = 1086,1152,1161
BOSS,LANA,BATTLE_CAGE,DAWN = 1182,1184,1264,1231
WONDROUS_PATCH,SACRED_ASH,HILDA = 1146,1129,1225
ENHANCED_HAMMER,RARE_CANDY = 1081,1079
BASIC_P,ENRICHING,TELEPATH_P = 5,13,19
MIST_ENERGY,ROCK_ENERGY = 11,20
NON_ATTACKER_IDS = {DUNSPARCE,DUNSPARCE2,DUDUNSPARCE,GENESECT,SHAYMIN,PSYDUCK,FEZ,ABRA}
ATTACKER_IDS = {ALAKAZAM}
PH_DMG = 20

NUMBER,YES,NO,CARD,TOOL_CARD,ENERGY_CARD,ENERGY,PLAY,ATTACH,EVOLVE,\
    ABILITY,DISCARD,RETREAT,ATTACK,END,SKILL,SPECIAL_CONDITION = range(17)
OT_NAMES = {0:'NUMBER',1:'YES',2:'NO',3:'CARD',4:'TOOL_CARD',5:'ENERGY_CARD',6:'ENERGY',
            7:'PLAY',8:'ATTACH',9:'EVOLVE',10:'ABILITY',11:'DISCARD',12:'RETREAT',
            13:'ATTACK',14:'END',15:'SKILL',16:'SPECIAL_CONDITION'}

def load_card_names(csv_path):
    names = {}
    try:
        with open(csv_path, encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    cid = int(row['Card ID'])
                except (ValueError, KeyError):
                    continue
                if cid not in names:
                    names[cid] = row.get('Card Name', '?')
    except FileNotFoundError:
        pass
    return names

def cname(names, cid):
    if cid is None: return 'None'
    return f"{names.get(cid, '?')}({cid})"

def pk_id(pk): return (pk or {}).get('id', -1)
def prize_val(pk):
    if not pk: return 1
    if pk.get('megaEx'): return 3
    if pk.get('ex'): return 2
    return 1

def get_players(obs):
    cur = obs.get('current') or {}
    yidx = cur.get('yourIndex')
    pl = cur.get('players') or []
    return cur, yidx, pl

def fmt_pk(names, pk):
    if not pk: return 'NONE'
    cid = pk_id(pk)
    hp = pk.get('hp', '?')
    maxhp = pk.get('maxHp', '?')
    tag = ''
    if pk.get('megaEx'): tag = '[megaEx]'
    elif pk.get('ex'): tag = '[ex]'
    return f"{names.get(cid,'?')}({cid}) {hp}/{maxhp}{tag}"

def resolve_chosen(names, prev_obs, action):
    """Given the observation the agent SAW and the action it picked, describe it."""
    sel = prev_obs.get('select')
    if not sel: return None
    opts = sel.get('option', [])
    if not action or action[0] >= len(opts) or action[0] < 0:
        return {'desc': 'EMPTY/OOB ACTION', 'stype': sel.get('type'), 'raw': action}
    o = opts[action[0]]
    stype = sel.get('type')
    cur, yidx, pl = get_players(prev_obs)
    me = pl[yidx] if yidx is not None and len(pl) > yidx else {}
    hand = me.get('hand') or []
    bench = me.get('bench') or []
    active = (me.get('active') or [None])[0]
    ot = o.get('type')
    info = {'stype': stype, 'otype': ot, 'otype_name': OT_NAMES.get(ot, str(ot)), 'raw_opt': o}
    if ot in (PLAY, ATTACH, EVOLVE):
        idx = o.get('index')
        if idx is not None and 0 <= idx < len(hand):
            info['card'] = cname(names, pk_id(hand[idx]))
        ipa = o.get('inPlayArea')
        if ipa == 4:
            info['target'] = f"ACTIVE:{fmt_pk(names, active)}"
        elif ipa == 5:
            ipi = o.get('inPlayIndex', 0)
            tgt = bench[ipi] if 0 <= ipi < len(bench) else None
            info['target'] = f"BENCH[{ipi}]:{fmt_pk(names, tgt)}"
    elif ot == ATTACK:
        info['attackId'] = o.get('attackId')
    elif ot == RETREAT:
        info['desc'] = 'RETREAT (target chosen in next select)'
    elif ot == ABILITY:
        area = o.get('area'); idx = o.get('index', 0)
        tgt = active if area == 4 else (bench[idx] if 0 <= idx < len(bench) else None)
        info['target'] = f"{'ACTIVE' if area==4 else 'BENCH['+str(idx)+']'}:{fmt_pk(names, tgt)}"
    elif ot == END:
        info['desc'] = 'END TURN'
    return info

PSYCHIC_CARD_IDS = {BASIC_P, TELEPATH_P}
PIVOT_FREE_RETREAT_IDS = {SHAYMIN}

def check_bad_energy_attach(names, prev_obs, sel, action):
    """Flags Psychic energy attached to a non-attacker with no legitimate reason.
    Legitimate = Alakazam (attacker) / Kadabra / Abra (pre-load before evolving),
    or the Active paying a real (non-zero, non-free) retreat cost into a bench
    Pokemon that's actually ready to attack. Everything else is 'preemptive
    giving' with zero payoff — the energy sits on a Pokemon that will never use it."""
    opts = sel.get('option', [])
    if not action or action[0] >= len(opts) or action[0] < 0:
        return None
    o = opts[action[0]]
    if o.get('type') != ATTACH:
        return None
    _, yidx, pl = get_players(prev_obs)
    if yidx is None or len(pl) <= yidx:
        return None
    me = pl[yidx]
    hand = me.get('hand') or []
    idx = o.get('index')
    cid = pk_id(hand[idx]) if idx is not None and 0 <= idx < len(hand) else None
    if cid not in PSYCHIC_CARD_IDS:
        return None
    ipa = o.get('inPlayArea'); ipi = o.get('inPlayIndex', 0)
    bench = me.get('bench') or []
    active = (me.get('active') or [None])[0]
    tgt = active if ipa == 4 else (bench[ipi] if ipa == 5 and 0 <= ipi < len(bench) else None)
    tid = pk_id(tgt)
    if tid in (ALAKAZAM, KADABRA, ABRA):
        tgt_has_psychic = 5 in (tgt.get('energies') or [])
        if tgt_has_psychic:
            # Powerful Hand only ever needs 1 Psychic on the card that attacks with
            # it — a 2nd copy on the SAME physical Pokemon is pure waste (hand size,
            # not energy count, is the damage stat).
            return f"-> {cname(names, tid)} ALREADY has Psychic — 2nd copy is pure waste (Powerful Hand only needs 1)"
        return None  # legitimate: the attacker itself, or pre-loading the evolution line
    if tid in PIVOT_FREE_RETREAT_IDS:
        return f"-> {cname(names, tid)} (FREE retreat — energy can never help it retreat, and it never attacks)"
    if ipa == 4:
        bench_has_ready_alak = any(
            pk_id(b) == ALAKAZAM and 5 in (b.get('energies') or []) for b in bench if b)
        if bench_has_ready_alak:
            return None  # legitimate: paying a real retreat cost into a waiting attacker
        return f"-> Active {cname(names, tid)} (no ready bench attacker to retreat into — energy fixes nothing this turn)"
    return f"-> bench support {cname(names, tid)} (no attack, no imminent retreat — pure waste)"

def can_lethal(prev_obs):
    """Replicate v15's can_ko logic approximately: active is Alakazam w/ psychic, attack available, dmg>=opp hp, no mist."""
    cur, yidx, pl = get_players(prev_obs)
    if yidx is None or len(pl) < 2: return None
    me = pl[yidx]; opp = pl[1-yidx]
    my_active = (me.get('active') or [None])[0]
    opp_active = (opp.get('active') or [None])[0]
    if not my_active or not opp_active: return None
    hand_n = me.get('handCount')
    if hand_n is None: hand_n = len(me.get('hand') or [])
    opp_mist = any(ec.get('id') in (MIST_ENERGY, ROCK_ENERGY) for ec in (opp_active.get('energyCards') or []))
    is_alak = pk_id(my_active) == ALAKAZAM
    has_psychic = 5 in (my_active.get('energies') or [])
    sel = prev_obs.get('select')
    attack_avail = sel and sel.get('type') == 0 and any(o.get('type') == ATTACK for o in sel.get('option', []))
    dmg = PH_DMG * hand_n if (is_alak and has_psychic and attack_avail and not opp_mist) else 0
    opp_hp = opp_active.get('hp', 99999) or 99999
    return {
        'is_alak': is_alak, 'has_psychic': has_psychic, 'attack_avail': attack_avail,
        'opp_mist': opp_mist, 'dmg': dmg, 'opp_hp': opp_hp,
        'lethal': bool(attack_avail and is_alak and has_psychic and not opp_mist and dmg >= opp_hp),
    }

def analyze(path, names):
    with open(path) as f:
        d = json.load(f)
    info = d.get('info', {})
    team_names = info.get('TeamNames', [])
    you_idx = team_names.index('Jason Anderson') if 'Jason Anderson' in team_names else 1
    opp_name = team_names[1-you_idx] if len(team_names) > 1 else '?'
    rewards = d.get('rewards', [None, None])
    your_result = rewards[you_idx]
    result_str = {1: 'WIN', -1: 'LOSS', 0: 'DRAW'}.get(your_result, str(your_result))

    lines = []
    lines.append(f"=== {path.split('/')[-1]} ===")
    lines.append(f"Opponent: {opp_name} | Result: {result_str} | Total steps: {len(d['steps'])}")
    lines.append("")

    steps = d['steps']
    n = len(steps)
    timeouts = []
    missed_lethals = []
    bad_retreats = []
    boss_plays = []
    bad_energy_attaches = []
    flagged_lines = []

    last_main_phase_idx = None  # index into steps where you last had a stype=0 decision
    pending_retreat_step = None

    for i in range(1, n):
        prev_rec = steps[i-1][you_idx]
        cur_rec = steps[i][you_idx]
        prev_obs = prev_rec.get('observation', {})
        action = cur_rec.get('action')
        sel = prev_obs.get('select')
        if sel is None:
            continue
        if cur_rec.get('observation', {}).get('current') is None and i < n - 1:
            # not your turn / no new info
            pass

        opts = sel.get('option', [])
        stype = sel.get('type')

        # detect empty/OOB action when there WAS a real decision to make
        if (not action) and len(opts) > 0:
            cur_state, yidx2, pl2 = get_players(prev_obs)
            timeouts.append((i, stype, len(opts)))
            continue
        if action and (action[0] >= len(opts) or action[0] < 0):
            timeouts.append((i, stype, len(opts), 'OOB:'+str(action)))
            continue

        if stype != 0:
            continue  # focus on main-phase decisions for flow; resolved below for retreat/boss target

        chosen = resolve_chosen(names, prev_obs, action)
        if chosen is None:
            continue

        cur_p, yidx, pl = get_players(prev_obs)
        me = pl[yidx] if yidx is not None and len(pl) > yidx else {}
        opp = pl[1-yidx] if yidx is not None and len(pl) == 2 else {}
        my_active = (me.get('active') or [None])[0]
        opp_active = (opp.get('active') or [None])[0]
        my_prizes = len(me.get('prize') or [])
        opp_prizes = len(opp.get('prize') or [])
        hand_n = me.get('handCount', len(me.get('hand') or []))

        # missed lethal check
        lethal_info = can_lethal(prev_obs)
        ot = chosen.get('otype')
        if lethal_info and lethal_info['lethal'] and ot != ATTACK:
            missed_lethals.append((i, lethal_info, chosen))

        # preemptive/wasted energy attach check
        bad_energy = check_bad_energy_attach(names, prev_obs, sel, action)
        if bad_energy:
            bad_energy_attaches.append((i, bad_energy))

        # Boss play detection -> look ahead for target selection
        if ot == PLAY and chosen.get('card', '').startswith('Boss'):
            # find next stype==1 selection targeting opponent bench
            target_desc = None
            for j in range(i, min(i+6, n)):
                rec_j = steps[j][you_idx]
                obs_j = rec_j.get('observation', {})
                sel_j = obs_j.get('select')
                if sel_j and sel_j.get('type') == 1:
                    optsj = sel_j.get('option', [])
                    if optsj and all(o.get('playerIndex') == (1-you_idx) and o.get('area') == 5 for o in optsj):
                        # the action that resolves THIS selection is in steps[j+1]
                        if j+1 < n:
                            act_j = steps[j+1][you_idx].get('action')
                            curj, yidxj, plj = get_players(obs_j)
                            oppj = plj[1-yidxj] if yidxj is not None and len(plj)==2 else {}
                            opp_bench = oppj.get('bench') or []
                            if act_j and 0 <= act_j[0] < len(optsj):
                                bi = optsj[act_j[0]].get('index', 0)
                                tgt = opp_bench[bi] if 0 <= bi < len(opp_bench) else None
                                target_desc = fmt_pk(names, tgt)
                        break
            boss_plays.append((i, my_prizes, opp_prizes, fmt_pk(names, opp_active), target_desc))

        # Retreat -> look ahead for bench target selection
        if ot == RETREAT:
            bench = me.get('bench') or []
            target_desc = None
            for j in range(i, min(i+6, n)):
                rec_j = steps[j][you_idx]
                obs_j = rec_j.get('observation', {})
                sel_j = obs_j.get('select')
                if sel_j and sel_j.get('type') == 1:
                    optsj = sel_j.get('option', [])
                    if optsj and any(o.get('area') == 5 for o in optsj):
                        if j+1 < n:
                            act_j = steps[j+1][you_idx].get('action')
                            if act_j and 0 <= act_j[0] < len(optsj):
                                oj = optsj[act_j[0]]
                                bi = oj.get('index', 0)
                                tgt = bench[bi] if 0 <= bi < len(bench) else None
                                target_desc = fmt_pk(names, tgt)
                                tgt_id = pk_id(tgt)
                                if tgt_id in NON_ATTACKER_IDS:
                                    # was there a better option (Alakazam w/ psychic or Dudunsparce) on bench?
                                    better = [b for b in bench if pk_id(b) == ALAKAZAM and 5 in (b.get('energies') or [])]
                                    if better:
                                        bad_retreats.append((i, target_desc, [fmt_pk(names,b) for b in better]))
                        break
                    elif sel_j.get('type') != 1:
                        break
            flagged_lines.append((i, 'RETREAT', target_desc, my_prizes, opp_prizes))

        # log every main-phase action compactly
        desc_extra = chosen.get('card') or chosen.get('desc') or chosen.get('attackId') or ''
        target = chosen.get('target', '')
        flagged_lines.append((i, chosen['otype_name'], f"{desc_extra} {target}".strip(),
                               my_prizes, opp_prizes))

    lines.append(f"--- Decision log ({len(flagged_lines)} main-phase actions) ---")
    for entry in flagged_lines:
        i, ot_name, desc, mp, op = entry[:5]
        lines.append(f"  step{i:3d} | you_prizes={mp} opp_prizes={op} | {ot_name}: {desc}")

    lines.append("")
    lines.append(f"--- FLAGS ---")
    lines.append(f"Timeouts/empty-or-OOB actions: {len(timeouts)}")
    for t in timeouts[:20]:
        lines.append(f"  step{t[0]}: stype={t[1]} n_opts={t[2]} {t[3] if len(t)>3 else ''}")

    lines.append(f"Missed lethal attacks: {len(missed_lethals)}")
    for i, li, chosen in missed_lethals[:20]:
        lines.append(f"  step{i}: could KO (dmg={li['dmg']} vs opp_hp={li['opp_hp']}) but chose {chosen['otype_name']} {chosen.get('card','')}")

    lines.append(f"Bad retreats (promoted non-attacker while ready Alakazam on bench): {len(bad_retreats)}")
    for i, tgt, better in bad_retreats[:20]:
        lines.append(f"  step{i}: promoted {tgt}, but had ready: {better}")

    lines.append(f"Wasted/preemptive Psychic energy attaches: {len(bad_energy_attaches)}")
    for i, desc in bad_energy_attaches[:20]:
        lines.append(f"  step{i}: {desc}")

    lines.append(f"Boss's Orders plays: {len(boss_plays)}")
    for i, mp, op, opp_act, tgt in boss_plays:
        lines.append(f"  step{i}: prizes you={mp} opp={op} | opp_active_was={opp_act} | gusted_target={tgt}")

    lines.append("")
    lines.append(f"Final state check (last few steps):")
    for i in range(max(1, n-6), n):
        rec = steps[i][you_idx]
        obs = rec.get('observation', {})
        cur, yidx, pl = get_players(obs)
        if yidx is None or len(pl) < 2:
            continue
        me = pl[yidx]; opp = pl[1-yidx]
        my_active = (me.get('active') or [None])[0]
        opp_active = (opp.get('active') or [None])[0]
        lines.append(f"  step{i}: you={fmt_pk(names,my_active)} prizes={len(me.get('prize') or [])} deck={me.get('deckCount')} "
                      f"| opp={fmt_pk(names,opp_active)} prizes={len(opp.get('prize') or [])} deck={opp.get('deckCount')}")

    return '\n'.join(lines)

if __name__ == '__main__':
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    names = load_card_names(os.path.join(repo_root, 'docs', 'EN_Card_Data.csv'))
    paths = sys.argv[1:]
    if not paths:
        print(f"Usage: python3 {sys.argv[0]} <replay.json> [more.json ...]")
        print("Writes <name>_summary.txt next to each input file.")
        sys.exit(1)
    for p in paths:
        try:
            out = analyze(p, names)
        except Exception as e:
            import traceback
            out = f"=== {p} ===\nERROR: {e}\n{traceback.format_exc()}"
        outpath = os.path.join(os.path.dirname(os.path.abspath(p)),
                                os.path.basename(p).replace('.json', '_summary.txt'))
        with open(outpath, 'w') as f:
            f.write(out)
        print(f"Wrote {outpath} ({len(out)} chars)")
