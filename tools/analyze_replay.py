#!/usr/bin/env python3
"""
Replay analyzer for PTCG AI Battle Challenge kaggle-env JSON logs.

KEY INSIGHT (reverse-engineered, verified against raw JSON): each per-step record
has a `status` field (ACTIVE/INACTIVE/DONE). The `select` object in a step's
observation ECHOES into the opponent's INACTIVE steps (where we correctly have
nothing to answer) as well as our own ACTIVE step. A real decision point is
`steps[i-1][you]['status'] == 'ACTIVE'` with a `select` present; the action that
answers it is read from `steps[i][you]['action']` (which may itself land on an
INACTIVE or DONE record — the action is NOT gated on step i's status, only the
select at step i-1 is). Treating every step with a stale echoed `select` as a
fresh decision (the previous version of this tool) fabricates dozens of phantom
"timeouts" per game and corrupts the decision log.

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
    energies = pk.get('energies') or []
    etag = f" E={energies}" if energies else ""
    return f"{names.get(cid,'?')}({cid}) {hp}/{maxhp}{tag}{etag}"

def cards_needed_for(hp):
    if hp is None or hp >= 99999: return None
    return math.ceil(hp / PH_DMG)

def resolve_chosen(names, prev_obs, action):
    """Given the observation the agent SAW (at a real ACTIVE decision point) and
    the action it picked, describe it."""
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

def real_decisions(you_idx, steps):
    """Yield (i, prev_obs, action) for every genuine decision point: prev record
    (steps[i-1][you_idx]) has status ACTIVE and a select; action is read from
    steps[i][you_idx] regardless of that record's status."""
    n = len(steps)
    out = []
    for i in range(1, n):
        prev_rec = steps[i-1][you_idx]
        if prev_rec.get('status') != 'ACTIVE':
            continue
        prev_obs = prev_rec.get('observation') or {}
        sel = prev_obs.get('select')
        if not sel:
            continue
        action = steps[i][you_idx].get('action')
        out.append((i, prev_obs, sel, action))
    return out

def classify_terminal(d, you_idx):
    """Terminal-cause triage: PRIZED_OUT / DECK_OUT / NO_POKEMON_IN_PLAY /
    EMPTY_OR_ILLEGAL_RETURN / OTHER."""
    steps = d['steps']
    n = len(steps)
    rewards = d.get('rewards', [None, None])
    your_result = rewards[you_idx]
    if your_result != -1:
        return None  # only classify losses
    # last real decision point: did we return a legal, non-empty action?
    decs = real_decisions(you_idx, steps)
    if decs:
        i, prev_obs, sel, action = decs[-1]
        opts = sel.get('option', [])
        illegal = (not action) or action[0] < 0 or action[0] >= len(opts)
        if illegal and len(opts) > 0:
            return 'EMPTY_OR_ILLEGAL_RETURN'
    # true last board state (reflects the actual end-of-game board)
    last_full = None
    for i in range(n-1, -1, -1):
        obs = steps[i][you_idx].get('observation') or {}
        cur, yidx, pl = get_players(obs)
        if yidx is not None and len(pl) == 2:
            last_full = (cur, yidx, pl)
            break
    if last_full:
        cur, yidx, pl = last_full
        me = pl[yidx]; opp = pl[1-yidx]
        opp_prizes_taken = 6 - len(opp.get('prize') or [])
        my_deck = me.get('deckCount', 99)
        no_pokemon = not me.get('active') and not (me.get('bench') or [])
        if no_pokemon:
            return 'NO_POKEMON_IN_PLAY'
        if my_deck == 0:
            return 'DECK_OUT'
        if opp_prizes_taken >= 6:
            return 'PRIZED_OUT'
    return 'OTHER'

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
    terminal_cause = classify_terminal(d, you_idx) if your_result == -1 else None

    steps = d['steps']
    n = len(steps)

    lines = []
    lines.append(f"=== {path.split('/')[-1]} ===")
    lines.append(f"Opponent: {opp_name} | Result: {result_str}"
                  + (f" | Terminal cause: {terminal_cause}" if terminal_cause else "")
                  + f" | Total steps: {n}")
    lines.append("")

    decs = real_decisions(you_idx, steps)
    lines.append(f"--- Decision log ({len(decs)} real decisions) ---")

    empty_illegal = []
    for i, prev_obs, sel, action in decs:
        opts = sel.get('option', [])
        stype = sel.get('type')
        cur, yidx, pl = get_players(prev_obs)
        me = pl[yidx] if yidx is not None and len(pl) > yidx else {}
        opp = pl[1-yidx] if yidx is not None and len(pl) == 2 else {}
        my_active = (me.get('active') or [None])[0]
        opp_active = (opp.get('active') or [None])[0]
        hand = me.get('hand') or []
        hand_names = ','.join(names.get(pk_id(c), '?') for c in hand) if hand else ''
        hand_n = me.get('handCount', len(hand))
        deck_n = me.get('deckCount')
        my_prizes = len(me.get('prize') or [])
        opp_prizes = len(opp.get('prize') or [])
        turn = cur.get('turn')

        illegal = (not action) or action[0] < 0 or action[0] >= len(opts)
        if illegal and len(opts) > 0:
            empty_illegal.append(i)
            lines.append(f"  step{i:3d} turn{turn} | stype={stype} n_opts={len(opts)} "
                         f"| EMPTY/ILLEGAL RETURN action={action}")
            continue
        if stype != 0:
            continue  # non-MAIN selects (targeting, deck search, etc.) are resolved
                      # by the surrounding main-phase action; skip for the readable log

        chosen = resolve_chosen(names, prev_obs, action)
        if chosen is None:
            continue
        cn = cards_needed_for((opp_active or {}).get('hp')) if opp_active else None
        desc_extra = chosen.get('card') or chosen.get('desc') or chosen.get('attackId') or ''
        target = chosen.get('target', '')
        lines.append(
            f"  step{i:3d} turn{turn} | you_prizes={my_prizes} opp_prizes={opp_prizes} "
            f"| hand={hand_n}[{hand_names}] deck={deck_n} "
            f"| my_active={fmt_pk(names,my_active)} opp_active={fmt_pk(names,opp_active)} "
            f"cards_needed={cn} "
            f"| {chosen['otype_name']}: {desc_extra} {target}".strip())

    lines.append("")
    lines.append("--- Final state check (last few steps) ---")
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

    if empty_illegal:
        lines.append("")
        lines.append(f"*** {len(empty_illegal)} EMPTY/ILLEGAL RETURN(S) at real ACTIVE decisions: steps {empty_illegal} ***")

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
        with open(outpath, 'w', encoding='utf-8') as f:
            f.write(out)
        print(f"Wrote {outpath} ({len(out)} chars)")
