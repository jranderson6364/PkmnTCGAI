"""Diagnostic for the Mega Lucario pilot: what our agent actually DOES.

The harness's observation carries no `logs`, so instead of parsing replays this
wraps the agent function and records every decision it makes: the select
context, the option type it chose, and (for attacks) which attack. That is
ground truth about the policy, not an inference from board state.
"""
import sys, os, argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import load_agent, play_game

OPT_NAME = {0:"NUMBER",1:"YES",2:"NO",3:"CARD",4:"TOOL_CARD",5:"ENERGY_CARD",
            6:"ENERGY",7:"PLAY",8:"ATTACH",9:"EVOLVE",10:"ABILITY",11:"DISCARD",
            12:"RETREAT",13:"ATTACK",14:"END",15:"SKILL",16:"SPECIAL_CONDITION"}
ATK_NAME = {982:"AuraJab",983:"MegaBrave",978:"WildPress",980:"CosmicBeam",
            981:"AccelStab",979:"PowerGem",976:"Corkscrew",977:"Confront"}
MLUCARIO, RIOLU = 678, 677
NAME = {678:"MegaLucario",677:"Riolu",674:"Hariyama",673:"Makuhita",
        676:"Solrock",675:"Lunatone"}


def make_probe(agent_fn, rec):
    """Wrap an agent so every decision it returns is recorded."""
    def probe(obs_dict):
        out = agent_fn(obs_dict)
        try:
            sel = obs_dict.get("select")
            if sel and out:
                opts = sel.get("option") or []
                ctx = sel.get("context")
                i = out[0]
                if 0 <= i < len(opts):
                    o = opts[i]
                    ot = o.get("type")
                    rec["ctx"][ctx] += 1
                    if ctx == 0:  # MAIN
                        rec["main_choice"][OPT_NAME.get(ot, str(ot))] += 1
                        if ot == 13:
                            rec["attacks"][ATK_NAME.get(o.get("attackId"), str(o.get("attackId")))] += 1
                        if ot == 14:
                            # what ELSE was on the table when we chose to end?
                            others = {OPT_NAME.get(x.get("type"), "?") for x in opts}
                            others.discard("END")
                            if others:
                                rec["end_with_options"][",".join(sorted(others))] += 1
                            # was an attack available and declined?
                            if any(x.get("type") == 13 for x in opts):
                                rec["declined_attack"] += 1
                    # turn-level: track whether we attacked at all this turn
                    cur = obs_dict.get("current") or {}
                    t = cur.get("turn")
                    if ot == 13:
                        rec["turns_attacked"].add(t)
                    rec["turns_seen"].add(t)
        except Exception:
            pass
        return out
    return probe


def board_summary(steps, me_idx):
    first_luc, max_e, max_board, final_turn = None, 0, 0, 0
    prizes = (None, None)
    for st in steps:
        if not isinstance(st, list) or len(st) <= me_idx: continue
        cur = (st[me_idx].get("observation") or {}).get("current") or {}
        players = cur.get("players") or []
        if len(players) != 2 or cur.get("yourIndex", 0) != me_idx: continue
        me, opp = players[me_idx], players[1 - me_idx]
        turn = cur.get("turn") or 0
        final_turn = max(final_turn, turn)
        active = (me.get("active") or [None])[0]
        board = ([active] if active else []) + [b for b in (me.get("bench") or []) if b]
        max_board = max(max_board, len(board))
        prizes = (len(me.get("prize") or []), len(opp.get("prize") or []))
        for p in board:
            if (p or {}).get("id") == MLUCARIO:
                if first_luc is None: first_luc = turn
                max_e = max(max_e, len((p or {}).get("energies") or []))
    return first_luc, max_e, max_board, final_turn, prizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent"); ap.add_argument("opponent")
    ap.add_argument("games", type=int, default=10, nargs="?")
    args = ap.parse_args()

    a0, d0, _ = load_agent(args.agent)
    a1, d1, _ = load_agent(args.opponent)

    rec = {"ctx": Counter(), "main_choice": Counter(), "attacks": Counter(),
           "end_with_options": Counter(), "declined_attack": 0,
           "turns_attacked": set(), "turns_seen": set()}
    wins = 0
    firsts, maxes, boards = [], [], []
    my_prizes_taken = []

    for g in range(args.games):
        me = g % 2
        rec["turns_attacked"] = set(); rec["turns_seen"] = set()
        probe = make_probe(a0, rec)
        if me == 0:
            r = play_game(probe, d0, a1, d1, keep_steps=True)
        else:
            r = play_game(a1, d1, probe, d0, keep_steps=True)
        rew = r["rewards"]
        won = rew[me] is not None and rew[1-me] is not None and rew[me] > rew[1-me]
        wins += bool(won)
        fl, me_, mb, ft, pz = board_summary(r["steps"], me)
        if fl is not None: firsts.append(fl)
        maxes.append(me_); boards.append(mb)
        if pz[0] is not None: my_prizes_taken.append(6 - pz[1])
        atk_turns = len(rec["turns_attacked"]); seen = len(rec["turns_seen"])
        print(f"  g{g:02d} {'W' if won else 'L'} turn={ft:3d} lucT={fl} maxE={me_} "
              f"board={mb} prizes(me_left,opp_left)={pz} prizesTaken={6-pz[1] if pz[1] is not None else '?'} "
              f"attacked_on {atk_turns}/{seen} turns")

    n = args.games
    print(f"\n=== {args.agent} vs {args.opponent}, n={n} ===")
    print(f"win rate             : {wins}/{n} = {wins/n:.1%}")
    print(f"Mega Lucario in play : {len(firsts)}/{n}"
          + (f", median turn {sorted(firsts)[len(firsts)//2]}" if firsts else ""))
    print(f"max energy on it     : mean {sum(maxes)/max(1,len(maxes)):.2f}")
    print(f"max board size       : mean {sum(boards)/max(1,len(boards)):.2f}")
    print(f"prizes WE took       : mean {sum(my_prizes_taken)/max(1,len(my_prizes_taken)):.2f} of 6")
    print(f"\nMAIN-phase choices   : {dict(rec['main_choice'])}")
    print(f"attacks chosen       : {dict(rec['attacks'])}")
    print(f"ENDed w/ attack avail: {rec['declined_attack']}")
    print(f"ENDed alongside      : {rec['end_with_options'].most_common(6)}")


if __name__ == "__main__":
    main()
