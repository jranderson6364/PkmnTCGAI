"""Self-play data collector for the RL program (S2, pre-registered 2026-07-23).

Runs the deep override search (twoply_collect.py) as the collecting side against a
strong search opponent (twoply_agent.py), logging one record per search decision:
the state, the per-candidate SEARCH values, the heuristic base scores, and the
heuristic-vs-search pick. Each record is then tagged with the game outcome.

From these records the training stage derives BOTH targets the two S1 branches
need, without recollecting:
  * VALUE target  (per state)      = search value of the chosen line, blended with
                                     the eventual game outcome (AlphaZero TD-style).
  * ADVANTAGE target (per candidate) = cand_val[i] - cand_val[heur_top], the only
                                     place the search carries info the heuristic
                                     lacks (the override signal).

Collecting side = twoply_collect (DEPTH/LEAF/N_DET inherited from env, defaulting
to the shipped d2/formula unless overridden). Opponent = twoply_agent (search, no
logging). Serial (workers=1) so game/outcome correlation is exact — the same bar
the AlphaZero-push collector was held to. Writes:
  <out>.jsonl   one search-decision record per line (state + values)
  <out>.outcomes.json   {game_id: +1 win / -1 loss / 0 tie for the collecting seat}

Run:  python training/nn/collect_search_data.py --games 200 --out training/nn/search_corpus
"""
import argparse
import json
import os
import sys
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "training", "local_cg"),
           os.path.join(_REPO, "training", "belief"),
           os.path.join(_REPO, "training", "nn"),
           os.path.join(_REPO, "training"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(_HERE, "search_corpus"))
    ap.add_argument("--opponent", default=os.path.join(_HERE, "twoply_agent.py"),
                    help="search opponent (no logging)")
    args = ap.parse_args()

    jsonl = args.out + ".jsonl"
    if os.path.exists(jsonl):
        os.remove(jsonl)
    os.environ["TWOPLY_COLLECT_LOG"] = jsonl

    from harness import play_game

    collector = _load_module(os.path.join(_HERE, "twoply_collect.py"), "tc_collect")
    opp = _load_module(args.opponent, "tc_opp")
    cdeck = list(collector.DECK)
    odeck = list(opp.DECK)

    outcomes = {}
    for g in range(args.games):
        gid = f"g{g}"
        collector._GAME_ID = gid          # per-game id read at record-write time
        me = g % 2                          # alternate seats
        if me == 0:
            r = play_game(collector.agent, cdeck, opp.agent, odeck)
        else:
            r = play_game(opp.agent, odeck, collector.agent, cdeck)
        rew = r["rewards"]
        if rew[me] is None or rew[1 - me] is None or rew[me] == rew[1 - me]:
            outcomes[gid] = 0
        else:
            outcomes[gid] = 1 if rew[me] > rew[1 - me] else -1
        if (g + 1) % 20 == 0:
            n_rec = sum(1 for _ in open(jsonl)) if os.path.exists(jsonl) else 0
            wins = sum(1 for v in outcomes.values() if v > 0)
            print(f"  {g+1}/{args.games} games | {n_rec} records | "
                  f"collector wins {wins}/{g+1}", flush=True)

    with open(args.out + ".outcomes.json", "w") as f:
        json.dump(outcomes, f)
    n_rec = sum(1 for _ in open(jsonl)) if os.path.exists(jsonl) else 0
    wins = sum(1 for v in outcomes.values() if v > 0)
    print(f"\nDONE: {args.games} games, {n_rec} search-decision records, "
          f"collector win rate {wins}/{args.games} = {wins/args.games:.1%}")
    print(f"  records:  {jsonl}")
    print(f"  outcomes: {args.out}.outcomes.json")


if __name__ == "__main__":
    main()
