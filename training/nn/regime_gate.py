"""Phase C two-part gate for the regime Q-net hybrid (issue #3).

GATE 1 (--scenario-pairs N): held-out scenario suite. For each of the 5
held-out exploiter-win seeds (never trained on, by construction), run N
PAIRED continuations: same determinization seed for both arms, hybrid_agent
vs plain main.py on the seed seat, plain main.py opponent both arms.
Pre-registered bar: CI-separable improvement — the paired mean win-rate
difference's 95% CI must exclude 0 from above.

GATE 2 (--anchor-games N): anchor non-regression. hybrid_agent and plain
main.py each play N games vs lucario + abomasnow (seats alternated, same
day, same machine). Pre-registered bar: hybrid must NOT be CI-separably
below plain v29d on either anchor (two-proportion 95% CI on the diff).

Usage (dry run):   python training/nn/regime_gate.py --scenario-pairs 3 --anchor-games 6
Usage (real gate): python training/nn/regime_gate.py --scenario-pairs 200 --anchor-games 200
"""
import argparse
import csv
import math
import os
import random
import statistics
import sys
import zlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_REPO_ROOT, "training"),
          os.path.join(_REPO_ROOT, "training", "local_cg"),
          os.path.join(_REPO_ROOT, "training", "belief"), _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import main as heuristic  # noqa: E402
import hybrid_agent  # noqa: E402
from regime_collect import load_seeds, _reset_stateful, _restore_stateful, MAX_PLIES  # noqa: E402
from mcts import MCTSSearcher, _obs_to_dict, _is_terminal, _terminal_value  # noqa: E402


def continue_once(seed_obs, seed_seat, our_agent, det_seed):
    """One continuation: determinize hidden zones with det_seed (global-RNG
    seeded, so paired arms get identical fills), play to a real terminal with
    our_agent on the seed seat and plain heuristic on the other. Returns the
    outcome in {-1, 0, 1} from the seed seat, or None on ply cap."""
    from cg.api import to_observation_class, search_begin, search_step, search_end

    observation = to_observation_class(seed_obs)
    state = observation.current
    my_p = state.players[seed_seat]
    opp_p = state.players[1 - seed_seat]
    random.seed(det_seed)
    filler = MCTSSearcher._filler
    your_deck = filler(my_p.deckCount, heuristic.DECK)
    your_prize = filler(len(my_p.prize), heuristic.DECK)
    opponent_deck = filler(opp_p.deckCount, heuristic.DECK)  # mirror
    opponent_prize = filler(len(opp_p.prize), heuristic.DECK)
    opponent_hand = filler(opp_p.handCount, heuristic.DECK)
    opponent_active = []
    if len(opp_p.active) > 0 and opp_p.active[0] is None:
        opponent_active = filler(1, heuristic.DECK)

    saved = _reset_stateful()
    try:
        ss = search_begin(observation, your_deck, your_prize, opponent_deck,
                          opponent_prize, opponent_hand, opponent_active,
                          manual_coin=False)
        cur_id, cur_obs = ss.searchId, seed_obs
        for _ply in range(MAX_PLIES):
            if _is_terminal(cur_obs):
                return _terminal_value(cur_obs, seed_seat)
            yours = cur_obs["current"]["yourIndex"]
            fn = our_agent if yours == seed_seat else heuristic.agent
            action = fn(cur_obs)
            action = list(action) if action else [0]
            ss = search_step(cur_id, action)
            cur_id, cur_obs = ss.searchId, _obs_to_dict(ss.observation)
        return None  # ply cap — discard the pair
    finally:
        _restore_stateful(saved)
        try:
            search_end()
        except Exception:
            pass


def _win(outcome):
    return 1.0 if outcome == 1 else (0.5 if outcome == 0 else 0.0)


def gate1_scenario(pairs, csv_w):
    _train, held = load_seeds()
    print(f"GATE 1: {len(held)} held-out seeds x {pairs} paired continuations")
    diffs = []
    per_seed = {}
    capped = 0
    override_start = hybrid_agent.overridden
    for sid, obs, seat in held:
        seed_diffs = []
        for k in range(pairs):
            det = zlib.crc32(f"{sid}:{k}".encode())
            out_h = continue_once(obs, seat, hybrid_agent.agent, det)
            out_c = continue_once(obs, seat, heuristic.agent, det)
            if out_h is None or out_c is None:
                capped += 1
                continue
            d = _win(out_h) - _win(out_c)
            seed_diffs.append(d)
            diffs.append(d)
            csv_w.writerow(["scenario", sid, k, out_h, out_c])
        wh = sum(1 for x in seed_diffs if x > 0)
        wc = sum(1 for x in seed_diffs if x < 0)
        per_seed[sid] = (statistics.mean(seed_diffs) if seed_diffs else 0.0,
                         len(seed_diffs), wh, wc)
        print(f"  {sid}: mean_diff={per_seed[sid][0]:+.3f} n={len(seed_diffs)} "
              f"(hybrid better in {wh}, worse in {wc})", file=sys.stderr)
    n = len(diffs)
    overrides = hybrid_agent.overridden - override_start
    assert overrides > 0, "hybrid never used the Q-net — gate would be vacuous"
    mean = statistics.mean(diffs)
    se = (statistics.stdev(diffs) / math.sqrt(n)) if n > 1 else float("inf")
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    print(f"GATE 1 RESULT: paired mean diff {mean:+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}] n={n} pairs (capped={capped}, "
          f"qnet overrides={overrides})")
    passed = lo > 0
    print(f"GATE 1 {'PASS' if passed else 'FAIL'} "
          f"(bar: 95% CI lower bound > 0)")
    return passed


def _run_block(agent_path, anchor_path, n_games, label, csv_w):
    from harness import run_matches
    w = l = t = crashed = 0
    for net_seat in (0, 1):
        n_seat = n_games - n_games // 2 if net_seat == 0 else n_games // 2
        paths = (agent_path, anchor_path) if net_seat == 0 else (anchor_path, agent_path)
        results = run_matches(paths[0], paths[1], n_seat, workers=None,
                              progress=False)
        for r in results:
            if "error" in r:
                crashed += 1
                continue
            ours, theirs = r["rewards"][net_seat], r["rewards"][1 - net_seat]
            if ours is None or theirs is None:
                crashed += 1
                continue
            if ours == 1:
                w += 1
            elif ours == -1:
                l += 1
            else:
                t += 1
            csv_w.writerow(["anchor", label, net_seat, ours, ""])
    n = w + l + t
    p = (w + 0.5 * t) / n if n else 0.0
    se = math.sqrt(p * (1 - p) / n) if n else float("inf")
    print(f"  {label}: {w}W-{l}L-{t}T (crashed={crashed}) "
          f"p={p:.3f} ±{1.96*se:.3f}", file=sys.stderr)
    return p, n


def gate2_anchors(n_games, csv_w):
    anchors = {"lucario": os.path.join(_REPO_ROOT, "opponents", "lucario_agent.py"),
               "abomasnow": os.path.join(_REPO_ROOT, "opponents", "abomasnow_agent.py")}
    agents = {"hybrid": os.path.join(_HERE, "hybrid_agent.py"),
              "v29d": os.path.join(_REPO_ROOT, "main.py")}
    print(f"GATE 2: {n_games} games/agent/anchor, seats alternated")
    stats = {}
    for aname, apath in agents.items():
        for anch, opath in anchors.items():
            stats[(aname, anch)] = _run_block(apath, opath, n_games,
                                              f"{aname} vs {anch}", csv_w)
    passed = True
    for anch in anchors:
        ph, nh = stats[("hybrid", anch)]
        pc, nc = stats[("v29d", anch)]
        diff = ph - pc
        se = math.sqrt(ph * (1 - ph) / nh + pc * (1 - pc) / nc)
        lo, hi = diff - 1.96 * se, diff + 1.96 * se
        verdict = "OK" if hi >= 0 else "REGRESSION"
        if hi < 0:
            passed = False
        print(f"GATE 2 {anch}: hybrid {ph:.3f} vs v29d {pc:.3f}, "
              f"diff {diff:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {verdict}")
    print(f"GATE 2 {'PASS' if passed else 'FAIL'} "
          f"(bar: not CI-separably below plain v29d on any anchor)")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-pairs", type=int, default=0,
                    help="Gate 1: paired continuations per held-out seed")
    ap.add_argument("--anchor-games", type=int, default=0,
                    help="Gate 2: games per agent per anchor")
    ap.add_argument("--csv", default=os.path.join(_REPO_ROOT, "training",
                                                  "regime_gate_games.csv"))
    args = ap.parse_args()

    print(f"checkpoint: {hybrid_agent._CKPT} (big={hybrid_agent._BIG})")
    with open(args.csv, "w", newline="") as f:
        csv_w = csv.writer(f)
        csv_w.writerow(["gate", "label", "k_or_seat", "outcome_h", "outcome_c"])
        g1 = gate1_scenario(args.scenario_pairs, csv_w) if args.scenario_pairs else None
        g2 = gate2_anchors(args.anchor_games, csv_w) if args.anchor_games else None
    print(f"SUMMARY: gate1={g1} gate2={g2}")


if __name__ == "__main__":
    main()
