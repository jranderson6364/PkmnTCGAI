"""Pre-registered gate battery for the rescue-mode heuristic fix
(report-log 2026-07-16, PRE-REGISTRATION A + same-day amendment).

Runs, in order, against the FROZEN pre-fix baseline
(training/baselines/v29d_pre_rescue.py):
  1. Mirror A/B, n=400, seats alternated. Adopt bar: fixed main.py win-rate
     95% CI lower bound > 0.50.
  2. Anchor non-regression: n=200/anchor vs lucario+abomasnow for BOTH the
     fixed main.py and the frozen baseline (same run, same machine). Bar:
     fixed not CI-separably below frozen on any anchor.
  3. Held-out scenario suite (directional read, NOT a bar): paired
     continuations from the 5 held-out seeds, fixed vs frozen on the seed
     seat, frozen baseline as the opponent both arms.

main.py MUST NOT be edited while this runs (the tie-gate contamination
lesson, same day).
"""
import argparse
import csv
import importlib.util
import math
import os
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

FIXED = os.path.join(_REPO_ROOT, "main.py")
FROZEN = os.path.join(_REPO_ROOT, "training", "baselines", "v29d_pre_rescue.py")

from regime_gate import _run_block  # noqa: E402


def _load_frozen():
    spec = importlib.util.spec_from_file_location("v29d_pre_rescue", FROZEN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v29d_pre_rescue"] = mod
    spec.loader.exec_module(mod)
    return mod


def scenario(pairs, csv_w):
    import main as fixed_mod
    frozen_mod = _load_frozen()
    from regime_collect import load_seeds, _reset_stateful, _restore_stateful, MAX_PLIES
    from mcts import MCTSSearcher, _obs_to_dict, _is_terminal, _terminal_value
    from cg.api import to_observation_class, search_begin, search_step, search_end
    import random

    def continue_once(seed_obs, seed_seat, our_agent, det_seed):
        observation = to_observation_class(seed_obs)
        state = observation.current
        my_p = state.players[seed_seat]
        opp_p = state.players[1 - seed_seat]
        random.seed(det_seed)
        filler = MCTSSearcher._filler
        your_deck = filler(my_p.deckCount, fixed_mod.DECK)
        your_prize = filler(len(my_p.prize), fixed_mod.DECK)
        opponent_deck = filler(opp_p.deckCount, fixed_mod.DECK)
        opponent_prize = filler(len(opp_p.prize), fixed_mod.DECK)
        opponent_hand = filler(opp_p.handCount, fixed_mod.DECK)
        opponent_active = []
        if len(opp_p.active) > 0 and opp_p.active[0] is None:
            opponent_active = filler(1, fixed_mod.DECK)
        saved = _reset_stateful()  # resets the fixed main's tracked state
        frozen_mod._STALL_MEMO = {}  # frozen module isn't in _STATEFUL_MODULES
        try:
            ss = search_begin(observation, your_deck, your_prize, opponent_deck,
                              opponent_prize, opponent_hand, opponent_active,
                              manual_coin=False)
            cur_id, cur_obs = ss.searchId, seed_obs
            for _ply in range(MAX_PLIES):
                if _is_terminal(cur_obs):
                    return _terminal_value(cur_obs, seed_seat)
                yours = cur_obs["current"]["yourIndex"]
                fn = our_agent if yours == seed_seat else frozen_mod.agent
                action = fn(cur_obs)
                action = list(action) if action else [0]
                ss = search_step(cur_id, action)
                cur_id, cur_obs = ss.searchId, _obs_to_dict(ss.observation)
            return None
        finally:
            _restore_stateful(saved)
            try:
                search_end()
            except Exception:
                pass

    _train, held = load_seeds()
    diffs = []
    capped = 0
    for sid, obs, seat in held:
        seed_diffs = []
        for k in range(pairs):
            det = zlib.crc32(f"fix:{sid}:{k}".encode())
            out_f = continue_once(obs, seat, fixed_mod.agent, det)
            out_c = continue_once(obs, seat, frozen_mod.agent, det)
            if out_f is None or out_c is None:
                capped += 1
                continue
            w = (1.0 if out_f == 1 else (0.5 if out_f == 0 else 0.0)) - \
                (1.0 if out_c == 1 else (0.5 if out_c == 0 else 0.0))
            seed_diffs.append(w)
            diffs.append(w)
            csv_w.writerow(["scenario", sid, k, out_f, out_c])
        print(f"  {sid}: mean_diff={statistics.mean(seed_diffs) if seed_diffs else 0:+.3f} "
              f"n={len(seed_diffs)}", file=sys.stderr)
    n = len(diffs)
    mean = statistics.mean(diffs) if diffs else 0.0
    se = (statistics.stdev(diffs) / math.sqrt(n)) if n > 1 else float("inf")
    print(f"SCENARIO (directional): paired mean diff {mean:+.4f} "
          f"[{mean - 1.96*se:+.4f}, {mean + 1.96*se:+.4f}] n={n} capped={capped}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", type=int, default=400)
    ap.add_argument("--anchor", type=int, default=200)
    ap.add_argument("--scenario-pairs", type=int, default=150)
    ap.add_argument("--csv", default=os.path.join(_REPO_ROOT, "training",
                                                  "fix_gate_games.csv"))
    args = ap.parse_args()

    anchors = {"lucario": os.path.join(_REPO_ROOT, "opponents", "lucario_agent.py"),
               "abomasnow": os.path.join(_REPO_ROOT, "opponents", "abomasnow_agent.py")}

    with open(args.csv, "w", newline="") as f:
        csv_w = csv.writer(f)
        csv_w.writerow(["gate", "label", "k_or_seat", "outcome_a", "outcome_b"])

        p, n = _run_block(FIXED, FROZEN, args.mirror, "fixed vs frozen (mirror)", csv_w)
        se = math.sqrt(p * (1 - p) / n)
        lo = p - 1.96 * se
        g1 = lo > 0.5
        print(f"GATE 1 (mirror): {p:.3f} [{lo:.3f}, {p + 1.96*se:.3f}] n={n} "
              f"-> {'PASS' if g1 else 'FAIL'} (bar: CI lower bound > 0.50)")

        g2 = True
        for anch, opath in anchors.items():
            pf, nf = _run_block(FIXED, opath, args.anchor, f"fixed vs {anch}", csv_w)
            pz, nz = _run_block(FROZEN, opath, args.anchor, f"frozen vs {anch}", csv_w)
            diff = pf - pz
            se = math.sqrt(pf * (1 - pf) / nf + pz * (1 - pz) / nz)
            hi = diff + 1.96 * se
            ok = hi >= 0
            g2 = g2 and ok
            print(f"GATE 2 {anch}: fixed {pf:.3f} vs frozen {pz:.3f}, diff "
                  f"{diff:+.3f} [{diff - 1.96*se:+.3f}, {hi:+.3f}] -> "
                  f"{'OK' if ok else 'REGRESSION'}")
        print(f"GATE 2 {'PASS' if g2 else 'FAIL'}")

        if args.scenario_pairs:
            scenario(args.scenario_pairs, csv_w)

        print(f"SUMMARY: gate1={g1} gate2={g2} adopt={'YES' if g1 and g2 else 'NO'}")


if __name__ == "__main__":
    main()
