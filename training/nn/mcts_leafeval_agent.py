"""agent(obs_dict) wrapper around MCTSSearcher(leaf_eval="net") -- Phase 0
step-0 probe (docs/nn-training.md Phase 0 amendment (d)): does PUCT search
with the existing ptcg_dmc_r2.pth value net at the leaf show any life over
raw argmax (training/nn/dmc_agent.py), before spending a week improving that
value net? No retraining -- reuses the checkpoint as-is.

Same agent contract as main.py so it can be evaluated with training/ab_test.py.
Needs `cg.api` on sys.path (training/setup_local_search.py for local dev).
"""
import os
import pickle
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_LOCAL_CG = os.path.join(_REPO_ROOT, "training", "local_cg")
if _LOCAL_CG not in sys.path:
    sys.path.insert(0, _LOCAL_CG)

from mcts import MCTSSearcher  # noqa: E402
from main import DECK  # noqa: E402

_SIMS = int(os.environ.get("MCTS_SIMS", "150"))
_C_PUCT = float(os.environ.get("MCTS_C_PUCT", "1.4"))
_PRIOR_TEMP = float(os.environ.get("MCTS_PRIOR_TEMP", "2.0"))
_NET_CKPT = os.environ.get("NET_CKPT") or os.path.join(_REPO_ROOT, "training", "ptcg_dmc_r2.pth")
_NET_LEAF_MAX_DEPTH = int(os.environ.get("MCTS_NET_LEAF_MAX_DEPTH", "40"))
# "qmax" (default) preserves this module's original behavior, correct for
# dmc_collect.py-trained checkpoints (this module's original Phase 0 probe
# target). Phase 2 collection (mcts_collect.py) sets this to "head" — see
# mcts.py's MCTSSearcher.__init__ docstring and docs/report-log.md 2026-07-07
# "mcts.py _net_leaf_value convention mismatch" for why the distinction matters.
_NET_VALUE_SOURCE = os.environ.get("MCTS_NET_VALUE_SOURCE", "qmax")
# Compute-budget check (docs/report-log.md 2026-07-05 timing probe): if set,
# appends "<elapsed_seconds>\n" per real decision to this path (one file per
# worker process, PID-suffixed, since ab_test.py fans out across
# multiprocessing workers) -- lets mcts_timing_probe.py measure real
# per-decision cost via the already-proven ab_test.py/harness.py runner
# instead of re-implementing game-running logic.
_TIMING_LOG = os.environ.get("MCTS_TIMING_LOG")
# Phase 2 (docs/nn-training.md "AlphaZero-Style Push"): if set, uses
# choose_with_stats() instead of choose() and appends one pickled
# {"obs":..., "action":..., "policy_target":[...], "root_value":...} record
# per real decision to this path (PID-suffixed, same multiprocess-safe
# pattern as MCTS_TIMING_LOG) -- lets mcts_collect.py extract real
# MCTS-derived training targets via the already-proven ab_test.py/
# harness.py multiprocess runner, instead of a hand-rolled direct-in-process
# game loop (which broke on repeated native search calls earlier this
# session -- see the compute-budget-check entry in report-log.md).
_COLLECT_LOG = os.environ.get("MCTS_COLLECT_LOG")
# Parallel collection support: harness.py's run_matches(extra_envs=...) sets
# this uniquely per job/game BEFORE this module is (re-)imported, so each
# fresh import picks up the correct per-game value -- lets mcts_collect.py
# correlate collect-log records to game outcomes correctly even when
# workers>1 interleaves multiple games' decisions across worker processes
# (previously required workers=1 + strict submission order).
_GAME_ID = os.environ.get("MCTS_GAME_ID")

_fallback_count = 0


def agent(obs_dict: dict) -> list:
    global _fallback_count
    sel = obs_dict.get("select")
    if sel is None:
        return DECK
    n = len(sel.get("option", []))
    if n == 0:
        return []
    t0 = time.time() if _TIMING_LOG else None
    try:
        searcher = MCTSSearcher(sims=_SIMS, c_puct=_C_PUCT, prior_temp=_PRIOR_TEMP,
                                 leaf_eval="net", net_ckpt=_NET_CKPT,
                                 net_leaf_max_depth=_NET_LEAF_MAX_DEPTH,
                                 net_value_source=_NET_VALUE_SOURCE)
        if _COLLECT_LOG:
            action, N, root_value = searcher.choose_with_stats(obs_dict)
            if N is not None:
                total = sum(N) or 1
                policy_target = [x / total for x in N]
                path = f"{_COLLECT_LOG}.{os.getpid()}"
                with open(path, "ab") as f:
                    pickle.dump({"obs": obs_dict, "action": action,
                                 "policy_target": policy_target,
                                 "root_value": root_value,
                                 "game_id": _GAME_ID}, f)
            return action
        return searcher.choose(obs_dict)
    except Exception as e:
        _fallback_count += 1
        print(f"[mcts_leafeval_agent] FALLBACK #{_fallback_count} to heuristic: {e!r}",
              file=sys.stderr)
        from main import agent as heuristic_agent
        return heuristic_agent(obs_dict)
    finally:
        if t0 is not None:
            path = f"{_TIMING_LOG}.{os.getpid()}"
            with open(path, "a") as f:
                f.write(f"{time.time() - t0:.4f}\n")
