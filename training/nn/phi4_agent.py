"""agent(obs_dict) wrapper around MCTSSearcher(leaf_eval="phi4") — Gate 2 of
the Φ v4 evaluation-function line (docs/report-log.md 2026-07-09): PIMC
search with a depth-limited rollout cut off by the Gate-1-passed Φ v4
linear eval, instead of full rollouts to terminal ("rollout") or the DMC
value net ("net").

Same agent contract as main.py; evaluated with training/ab_test.py.
Needs `cg.api` on sys.path (training/setup_local_search.py for local dev).
Set MCTS_OPPONENT_MODULE to the anchor being played (belief-driven in a
real ship; explicit here, same convention as the closed search lines).
"""
import os
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

_SIMS = int(os.environ.get("MCTS_SIMS", "60"))
_C_PUCT = float(os.environ.get("MCTS_C_PUCT", "1.4"))
_PRIOR_TEMP = float(os.environ.get("MCTS_PRIOR_TEMP", "2.0"))
_PHI4_TURNS = int(os.environ.get("PHI4_TURNS", "2"))
_PHI4_MAX_PLIES = int(os.environ.get("PHI4_MAX_PLIES", "60"))
_TIMING_LOG = os.environ.get("MCTS_TIMING_LOG")

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
                                leaf_eval="phi4", phi4_turns=_PHI4_TURNS,
                                phi4_max_plies=_PHI4_MAX_PLIES)
        return searcher.choose(obs_dict)
    except Exception as e:
        _fallback_count += 1
        print(f"[phi4_agent] FALLBACK #{_fallback_count} to heuristic: {e!r}",
              file=sys.stderr)
        from main import agent as heuristic_agent
        return heuristic_agent(obs_dict)
    finally:
        if t0 is not None:
            path = f"{_TIMING_LOG}.{os.getpid()}"
            with open(path, "a") as f:
                f.write(f"{time.time() - t0:.4f}\n")
