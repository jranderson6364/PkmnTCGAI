"""agent(obs_dict) wrapper around MCTSSearcher — satisfies the same contract
as main.py / opponents/*.py so it can be evaluated with training/ab_test.py.

Needs `cg.api` on sys.path: either the live Kaggle environment, or the local
dev shim from `training/setup_local_search.py` (run once first; see that
file's docstring for why it works and what it does NOT change about the
actual ladder submission).

Sims/decision, rollout depth, etc. are tunable via env vars so ab_test.py runs
don't need code edits between gates.
"""
import os
import sys

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
_MAX_DEPTH = int(os.environ.get("MCTS_MAX_ROLLOUT_DEPTH", "250"))
_C_PUCT = float(os.environ.get("MCTS_C_PUCT", "1.4"))
_PRIOR_TEMP = float(os.environ.get("MCTS_PRIOR_TEMP", "2.0"))

# Counts fallbacks-to-heuristic due to an internal exception, so a silent
# per-decision failure can't masquerade as a clean MCTS game in ab_test.py's
# game-level error count (which wouldn't see this). Printed to stderr (not
# raised) so one bad decision doesn't cost the whole game, but is still
# visible in the worker's captured output.
_fallback_count = 0


def agent(obs_dict: dict) -> list:
    global _fallback_count
    sel = obs_dict.get("select")
    if sel is None:
        return DECK
    n = len(sel.get("option", []))
    if n == 0:
        return []
    try:
        searcher = MCTSSearcher(sims=_SIMS, max_rollout_depth=_MAX_DEPTH,
                                 c_puct=_C_PUCT, prior_temp=_PRIOR_TEMP)
        return searcher.choose(obs_dict)
    except Exception as e:
        _fallback_count += 1
        print(f"[mcts_agent] FALLBACK #{_fallback_count} to heuristic: {e!r}",
              file=sys.stderr)
        from main import agent as heuristic_agent
        return heuristic_agent(obs_dict)
