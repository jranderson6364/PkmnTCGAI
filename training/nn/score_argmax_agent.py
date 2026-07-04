"""Diagnostic-only agent: argmax(main.score_options(obs, sel)) with no search
at all. Exists to answer one question (per advisor guidance 2026-07-04, given
the MCTS gates' sub-50% results, which "echo chamber" alone doesn't explain):
is `score_options` — the prior/rollout/opponent-model function `mcts.py`
relies on for everything — itself a faithful reconstruction of the real
`_choose` teacher, or a materially weaker stand-in? An A/B against main.py
isolates that one variable before any more MCTS tuning.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import main as heuristic  # noqa: E402
from main import DECK, _safe_return  # noqa: E402


def agent(obs_dict: dict) -> list:
    sel = obs_dict.get("select")
    if not sel:
        return DECK
    opts = sel.get("option") or []
    n = len(opts)
    if n == 0:
        return []
    scores = heuristic.score_options(obs_dict, sel)
    if not scores or len(scores) != n:
        scores = [0.0] * n
    best = max(range(n), key=lambda i: scores[i])
    return _safe_return([best], sel)
