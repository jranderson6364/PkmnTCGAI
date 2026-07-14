"""Exploration wrapper for regime_collect.py Source B: plays main.agent
(v29d) verbatim EXCEPT at in-regime single-select decisions, where with
probability REGIME_EPS it picks a uniform random legal option. Out-of-regime
play is untouched so fresh games stay realistic.
"""
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import main as heuristic  # noqa: E402
from regime_detector import regime_fires  # noqa: E402

_EPS = float(os.environ.get("REGIME_EPS", "0.25"))
_rng = random.Random()


def agent(obs_dict: dict) -> list:
    sel = obs_dict.get("select") or {}
    opts = sel.get("option") or []
    action = heuristic.agent(obs_dict)
    if (len(opts) > 1 and (sel.get("minCount", 0) or 0) < 2
            and regime_fires(obs_dict.get("current") or {},
                             obs_dict["current"]["yourIndex"])
            and _rng.random() < _EPS):
        return [_rng.randrange(len(opts))]
    return action
