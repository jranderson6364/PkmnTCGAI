"""Candidate wrapper for the PANEL-fitness weight search.

Same idea as wsearch_agent.py but bound to the refreshed v30 policy snapshot
(frozen_main_v30.py) so the search optimizes the CURRENTLY shipped heuristic.
Candidate weights come from the JSON named by env PANEL_WEIGHTS; absent -> stock.
"""
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_FROZEN = os.path.join(_HERE, "frozen_main_v30.py")

_spec = importlib.util.spec_from_file_location("panel_candidate_main", _FROZEN)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["panel_candidate_main"] = _mod
_spec.loader.exec_module(_mod)

_wp = os.environ.get("PANEL_WEIGHTS")
if _wp and os.path.exists(_wp):
    with open(_wp) as f:
        _mod.W.update({k: float(v) for k, v in json.load(f).items()})

agent = _mod.agent
DECK = list(_mod.DECK)
