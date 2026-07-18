"""W-search candidate wrapper (pre-registered docs/report-log.md 2026-07-18).

Plays frozen_main.py's exact policy with the W dict patched from the JSON
file named by env WSEARCH_WEIGHTS (absent/empty -> stock weights). Imports
the FROZEN snapshot, not live main.py, so weights stay the only variable
even if main.py is edited while a search runs. Loaded fresh per game by
training/harness.py's worker, which applies each job's extra_env before
import — per-candidate weights ride the existing extra_envs mechanism.
"""
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_FROZEN = os.path.join(_HERE, "frozen_main.py")

_spec = importlib.util.spec_from_file_location("wsearch_candidate_main", _FROZEN)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["wsearch_candidate_main"] = _mod
_spec.loader.exec_module(_mod)

_wp = os.environ.get("WSEARCH_WEIGHTS")
if _wp:
    with open(_wp) as f:
        _mod.W.update({k: float(v) for k, v in json.load(f).items()})

agent = _mod.agent
DECK = list(_mod.DECK)
