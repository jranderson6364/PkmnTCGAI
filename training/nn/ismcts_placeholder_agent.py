"""Placeholder-determinization arm of the pre-registered belief-vs-
placeholder ablation (docs/report-log.md 2026-07-07 ISMCTS gate 1 entry).
Same search as ismcts_agent.py, filler hidden zones — flipped per-call via
the module flag because both A/B sides share one worker process, so an
env var read at import time couldn't differ between them.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ismcts_agent as _base  # noqa: E402

DECK = _base.DECK


def agent(obs_dict: dict) -> list:
    prev = _base.DET_MODE
    _base.DET_MODE = "placeholder"
    try:
        return _base.agent(obs_dict)
    finally:
        _base.DET_MODE = prev
