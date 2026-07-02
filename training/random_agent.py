"""Minimal random agent for BC-net gate evaluation (65%+ vs random target)."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import DECK  # noqa: E402


def agent(obs_dict: dict) -> list:
    sel = obs_dict.get("select")
    if sel is None:
        return DECK
    n = len(sel.get("option", []))
    if n == 0:
        return []
    mx = sel.get("maxCount", 1) or 1
    mn = sel.get("minCount", 0) or 0
    k = max(mn, min(mx, n))
    return random.sample(range(n), k)
