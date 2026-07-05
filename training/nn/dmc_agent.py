"""Epsilon-greedy Q-value agent for DMC self-play collection (exploration).
Treats the net's per-action logit as Q(s,a): with prob epsilon picks a
uniform-random legal action, else argmax over Q. Never used for ladder/eval.

Env vars: NET_CKPT (checkpoint path), NET_EPS (epsilon, default 0.1).
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from net_common import load_model, encode_batch, clamp, _HERE
from main import DECK

_CKPT = os.environ.get("NET_CKPT") or os.path.join(_HERE, "..", "ptcg_dmc_r1.pth")
_EPS = float(os.environ.get("NET_EPS", "0.1"))


def agent(obs_dict: dict) -> list:
    sel = obs_dict.get("select")
    if sel is None:
        return DECK
    n = len(sel.get("option", []))
    if n == 0:
        return []
    mx = sel.get("maxCount", 1) or 1
    k = min(mx, n)
    try:
        if k >= n:
            return clamp(list(range(n)), sel)
        if random.random() < _EPS:
            picks = random.sample(range(n), k)
            return clamp(picks, sel)
        model = load_model(_CKPT)
        batch, n_actions = encode_batch(obs_dict, sel)
        with torch.no_grad():
            logits, _ = model(*batch)
        q = logits[0, :n_actions]
        picks = torch.topk(q, k).indices.tolist()
        return clamp(picks, sel)
    except Exception:
        mn = sel.get("minCount", 1) or 0
        return clamp(list(range(max(mn, 1))), sel)
