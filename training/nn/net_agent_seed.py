"""Fixed-checkpoint twin of net_agent.py — always loads ptcg_dagger_r2.pth
regardless of $NET_CKPT, so it can be A/B'd head-to-head against a
net_agent.py invocation pointed at a different checkpoint in the same run
(ab_test.py sets one shared env var for the whole process pool, which can't
distinguish two net_agent.py instances). Used for the Stage 2 AWR
seed-comparison gate (docs/nn-training.md Stage 2) — vs-teacher alone can't
tell "AWR improved nothing" from "the seed was already near parity".
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from net_common import load_model, encode_batch, clamp, _HERE
from main import DECK

_CKPT = os.path.join(_HERE, "..", "ptcg_dagger_r2.pth")


def agent(obs_dict: dict) -> list:
    sel = obs_dict.get("select")
    if sel is None:
        return DECK
    n = len(sel.get("option", []))
    if n == 0:
        return []
    try:
        model = load_model(_CKPT)
        batch, n_actions = encode_batch(obs_dict, sel)
        with torch.no_grad():
            logits, _ = model(*batch)
        logits = logits[0, :n_actions]
        mx = sel.get("maxCount", 1) or 1
        k = min(mx, n_actions)
        top = torch.topk(logits, k=k).indices.tolist()
        return clamp(top, sel)
    except Exception:
        mn = sel.get("minCount", 1) or 0
        return clamp(list(range(max(mn, 1))), sel)
