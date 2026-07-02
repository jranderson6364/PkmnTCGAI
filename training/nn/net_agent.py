"""agent(obs_dict) wrapper around a trained checkpoint — satisfies the same
contract as main.py / opponents/*.py so it can be evaluated with
training/ab_test.py and, eventually, submitted to the ladder.

Single forward pass per decision (argmax over valid actions), CPU or GPU —
no search, no MCTS. This is the "ship the plain BC net" step from
docs/competition-strategy.md.

Checkpoint path: $NET_CKPT env var, else training/ptcg_bc_v1.pth.
"""
import os
import sys

# Must NOT rely on the top-level launching script's sys.path — a spawned
# multiprocessing worker only inherits the sys.path of whichever script
# started the pool (ab_test.py vs selfplay_collect.py insert different dirs),
# so this module has to set up its own path regardless of caller.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from net_common import load_model, encode_batch, clamp, _HERE
from main import DECK  # reuse the same 60-card deck as the heuristic

_CKPT = os.environ.get("NET_CKPT") or os.path.join(_HERE, "..", "ptcg_bc_v1.pth")


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
