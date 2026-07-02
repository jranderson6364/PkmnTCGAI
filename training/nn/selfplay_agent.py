"""Temperature-sampling variant of net_agent.py, used ONLY for self-play data
collection (exploration) — never for ladder/eval, where net_agent.py's
deterministic argmax is used instead.

Env vars: NET_CKPT (checkpoint path), NET_TEMP (softmax temperature, default 1.0).
"""
import os
import sys

# See net_agent.py — must not rely on the top-level launching script's sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from net_common import load_model, encode_batch, clamp, _HERE
from main import DECK

_CKPT = os.environ.get("NET_CKPT") or os.path.join(_HERE, "..", "ptcg_bc_v1.pth")
_TEMP = float(os.environ.get("NET_TEMP", "1.0"))


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
        if k >= n_actions:
            picks = list(range(n_actions))
        else:
            probs = F.softmax(logits / max(_TEMP, 1e-3), dim=-1)
            picks = torch.multinomial(probs, k, replacement=False).tolist()
        return clamp(picks, sel)
    except Exception:
        mn = sel.get("minCount", 1) or 0
        return clamp(list(range(max(mn, 1))), sel)
