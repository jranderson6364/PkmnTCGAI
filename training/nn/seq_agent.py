"""agent(obs_dict) wrapper around a trained SeqPTCGNet checkpoint — same
contract as net_agent.py, but maintains the running per-game decision
history (module-level list) and does ONE causal forward pass over the
accumulated sequence per decision, reading out the argmax/top-k at the
LAST position. Relies on harness.py's load_agent() re-executing this
module fresh per game (see harness.py's `_worker` docstring) so the
module-level `_history` list naturally resets at the start of every game
— no explicit game-boundary detection needed.

Checkpoint path: $SEQ_CKPT env var, else training/ptcg_seq_main.pth.
Context is capped at the last MAX_LEN decisions (matches train_seq.py's
default) purely as a safety bound, not because games are expected to
exceed it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from encode_seq import encode_sample_seq, PHI4_DIM
from encode import MAX_ACTIONS, NUM_FEATS
from model_seq import SeqPTCGNet
from net_common import clamp
from main import DECK

_CKPT = os.environ.get("SEQ_CKPT") or os.path.join(os.path.dirname(__file__), "..", "ptcg_seq_main.pth")
_MAX_LEN = 160

_net = None
_history = []


def _load():
    global _net
    if _net is None:
        ckpt = torch.load(_CKPT, map_location="cpu")
        net = SeqPTCGNet(n_layers=ckpt.get("n_layers", 2),
                          use_history=ckpt.get("use_history", True),
                          use_phi4=ckpt.get("use_phi4", True))
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        _net = net
    return _net


def _build_tensors(steps):
    T = len(steps)
    board_ids = torch.zeros(1, T, 13, dtype=torch.long)
    hand_ids = torch.zeros(1, T, 20, dtype=torch.long)
    discard_ids = torch.zeros(1, T, 20, dtype=torch.long)
    numeric = torch.zeros(1, T, NUM_FEATS, dtype=torch.float)
    phi4 = torch.zeros(1, T, PHI4_DIM, dtype=torch.float)
    action_type = torch.zeros(1, T, MAX_ACTIONS, dtype=torch.long)
    action_card = torch.zeros(1, T, MAX_ACTIONS, dtype=torch.long)
    action_attack = torch.zeros(1, T, MAX_ACTIONS, dtype=torch.long)
    action_numeric = torch.zeros(1, T, MAX_ACTIONS, 4, dtype=torch.float)
    action_mask = torch.zeros(1, T, MAX_ACTIONS, dtype=torch.float)
    step_mask = torch.ones(1, T, dtype=torch.float)
    for t, enc in enumerate(steps):
        board_ids[0, t] = torch.tensor(enc["board_ids"], dtype=torch.long)
        h = enc["hand_ids"][:20]
        hand_ids[0, t, :len(h)] = torch.tensor(h, dtype=torch.long)
        dcd = enc["discard_ids"][:20]
        discard_ids[0, t, :len(dcd)] = torch.tensor(dcd, dtype=torch.long)
        numeric[0, t] = torch.tensor(enc["numeric"], dtype=torch.float)
        phi4[0, t] = torch.tensor(enc["phi4"], dtype=torch.float)
        for j, a in enumerate(enc["actions"][:MAX_ACTIONS]):
            action_type[0, t, j] = a["type"]
            action_card[0, t, j] = a["card_id"]
            action_attack[0, t, j] = a["attack_id"]
            action_numeric[0, t, j] = torch.tensor(a["numeric"], dtype=torch.float)
            action_mask[0, t, j] = 1.0
    return (board_ids, hand_ids, discard_ids, numeric, action_type, action_card,
            action_attack, action_numeric, action_mask, phi4, step_mask)


def agent(obs_dict: dict) -> list:
    global _history
    sel = obs_dict.get("select")
    if sel is None:
        return DECK
    n = len(sel.get("option", []))
    if n == 0:
        return []
    try:
        net = _load()
        enc = encode_sample_seq(obs_dict, sel)
        _history.append(enc)
        steps = _history[-_MAX_LEN:]
        tensors = _build_tensors(steps)
        with torch.no_grad():
            logits, _ = net(*tensors)
        last_logits = logits[0, -1, :n]
        mx = sel.get("maxCount", 1) or 1
        k = min(mx, n)
        top = torch.topk(last_logits, k=k).indices.tolist()
        return clamp(top, sel)
    except Exception:
        mn = sel.get("minCount", 1) or 0
        return clamp(list(range(max(mn, 1))), sel)
