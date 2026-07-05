"""Shared model-loading + encoding glue for net_agent.py / selfplay_agent.py."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO_ROOT)

import torch

from encode import encode_sample, MAX_ACTIONS
from model import PTCGNet

DEVICE = torch.device("cpu")
_models = {}


def load_model(ckpt_path):
    """Cache by path so a process that loads two different checkpoints (e.g.
    self-play mirror vs a frozen teacher) doesn't reload on every call."""
    m = _models.get(ckpt_path)
    if m is None:
        m = PTCGNet()
        state = torch.load(ckpt_path, map_location=DEVICE)
        # older checkpoints' value_head was narrower (no oracle features) —
        # drop only the mismatched keys rather than error; a fresh value head
        # still gives usable (if uncalibrated) values, and the policy path
        # (unaffected by the oracle widening) loads and behaves unchanged.
        own = m.state_dict()
        state = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
        m.load_state_dict(state, strict=False)
        m.eval()
        _models[ckpt_path] = m
    return m


def encode_batch(obs, sel):
    enc = encode_sample(obs, sel)
    n = enc["n_actions"]
    board_ids = torch.tensor([enc["board_ids"]], dtype=torch.long)
    h = enc["hand_ids"][:20]
    hand_ids = torch.zeros(1, 20, dtype=torch.long)
    hand_ids[0, :len(h)] = torch.tensor(h, dtype=torch.long)
    d = enc["discard_ids"][:20]
    discard_ids = torch.zeros(1, 20, dtype=torch.long)
    discard_ids[0, :len(d)] = torch.tensor(d, dtype=torch.long)
    numeric = torch.tensor([enc["numeric"]], dtype=torch.float)

    action_type = torch.zeros(1, MAX_ACTIONS, dtype=torch.long)
    action_card = torch.zeros(1, MAX_ACTIONS, dtype=torch.long)
    action_attack = torch.zeros(1, MAX_ACTIONS, dtype=torch.long)
    action_numeric = torch.zeros(1, MAX_ACTIONS, 4, dtype=torch.float)
    action_mask = torch.zeros(1, MAX_ACTIONS, dtype=torch.float)
    for j, a in enumerate(enc["actions"][:MAX_ACTIONS]):
        action_type[0, j] = a["type"]
        action_card[0, j] = a["card_id"]
        action_attack[0, j] = a["attack_id"]
        action_numeric[0, j] = torch.tensor(a["numeric"], dtype=torch.float)
        action_mask[0, j] = 1.0
    return (board_ids, hand_ids, discard_ids, numeric, action_type, action_card,
            action_attack, action_numeric, action_mask), n


def clamp(indices, sel):
    mn = sel.get("minCount", 0) or 0
    mx = sel.get("maxCount", 1) or 1
    n = len(sel.get("option", []))
    out = []
    for i in indices:
        if 0 <= i < n and i not in out:
            out.append(i)
        if len(out) >= mx:
            break
    i = 0
    while len(out) < mn and i < n:
        if i not in out:
            out.append(i)
        i += 1
    return out if out else ([0] if n > 0 else [])


def value_estimate(model, obs, sel):
    """Run just the value head for one decision point (used for n-step
    bootstrapping during self-play data collection)."""
    batch, n_actions = encode_batch(obs, sel)
    with torch.no_grad():
        _, value = model(*batch)
    return value.item()
