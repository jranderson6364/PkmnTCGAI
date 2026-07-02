"""Turns training/bc_data*.pkl(.gz) shards into batched tensors for PTCGNet.

BC value target = game outcome (+1/-1/0), the simplest valid target for the
imitation warmup. n-step bootstrapped value targets (docs/nn-training.md
§Value Targets) apply to the self-play phase, not this warmup.
"""
import glob
import gzip
import pickle

import torch
from torch.utils.data import Dataset

from encode import encode_sample, MAX_ACTIONS


def _opener(path):
    return gzip.open if path.endswith(".gz") else open


def load_shards(pattern):
    """pattern e.g. '/kaggle/input/**/bc_data*.pkl' (recursive — Kaggle's exact
    mount subdirectory name can differ from the dataset slug)."""
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(f"no shards matched {pattern}")
    samples = []
    for p in paths:
        with _opener(p)(p, "rb") as f:
            samples.extend(pickle.load(f))
    return samples


class BCDataset(Dataset):
    """Wraps raw {obs, action, outcome} samples; encodes lazily in __getitem__
    (cheap — pure python dict indexing, no torch ops until collate)."""

    def __init__(self, raw_samples):
        self.raw = raw_samples

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, i):
        d = self.raw[i]
        obs, action = d["obs"], d["action"]
        # SP samples carry an n-step bootstrapped value_target (see
        # selfplay_collect.py); plain BC samples fall back to terminal outcome.
        value = d.get("value_target", d.get("outcome", 0))
        sel = obs.get("select")
        enc = encode_sample(obs, sel)
        n_actions = enc["n_actions"]
        label = action[0] if action else 0
        label = min(label, n_actions - 1) if n_actions else 0
        # MCTS samples (future: mcts_collect.py) carry a soft policy_target —
        # normalized root visit counts, aligned to sel['option']. BC/direct-SP
        # samples have none; collate() falls back to a one-hot of `label`.
        policy_target = d.get("policy_target")
        if policy_target is not None:
            policy_target = list(policy_target[:n_actions]) if n_actions else []
        return enc, label, float(value or 0), policy_target


def collate(batch):
    B = len(batch)
    board_ids = torch.zeros(B, 13, dtype=torch.long)
    hand_ids = torch.zeros(B, 20, dtype=torch.long)
    discard_ids = torch.zeros(B, 20, dtype=torch.long)
    numeric = torch.zeros(B, 13, dtype=torch.float)
    action_type = torch.zeros(B, MAX_ACTIONS, dtype=torch.long)
    action_card = torch.zeros(B, MAX_ACTIONS, dtype=torch.long)
    action_attack = torch.zeros(B, MAX_ACTIONS, dtype=torch.long)
    action_numeric = torch.zeros(B, MAX_ACTIONS, 4, dtype=torch.float)
    action_mask = torch.zeros(B, MAX_ACTIONS, dtype=torch.float)
    labels = torch.zeros(B, dtype=torch.long)
    values = torch.zeros(B, dtype=torch.float)
    policy_targets = torch.zeros(B, MAX_ACTIONS, dtype=torch.float)

    for i, (enc, label, outcome, policy_target) in enumerate(batch):
        board_ids[i] = torch.tensor(enc["board_ids"], dtype=torch.long)
        h = enc["hand_ids"][:20]
        hand_ids[i, :len(h)] = torch.tensor(h, dtype=torch.long)
        dcd = enc["discard_ids"][:20]
        discard_ids[i, :len(dcd)] = torch.tensor(dcd, dtype=torch.long)
        numeric[i] = torch.tensor(enc["numeric"], dtype=torch.float)
        n = enc["n_actions"]
        for j, a in enumerate(enc["actions"][:MAX_ACTIONS]):
            action_type[i, j] = a["type"]
            action_card[i, j] = a["card_id"]
            action_attack[i, j] = a["attack_id"]
            action_numeric[i, j] = torch.tensor(a["numeric"], dtype=torch.float)
            action_mask[i, j] = 1.0
        labels[i] = label
        values[i] = outcome
        if policy_target and len(policy_target) == n:
            policy_targets[i, :n] = torch.tensor(policy_target, dtype=torch.float)
        elif n > 0:
            policy_targets[i, label] = 1.0  # one-hot fallback (BC / direct-SP samples)

    return {
        "board_ids": board_ids, "hand_ids": hand_ids, "discard_ids": discard_ids,
        "numeric": numeric, "action_type": action_type, "action_card": action_card,
        "action_attack": action_attack, "action_numeric": action_numeric,
        "action_mask": action_mask, "labels": labels, "values": values,
        "policy_targets": policy_targets,
    }
