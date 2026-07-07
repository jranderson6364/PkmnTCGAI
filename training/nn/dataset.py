"""Turns training/bc_data*.pkl(.gz) shards into batched tensors for PTCGNet.

BC value target = game outcome (+1/-1/0), the simplest valid target for the
imitation warmup. n-step bootstrapped value targets (docs/nn-training.md
§Value Targets) apply to the self-play phase, not this warmup.
"""
import glob
import gzip
import pickle
import random

import torch
from torch.utils.data import Dataset

from encode import encode_sample, MAX_ACTIONS, CARD_VOCAB, NUM_FEATS

# probability of zeroing an available oracle in collate() (regularizes the
# value head so it doesn't collapse onto always expecting the oracle feature
# at inference, where main.py never supplies one). Eval scripts that want the
# pure oracle/no-oracle path unconditionally can override: dataset.ORACLE_DROPOUT = 0
ORACLE_DROPOUT = 0.25


def _opener(path):
    return gzip.open if path.endswith(".gz") else open


def load_shards(pattern, limit=None):
    """pattern e.g. '/kaggle/input/**/bc_data*.pkl' (recursive — Kaggle's exact
    mount subdirectory name can differ from the dataset slug). Comma-separate
    multiple patterns to combine sources, e.g. BC + DAgger-round shards.

    `limit`, if given, stops reading further shards once enough samples are
    loaded — capping AFTER a full glob-matched load (e.g. slicing the return
    value) still momentarily materializes every shard at once, which is what
    actually exhausted RAM (~37GB transient peak) even with a small final
    sample count; this avoids ever reading past the limit."""
    paths = []
    for part in pattern.split(","):
        paths.extend(glob.glob(part.strip(), recursive=True))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"no shards matched {pattern}")
    if limit:
        # shuffle which shards get read (not the samples within — that would
        # need everything in memory first, the exact problem `limit` avoids)
        random.Random(0).shuffle(paths)
    samples = []
    for p in paths:
        with _opener(p)(p, "rb") as f:
            samples.extend(pickle.load(f))
        if limit and len(samples) >= limit:
            break
    if limit:
        samples = samples[:limit]
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
        # advantage = value_target - v_pred (Stage 2 AWR, docs/nn-training.md
        # §3): only self-play samples carry v_pred (the collecting net's own
        # value-head estimate at s_t); BC samples have none, so advantage is
        # None there and train_sp.py falls back to a flat policy-loss weight.
        advantage = (value - d["v_pred"]) if "v_pred" in d else None
        # oracle: privileged opponent-hand card ids (oracle-critic collections
        # only; absent on plain BC/DAgger/SP samples → no-oracle training).
        opp_hand = d.get("opp_hand") or []
        opp_hand = [min(c, CARD_VOCAB - 1) for c in opp_hand if c]
        return enc, label, float(value or 0), policy_target, advantage, opp_hand


def collate(batch):
    B = len(batch)
    board_ids = torch.zeros(B, 13, dtype=torch.long)
    hand_ids = torch.zeros(B, 20, dtype=torch.long)
    discard_ids = torch.zeros(B, 20, dtype=torch.long)
    numeric = torch.zeros(B, NUM_FEATS, dtype=torch.float)
    action_type = torch.zeros(B, MAX_ACTIONS, dtype=torch.long)
    action_card = torch.zeros(B, MAX_ACTIONS, dtype=torch.long)
    action_attack = torch.zeros(B, MAX_ACTIONS, dtype=torch.long)
    action_numeric = torch.zeros(B, MAX_ACTIONS, 4, dtype=torch.float)
    action_mask = torch.zeros(B, MAX_ACTIONS, dtype=torch.float)
    labels = torch.zeros(B, dtype=torch.long)
    values = torch.zeros(B, dtype=torch.float)
    policy_targets = torch.zeros(B, MAX_ACTIONS, dtype=torch.float)
    advantages = torch.zeros(B, dtype=torch.float)
    has_advantage = torch.zeros(B, dtype=torch.float)
    oracle_ids = torch.zeros(B, 20, dtype=torch.long)
    oracle_flag = torch.zeros(B, 1, dtype=torch.float)

    for i, (enc, label, outcome, policy_target, advantage, opp_hand) in enumerate(batch):
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
        if advantage is not None:
            advantages[i] = advantage
            has_advantage[i] = 1.0
        oh = opp_hand[:20]
        if oh and random.random() >= ORACLE_DROPOUT:
            oracle_ids[i, :len(oh)] = torch.tensor(oh, dtype=torch.long)
            oracle_flag[i, 0] = 1.0

    return {
        "board_ids": board_ids, "hand_ids": hand_ids, "discard_ids": discard_ids,
        "numeric": numeric, "action_type": action_type, "action_card": action_card,
        "action_attack": action_attack, "action_numeric": action_numeric,
        "action_mask": action_mask, "labels": labels, "values": values,
        "policy_targets": policy_targets,
        "advantages": advantages, "has_advantage": has_advantage,
        "oracle_ids": oracle_ids, "oracle_flag": oracle_flag,
    }
