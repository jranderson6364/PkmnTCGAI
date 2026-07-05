"""BC/self-play actor-critic net. See encode.py for the feature design and
docs/nn-training.md for the architecture rationale. Small on purpose: this is
a warm-start policy, not the final word — self-play (Phase 1+) can grow it.
"""
import torch
import torch.nn as nn

from encode import (
    CARD_VOCAB, ATTACK_VOCAB, OPTION_TYPE_VOCAB, N_BOARD_SLOTS, NUM_FEATS,
)

D_CARD = 128
D_ATTACK = 64
D_TYPE = 32
D_MODEL = 128


class PTCGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.card_embed = nn.Embedding(CARD_VOCAB, D_CARD, padding_idx=0)
        self.attack_embed = nn.Embedding(ATTACK_VOCAB, D_ATTACK, padding_idx=0)
        self.type_embed = nn.Embedding(OPTION_TYPE_VOCAB, D_TYPE)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_CARD, nhead=2, dim_feedforward=256, batch_first=True)
        self.board_transformer = nn.TransformerEncoder(enc_layer, num_layers=1)

        self.hand_bag = nn.EmbeddingBag(CARD_VOCAB, D_CARD, mode="sum", padding_idx=0)
        self.discard_bag = nn.EmbeddingBag(CARD_VOCAB, D_CARD, mode="sum", padding_idx=0)

        self.numeric_proj = nn.Sequential(nn.Linear(NUM_FEATS, D_CARD), nn.ReLU())
        self.trunk = nn.Sequential(
            nn.Linear(D_CARD * 4, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        # oracle: privileged opponent-hand feature for the value head only
        # (PerfectDou-style critic — see docs/report-log.md 2026-07-04 lit review).
        # Not available at inference (main.py never sees the opponent's hand), so
        # it's zeroed + flagged off whenever the caller doesn't supply it.
        self.oracle_embed = nn.EmbeddingBag(CARD_VOCAB, D_CARD, mode="sum", padding_idx=0)
        self.value_head = nn.Linear(256 + D_CARD + 1, 1)

        act_in_dim = D_TYPE + D_CARD + D_ATTACK + 4
        self.action_mlp = nn.Sequential(nn.Linear(act_in_dim, D_MODEL), nn.ReLU())
        self.logit_mlp = nn.Sequential(
            nn.Linear(D_MODEL + 256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, board_ids, hand_ids, discard_ids, numeric,
                action_type, action_card, action_attack, action_numeric, action_mask,
                oracle_ids=None, oracle_offsets=None, oracle_flag=None):
        board_emb = self.card_embed(board_ids)                  # (B,13,128)
        board_ctx = self.board_transformer(board_emb)            # (B,13,128)
        board_vec = board_ctx.mean(dim=1)                        # (B,128)
        hand_vec = self.hand_bag(hand_ids)                       # (B,128)
        discard_vec = self.discard_bag(discard_ids)              # (B,128)
        feat_vec = self.numeric_proj(numeric)                    # (B,128)

        trunk_in = torch.cat([board_vec, hand_vec, discard_vec, feat_vec], dim=-1)
        trunk = self.trunk(trunk_in)                             # (B,256)
        B = trunk.shape[0]
        if oracle_ids is not None:
            oracle_vec = self.oracle_embed(oracle_ids)            # (B,128)
        else:
            oracle_vec = torch.zeros(B, D_CARD, device=trunk.device, dtype=trunk.dtype)
        if oracle_flag is None:
            oracle_flag = torch.zeros(B, 1, device=trunk.device, dtype=trunk.dtype)
        value = torch.tanh(self.value_head(
            torch.cat([trunk, oracle_vec, oracle_flag], dim=-1)).squeeze(-1))  # (B,)

        B, A = action_type.shape
        type_emb = self.type_embed(action_type)                  # (B,A,32)
        card_emb = self.card_embed(action_card)                  # (B,A,128)
        attack_emb = self.attack_embed(action_attack)             # (B,A,64)
        act_in = torch.cat([type_emb, card_emb, attack_emb, action_numeric], dim=-1)
        act_vec = self.action_mlp(act_in)                         # (B,A,128)

        trunk_exp = trunk.unsqueeze(1).expand(-1, A, -1)          # (B,A,256)
        logits = self.logit_mlp(torch.cat([act_vec, trunk_exp], dim=-1)).squeeze(-1)  # (B,A)
        logits = logits.masked_fill(action_mask == 0, -1e9)
        return logits, value
