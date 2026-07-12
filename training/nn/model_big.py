"""2x-wider variant of model.PTCGNet, for testing whether DMC's ~5-7%
win-rate plateau (docs/report-log.md 2026-07-09/10 "DMC round 4 at real
scale") is a capacity ceiling — the one lever this project's extensive
DMC testing never varied. Deliberately a SEPARATE file/class (not a
parameterized PTCGNet) so existing checkpoints and consumers of the
original architecture are completely unaffected.
"""
import torch
import torch.nn as nn

from encode import CARD_VOCAB, ATTACK_VOCAB, OPTION_TYPE_VOCAB, NUM_FEATS

D_CARD = 256
D_ATTACK = 128
D_TYPE = 64
D_MODEL = 256
D_TRUNK = 512


class PTCGNetBig(nn.Module):
    def __init__(self):
        super().__init__()
        self.card_embed = nn.Embedding(CARD_VOCAB, D_CARD, padding_idx=0)
        self.attack_embed = nn.Embedding(ATTACK_VOCAB, D_ATTACK, padding_idx=0)
        self.type_embed = nn.Embedding(OPTION_TYPE_VOCAB, D_TYPE)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_CARD, nhead=4, dim_feedforward=512, batch_first=True)
        self.board_transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        self.hand_bag = nn.EmbeddingBag(CARD_VOCAB, D_CARD, mode="sum", padding_idx=0)
        self.discard_bag = nn.EmbeddingBag(CARD_VOCAB, D_CARD, mode="sum", padding_idx=0)

        self.numeric_proj = nn.Sequential(nn.Linear(NUM_FEATS, D_CARD), nn.ReLU())
        self.trunk = nn.Sequential(
            nn.Linear(D_CARD * 4, D_TRUNK), nn.ReLU(),
            nn.Linear(D_TRUNK, D_TRUNK), nn.ReLU(),
        )
        self.oracle_embed = nn.EmbeddingBag(CARD_VOCAB, D_CARD, mode="sum", padding_idx=0)
        self.value_head = nn.Linear(D_TRUNK + D_CARD + 1, 1)

        act_in_dim = D_TYPE + D_CARD + D_ATTACK + 4
        self.action_mlp = nn.Sequential(nn.Linear(act_in_dim, D_MODEL), nn.ReLU())
        self.logit_mlp = nn.Sequential(
            nn.Linear(D_MODEL + D_TRUNK, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, board_ids, hand_ids, discard_ids, numeric,
                action_type, action_card, action_attack, action_numeric, action_mask,
                oracle_ids=None, oracle_offsets=None, oracle_flag=None):
        board_emb = self.card_embed(board_ids)
        board_ctx = self.board_transformer(board_emb)
        board_vec = board_ctx.mean(dim=1)
        hand_vec = self.hand_bag(hand_ids)
        discard_vec = self.discard_bag(discard_ids)
        feat_vec = self.numeric_proj(numeric)

        trunk_in = torch.cat([board_vec, hand_vec, discard_vec, feat_vec], dim=-1)
        trunk = self.trunk(trunk_in)
        B = trunk.shape[0]
        if oracle_ids is not None:
            oracle_vec = self.oracle_embed(oracle_ids)
        else:
            oracle_vec = torch.zeros(B, D_CARD, device=trunk.device, dtype=trunk.dtype)
        if oracle_flag is None:
            oracle_flag = torch.zeros(B, 1, device=trunk.device, dtype=trunk.dtype)
        value = torch.tanh(self.value_head(
            torch.cat([trunk, oracle_vec, oracle_flag], dim=-1)).squeeze(-1))

        B, A = action_type.shape
        type_emb = self.type_embed(action_type)
        card_emb = self.card_embed(action_card)
        attack_emb = self.attack_embed(action_attack)
        act_in = torch.cat([type_emb, card_emb, attack_emb, action_numeric], dim=-1)
        act_vec = self.action_mlp(act_in)

        trunk_exp = trunk.unsqueeze(1).expand(-1, A, -1)
        logits = self.logit_mlp(torch.cat([act_vec, trunk_exp], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(action_mask == 0, -1e9)
        return logits, value
