"""Sequence-policy net for the capacity-vs-information experiment
(docs/next-session-plan.md). Reuses PTCGNet's per-step state/action encoders
verbatim (same embeddings, same trunk) so the only new capacity is a causal
transformer mixing trunk vectors ACROSS a game's decision history before the
per-step action-scoring head runs — isolates "does history context help"
from "is the per-step encoder any different."

use_history=False collapses this to the existing PTCGNet with the phi4
side-channel added (used as the "no history, same capacity-ish" sanity
check); use_phi4=False drops the Φ v4 features (control (b) in the plan).
"""
import torch
import torch.nn as nn

from encode import CARD_VOCAB, ATTACK_VOCAB, OPTION_TYPE_VOCAB, NUM_FEATS
from encode_seq import PHI4_DIM

D_CARD = 128
D_ATTACK = 64
D_TYPE = 32
D_MODEL = 128
D_TRUNK = 256


class SeqPTCGNet(nn.Module):
    def __init__(self, n_layers=2, n_heads=4, use_history=True, use_phi4=True):
        super().__init__()
        self.use_history = use_history
        self.use_phi4 = use_phi4
        self.card_embed = nn.Embedding(CARD_VOCAB, D_CARD, padding_idx=0)
        self.attack_embed = nn.Embedding(ATTACK_VOCAB, D_ATTACK, padding_idx=0)
        self.type_embed = nn.Embedding(OPTION_TYPE_VOCAB, D_TYPE)

        board_layer = nn.TransformerEncoderLayer(
            d_model=D_CARD, nhead=2, dim_feedforward=256, batch_first=True)
        self.board_transformer = nn.TransformerEncoder(board_layer, num_layers=1)

        self.hand_bag = nn.EmbeddingBag(CARD_VOCAB, D_CARD, mode="sum", padding_idx=0)
        self.discard_bag = nn.EmbeddingBag(CARD_VOCAB, D_CARD, mode="sum", padding_idx=0)

        numeric_in = NUM_FEATS + (PHI4_DIM if use_phi4 else 0)
        self.numeric_proj = nn.Sequential(nn.Linear(numeric_in, D_CARD), nn.ReLU())
        self.trunk = nn.Sequential(
            nn.Linear(D_CARD * 4, D_TRUNK), nn.ReLU(),
            nn.Linear(D_TRUNK, D_TRUNK), nn.ReLU(),
        )

        if use_history:
            hist_layer = nn.TransformerEncoderLayer(
                d_model=D_TRUNK, nhead=n_heads, dim_feedforward=D_TRUNK * 4,
                batch_first=True)
            self.history_transformer = nn.TransformerEncoder(hist_layer, num_layers=n_layers)
            self.pos_embed = nn.Embedding(256, D_TRUNK)  # max ~150 decisions/game, margin to 256

        self.value_head = nn.Linear(D_TRUNK, 1)

        act_in_dim = D_TYPE + D_CARD + D_ATTACK + 4
        self.action_mlp = nn.Sequential(nn.Linear(act_in_dim, D_MODEL), nn.ReLU())
        self.logit_mlp = nn.Sequential(
            nn.Linear(D_MODEL + D_TRUNK, 128), nn.ReLU(), nn.Linear(128, 1))

    def encode_step(self, board_ids, hand_ids, discard_ids, numeric, phi4=None):
        """Per-step trunk vector, batched over (B, T, ...) or (B, ...) — flattens
        leading dims, runs the state encoder once, reshapes back."""
        lead_shape = board_ids.shape[:-1]  # (...,) before the 13 board slots
        B = board_ids.numel() // 13
        board_ids = board_ids.reshape(B, 13)
        hand_ids = hand_ids.reshape(B, -1)
        discard_ids = discard_ids.reshape(B, -1)
        numeric = numeric.reshape(B, -1)

        board_emb = self.card_embed(board_ids)
        board_ctx = self.board_transformer(board_emb)
        board_vec = board_ctx.mean(dim=1)
        hand_vec = self.hand_bag(hand_ids)
        discard_vec = self.discard_bag(discard_ids)
        if self.use_phi4:
            phi4 = phi4.reshape(B, -1)
            numeric = torch.cat([numeric, phi4], dim=-1)
        feat_vec = self.numeric_proj(numeric)

        trunk_in = torch.cat([board_vec, hand_vec, discard_vec, feat_vec], dim=-1)
        trunk = self.trunk(trunk_in)  # (B, D_TRUNK)
        return trunk.reshape(*lead_shape, D_TRUNK)

    def forward(self, board_ids, hand_ids, discard_ids, numeric,
                action_type, action_card, action_attack, action_numeric, action_mask,
                phi4=None, step_mask=None):
        """Sequence-shaped inputs: board_ids (B,T,13), hand_ids (B,T,20),
        numeric (B,T,NUM_FEATS), phi4 (B,T,11), action_* (B,T,A,...),
        step_mask (B,T) 1 where a real decision exists (0 = padding).
        Also accepts non-sequence (B,...) inputs when T is absent — degrades
        to per-state scoring (T=1) for reuse in the plain-MLP control path."""
        if board_ids.dim() == 2:  # (B,13) — no time dim, add one
            board_ids = board_ids.unsqueeze(1)
            hand_ids = hand_ids.unsqueeze(1)
            discard_ids = discard_ids.unsqueeze(1)
            numeric = numeric.unsqueeze(1)
            action_type = action_type.unsqueeze(1)
            action_card = action_card.unsqueeze(1)
            action_attack = action_attack.unsqueeze(1)
            action_numeric = action_numeric.unsqueeze(1)
            action_mask = action_mask.unsqueeze(1)
            if phi4 is not None:
                phi4 = phi4.unsqueeze(1)
            if step_mask is not None:
                step_mask = step_mask.unsqueeze(1)

        B, T = board_ids.shape[0], board_ids.shape[1]
        trunk = self.encode_step(board_ids, hand_ids, discard_ids, numeric, phi4)  # (B,T,D_TRUNK)

        if self.use_history:
            pos_ids = torch.arange(T, device=trunk.device).unsqueeze(0).expand(B, T)
            trunk_pos = trunk + self.pos_embed(pos_ids)
            causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=trunk.device), diagonal=1)
            key_padding_mask = None
            if step_mask is not None:
                key_padding_mask = (step_mask == 0)  # (B,T) True = ignore
            trunk = self.history_transformer(
                trunk_pos, mask=causal_mask, src_key_padding_mask=key_padding_mask)  # (B,T,D_TRUNK)

        value = torch.tanh(self.value_head(trunk).squeeze(-1))  # (B,T)

        A = action_type.shape[2]
        type_emb = self.type_embed(action_type)
        card_emb = self.card_embed(action_card)
        attack_emb = self.attack_embed(action_attack)
        act_in = torch.cat([type_emb, card_emb, attack_emb, action_numeric], dim=-1)
        act_vec = self.action_mlp(act_in)  # (B,T,A,D_MODEL)

        trunk_exp = trunk.unsqueeze(2).expand(-1, -1, A, -1)  # (B,T,A,D_TRUNK)
        logits = self.logit_mlp(torch.cat([act_vec, trunk_exp], dim=-1)).squeeze(-1)  # (B,T,A)
        logits = logits.masked_fill(action_mask == 0, -1e9)
        return logits, value
