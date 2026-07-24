"""ACAN — Action-Conditioned Advantage Net (S3, pre-registered 2026-07-23).

The architecture axis this project never varied. Every net we have ever trained
(BC, DAgger, AWR, DMC, sequence-policy, Phi v4) scores either a FIXED action-slot
vector or the BOARD ALONE. This one embeds each *candidate action* and scores it
against the state, one scalar per (state, action) -- the pattern the official
sample and the strongest public nets use.

What it learns: the ADVANTAGE of each candidate over the heuristic's own pick, as
measured by the shipped d2/formula belief-determinized search (the 776 agent).

Why it is not another entry in the graveyard: every closed method distilled the
HEURISTIC and is therefore capped below it by construction. This distills the
SEARCH, which is measurably above the heuristic on the live ladder (776 vs 673).

Target handling (the make-or-break detail): raw advantages span [-5e3, +1e7]
because the leaf formula uses terminal sentinels. Plain MSE would be owned by a
handful of terminal decisions; log-squashing would compress exactly the near-
MARGIN region the override decision turns on. So targets are CLIPPED to a
margin-scaled window and linearly normalised to [-1, 1] -- full resolution where
the decision actually happens, saturation for "clearly winning".
"""
import torch
import torch.nn as nn

N_TYPE = 32          # option type ids are small ints; +1 shift for the -1 sentinel
N_AREA = 16          # inPlayArea: ACTIVE=4, BENCH=5, ... (+1 shift)
N_IDX = 12           # inPlayIndex: bench slot (+1 shift)
D_TYPE, D_CARD, D_ATK, D_AREA, D_IDX = 16, 32, 16, 8, 8
D_ACT = 64
# Continuous per-action features. These carry the target's ZERO POINT: the label
# is the advantage over the HEURISTIC's pick, so without knowing which candidate
# that is (and by how much the heuristic prefers it) the net cannot know what
# "advantage 0" means for this decision, and collapses to predicting the mean.
# Both are available at runtime -- we always score the heuristic first.
N_ACT_FLOAT = 2      # [is_heur_top, (base[i] - base[heur_top]) / base_scale]


def act_index(desc, card_vocab, atk_vocab):
    """[type, src_card, attackId, inPlayArea, inPlayIndex, target_card]
    -> embedding indices (0 = unknown/absent).

    The last three are what make candidates distinguishable: two options that
    play the same card to different targets share the first three fields.
    """
    t, c, a, ar, ix, tg = desc

    def _clamp(v, n):
        v = int(v) + 1
        return v if 0 <= v < n else 0

    return [_clamp(t, N_TYPE), card_vocab.get(int(c), 0), atk_vocab.get(int(a), 0),
            _clamp(ar, N_AREA), _clamp(ix, N_IDX), card_vocab.get(int(tg), 0)]


class ACAN(nn.Module):
    def __init__(self, n_state, n_card, n_atk, hidden=128):
        super().__init__()
        self.state = nn.Sequential(
            nn.Linear(n_state, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.type_emb = nn.Embedding(N_TYPE, D_TYPE)
        # one card table shared by the SOURCE card and the TARGET Pokemon --
        # same vocabulary, and sharing is more sample-efficient
        self.card_emb = nn.Embedding(n_card, D_CARD)
        self.atk_emb = nn.Embedding(n_atk, D_ATK)
        self.area_emb = nn.Embedding(N_AREA, D_AREA)
        self.idx_emb = nn.Embedding(N_IDX, D_IDX)
        self.act = nn.Sequential(
            nn.Linear(D_TYPE + 2 * D_CARD + D_ATK + D_AREA + D_IDX + N_ACT_FLOAT,
                      D_ACT), nn.ReLU(),
        )
        # multiplicative state x action interaction -- lets the net say "this card
        # is good HERE", which a concat-only net can only express weakly
        self.s_proj = nn.Linear(hidden, D_ACT)
        self.head = nn.Sequential(
            nn.Linear(hidden + D_ACT + D_ACT, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, s, a, af):
        """s: [B, n_state], a: [B, 6] long, af: [B, 2] -> [B] advantage in [-1, 1]."""
        hs = self.state(s)
        ha = self.act(torch.cat([self.type_emb(a[:, 0]),
                                 self.card_emb(a[:, 1]),
                                 self.atk_emb(a[:, 2]),
                                 self.area_emb(a[:, 3]),
                                 self.idx_emb(a[:, 4]),
                                 self.card_emb(a[:, 5]), af], dim=-1))
        inter = self.s_proj(hs) * ha
        out = self.head(torch.cat([hs, ha, inter], dim=-1)).squeeze(-1)
        return torch.tanh(out)
