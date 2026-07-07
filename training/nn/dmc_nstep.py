"""Phase 0 items 1+2 (docs/nn-training.md Phase 0): n-step bootstrapped value
targets, and an orthogonal potential-based Φ-shaping arm, for DMC-style
(s, taken_action, outcome) samples -- alternatives/additions to
train_dmc.py's current full-episode Monte Carlo target.

DMC treats the model's per-action logit as Q(s,a) (see train_dmc.py), so the
bootstrap value at a future state is max_a Q(s,a) at that state -- NOT the
separate value_head output (untrained/uncalibrated for DMC checkpoints,
since train_dmc.py never regresses it).

n-step alone (use_phi_shaping=False): pure n-step TD bootstrapping, no
intermediate reward --

    target_t = Q_max(s_{t+n})   if t+n is still within the game
             = outcome           if t+n reaches or passes the terminal decision

n_step=None (or >= the game's decision count) reduces exactly to the
existing full-Monte-Carlo target (every decision gets the flat game
outcome) -- this is the n="full" arm of the sweep and the current
train_dmc.py behavior, so this module is a strict superset, not a
replacement.

Φ-shaping (use_phi_shaping=True, orthogonal to n_step -- can combine with
any n_step value, including None/full-MC): adds the potential-based shaping
reward F_k = phi_gamma*Φ(s_{k+1}) - Φ(s_k) at every step along the
accumulation window (Ng/Harada/Russell 1999 -- provably policy-invariant,
i.e. doesn't change which action is optimal, only rescales/recenters the
absolute Q values). Φ is `phi_baseline.phi()`, the SAME fixed, never-fit
potential function already gated against 1356 real replays (ALL sign_acc
0.563, LATE 0.606 -- docs/report-log.md 2026-07-05 "Φ-only real-replay
baseline" entry) -- reused rather than reimplemented so the same validated
function is what both diagnoses the honest gate baseline AND drives
training. Φ(terminal) is defined as 0 by convention. Combined with n_step,
this is exactly n-step TD with potential-based shaping:

    G_t^(n) = sum_{k=t}^{end-1} phi_gamma^(k-t) * F_k
              + phi_gamma^(end-t) * [Q_max(s_end) if end<m else outcome]

which gives every decision in a game its OWN target shaped by how much
closer to winning that specific decision's follow-up state got, instead of
every decision in the game sharing one identical sparse ±1 label -- the
credit-assignment fix the Phase 0 diagnosis is about. Clipped to [-1, 1]
since combined with shaping the raw sum can exceed the Q-head's tanh-free
but Huber-loss-fit range.

CONFIRMED NEGATIVE RESULT (2026-07-05, see docs/report-log.md "Phase 0
ablation grid" + its root-cause follow-up entry): use_phi_shaping=True
trains the net toward target_t ~= outcome - Phi(s_t) (the full-MC telescoped
form), which is NOT comparable in sign to the true outcome across different
states -- Ng/Harada/Russell's policy-invariance guarantee is about
preserving the optimal ACTION RANKING WITHIN one state (Phi(s_t) is a
constant term across actions from that state), not about the shaped value's
sign meaning "win" vs "loss" relative to an external label. Measured: 11.5%
of training labels flip sign vs the true game outcome; that rate nearly
DOUBLES (21.1%) on real ladder replay states vs the self-play training
distribution, so a net that fits this target well in-distribution is
mechanically MORE wrong on real-replay sign-accuracy specifically, not just
noisier. Do not evaluate a Phi-shaped checkpoint with a cross-state sign-
accuracy gate; do not re-enable use_phi_shaping for training without first
redesigning how Phi is consumed (e.g. as a fixed leaf-eval/warm-start prior
kept OUTSIDE the regression target, not folded into it).

Must be called with each game's decisions in their real per-game, per-seat
order (exactly what bc_collect.extract_decisions returns) -- BEFORE they get
concatenated into a flat cross-game samples list, since game boundaries
can't be recovered after that point.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch  # noqa: E402

from net_common import load_model, encode_batch  # noqa: E402
from phi_baseline import phi as phi_state  # noqa: E402

PHI_GAMMA = 0.997  # matches selfplay_collect.py's GAMMA for consistency


def q_max_value(model, obs, sel):
    """max_a Q(s,a) at one decision point, using the DMC convention (action
    logit AS Q), not the model's separate value_head."""
    batch, n_actions = encode_batch(obs, sel)
    with torch.no_grad():
        logits, _ = model(*batch)
    return float(logits[0, :n_actions].max().item())


def _phi_at(d):
    """Reads the acting seat straight from the sample's own obs
    ('current'.'yourIndex') rather than assuming 0 -- extract_decisions()
    does NOT re-perspective samples to a fixed seat, so a learner playing as
    seat 1 in a given game has yourIndex=1 in every one of its samples."""
    obs = d.get("obs")
    cur = (obs or {}).get("current") or {}
    me_idx = cur.get("yourIndex")
    if me_idx is None:
        return 0.0
    try:
        v = phi_state(cur, me_idx)
    except Exception:
        v = None
    return v if v is not None else 0.0


def compute_nstep_targets(bootstrap_ckpt, decisions, outcome, n_step=None,
                           use_phi_shaping=False, phi_gamma=PHI_GAMMA):
    """decisions: ordered list of {'obs':..., 'action':...} for ONE game, ONE
    seat, in real play order (each obs's own 'current'.'yourIndex' identifies
    which seat -- extract_decisions does NOT renormalize this to 0/1). Adds
    'outcome' (the training target, in place -- keeps train_dmc.py's
    existing field name so it needs no changes) and 'mc_outcome' (the true
    full-episode label, always preserved for gating/diagnostics regardless
    of n_step/shaping)."""
    m = len(decisions)
    if m == 0:
        return
    use_bootstrap = n_step is not None and n_step < m
    values = None
    if use_bootstrap:
        model = load_model(bootstrap_ckpt)
        values = []
        for d in decisions:
            sel = d["obs"].get("select") if d.get("obs") else None
            if not sel or not sel.get("option"):
                values.append(0.0)
                continue
            try:
                values.append(q_max_value(model, d["obs"], sel))
            except Exception:
                values.append(0.0)

    phis = None
    if use_phi_shaping:
        phis = [_phi_at(d) for d in decisions]  # phis[m] (one past last) treated as 0.0 (terminal)

    for t, d in enumerate(decisions):
        d["mc_outcome"] = float(outcome)
        end = (t + n_step) if (n_step is not None) else m
        end = min(end, m)

        if not use_phi_shaping:
            if not use_bootstrap:
                d["outcome"] = float(outcome)
            else:
                d["outcome"] = values[end] if end < m else float(outcome)
            continue

        G, disc = 0.0, 1.0
        for k in range(t, end):
            phi_next = phis[k + 1] if k + 1 < m else 0.0
            F_k = phi_gamma * phi_next - phis[k]
            G += disc * F_k
            disc *= phi_gamma
        G += disc * (values[end] if (use_bootstrap and end < m) else float(outcome))
        d["outcome"] = max(-1.0, min(1.0, G))
