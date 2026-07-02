"""Blended MCTS prior: heuristic score_options() + net policy logits -> one
distribution. See the plan at the top of docs/nn-training.md's "Heuristic-
blended MCTS" section (and the approved plan file this was built from) for
the full design rationale. This module has NO Kaggle/cg-lib dependency and no
tree — it's the local, testable piece: the math that will feed the MCTS prior
once the tree itself exists (Kaggle-only, deferred).

Formula (mixes DISTRIBUTIONS, not raw scores — the heuristic's scale, ~4 to
600+, is wildly different from the net's logits):
    p_h = softmax(score_options(obs, sel) / T_h)
    p_n = softmax(net_logits(obs, sel)[:n] / T_n)
    prior = normalize(lambda_ * p_h + (1 - lambda_) * p_n)

lambda_ starts high (0.8 — the net is currently only ~22% vs the heuristic in
real games) and anneals down only when a self-play iteration proves a real
improvement over the previous checkpoint (evidence-gated, not wall-clock —
see the plan). Floored at 0.2: the heuristic never fully drops out, staying a
permanent exploration/anti-collapse regularizer (recall the documented prior
incident: SP-only training collapsed a policy 46%->20% over 3 iterations
without a stabilizing signal).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

import torch

import main as heuristic
from net_common import load_model, encode_batch


def _softmax(xs, T):
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp((x - m) / max(T, 1e-6)) for x in xs]
    s = sum(exps)
    return [e / s for e in exps] if s > 0 else [1.0 / len(xs)] * len(xs)


def heuristic_prior(obs, sel):
    """softmax(score_options(obs, sel) / T_h) — the heuristic half of the blend.
    T_h is picked per select-type family since score_options' scale differs
    sharply between MAIN-phase (stype==0, spans ~4..600) and everything else
    (flatter, ~0..100)."""
    scores = heuristic.score_options(obs, sel)
    if not scores:
        return []
    T = heuristic.W['prior_T_h_main'] if sel.get('type') == 0 else heuristic.W['prior_T_h_default']
    return _softmax(scores, T)


def net_prior(model, obs, sel):
    """softmax(net policy logits / T_net) — the learned half of the blend."""
    batch, n_actions = encode_batch(obs, sel)
    with torch.no_grad():
        logits, _ = model(*batch)
    logits = logits[0, :n_actions].tolist()
    T = heuristic.W['prior_T_net']
    return _softmax(logits, T)


def blended_prior(model, obs, sel, lambda_, dirichlet_eps=0.0, dirichlet_alpha=0.3):
    """prior = normalize(lambda_ * p_h + (1 - lambda_) * p_n), optionally with
    Dirichlet noise mixed in (root-node exploration only — pass dirichlet_eps=0
    for non-root nodes)."""
    p_h = heuristic_prior(obs, sel)
    p_n = net_prior(model, obs, sel)
    n = len(p_h)
    if n == 0:
        return []
    prior = [lambda_ * p_h[i] + (1 - lambda_) * p_n[i] for i in range(n)]
    if dirichlet_eps > 0:
        noise = torch.distributions.Dirichlet(torch.full((n,), dirichlet_alpha)).sample().tolist()
        prior = [(1 - dirichlet_eps) * prior[i] + dirichlet_eps * noise[i] for i in range(n)]
    s = sum(prior)
    return [p / s for p in prior] if s > 0 else [1.0 / n] * n


def anneal_lambda(current, new_beats_prev, ci_half_width, margin=0.0, step=0.15, floor=0.2, ceiling=0.8):
    """Evidence-gated annealing step (see plan): step lambda_ down by `step`
    only if the new checkpoint beat the previous one by more than the printed
    95% CI half-width (real signal, not noise); hold or step back up on
    regression. `new_beats_prev` = (new_winrate - 0.5); `margin` defaults to 0
    (beat by more than just the CI). Never drops below `floor`."""
    if new_beats_prev > ci_half_width + margin:
        return max(floor, current - step)
    if new_beats_prev < -(ci_half_width + margin):
        return min(ceiling, current + step)
    return current


if __name__ == "__main__":
    import argparse
    import glob

    from dataset import load_shards

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_REPO_ROOT, "training", "bc_data*.pkl.gz"))
    ap.add_argument("--ckpt", default=os.path.join(_REPO_ROOT, "training", "ptcg_bc_v1.pth"))
    ap.add_argument("--lambda-", dest="lambda_", type=float, default=0.8)
    ap.add_argument("--n", type=int, default=200, help="samples to diagnose")
    args = ap.parse_args()

    print(f"Loading data from {args.data} ...")
    raw = load_shards(args.data)
    import random
    random.Random(0).shuffle(raw)
    raw = raw[: args.n]

    model = load_model(args.ckpt)

    n_nan = 0
    agree_h = agree_n = agree_blend = 0
    checked = 0
    for d in raw:
        obs, action = d["obs"], d["action"]
        sel = obs.get("select")
        if not sel or not sel.get("option"):
            continue
        try:
            p_h = heuristic_prior(obs, sel)
            p_n = net_prior(model, obs, sel)
            prior = blended_prior(model, obs, sel, args.lambda_)
        except Exception as e:
            print("ERROR encoding sample:", e)
            continue
        if any(math.isnan(p) for p in p_h + p_n + prior):
            n_nan += 1
            continue
        checked += 1
        taken = action[0] if action else 0
        if taken >= len(p_h):
            continue
        if max(range(len(p_h)), key=lambda i: p_h[i]) == taken:
            agree_h += 1
        if max(range(len(p_n)), key=lambda i: p_n[i]) == taken:
            agree_n += 1
        if max(range(len(prior)), key=lambda i: prior[i]) == taken:
            agree_blend += 1

    print(f"checked={checked} nan_samples={n_nan}")
    print(f"heuristic-argmax matches action taken: {agree_h}/{checked} ({agree_h/max(checked,1):.3f})")
    print(f"net-argmax matches action taken:       {agree_n}/{checked} ({agree_n/max(checked,1):.3f})")
    print(f"blend(lambda={args.lambda_})-argmax matches: {agree_blend}/{checked} ({agree_blend/max(checked,1):.3f})")
