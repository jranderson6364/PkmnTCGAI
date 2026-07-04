"""Self-play training step: warm-starts from a BC (or prior SP) checkpoint,
trains on a 40% BC / 60% SP mix (non-negotiable per docs/nn-training.md —
SP-only collapsed the prior project's attempt: 46%->20% vs teacher in 3 iters).

Usage:
  python train_sp.py --bc-data "../bc_data*.pkl.gz" --sp-data "../sp_data.pkl.gz" \
      --init ../ptcg_bc_v1.pth --out ../ptcg_sp_iter1.pth --epochs 3
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset

from dataset import BCDataset, collate, load_shards
from model import PTCGNet


def awr_normalizer(sp_raw, beta):
    """Mean of exp(advantage/beta) over SP samples that carry a v_pred
    (docs/nn-training.md Stage 2) — used to rescale weights to mean ~1.0
    within the SP portion, so AWR doesn't silently shift the "non-negotiable"
    40% BC / 60% SP batch composition away from its target ratio."""
    ws = [math.exp(max(-10.0, min(10.0, (s["value_target"] - s["v_pred"]) / beta)))
          for s in sp_raw if "v_pred" in s]
    return (sum(ws) / len(ws)) if ws else 1.0


def build_mixed_loader(bc_raw, sp_raw, batch_size, bc_frac=0.4):
    bc_ds = BCDataset(bc_raw)
    sp_ds = BCDataset(sp_raw)
    combined = ConcatDataset([bc_ds, sp_ds])
    n_bc, n_sp = len(bc_ds), len(sp_ds)
    # per-sample weight so the EXPECTED batch composition is bc_frac/  (1-bc_frac),
    # regardless of how imbalanced the two pools are in raw sample count.
    w_bc = bc_frac / max(n_bc, 1)
    w_sp = (1 - bc_frac) / max(n_sp, 1)
    weights = [w_bc] * n_bc + [w_sp] * n_sp
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(combined, batch_size=batch_size, sampler=sampler, collate_fn=collate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-data", default="../bc_data*.pkl.gz")
    ap.add_argument("--sp-data", default="../sp_data.pkl.gz")
    ap.add_argument("--init", default="../ptcg_bc_v1.pth")
    ap.add_argument("--out", default="../ptcg_sp_iter1.pth")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)  # lower than BC warmup — fine-tuning
    ap.add_argument("--bc-frac", type=float, default=0.4)
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--bc-limit", type=int, default=None,
                     help="cap BC raw samples loaded into RAM (each source is "
                          "resampled with replacement anyway, so a large corpus "
                          "gains nothing but memory pressure over a capped one)")
    ap.add_argument("--sp-limit", type=int, default=None)
    ap.add_argument("--awr-beta", type=float, default=1.0,
                     help="temperature for the AWR policy-loss weight "
                          "exp(advantage/beta) (Stage 2, docs/nn-training.md); "
                          "advantages are ~[-2,2] (both terms tanh-clipped), "
                          "beta=1.0 is the calibrated default")
    ap.add_argument("--awr-clip", type=float, default=20.0,
                     help="clip normalized AWR weights to [1/awr-clip, awr-clip] "
                          "so a few high-advantage samples can't dominate the gradient")
    ap.add_argument("--winner-only", action="store_true",
                     help="dumb-baseline ablation (docs/nn-training.md Stage 2): "
                          "drop AWR weighting, just filter SP samples to winning "
                          "games (outcome > 0)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    bc_raw = load_shards(args.bc_data, limit=args.bc_limit)
    sp_raw = load_shards(args.sp_data, limit=args.sp_limit)
    if args.winner_only:
        sp_raw = [s for s in sp_raw if s.get("outcome", 0) > 0]
        print(f"--winner-only: filtered sp_samples to {len(sp_raw)} (outcome>0)")
    print(f"bc_samples={len(bc_raw)} sp_samples={len(sp_raw)}")

    awr_norm = 1.0 if args.winner_only else awr_normalizer(sp_raw, args.awr_beta)
    print(f"awr_norm={awr_norm:.4f} beta={args.awr_beta} winner_only={args.winner_only}")

    loader = build_mixed_loader(bc_raw, sp_raw, args.batch_size, args.bc_frac)

    model = PTCGNet().to(device)
    model.load_state_dict(torch.load(args.init, map_location=device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    value_loss_fn = nn.HuberLoss(delta=0.2)

    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            if n_batches >= args.steps_per_epoch:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, value = model(
                batch["board_ids"], batch["hand_ids"], batch["discard_ids"],
                batch["numeric"], batch["action_type"], batch["action_card"],
                batch["action_attack"], batch["action_numeric"], batch["action_mask"])
            # Soft-target cross-entropy against policy_targets: a strict
            # generalization of CrossEntropyLoss(logits, labels) — dataset.py's
            # collate() fills policy_targets with a one-hot(label) fallback for
            # plain BC/direct-SP samples (no MCTS visit counts yet), so this is
            # mathematically identical to hard CE until mcts_collect.py starts
            # producing real soft targets.
            log_probs = torch.log_softmax(logits, dim=-1)
            per_sample_p_loss = -(batch["policy_targets"] * log_probs).sum(dim=-1)
            if args.winner_only:
                weight = torch.ones_like(per_sample_p_loss)
            else:
                # AWR (Stage 2): BC samples (has_advantage==0) get weight 1.0;
                # SP samples get exp(advantage/beta), rescaled by the corpus-
                # level normalizer so the SP portion's mean weight stays ~1.0
                # (protects the non-negotiable 40/60 BC/SP mix) and clipped so
                # a few high-advantage samples can't dominate the gradient.
                raw_w = torch.exp(torch.clamp(batch["advantages"] / args.awr_beta, -10.0, 10.0))
                sp_w = torch.clamp(raw_w / awr_norm, 1.0 / args.awr_clip, args.awr_clip)
                weight = torch.where(batch["has_advantage"] > 0, sp_w, torch.ones_like(sp_w))
            p_loss = (weight * per_sample_p_loss).mean()
            v_loss = value_loss_fn(value, batch["values"])
            loss = p_loss + 0.5 * v_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"epoch {epoch}: avg_loss={total_loss/max(n_batches,1):.4f} ({n_batches} steps)")

    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
