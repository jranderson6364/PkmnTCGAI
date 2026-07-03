"""Self-play training step: warm-starts from a BC (or prior SP) checkpoint,
trains on a 40% BC / 60% SP mix (non-negotiable per docs/nn-training.md —
SP-only collapsed the prior project's attempt: 46%->20% vs teacher in 3 iters).

Usage:
  python train_sp.py --bc-data "../bc_data*.pkl.gz" --sp-data "../sp_data.pkl.gz" \
      --init ../ptcg_bc_v1.pth --out ../ptcg_sp_iter1.pth --epochs 3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset

from dataset import BCDataset, collate, load_shards
from model import PTCGNet


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
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    bc_raw = load_shards(args.bc_data, limit=args.bc_limit)
    sp_raw = load_shards(args.sp_data, limit=args.sp_limit)
    print(f"bc_samples={len(bc_raw)} sp_samples={len(sp_raw)}")

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
            p_loss = -(batch["policy_targets"] * log_probs).sum(dim=-1).mean()
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
