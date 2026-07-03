"""BC training loop. Runs locally (CPU, small subset, for smoke-testing) or on
Kaggle (GPU, full data) — same code, just point --data at different globs.

Usage:
  python train_bc.py --data "../bc_data*.pkl.gz" --epochs 10 --out ptcg_bc.pth
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import BCDataset, collate, load_shards
from model import PTCGNet


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, value = model(
                batch["board_ids"], batch["hand_ids"], batch["discard_ids"],
                batch["numeric"], batch["action_type"], batch["action_card"],
                batch["action_attack"], batch["action_numeric"], batch["action_mask"])
            pred = logits.argmax(dim=-1)
            correct += (pred == batch["labels"]).sum().item()
            total += pred.shape[0]
    model.train()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../bc_data*.pkl.gz")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="ptcg_bc.pth")
    ap.add_argument("--limit", type=int, default=None, help="cap samples (smoke tests)")
    ap.add_argument("--val-frac", type=float, default=0.05)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    raw = load_shards(args.data, limit=args.limit)
    print(f"loaded {len(raw)} samples")

    ds = BCDataset(raw)
    n_val = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = PTCGNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.HuberLoss(delta=0.2)

    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, value = model(
                batch["board_ids"], batch["hand_ids"], batch["discard_ids"],
                batch["numeric"], batch["action_type"], batch["action_card"],
                batch["action_attack"], batch["action_numeric"], batch["action_mask"])
            p_loss = policy_loss_fn(logits, batch["labels"])
            v_loss = value_loss_fn(value, batch["values"])
            loss = p_loss + 0.5 * v_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: train_loss={total_loss/max(n_batches,1):.4f} "
              f"val_top1_acc={val_acc:.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
