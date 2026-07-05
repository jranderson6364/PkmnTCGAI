"""Oracle-critic value retrain: warm-starts from a policy checkpoint (DAgger
round 2 by default) and retrains the (now oracle-widened) value head on
vd_collect.py's diverse, opponent-hand-labeled corpus. Policy head is kept
warm and only lightly regularized (0.1x CE) so it doesn't drift off the
DAgger-trained policy while the value head learns to use the oracle feature.

Usage:
  python train_value.py --data "../vd_diverse*.pkl.gz" --epochs 3 --out ../ptcg_value_oracle.pth
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
            _, value = model(
                batch["board_ids"], batch["hand_ids"], batch["discard_ids"],
                batch["numeric"], batch["action_type"], batch["action_card"],
                batch["action_attack"], batch["action_numeric"], batch["action_mask"],
                oracle_ids=batch["oracle_ids"], oracle_flag=batch["oracle_flag"])
            correct += ((value >= 0) == (batch["values"] >= 0)).sum().item()
            total += value.shape[0]
    model.train()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../vd_diverse*.pkl.gz,../vd_ladder.pkl.gz")
    ap.add_argument("--init", default="../ptcg_dagger_r2.pth")
    ap.add_argument("--out", default="../ptcg_value_oracle.pth")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
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
    # value_head's input dim widened (oracle features) so its weights can't be
    # warm-started from an old checkpoint — drop those keys and let them
    # init fresh; everything else (trunk, embeddings, policy head) loads as-is.
    state = torch.load(args.init, map_location=device)
    own = model.state_dict()
    state = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
    result = model.load_state_dict(state, strict=False)
    print(f"warm-start from {args.init}: missing={result.missing_keys} "
          f"unexpected={result.unexpected_keys}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.HuberLoss(delta=0.2)

    best_acc = -1.0
    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, value = model(
                batch["board_ids"], batch["hand_ids"], batch["discard_ids"],
                batch["numeric"], batch["action_type"], batch["action_card"],
                batch["action_attack"], batch["action_numeric"], batch["action_mask"],
                oracle_ids=batch["oracle_ids"], oracle_flag=batch["oracle_flag"])
            v_loss = value_loss_fn(value, batch["values"])
            p_loss = policy_loss_fn(logits, batch["labels"])
            loss = v_loss + 0.1 * p_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: train_loss={total_loss/max(n_batches,1):.4f} "
              f"val_sign_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.out)

    print(f"saved best (val_sign_acc={best_acc:.4f}) to {args.out}")


if __name__ == "__main__":
    main()
