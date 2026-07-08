"""Winner-filtered behavioral cloning from REAL ladder replays (2026-07-08,
method-survey #10/#23 — the one imitation variant that uses an external
data source, so the teacher-parity ceiling argument doesn't bind).

Rows are (state, action-actually-taken) pairs from replay sides that WON
their game — both seats, any archetype (the model conditions on the
acting player's own observation; archetype mix is logged). Alignment
convention verified empirically 2026-07-08: `steps[t][seat]["action"]`
was taken from `steps[t-1][seat]["observation"]` (98.3% index-valid vs
89.6% same-record) — extraction pairs action t with observation t-1.

Skips: multi-action selects (len(action)!=1), label >= MAX_ACTIONS,
selects with <2 options, records where the observation isn't the acting
seat's own.

Usage:
  python training/nn/replay_policy.py --init-ckpt training/ptcg_dagger_r2.pth \
      --out training/ptcg_rp_r1.pth [--epochs 3] [--all-outcomes]
"""
import argparse
import glob
import json
import os
import random
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

from dataset import collate, MAX_ACTIONS  # noqa: E402
from encode import encode_sample  # noqa: E402
from model import PTCGNet  # noqa: E402


_ALAKAZAM_LINE = {741, 742, 743}


def _seat_plays_alakazam(cur, seat):
    """True if this seat's own visible zones show any Abra/Kadabra/Alakazam."""
    p = (cur.get("players") or [{}, {}])[seat]
    for zone in ("active", "bench", "hand", "discard"):
        for card in (p.get(zone) or []):
            if isinstance(card, dict) and card.get("id") in _ALAKAZAM_LINE:
                return True
    return False


def extract_rows(path, winners_only=True, alakazam_only=False):
    """Per-file list of (enc, action_label, outcome) rows."""
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return []
    rewards = d.get("rewards")
    if not rewards or len(rewards) != 2:
        return []
    steps = d.get("steps") or []
    rows = []
    for seat in (0, 1):
        if rewards[seat] not in (1, -1):
            continue
        if winners_only and rewards[seat] != 1:
            continue
        seat_rows, seen_alakazam = [], False
        for t in range(1, len(steps)):
            rec = steps[t][seat] if len(steps[t]) > seat else None
            prev = steps[t - 1][seat] if len(steps[t - 1]) > seat else None
            if not rec or not prev:
                continue
            act = rec.get("action")
            if not isinstance(act, list) or len(act) != 1:
                continue
            obs = prev.get("observation") or {}
            cur = obs.get("current") or {}
            if cur.get("yourIndex") != seat:
                continue
            if alakazam_only and not seen_alakazam:
                seen_alakazam = _seat_plays_alakazam(cur, seat)
            sel = obs.get("select")
            opts = (sel or {}).get("option") or []
            if len(opts) < 2:
                continue
            a = act[0]
            if not (0 <= a < min(len(opts), MAX_ACTIONS)):
                continue
            try:
                enc = encode_sample(obs, sel)
            except Exception:
                continue
            seat_rows.append((enc, a, float(rewards[seat])))
        if not alakazam_only or seen_alakazam:
            rows.extend(seat_rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays-dir", default=os.path.join(_REPO_ROOT, "replays", "bulk"))
    ap.add_argument("--extra-dirs", default="replays/v28,replays/v26remake,replays/v29")
    ap.add_argument("--init-ckpt", default=os.path.join(_REPO_ROOT, "training", "ptcg_dagger_r2.pth"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--all-outcomes", action="store_true",
                    help="train on losers too (ablation arm)")
    ap.add_argument("--alakazam-only", action="store_true",
                    help="keep only sides whose visible zones show the "
                         "Abra/Kadabra/Alakazam line (on-distribution arm)")
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-games", type=int, default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    paths = sorted(glob.glob(os.path.join(args.replays_dir, "*.json")))
    for extra in (args.extra_dirs or "").split(","):
        extra = extra.strip()
        if extra:
            paths.extend(sorted(glob.glob(os.path.join(_REPO_ROOT, extra, "*.json"))))
    if args.max_games:
        paths = paths[: args.max_games]
    rng.shuffle(paths)
    n_hold = int(len(paths) * args.holdout_frac)
    hold_paths, train_paths = paths[:n_hold], paths[n_hold:]

    train_rows, hold_rows = [], []
    for dst, plist in ((train_rows, train_paths), (hold_rows, hold_paths)):
        for p in plist:
            dst.extend(extract_rows(p, winners_only=not args.all_outcomes,
                                    alakazam_only=args.alakazam_only))
    print(f"files={len(paths)} train_rows={len(train_rows)} holdout_rows={len(hold_rows)}",
          flush=True)

    model = PTCGNet()
    state = torch.load(args.init_ckpt, map_location="cpu")
    sd = state.get("model", state)
    own = model.state_dict()
    compat = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    skipped = sorted(set(sd) - set(compat))
    model.load_state_dict(compat, strict=False)
    if skipped:
        print(f"init: warm-started {len(compat)} tensors, skipped shape-mismatched: "
              f"{skipped}", flush=True)  # pre-richenc ckpts lack 25-feat numeric_proj
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    def batchify(rows):
        return collate([(enc, a, o, None, None, []) for enc, a, o in rows])

    def forward(b):
        return model(b["board_ids"], b["hand_ids"], b["discard_ids"], b["numeric"],
                     b["action_type"], b["action_card"], b["action_attack"],
                     b["action_numeric"], b["action_mask"])

    def holdout_acc():
        model.eval()
        correct = seen = 0
        with torch.no_grad():
            for i in range(0, len(hold_rows), 512):
                chunk = hold_rows[i:i + 512]
                b = batchify(chunk)
                logits, _ = forward(b)
                correct += (logits.argmax(dim=-1) == b["labels"]).sum().item()
                seen += len(chunk)
        model.train()
        return correct / max(seen, 1)

    order = list(range(len(train_rows)))
    for epoch in range(args.epochs):
        rng.shuffle(order)
        tot, seen = 0.0, 0
        for i in range(0, len(order), args.batch):
            chunk = [train_rows[j] for j in order[i:i + args.batch]]
            b = batchify(chunk)
            logits, _ = forward(b)
            loss = ce(logits, b["labels"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach())
            seen += 1
        acc = holdout_acc()
        print(f"epoch {epoch}: ce={tot/max(seen,1):.4f} holdout_top1={acc:.4f}", flush=True)
        torch.save({"model": model.state_dict()}, f"{args.out}.ep{epoch}")
        torch.save({"model": model.state_dict()}, args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
