"""Phase E (2026-07-09 overnight): eval-accelerated loss mining. Runs the
champion Φ v4-MLP state eval over OUR decision states in LOSS replays and
ranks games by squandered advantage (peak value later lost) and by the
sharpest value collapses — a shortlist for the manual replay-mining
workflow (the project's one reliably positive loop), so mining starts from
the eval's best guesses instead of random losses.

Non-override consumer: the eval never picks actions here, it only
prioritizes which of our own losses to study.

Usage:
  python training/nn/blunder_scan.py --replays-dir replays/v29d_ladder [--top 15]
"""
import argparse
import glob
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from eval_v4 import extract_game  # noqa: E402
from eval_v4_mlp import MLP  # noqa: E402


def load_mlp():
    m = MLP(12, 64, 2)
    m.load_state_dict(torch.load(os.path.join(_REPO_ROOT, "training", "eval_v4_mlp.pth")))
    m.eval()
    return m


def game_values(model, rows):
    X = np.array([np.concatenate([f, [min(t, 30) / 6.0 / 5.0]]) for f, _p, t in rows],
                 dtype=np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(X)).numpy()
    return np.tanh(logits / 2.0), [t for _f, _p, t in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays-dir", default=os.path.join(_REPO_ROOT, "replays", "v29d_ladder"))
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    model = load_mlp()
    records = []
    paths = sorted(glob.glob(os.path.join(args.replays_dir, "*.json")))
    for p in paths:
        g = extract_game(p)
        if not g:
            continue
        rows, outcome = g
        if outcome != -1:
            continue  # losses only
        v, turns = game_values(model, rows)
        peak_i = int(np.argmax(v))
        drops = v[1:] - v[:-1]
        worst_i = int(np.argmin(drops)) if len(drops) else 0
        records.append({
            "file": os.path.basename(p),
            "n_dec": len(v),
            "peak_v": float(v[peak_i]), "peak_turn": turns[peak_i],
            "worst_drop": float(drops[worst_i]) if len(drops) else 0.0,
            "worst_drop_turn": turns[worst_i + 1] if len(drops) else None,
            "final_v": float(v[-1]), "final_turn": turns[-1],
        })

    print(f"losses scanned: {len(records)} (of {len(paths)} files)")
    squandered = sorted([r for r in records if r["peak_v"] >= 0.3],
                        key=lambda r: -r["peak_v"])
    print(f"\nSQUANDERED-ADVANTAGE losses (peak eval >= +0.3, we lost anyway): "
          f"{len(squandered)}")
    for r in squandered[:args.top]:
        print(f"  {r['file']}: peak {r['peak_v']:+.2f}@T{r['peak_turn']} -> "
              f"final {r['final_v']:+.2f}@T{r['final_turn']}; "
              f"worst drop {r['worst_drop']:+.2f}@T{r['worst_drop_turn']} ({r['n_dec']} dec)")
    others = sorted([r for r in records if r["peak_v"] < 0.3],
                    key=lambda r: r["worst_drop"])
    print(f"\nOther losses, sharpest collapses first: {len(others)}")
    for r in others[:args.top]:
        print(f"  {r['file']}: peak {r['peak_v']:+.2f}@T{r['peak_turn']}; "
              f"worst drop {r['worst_drop']:+.2f}@T{r['worst_drop_turn']}; "
              f"final {r['final_v']:+.2f}@T{r['final_turn']}")


if __name__ == "__main__":
    main()
