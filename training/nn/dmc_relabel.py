"""Phase 0 ablation grid support: relabel ONE shared, already-collected raw
game corpus into multiple n-step/Φ-shaping arms offline, instead of
re-playing the same (expensive) games once per arm. Requires the corpus to
have been collected with `dmc_collect.py` (any point after the `game_id`
field was added, 2026-07-05) since game boundaries can't be recovered from a
flat decision list otherwise.

--in accepts a glob pattern so multi-shard corpora (dmc_collect.py writes a
new .partN shard past 100k samples) load in one call.

Usage:
  python training/nn/dmc_relabel.py --in training/dmc_raw.pkl.gz \
      --out training/dmc_r5_n5.pkl.gz --n-step 5
  python training/nn/dmc_relabel.py --in "training/dmc_raw*.pkl.gz" \
      --out training/dmc_r5_phi.pkl.gz --phi-shaping
  python training/nn/dmc_relabel.py --in training/dmc_raw.pkl.gz \
      --out training/dmc_r5_n5_phi.pkl.gz --n-step 5 --phi-shaping
"""
import argparse
import glob
import gzip
import os
import pickle
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from dmc_nstep import compute_nstep_targets  # noqa: E402


def load_shard(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        return pickle.load(f)


def load_shards(pattern):
    paths = sorted(glob.glob(pattern)) or [pattern]
    out = []
    for p in paths:
        shard = load_shard(p)
        print(f"  loaded {len(shard)} samples from {p}")
        out.extend(shard)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-step", type=int, default=None)
    ap.add_argument("--bootstrap-ckpt", default=os.path.join(
        os.path.dirname(os.path.dirname(_HERE)), "training", "ptcg_dmc_r2.pth"))
    ap.add_argument("--phi-shaping", action="store_true")
    args = ap.parse_args()

    samples = load_shards(args.inp)
    print(f"loaded {len(samples)} raw samples total from {args.inp}")

    games = {}
    for d in samples:
        gid = d.get("game_id")
        if gid is None:
            raise SystemExit(
                f"{args.inp} has no 'game_id' field -- collected before "
                "dmc_collect.py's game_id addition (2026-07-05); can't relabel offline.")
        games.setdefault(gid, []).append(d)

    out = []
    for gid, decisions in games.items():
        outcome = decisions[0].get("mc_outcome", decisions[0].get("outcome", 0))
        compute_nstep_targets(args.bootstrap_ckpt, decisions, outcome,
                               n_step=args.n_step, use_phi_shaping=args.phi_shaping)
        out.extend(decisions)

    print(f"relabeled {len(out)} samples across {len(games)} games "
          f"(n_step={args.n_step}, phi_shaping={args.phi_shaping})")

    opener = gzip.open if args.out.endswith(".gz") else open
    with opener(args.out, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
