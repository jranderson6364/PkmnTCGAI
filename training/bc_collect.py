"""Collect behavior-cloning data from v22 self-play (and vs the opponent pool).

Each game's step trace is reduced to (obs_dict, chosen_action, player, outcome)
tuples for OUR seats and pickled. The NN training notebook converts these into
encoder/decoder tensors on Kaggle (where cg.api / all_card_data is available).

Usage:
  python training/bc_collect.py --games 700 --out bc_data.pkl [--opponent opponents/lucario_agent.py] [--workers N]

Default opponent is self (mirror). ~150+ decisions per game, so 700 games
≈ 100k+ samples. On 16 cores this is well under an hour.
"""
import argparse
import gzip
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import load_agent, run_matches

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(REPO_ROOT, "main.py")


def extract_decisions(steps, seat):
    """kaggle-env step semantics (reverse-engineered, see tools/analyze_replay.py):
    steps[i]['action'] resolves steps[i-1]['observation'].select's option list."""
    out = []
    for i in range(1, len(steps)):
        prev = steps[i - 1][seat].get("observation", {})
        act = steps[i][seat].get("action")
        sel = prev.get("select")
        if sel is None or act is None:
            continue
        if not sel.get("option"):
            continue
        out.append({"obs": prev, "action": act})
    return out


def write_shard(out, idx, samples):
    path = out if idx == 0 else out.replace(".pkl", f".part{idx}.pkl")
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB, {len(samples)} samples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=700)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "training", "bc_data.pkl.gz"))
    ap.add_argument("--opponent", default=MAIN)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    print(f"Collecting {args.games} games: {MAIN} vs {args.opponent}")
    samples = []
    games = wins = 0
    shard_idx = total_samples = 0
    CHUNK = 100  # full step traces are heavy — extract and discard per chunk
    SHARD_SAMPLES = 100_000  # flush to disk so memory stays bounded
    remaining = args.games
    while remaining > 0:
        n = min(CHUNK, remaining)
        remaining -= n
        results = run_matches(MAIN, args.opponent, n, workers=args.workers,
                              keep_steps=True, progress=False)
        for r in results:
            if "error" in r or "steps" not in r:
                continue
            games += 1
            outcome = r["rewards"][0]  # our seat is 0
            if outcome == 1:
                wins += 1
            for d in extract_decisions(r["steps"], seat=0):
                d["outcome"] = outcome
                samples.append(d)
            # mirror games: seat 1 is also our agent — harvest it too
            if os.path.abspath(args.opponent) == os.path.abspath(MAIN):
                outcome1 = r["rewards"][1]
                for d in extract_decisions(r["steps"], seat=1):
                    d["outcome"] = outcome1
                    samples.append(d)
        print(f"  {games}/{args.games} games, {len(samples)} samples (shard {shard_idx})", file=sys.stderr)
        if len(samples) >= SHARD_SAMPLES:
            write_shard(args.out, shard_idx, samples)
            total_samples += len(samples)
            samples = []
            shard_idx += 1

    total_samples += len(samples)
    if samples or shard_idx == 0:
        write_shard(args.out, shard_idx, samples)
        shard_idx += 1
    print(f"games={games} our_p0_winrate={wins/max(games,1):.3f} "
          f"samples={total_samples} shards={shard_idx}")


if __name__ == "__main__":
    main()
