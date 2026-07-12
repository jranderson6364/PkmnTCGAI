"""Collect BC data from the CURRENT teacher (v29d main.py, mirror self-play),
grouped by game for sequence-policy training.

Unlike bc_collect.py's flat sample list, each output shard is a list of GAMES:
[{"decisions": [{"obs":..., "action":...}, ...], "outcome": int}, ...]
in original in-game order, one entry per (obs, action) decision for our seat.
Needed because the sequence model consumes ordered game history, not iid
states. See docs/next-session-plan.md Phase 1.

Usage:
  python training/nn/seq_collect.py --games 2000 --out training/seq_data_v29d.pkl.gz [--workers N]
"""
import argparse
import gzip
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import run_matches

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAIN = os.path.join(REPO_ROOT, "main.py")


def extract_decisions(steps, seat):
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


def write_shard(out, idx, games):
    path = out if idx == 0 else out.replace(".pkl", f".part{idx}.pkl")
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wb") as f:
        pickle.dump(games, f, protocol=pickle.HIGHEST_PROTOCOL)
    n_dec = sum(len(g["decisions"]) for g in games)
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB, {len(games)} games, {n_dec} decisions)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "training", "seq_data_v29d.pkl.gz"))
    ap.add_argument("--opponent", default=MAIN)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    print(f"Collecting {args.games} games (sequence-grouped): {MAIN} vs {args.opponent}")
    games_out = []
    games = wins = 0
    shard_idx = total_games = 0
    CHUNK = 100
    SHARD_GAMES = 2000  # ~2000 games/shard keeps shard files a manageable size
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
            outcome0 = r["rewards"][0]
            if outcome0 == 1:
                wins += 1
            dec0 = extract_decisions(r["steps"], seat=0)
            if dec0:
                games_out.append({"decisions": dec0, "outcome": outcome0})
            if os.path.abspath(args.opponent) == os.path.abspath(MAIN):
                outcome1 = r["rewards"][1]
                dec1 = extract_decisions(r["steps"], seat=1)
                if dec1:
                    games_out.append({"decisions": dec1, "outcome": outcome1})
        print(f"  {games}/{args.games} games, {len(games_out)} sequences (shard {shard_idx})", file=sys.stderr)
        if len(games_out) >= SHARD_GAMES:
            write_shard(args.out, shard_idx, games_out)
            total_games += len(games_out)
            games_out = []
            shard_idx += 1

    total_games += len(games_out)
    if games_out or shard_idx == 0:
        write_shard(args.out, shard_idx, games_out)
        shard_idx += 1
    print(f"games={games} our_p0_winrate={wins/max(games,1):.3f} "
          f"sequences={total_games} shards={shard_idx}")


if __name__ == "__main__":
    main()
