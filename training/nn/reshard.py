"""One-off: re-shard seq_collect.py output into smaller chunks so
load_seq_shards() never has to hold more than ~200 games' raw (unencoded)
obs dicts in memory at once. A 2000-game shard's raw pickle was observed to
peak at ~17GB alone (Python object overhead on deeply nested obs dicts is
much larger than the gzip-compressed size suggests) — see
docs/report-log.md 2026-07-09 OOM entries. Reads existing large shards,
writes smaller ones, does NOT delete the originals (caller's call).
"""
import argparse
import gzip
import os
import pickle
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--in-glob", required=True)
ap.add_argument("--out-prefix", required=True)
ap.add_argument("--chunk-size", type=int, default=200)
args = ap.parse_args()

import glob
paths = sorted(glob.glob(args.in_glob))
print(f"resharding {paths} into chunks of {args.chunk_size}")

chunk = []
chunk_idx = 0
total = 0


def flush():
    global chunk, chunk_idx
    if not chunk:
        return
    path = f"{args.out_prefix}.{chunk_idx}.pkl.gz"
    with gzip.open(path, "wb") as f:
        pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {path} ({len(chunk)} games, {os.path.getsize(path)/1e6:.1f} MB)")
    chunk = []
    chunk_idx += 1


for p in paths:
    with gzip.open(p, "rb") as f:
        games = pickle.load(f)
    for g in games:
        chunk.append(g)
        total += 1
        if len(chunk) >= args.chunk_size:
            flush()
    del games

flush()
print(f"total games resharded: {total}, shards written: {chunk_idx}")
