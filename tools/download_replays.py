"""Bulk-download ladder replay episodes via the Kaggle CLI (docs/belief-model.md
Phase B blocker: "only 8 replay JSONs on disk today - need bulk download").

For every COMPLETE submission to the ladder, lists its episodes and downloads
any replay not already on disk. Resumable: re-running skips episode ids that
already have a file in the output directory.

Usage:
  python tools/download_replays.py --out replays/bulk
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

COMPETITION = "pokemon-tcg-ai-battle"


def run_json(args, retries=1):
    for attempt in range(retries + 1):
        r = subprocess.run(["kaggle", *args, "--format", "json"], capture_output=True, text=True)
        if r.stdout.strip():
            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(r.stdout)
                return obj
            except json.JSONDecodeError:
                pass
        if attempt < retries:
            time.sleep(1.0)
    print(f"  WARN: no parseable JSON for {args} (stderr: {r.stderr.strip()[:200]})", file=sys.stderr)
    return None


def list_submissions():
    subs = run_json(["competitions", "submissions", COMPETITION]) or []
    return [s["ref"] for s in subs if s.get("status") == "SubmissionStatus.COMPLETE"]


def list_episodes(submission_id):
    return run_json(["competitions", "episodes", str(submission_id)])


def existing_episode_ids(out_dir):
    ids = set()
    for pattern in ("*.json",):
        for path in glob.glob(os.path.join(out_dir, "**", pattern), recursive=True):
            m = re.search(r"(\d{6,})", os.path.basename(path))
            if m:
                ids.add(int(m.group(1)))
    return ids


def download_episode(episode_id, out_dir, retries=1):
    for attempt in range(retries + 1):
        r = subprocess.run(["kaggle", "competitions", "replay", str(episode_id), "-p", out_dir, "-q"],
                            capture_output=True, text=True)
        if r.returncode == 0:
            return True
        if attempt < retries:
            time.sleep(1.0)
    print(f"  FAILED episode {episode_id}: {r.stderr.strip()[:200]}", file=sys.stderr)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("replays", "bulk"))
    ap.add_argument("--sleep", type=float, default=0.2, help="seconds between downloads (rate-limit courtesy)")
    ap.add_argument("--limit", type=int, default=None, help="cap total new downloads this run")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    have = existing_episode_ids(args.out) | existing_episode_ids("replays")
    print(f"{len(have)} episode ids already on disk", file=sys.stderr)

    subs = list_submissions()
    print(f"{len(subs)} complete submissions", file=sys.stderr)

    all_episode_ids = []
    for s in subs:
        eps = list_episodes(s) or []
        all_episode_ids.extend(e["id"] for e in eps)
    all_episode_ids = sorted(set(all_episode_ids))
    print(f"{len(all_episode_ids)} unique episodes across all submissions", file=sys.stderr)

    todo = [e for e in all_episode_ids if e not in have]
    print(f"{len(todo)} episodes not yet downloaded", file=sys.stderr)
    if args.limit:
        todo = todo[:args.limit]
        print(f"capped to {len(todo)} this run", file=sys.stderr)

    downloaded = failed = 0
    for i, ep_id in enumerate(todo):
        ok = download_episode(ep_id, args.out)
        if ok:
            downloaded += 1
        else:
            failed += 1
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(todo)} ({downloaded} ok, {failed} failed)", file=sys.stderr)
        time.sleep(args.sleep)

    print(f"done: {downloaded} downloaded, {failed} failed, {len(have)+downloaded} total on disk in {args.out}")


if __name__ == "__main__":
    main()
