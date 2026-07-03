"""DAgger data collection (Stage 1): the NET pilots mirror self-play games: at
every decision, record the teacher's (main.score_options) argmax as the label
instead of whatever action the net actually took. Trains the net exactly on
the states it reaches, the direct fix for BC's compounding-error gap (85.9%
action-match but 22% head-to-head vs teacher) -- see docs/nn-training.md
"Resume Here" step 2.

Usage:
  python training/nn/dagger_collect.py --games 1000 --ckpt training/ptcg_bc_v1.pth \
      --temp 1.0 --workers 14 --out training/dagger_data_r1.pkl
"""
import argparse
import gzip
import os
import pickle
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "training"))
sys.path.insert(0, REPO_ROOT)

from harness import run_matches  # noqa: E402

AGENT_PATH = os.path.join(_HERE, "selfplay_agent.py")
DEFAULT_CKPT = os.path.join(REPO_ROOT, "training", "ptcg_bc_v1.pth")


def extract_decisions(steps, seat):
    """Same step semantics as bc_collect.extract_decisions."""
    out = []
    for i in range(1, len(steps)):
        prev = steps[i - 1][seat].get("observation", {})
        act = steps[i][seat].get("action")
        sel = prev.get("select")
        if sel is None or act is None:
            continue
        if not sel.get("option"):
            continue
        out.append({"obs": prev, "sel": sel})
    return out


def relabel_with_teacher(obs, sel):
    """argmax(main.score_options) -> single-index [action] list, matching the
    action-list schema dataset.py already expects (label = action[0])."""
    import main  # local import: only needed in the (already-forked) collector process
    scores = main.score_options(obs, sel)
    if not scores:
        return None
    best = max(range(len(scores)), key=lambda i: scores[i])
    return [best]


def write_shard(out, idx, samples):
    path = out if idx == 0 else out.replace(".pkl", f".part{idx}.pkl")
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB, {len(samples)} samples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--temp", type=float, default=1.0,
                     help="softmax temperature for the net's own move selection "
                          "(exploration); the recorded LABEL is always the teacher's, "
                          "regardless of temperature.")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "training", "dagger_data.pkl"))
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    os.environ["NET_CKPT"] = os.path.abspath(args.ckpt)
    os.environ["NET_TEMP"] = str(args.temp)

    print(f"Collecting {args.games} DAgger games: net-piloted mirror self-play "
          f"(ckpt={args.ckpt}, temp={args.temp}), teacher-relabeled")
    samples = []
    games = wins = relabel_errors = 0
    shard_idx = total_samples = 0
    CHUNK = 100  # full step traces are heavy -- extract and discard per chunk
    SHARD_SAMPLES = 100_000  # flush to disk so memory stays bounded
    remaining = args.games
    while remaining > 0:
        n = min(CHUNK, remaining)
        remaining -= n
        results = run_matches(AGENT_PATH, AGENT_PATH, n, workers=args.workers,
                               keep_steps=True, progress=False)
        for r in results:
            if "error" in r or "steps" not in r:
                continue
            games += 1
            r0, r1 = r["rewards"]
            if r0 == 1:
                wins += 1
            for seat, outcome in ((0, r0), (1, r1)):
                for d in extract_decisions(r["steps"], seat):
                    label = relabel_with_teacher(d["obs"], d["sel"])
                    if label is None:
                        relabel_errors += 1
                        continue
                    samples.append({"obs": d["obs"], "action": label, "outcome": outcome or 0})
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
    print(f"games={games} net_p0_winrate={wins/max(games,1):.3f} "
          f"samples={total_samples} shards={shard_idx} relabel_errors={relabel_errors}")


if __name__ == "__main__":
    main()
