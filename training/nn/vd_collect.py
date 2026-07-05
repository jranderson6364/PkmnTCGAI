"""Oracle-aware value-data collector: diverse value-head corpus with privileged
oracle features (the opponent's true hand) for the oracle-critic retrain
(PerfectDou-style; see docs/report-log.md 2026-07-04 literature review).

Plays main.py vs each of the 4 fixed anchors plus a main.py mirror, and for
every OUR-seat decision also records the opponent's true hand at that step
(read off the OPPONENT's own observation, which the harness keeps for both
seats) as `opp_hand`. train_value.py trains the widened value head on this;
main.py/net_agent.py never see `opp_hand` at inference (oracle_flag=0 there).

Usage:
  python training/nn/vd_collect.py --games 400 --out training/vd_diverse.pkl.gz
  python training/nn/vd_collect.py --games 4 --opponents lucario --out /tmp/smoke.pkl
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
from encode import hand_ids as _hand_ids  # noqa: E402

MAIN = os.path.join(REPO_ROOT, "main.py")
ANCHORS = {
    "lucario": os.path.join(REPO_ROOT, "opponents", "lucario_agent.py"),
    "dragapult": os.path.join(REPO_ROOT, "opponents", "dragapult_agent.py"),
    "abomasnow": os.path.join(REPO_ROOT, "opponents", "abomasnow_agent.py"),
    "starmie": os.path.join(REPO_ROOT, "opponents", "starmie_agent.py"),
    "mirror": MAIN,
}


def _opp_handcount(prev_obs, seat):
    """opponent's handCount as visible from OUR OWN observation (prev_obs) —
    ground truth for the opponent's CURRENT hand size at our decision."""
    cur = prev_obs.get("current") or {}
    me_idx = cur.get("yourIndex", 0)
    pl = cur.get("players") or []
    opp = pl[1 - me_idx] if len(pl) == 2 else {}
    return opp.get("handCount")


WALK_LIMIT = 30  # backward-walk bound, ~current + previous turn


def _oracle_hand(steps, i, seat, expected_n):
    """Validated backward-walk join: the idle seat's same-step observation is
    often a turn stale (~50% mismatch measured vs our observed opp handCount),
    so walk back from step i-1 and take the most recent opponent observation
    whose own hand length EXACTLY equals expected_n. No match within
    WALK_LIMIT -> None, so only verifiably-current privileged info is used
    (oracle_flag stays 0 for that sample)."""
    if expected_n is None:
        return None
    for j in range(i - 1, max(i - 1 - WALK_LIMIT, -1), -1):
        opp_obs = steps[j][1 - seat].get("observation", {})
        if not opp_obs.get("current"):
            continue
        hand = [c for c in _hand_ids(opp_obs) if c]
        if len(hand) == expected_n:
            return hand
    return None


def extract_decisions_oracle(steps, seat, mismatches):
    """Same step semantics as bc_collect.extract_decisions, plus opp_hand
    joined from the opponent's own observations via _oracle_hand."""
    out = []
    for i in range(1, len(steps)):
        prev = steps[i - 1][seat].get("observation", {})
        act = steps[i][seat].get("action")
        sel = prev.get("select")
        if sel is None or act is None:
            continue
        if not sel.get("option"):
            continue
        expected_n = _opp_handcount(prev, seat)
        opp_hand = _oracle_hand(steps, i, seat, expected_n)
        mismatches[1] += 1
        if opp_hand is None:
            mismatches[0] += 1
            opp_hand = []
        out.append({"obs": prev, "action": act, "opp_hand": opp_hand})
    return out


def write_shard(out, idx, samples):
    path = out if idx == 0 else out.replace(".pkl", f".part{idx}.pkl")
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB, {len(samples)} samples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400, help="games PER opponent")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "training", "vd_diverse.pkl.gz"))
    ap.add_argument("--opponents", default="lucario,dragapult,abomasnow,starmie,mirror",
                     help="comma-separated subset of: " + ",".join(ANCHORS))
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    opp_names = [o.strip() for o in args.opponents.split(",") if o.strip()]
    for name in opp_names:
        if name not in ANCHORS:
            raise SystemExit(f"unknown opponent {name!r}, choices: {list(ANCHORS)}")

    samples = []
    games = wins = 0
    mismatches = [0, 0]  # [mismatched, total]
    shard_idx = total_samples = 0
    CHUNK = 100
    SHARD_SAMPLES = 100_000
    for name in opp_names:
        opp_path = ANCHORS[name]
        is_mirror = os.path.abspath(opp_path) == os.path.abspath(MAIN)
        print(f"Collecting {args.games} games: {MAIN} vs {name} ({opp_path})")
        remaining = args.games
        while remaining > 0:
            n = min(CHUNK, remaining)
            remaining -= n
            results = run_matches(MAIN, opp_path, n, workers=args.workers,
                                   keep_steps=True, progress=False)
            for r in results:
                if "error" in r or "steps" not in r:
                    continue
                games += 1
                outcome = r["rewards"][0]
                if outcome == 1:
                    wins += 1
                for d in extract_decisions_oracle(r["steps"], seat=0, mismatches=mismatches):
                    d["outcome"] = outcome
                    samples.append(d)
                if is_mirror:
                    outcome1 = r["rewards"][1]
                    for d in extract_decisions_oracle(r["steps"], seat=1, mismatches=mismatches):
                        d["outcome"] = outcome1
                        samples.append(d)
            print(f"  {name}: {games} games total, {len(samples)} buffered samples "
                  f"(shard {shard_idx})", file=sys.stderr)
            if len(samples) >= SHARD_SAMPLES:
                write_shard(args.out, shard_idx, samples)
                total_samples += len(samples)
                samples = []
                shard_idx += 1

    total_samples += len(samples)
    if samples or shard_idx == 0:
        write_shard(args.out, shard_idx, samples)
        shard_idx += 1
    miss_rate = mismatches[0] / max(mismatches[1], 1)
    print(f"games={games} our_winrate={wins/max(games,1):.3f} "
          f"samples={total_samples} shards={shard_idx} "
          f"oracle_exact_match_rate={1 - miss_rate:.4f} "
          f"(misses {mismatches[0]}/{mismatches[1]})")


if __name__ == "__main__":
    main()
