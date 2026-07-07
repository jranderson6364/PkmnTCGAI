"""Self-play data collection: current net (temperature-sampling, for
exploration) plays full games against itself via the local engine. Computes
n-step bootstrapped value targets per docs/nn-training.md §Value Targets:

    G_t = sum_{k<n} gamma^k * r_{t+k} + gamma^n * V(s_{t+n})
    target = 0.7 * terminal_outcome + 0.3 * G_t

using the CURRENT net's own value head for V() (bootstrap) and a shaped
intermediate reward (prizes taken/conceded + hand-vs-KO-threshold progress,
see training/README.md §Curriculum & Reward Shaping). Saved samples add a
`value_target` field on top of the BC sample schema; dataset.py falls back to
plain `outcome` when that field is absent, so BC and SP shards share one
Dataset/collate implementation.

Usage:
  python selfplay_collect.py --games 300 --ckpt ../ptcg_bc_v1.pth --out ../sp_data.pkl.gz
"""
import argparse
import glob
import gzip
import math
import os
import pickle
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from harness import run_matches
from bc_collect import extract_decisions
from net_common import load_model, value_estimate

GAMMA = 0.997
N_STEP = 10
TERMINAL_WEIGHT = 0.7


def _pk_hp(obs, me_idx, opp=True):
    cur = obs.get("current") or {}
    pl = cur.get("players") or []
    idx = (1 - me_idx) if opp else me_idx
    p = pl[idx] if len(pl) > idx else {}
    a = (p.get("active") or [None])
    active = a[0] if a and a[0] else None
    return (active or {}).get("hp", 99999) or 99999


def shaped_reward(obs_before, obs_after, me_idx):
    """One decision-to-next-own-decision shaped reward (the turn-level design
    from training/README.md adapted to our per-decision spacing)."""
    cur_b = obs_before.get("current") or {}
    cur_a = obs_after.get("current") or {}
    pl_b, pl_a = cur_b.get("players") or [], cur_a.get("players") or []
    if len(pl_b) < 2 or len(pl_a) < 2:
        return 0.0
    my_b, my_a = pl_b[me_idx], pl_a[me_idx]
    opp_b, opp_a = pl_b[1 - me_idx], pl_a[1 - me_idx]
    r = 0.0
    r += (len(opp_b.get("prize") or []) - len(opp_a.get("prize") or [])) * 1.0
    r -= (len(my_b.get("prize") or []) - len(my_a.get("prize") or [])) * 0.5
    opp_active = (opp_a.get("active") or [None])
    opp_active = opp_active[0] if opp_active and opp_active[0] else None
    opp_hp = (opp_active or {}).get("hp", 99999) or 99999
    hand_n = my_a.get("handCount") or len(my_a.get("hand") or [])
    needed = max(1, math.ceil(opp_hp / 20)) if opp_hp < 99999 else 999
    r += min(hand_n / needed, 1.0) * 0.1
    return r


def compute_value_targets(model, decisions, outcome, mcts_root_values=None):
    """decisions: list of {obs, action}. Adds 'value_target' in place.

    mcts_root_values: optional list (same length as decisions) of MCTS-backed-up
    root Q values, one per decision. Where a decision was actually searched
    (mcts_root_values[t] is not None), that replaces the raw value-head estimate
    as the bootstrap V(s_t) — a stronger leaf/bootstrap signal than the net's
    un-searched value head alone. Decisions without a tree search (None, or when
    this whole argument is omitted) fall back to the plain value head, exactly
    as before — this keeps selfplay_collect.py's direct self-play path
    (no search) working unchanged; mcts_collect.py (Kaggle-only, not yet built)
    is the intended caller that will pass real root values."""
    n = len(decisions)
    if n == 0:
        return
    if mcts_root_values is None:
        mcts_root_values = [None] * n
    values = [
        mcts_root_values[i] if mcts_root_values[i] is not None
        else value_estimate(model, d["obs"], d["obs"]["select"])
        for i, d in enumerate(decisions)
    ]
    rewards = [0.0] * n
    for t in range(n - 1):
        # 2026-07-06 fix: me_idx was hardcoded 0, but `decisions` comes from
        # extract_decisions(steps, seat=net_seat) and net_seat is 1 for half
        # of any collection run that alternates seats (as mcts_collect.py and
        # dmc_collect.py both do) -- for those games this silently computed
        # the shaped reward from the OPPONENT's perspective (swapping "my"
        # and "opp" prize/hand progress). Read the real seat from each
        # decision's own obs instead of assuming 0. Same bug class as the
        # one already found and fixed in dmc_nstep.py's _phi_at this session.
        me_idx = (decisions[t]["obs"].get("current") or {}).get("yourIndex", 0)
        rewards[t] = shaped_reward(decisions[t]["obs"], decisions[t + 1]["obs"], me_idx)
    for t in range(n):
        G = 0.0
        disc = 1.0
        end = min(t + N_STEP, n)
        for k in range(t, end):
            G += disc * rewards[k]
            disc *= GAMMA
        if end < n:
            G += disc * values[end]
        else:
            G += disc * outcome
        target = TERMINAL_WEIGHT * outcome + (1 - TERMINAL_WEIGHT) * G
        # value head is tanh-bounded [-1,1]; shaped-reward accumulation can
        # push raw targets slightly outside that range — clip so training
        # doesn't chase an unreachable label.
        decisions[t]["value_target"] = max(-1.0, min(1.0, target))
        decisions[t]["outcome"] = outcome
        # raw V(s_t) from the collecting net, stored so AWR training (Stage 2)
        # can compute advantage = value_target - v_pred without re-running the
        # value head at train time.
        decisions[t]["v_pred"] = values[t]


def _shard_path(out, idx):
    return out if idx == 0 else out.replace(".pkl", f".part{idx}.pkl")


def _next_shard_idx(out):
    """Scan for existing shards matching `out`'s naming so a second invocation
    (--out pointing at the same base path) ADDS new shards instead of
    overwriting shard 0 — supports the "collect a bit now, add more later"
    workflow without clobbering prior runs."""
    base = out[:-len(".pkl.gz")] if out.endswith(".pkl.gz") else out[:-len(".pkl")]
    ext = ".pkl.gz" if out.endswith(".pkl.gz") else ".pkl"
    pattern = re.compile(re.escape(os.path.basename(base)) + r"(?:\.part(\d+))?" + re.escape(ext) + r"$")
    max_idx = -1
    for f in glob.glob(os.path.join(os.path.dirname(out) or ".", "*")):
        m = pattern.match(os.path.basename(f))
        if m:
            idx = int(m.group(1)) if m.group(1) else 0
            max_idx = max(max_idx, idx)
    return max_idx + 1


def write_shard(out, idx, samples):
    path = _shard_path(out, idx)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB, {len(samples)} samples)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ptcg_bc_v1.pth"))
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sp_data.pkl.gz"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--chunk-games", type=int, default=100,
                     help="games per collect+value-target+write cycle — bounds how much "
                          "work a kill can lose, since each chunk is written to disk "
                          "before starting the next")
    ap.add_argument("--shard-samples", type=int, default=50_000,
                     help="flush a new shard file once the in-memory sample buffer "
                          "reaches this size (keeps memory bounded on long runs)")
    args = ap.parse_args()

    os.environ["NET_CKPT"] = os.path.abspath(args.ckpt)
    os.environ["NET_TEMP"] = str(args.temp)
    agent_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selfplay_agent.py")

    model = load_model(os.path.abspath(args.ckpt))  # for value-target bootstrapping
    shard_idx = _next_shard_idx(args.out)
    if shard_idx > 0:
        print(f"found existing shards — continuing at shard {shard_idx}", file=sys.stderr)

    print(f"Collecting {args.games} self-play games with ckpt={args.ckpt} temp={args.temp}")
    samples = []
    games = total_samples = 0
    remaining = args.games
    while remaining > 0:
        n = min(args.chunk_games, remaining)
        remaining -= n
        results = run_matches(agent_path, agent_path, n, workers=args.workers,
                               keep_steps=True, progress=False)
        for r in results:
            if "error" in r or "steps" not in r:
                continue
            games += 1
            for seat in (0, 1):
                outcome = r["rewards"][seat]
                decs = extract_decisions(r["steps"], seat=seat)
                compute_value_targets(model, decs, outcome or 0)
                samples.extend(decs)
        print(f"  {games}/{args.games} games, {len(samples)} buffered samples "
              f"(next shard {shard_idx})", file=sys.stderr)
        if len(samples) >= args.shard_samples:
            write_shard(args.out, shard_idx, samples)
            total_samples += len(samples)
            samples = []
            shard_idx += 1

    total_samples += len(samples)
    if samples or shard_idx == 0:
        write_shard(args.out, shard_idx, samples)
        shard_idx += 1
    print(f"games={games} samples={total_samples} shards_written_this_run={shard_idx}")


if __name__ == "__main__":
    main()
