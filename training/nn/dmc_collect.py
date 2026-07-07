"""DMC curriculum data collection: the current learner (dmc_agent.py,
epsilon-greedy) plays a MIX of (a) frozen main.py (as before, exploiter_collect.py's
recipe), (b) past DMC checkpoints from an opponent pool (dmc_agent_pool.py,
greedy), and (c) real coded opponent bots (opponents/*_agent.py) -- seats
alternated in all cases. Records the learner's own chosen actions + game
outcome, same schema as exploiter_collect.py, so train_dmc.py needs no
changes.

Rationale (docs/report-log.md 2026-07-05 "DMC round 4"): rounds 1-3 trained
and evaluated only against a single frozen, non-adapting opponent (main.py),
giving no curriculum/intermediate difficulty -- diagnosed as the likely reason
learning was real but slow (1.0%->1.7%->2.5% over 3 rounds). Mixing in
self-play against the learner's own past checkpoints gives it opponents at
its own skill level to climb through, in addition to the fixed heuristic
target used for the pre-registered win-rate gate.

Phase 0 item 3 (docs/nn-training.md §Phase 0, 2026-07-05): real coded bots
(different decks, different piloting logic, not mirror/checkpoint self-play)
are mixed in via --bots/--bots-frac -- the closed PIMC-rollout line's own
root-cause diagnosis was that mirror-only opponents give ZERO discriminating
signal (90/90 vs mirror-self, confirmed via a targeted diagnostic), so this
collector must never default to mirror-only diversity.

Usage:
  python training/nn/dmc_collect.py --games 1000 --ckpt training/ptcg_dmc_r2.pth \
      --pool training/ptcg_dmc_r2.pth --pool-frac 0.4 \
      --bots opponents/lucario_agent.py,opponents/dragapult_agent.py,opponents/abomasnow_agent.py,opponents/starmie_agent.py \
      --bots-frac 0.3 --out training/dmc_r4.pkl.gz
"""
import argparse
import csv
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
from bc_collect import extract_decisions  # noqa: E402
from dmc_nstep import compute_nstep_targets  # noqa: E402

NET_AGENT = os.path.join(_HERE, "dmc_agent.py")
POOL_AGENT = os.path.join(_HERE, "dmc_agent_pool.py")
MAIN = os.path.join(REPO_ROOT, "main.py")
DEFAULT_CKPT = os.path.join(REPO_ROOT, "training", "ptcg_dmc_r2.pth")


def write_shard(out, idx, samples):
    path = out if idx == 0 else out.replace(".pkl", f".part{idx}.pkl")
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB, {len(samples)} samples)")


def _csv_path(out):
    base = out
    for ext in (".pkl.gz", ".pkl"):
        if base.endswith(ext):
            return base[:-len(ext)] + "_games.csv"
    return base + "_games.csv"


def play_chunk(csv_w, opponent_path, opponent_label, n, samples, counters,
               n_step=None, bootstrap_ckpt=None, use_phi_shaping=False):
    """Play n games of the current learner vs opponent_path, seats alternated,
    appending outcome-labeled decisions (from the learner's seat) to samples.

    n_step/bootstrap_ckpt/use_phi_shaping (Phase 0 items 1+2,
    docs/nn-training.md §Phase 0): when either is set, each game's decisions
    get an n-step-bootstrapped and/or Φ-shaped target (see dmc_nstep.py)
    instead of the flat full-episode outcome. All-default (n_step=None,
    use_phi_shaping=False) is the original, unchanged full-Monte-Carlo
    behavior."""
    for net_seat, n_seat in ((0, n - n // 2), (1, n // 2)):
        if n_seat == 0:
            continue
        paths = (NET_AGENT, opponent_path) if net_seat == 0 else (opponent_path, NET_AGENT)
        results = run_matches(paths[0], paths[1], n_seat, workers=None, keep_steps=True, progress=False)
        for r in results:
            if "error" in r or "steps" not in r:
                continue
            outcome = r["rewards"][net_seat]
            result = "win" if outcome == 1 else ("loss" if outcome == -1 else "tie")
            game_id = counters["games"]
            csv_w.writerow([game_id, net_seat, opponent_label, result])
            counters["games"] += 1
            if outcome == 1:
                counters["wins"] += 1
            game_decisions = extract_decisions(r["steps"], seat=net_seat)
            for d in game_decisions:
                d["game_id"] = game_id  # preserves game boundaries for offline relabeling (dmc_relabel.py)
            if n_step is not None or use_phi_shaping:
                compute_nstep_targets(bootstrap_ckpt, game_decisions, outcome or 0,
                                       n_step=n_step, use_phi_shaping=use_phi_shaping)
            else:
                for d in game_decisions:
                    d["outcome"] = outcome or 0
            samples.extend(game_decisions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="current learner checkpoint (NET_CKPT)")
    ap.add_argument("--eps", type=float, default=0.1, help="learner exploration epsilon (NET_EPS)")
    ap.add_argument("--pool", default="", help="comma-separated past checkpoint paths to self-play against")
    ap.add_argument("--pool-frac", type=float, default=0.4, help="fraction of games played vs the pool (rest vs frozen main.py)")
    ap.add_argument("--bots", default="",
                     help="Phase 0 item 3: comma-separated real coded opponent module paths "
                          "(e.g. opponents/lucario_agent.py,opponents/dragapult_agent.py,"
                          "opponents/abomasnow_agent.py,opponents/starmie_agent.py). Opt-in, "
                          "like --pool -- empty (default) preserves prior behavior.")
    ap.add_argument("--bots-frac", type=float, default=0.3,
                     help="fraction of games played vs --bots (only applied if --bots is non-empty)")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "training", "dmc_r4.pkl.gz"))
    ap.add_argument("--n-step", type=int, default=None,
                     help="Phase 0 item 1: bootstrap targets at this horizon instead of "
                          "full-episode Monte Carlo (default). E.g. 1, 5, 15.")
    ap.add_argument("--bootstrap-ckpt", default=None,
                     help="checkpoint whose Q-head supplies the n-step bootstrap value "
                          "(defaults to --ckpt, i.e. the current learner, if --n-step is set)")
    ap.add_argument("--phi-shaping", action="store_true",
                     help="Phase 0 item 2: add potential-based Φ-shaping to the target "
                          "(orthogonal to --n-step -- can combine, or use alone with full-MC)")
    args = ap.parse_args()

    os.environ["NET_CKPT"] = os.path.abspath(args.ckpt)
    os.environ["NET_EPS"] = str(args.eps)
    bootstrap_ckpt = os.path.abspath(args.bootstrap_ckpt or args.ckpt) if args.n_step is not None else None

    pool = [p.strip() for p in args.pool.split(",") if p.strip()]
    pool_frac = args.pool_frac if pool else 0.0
    bots = [os.path.abspath(os.path.join(REPO_ROOT, b.strip())) if not os.path.isabs(b.strip())
            else b.strip() for b in args.bots.split(",") if b.strip()]
    bots_frac = args.bots_frac if bots else 0.0

    mix_desc = []
    if bots:
        mix_desc.append(f"{bots_frac:.0%} real bots{[os.path.basename(b) for b in bots]}")
    if pool:
        mix_desc.append(f"{pool_frac:.0%} pool{[os.path.basename(p) for p in pool]}")
    mix_desc.append(f"{max(0.0, 1 - pool_frac - bots_frac):.0%} frozen main.py")
    print(f"Collecting {args.games} games: learner (ckpt={args.ckpt}, eps={args.eps}) vs "
          + ", ".join(mix_desc))

    csv_path = _csv_path(args.out)
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["game_idx", "net_seat", "opponent", "result"])

    samples = []
    counters = {"games": 0, "wins": 0}
    shard_idx = total_samples = 0
    CHUNK = 100  # full step traces are heavy -- extract and discard per chunk
    SHARD_SAMPLES = 100_000
    remaining = args.games
    pool_i = bots_i = 0
    while remaining > 0:
        n = min(CHUNK, remaining)
        remaining -= n
        n_pool = round(n * pool_frac)
        n_bots = round(n * bots_frac)
        n_main = max(0, n - n_pool - n_bots)

        if n_main > 0:
            play_chunk(csv_w, MAIN, "main.py", n_main, samples, counters,
                       n_step=args.n_step, bootstrap_ckpt=bootstrap_ckpt,
                       use_phi_shaping=args.phi_shaping)

        if n_bots > 0 and bots:
            # round-robin the real bots across a chunk so each gets roughly
            # equal representation, same pattern as the checkpoint pool below
            per_bot = max(1, n_bots // len(bots))
            done = 0
            while done < n_bots:
                bot = bots[bots_i % len(bots)]
                bots_i += 1
                take = min(per_bot, n_bots - done)
                play_chunk(csv_w, bot, os.path.basename(bot), take, samples, counters,
                           n_step=args.n_step, bootstrap_ckpt=bootstrap_ckpt,
                           use_phi_shaping=args.phi_shaping)
                done += take

        if n_pool > 0 and pool:
            # round-robin the pool games across the given checkpoints so
            # each one gets roughly equal representation within a chunk
            per_ckpt = max(1, n_pool // len(pool))
            done = 0
            while done < n_pool:
                ckpt = pool[pool_i % len(pool)]
                pool_i += 1
                take = min(per_ckpt, n_pool - done)
                os.environ["NET_CKPT_POOL"] = os.path.abspath(ckpt)
                play_chunk(csv_w, POOL_AGENT, os.path.basename(ckpt), take, samples, counters,
                           n_step=args.n_step, bootstrap_ckpt=bootstrap_ckpt,
                           use_phi_shaping=args.phi_shaping)
                done += take

        print(f"  {counters['games']}/{args.games} games, {len(samples)} samples (shard {shard_idx})", file=sys.stderr)
        if len(samples) >= SHARD_SAMPLES:
            write_shard(args.out, shard_idx, samples)
            total_samples += len(samples)
            samples = []
            shard_idx += 1

    total_samples += len(samples)
    if samples or shard_idx == 0:
        write_shard(args.out, shard_idx, samples)
        shard_idx += 1
    csv_f.close()
    print(f"wrote {csv_path} ({counters['games']} rows)")
    print(f"games={counters['games']} learner_winrate_overall={counters['wins']/max(counters['games'],1):.3f} "
          f"samples={total_samples} shards={shard_idx}")


if __name__ == "__main__":
    main()
