"""Phase 2 (docs/nn-training.md "AlphaZero-Style Push"): self-play data
collection with REAL search-derived targets -- closing the training loop
that Phase 0-era work only used at inference time.

Asymmetric: OUR side plays via `mcts_leafeval_agent.py` (PUCT + leaf-eval
via the value net's own value head, reduced sims -- eval-time's 100 sims/
decision would make a full self-play generation intractable within this
project's runway), which -- when MCTS_COLLECT_LOG is set -- logs the real
MCTS visit counts (-> policy_target) and backed-up root value
(-> mcts_root_value) per decision. The OPPONENT side is sampled per-game
from `opponent_pool.py`'s real-ladder-meta-weighted pool of the project's
existing rule-based archetype bots (lucario/dragapult/starmie/abomasnow
with their own real decks + logic, several more archetypes' reconstructed
decks piloted by `generic_pilot.py`, and a mirror slice piloted by the real
heuristic `main.py`) -- NOT the same checkpoint's own unsearched policy.

**2026-07-07 redesign, replacing the original same-checkpoint-mirror
opponent:** the mirror opponent was far too weak (searching side won 96.7%
of games), which produced a training corpus with severely imbalanced
win/loss labels and collapsed the trained value head toward unconditionally
predicting "winning" -- confirmed as the root cause of round 2's regression
(see docs/report-log.md 2026-07-07 "Root cause of the Phase 2 round 2
regression"). A real, diverse, competitive opponent pool is the actual fix;
see opponent_pool.py's docstring for the exact weights and sourcing.

PARALLEL collection (2026-07-06, Fable-directed follow-up to the 30-game
serial validation milestone): each game is assigned a unique MCTS_GAME_ID via
harness.run_matches(extra_envs=...) (new -- see harness.py), which
mcts_leafeval_agent.py embeds in every collect-log record it writes. This
lets records from DIFFERENT worker processes (each potentially handling many
games over its lifetime, in whatever order the pool schedules them) be
grouped back to the correct game by game_id rather than by process/file
order -- the earlier serial-only version relied on strict submission order
within one process, which doesn't hold once workers>1.

Usage:
  python training/nn/mcts_collect.py --games 150 --workers 10 \
      --ckpt training/ptcg_dmc_p0_v2_n1_richenc_v2.pth --sims 40 \
      --out training/mcts_p2_r2.pkl.gz
"""
import argparse
import glob
import gzip
import json
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
from selfplay_collect import compute_value_targets  # noqa: E402
from net_common import load_model  # noqa: E402
import opponent_pool  # noqa: E402

OUR_AGENT = os.path.join(_HERE, "mcts_leafeval_agent.py")


def _read_all_collect_logs(collect_log_base):
    """Reads back sequentially-appended pickle records from EVERY worker's
    log file (PID-suffixed, glob-matched) -- with workers>1, many distinct
    processes may each have written some of the games' records."""
    records = []
    for path in glob.glob(f"{collect_log_base}.*"):
        with open(path, "rb") as f:
            while True:
                try:
                    records.append(pickle.load(f))
                except EOFError:
                    break
    return records


def _obs_key(obs):
    """Stable key for matching the SAME logical obs across two different
    code paths (a live agent() call vs. the same state reconstructed from
    env.steps afterward). Two things confirmed empirically NOT safe here:
    (1) pickle.dumps() comparison -- two dicts can be `==` equal with
    different internal key insertion order, which pickle serializes
    byte-differently; (2) including the full obs -- `remainingOverageTime`
    (a live decrementing clock) and `step` were found to differ/be
    inconsistently populated (None in one reconstruction path, a real int in
    the other) between the two representations for at least one seat
    direction, even for the logically same decision. Only `current` (full
    game state: turn, players, board) + `select` (the options being chosen
    from) are used -- together they uniquely identify a decision without
    the volatile metadata fields."""
    return json.dumps({"current": obs.get("current"), "select": obs.get("select")},
                       sort_keys=True, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--ckpt", required=True, help="checkpoint for BOTH sides (true self-play)")
    ap.add_argument("--sims", type=int, default=40, help="reduced sims/decision (vs eval-time's 100)")
    ap.add_argument("--temp", type=float, default=1.0, help="opponent-side temperature (selfplay_agent.py)")
    ap.add_argument("--workers", type=int, default=None, help="None = auto (cpu_count-1)")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "training", "mcts_p2_r2.pkl.gz"))
    args = ap.parse_args()

    os.environ["NET_CKPT"] = os.path.abspath(args.ckpt)
    os.environ["MCTS_SIMS"] = str(args.sims)
    os.environ["NET_TEMP"] = str(args.temp)
    # Phase 2 checkpoints are train_sp.py-lineage (separate value_head, logits
    # are policy preferences) — mcts.py's MCTSSearcher net-leaf-eval must read
    # the value_head, not the DMC "max policy logit as Q" convention its
    # net_value_source default preserves for the older Phase 0 probe. See
    # docs/report-log.md 2026-07-07 "mcts.py _net_leaf_value convention
    # mismatch" — the missing override here is exactly the bug that corrupted
    # the round-2 (mcts_p2_r3.pkl.gz) training corpus's bootstrap values.
    os.environ["MCTS_NET_VALUE_SOURCE"] = "head"
    collect_log_base = os.path.join(REPO_ROOT, "training", "mcts_collect")
    os.environ["MCTS_COLLECT_LOG"] = collect_log_base
    for stale in glob.glob(f"{collect_log_base}.*"):
        os.remove(stale)

    bootstrap_model = load_model(os.path.abspath(args.ckpt))

    samples = []
    counters = {"games": 0, "wins": 0, "relabel_errors": 0, "decisions_seen": 0, "decisions_matched": 0}
    per_opp = {}  # label -> {"games":n, "wins":n}
    game_id_counter = 0

    for net_seat, n_seat in ((0, args.games - args.games // 2), (1, args.games // 2)):
        if n_seat == 0:
            continue
        # opponent_pool.allocate splits this seat's games across the real-
        # meta-weighted archetype pool -- one run_matches call per archetype
        # (not per-game) since run_matches binds one fixed opponent agent
        # path for its whole batch; deck overrides (for generic_pilot.py
        # archetypes) still vary via the `decks` param.
        for label, opp_path, opp_deck, n_this_opp in opponent_pool.allocate(n_seat, seed=net_seat):
            paths = (OUR_AGENT, opp_path) if net_seat == 0 else (opp_path, OUR_AGENT)
            deck_override = (None, opp_deck) if net_seat == 0 else (opp_deck, None)
            decks = [deck_override] * n_this_opp
            game_ids = list(range(game_id_counter, game_id_counter + n_this_opp))
            game_id_counter += n_this_opp
            extra_envs = [{"MCTS_GAME_ID": str(gid)} for gid in game_ids]

            print(f"[mcts_collect] net_seat={net_seat} opponent={label} n={n_this_opp}",
                  file=sys.stderr)
            results = run_matches(paths[0], paths[1], n_this_opp, workers=args.workers,
                                   keep_steps=True, progress=True, extra_envs=extra_envs,
                                   decks=decks)

            all_records = _read_all_collect_logs(collect_log_base)
            records_by_game = {}
            for rec in all_records:
                records_by_game.setdefault(rec.get("game_id"), []).append(rec)

            opp_stats = per_opp.setdefault(label, {"games": 0, "wins": 0})
            for r in results:
                if "error" in r or "steps" not in r:
                    continue
                gid = (r.get("extra_env") or {}).get("MCTS_GAME_ID")
                outcome = r["rewards"][net_seat]
                if outcome not in (1, -1, 0):
                    # A game can end without a clean win/loss/tie reward (e.g.
                    # truncated at the engine's internal step limit) -- see
                    # docs/report-log.md 2026-07-07 "Round 3 collection
                    # None-outcome crash". Skip it rather than crash the whole
                    # collection run on compute_value_targets's outcome*float.
                    counters["relabel_errors"] += 1
                    print(f"[mcts_collect] game (id={gid}, opponent={label}): "
                          f"non-numeric outcome {outcome!r} (statuses={r.get('statuses')}) "
                          "-- skipping", file=sys.stderr)
                    continue
                counters["games"] += 1
                opp_stats["games"] += 1
                if outcome == 1:
                    counters["wins"] += 1
                    opp_stats["wins"] += 1
                game_records_raw = records_by_game.get(gid, [])
                record_by_obs = {_obs_key(rec["obs"]): rec for rec in game_records_raw}

                all_decisions = extract_decisions(r["steps"], seat=net_seat)
                counters["decisions_seen"] += len(all_decisions)
                game_decisions, game_records = [], []
                for d in all_decisions:
                    rec = record_by_obs.get(_obs_key(d["obs"]))
                    if rec is not None:
                        game_decisions.append(d)
                        game_records.append(rec)
                counters["decisions_matched"] += len(game_decisions)
                if not game_decisions:
                    counters["relabel_errors"] += 1
                    print(f"[mcts_collect] game {counters['games']} (id={gid}, "
                          f"opponent={label}): 0/{len(all_decisions)} decisions matched "
                          "a search record -- skipping", file=sys.stderr)
                    continue
                mcts_root_values = [rec["root_value"] for rec in game_records]
                for d, rec in zip(game_decisions, game_records):
                    d["policy_target"] = rec["policy_target"]
                compute_value_targets(bootstrap_model, game_decisions, outcome,
                                       mcts_root_values=mcts_root_values)
                samples.extend(game_decisions)
            # Checkpoint after every opponent batch -- a 300-game run with
            # real search is slow enough (see docs/report-log.md 2026-07-07
            # "Round 3 collection timeout") that a kill/timeout partway
            # through must not lose everything collected so far.
            opener = gzip.open if args.out.endswith(".gz") else open
            with opener(args.out, "wb") as f:
                pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[mcts_collect] checkpoint: {len(samples)} samples from "
                  f"{counters['games']} games written to {args.out}", file=sys.stderr)
            # clear this archetype's collect logs before the next run_matches
            # call reuses the same worker pool -- otherwise the next
            # archetype's _read_all_collect_logs would also pick up stale
            # records from this one (game_ids never repeat, so matching would
            # still be correct, but the file would grow unboundedly).
            for stale in glob.glob(f"{collect_log_base}.*"):
                os.remove(stale)

    match_rate = counters["decisions_matched"] / max(counters["decisions_seen"], 1)
    print(f"games={counters['games']} wins={counters['wins']} "
          f"winrate={counters['wins']/max(counters['games'],1):.3f} "
          f"samples={len(samples)} relabel_errors={counters['relabel_errors']} "
          f"decisions_seen={counters['decisions_seen']} "
          f"decisions_matched={counters['decisions_matched']} match_rate={match_rate:.3f}")
    for label, s in sorted(per_opp.items(), key=lambda kv: -kv[1]["games"]):
        wr = s["wins"] / max(s["games"], 1)
        print(f"  opponent={label:16s} games={s['games']:4d} winrate={wr:.3f}", file=sys.stderr)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
