"""Phase B of the learn-inside-the-champion pivot (issue #3): from-state
self-play collection for the board-thinning/deck-out regime Q-net.

Two sources, one schema (train_dmc.py-compatible: obs/action/outcome/game_id):

SOURCE A (dense, default): continuations from mined failure states.
  For each TRAIN-split seed (the first in-regime decision of each
  replays/exploiter_wins mirror loss), re-determinize hidden zones fresh per
  continuation (mcts.py's proven filler + search_begin path, mirror decks
  both sides), then play to a real terminal with main.agent on BOTH sides,
  epsilon-random exploration on OUR side at in-regime single-select
  decisions only. Every OUR in-regime decision is recorded, labeled with the
  continuation's real terminal outcome (full MC, the confirmed-best DMC
  target on this problem).

  Seed split is at GAME level (fixed RNG seed 7, 30% held out): held-out
  seeds are NEVER collected here — they and the 10 v29d_ladder thinning
  losses form Phase C's scenario-suite gate, so gate states are untrained-on
  by construction.

SOURCE B (--fresh-games): whole games of the exploration wrapper
  (regime_explore_agent.py) vs mirror + real anchor bots via the standard
  harness, seats ALTERNATED, keeping only in-regime decisions. Low yield but
  organic + diverse, and the vehicle for the mandatory seat-swap
  verification (labels must flip with the learner's seat).

Seat-swap verification (issue #3 acceptance criterion 2) = --verify-seats:
  runs Source B for a small batch and asserts per-seat outcome accounting
  matches harness rewards exactly, plus the _terminal_value sign-flip unit
  check on real terminals from Source A. Collection refuses to run the full
  haul until this passes in the same invocation (--verify-seats runs first).

Usage (smoke):
  python training/nn/regime_collect.py --continuations 3 --verify-seats
Usage (overnight):
  python training/nn/regime_collect.py --continuations 600 --fresh-games 2000 \
      --verify-seats --out training/regime_r1.pkl.gz
"""
import argparse
import csv
import glob
import gzip
import json
import os
import pickle
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_REPO_ROOT, "training"),
          os.path.join(_REPO_ROOT, "training", "local_cg"),
          os.path.join(_REPO_ROOT, "training", "belief"), _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import main as heuristic  # noqa: E402
import mcts as _mcts  # noqa: E402
from mcts import (MCTSSearcher, _obs_to_dict, _is_terminal,  # noqa: E402
                  _terminal_value, _is_multiselect)
from regime_detector import regime_fires, walk_game  # noqa: E402
from eval_v4 import our_seat  # noqa: E402

SPLIT_SEED = 7
HELDOUT_FRAC = 0.30
MAX_PLIES = 300
DEVIATE_AT_MAX = 24  # ~2x the observed median in-regime decisions/continuation


def load_seeds():
    """Returns (train_seeds, heldout_seeds): lists of (seed_id, obs_dict,
    our_seat) — the FIRST in-regime our-decision obs of each exploiter-win
    mirror loss. Ladder losses are NOT seeds (reserved for the gate suite)."""
    seeds = []
    for path in sorted(glob.glob(os.path.join(_REPO_ROOT, "replays",
                                              "exploiter_wins", "*.json"))):
        d = json.load(open(path, encoding="utf-8"))
        you = our_seat(d.get("info", {}))
        if you is None or not d.get("rewards") or d["rewards"][you] != -1:
            continue
        for step in d.get("steps", []):
            if len(step) <= you:
                continue
            rec = step[you]
            obs = rec.get("observation") if rec else None
            if not obs:
                continue
            cur = obs.get("current") or {}
            if cur.get("yourIndex") != you:
                continue
            sel = obs.get("select")
            if not sel or not sel.get("option"):
                continue
            if regime_fires(cur, you):
                seeds.append((os.path.basename(path), obs, you))
                break
    rng = random.Random(SPLIT_SEED)
    idx = list(range(len(seeds)))
    rng.shuffle(idx)
    n_held = max(1, round(len(seeds) * HELDOUT_FRAC))
    held = {i for i in idx[:n_held]}
    train = [s for i, s in enumerate(seeds) if i not in held]
    heldout = [s for i, s in enumerate(seeds) if i in held]
    return train, heldout


def _reset_stateful():
    saved = {}
    for module, attrs in _mcts._STATEFUL_MODULES.items():
        saved[module] = {a: getattr(module, a) for a in attrs if hasattr(module, a)}
        for a, factory in attrs.items():
            if hasattr(module, a):
                setattr(module, a, factory())
    return saved


def _restore_stateful(saved):
    for module, vals in saved.items():
        for a, v in vals.items():
            setattr(module, a, v)


def continue_from(seed_obs, seed_seat, eps, rng, unit_checks, deviate_at=None):
    """One continuation: fresh mirror determinization -> play to terminal.
    Returns (regime_decisions, outcome, plies, deviated) or None on ply cap.
    regime_decisions: [{"obs":…, "action":[i]}] for OUR in-regime decisions.

    deviate_at (round 2, the pre-registered iteration): instead of eps-random
    at EVERY in-regime decision — which round 1 proved destroys all win
    contrast (0/13,000 wins; a rescue needs a ~30-decision clean sequence and
    0.75^30 ≈ 2e-4) — take ONE uniform-random action at the deviate_at-th
    in-regime decision and play pure heuristic everywhere else. This is the
    one-step-deviation estimator of Q^v29d(s, a), the exact counterfactual
    the hybrid deploys (override v29d at a state, v29d plays on)."""
    from cg.api import to_observation_class, search_begin, search_step, search_end

    observation = to_observation_class(seed_obs)
    state = observation.current
    my_p = state.players[seed_seat]
    opp_p = state.players[1 - seed_seat]
    filler = MCTSSearcher._filler
    your_deck = filler(my_p.deckCount, heuristic.DECK)
    your_prize = filler(len(my_p.prize), heuristic.DECK)
    opponent_deck = filler(opp_p.deckCount, heuristic.DECK)   # mirror
    opponent_prize = filler(len(opp_p.prize), heuristic.DECK)
    opponent_hand = filler(opp_p.handCount, heuristic.DECK)
    opponent_active = []
    if len(opp_p.active) > 0 and opp_p.active[0] is None:
        opponent_active = filler(1, heuristic.DECK)

    saved = _reset_stateful()
    try:
        ss = search_begin(observation, your_deck, your_prize, opponent_deck,
                          opponent_prize, opponent_hand, opponent_active,
                          manual_coin=False)
        cur_id, cur_obs = ss.searchId, seed_obs
        decisions = []
        in_regime_seen = 0
        for ply in range(MAX_PLIES):
            if _is_terminal(cur_obs):
                out = _terminal_value(cur_obs, seed_seat)
                # unit seat check on real terminals: the same obs valued
                # from the opposite seat must be the exact negation
                if unit_checks is not None and out != 0.0:
                    flipped = _terminal_value(cur_obs, 1 - seed_seat)
                    assert flipped == -out, (
                        f"seat-flip check FAILED: {out} vs {flipped}")
                    unit_checks.append(True)
                deviated = deviate_at is not None and in_regime_seen >= deviate_at
                return decisions, out, ply, deviated
            yours = cur_obs["current"]["yourIndex"]
            sel = cur_obs.get("select") or {}
            opts = sel.get("option") or []
            action = heuristic.agent(cur_obs)
            action = list(action) if action else [0]
            if yours == seed_seat and not _is_multiselect(sel) and len(opts) > 1 \
                    and regime_fires(cur_obs["current"], seed_seat):
                in_regime_seen += 1
                if deviate_at is not None:
                    if in_regime_seen == deviate_at:
                        action = [rng.randrange(len(opts))]
                elif rng.random() < eps:
                    action = [rng.randrange(len(opts))]
                decisions.append({"obs": cur_obs, "action": list(action)})
            ss = search_step(cur_id, action)
            cur_id, cur_obs = ss.searchId, _obs_to_dict(ss.observation)
        return None  # ply cap — unlabeled, discard
    finally:
        _restore_stateful(saved)
        try:
            search_end()
        except Exception:
            pass


SHARD_SAMPLES = 50_000


def maybe_flush(args, samples, counters, force=False):
    """Incremental shard write — the mcts_collect lesson (a 9-11h run once
    lost everything to a crash with no interim checkpoints; this collector
    repeated that on 2026-07-14 night 1 and lost Source A to a Source B
    crash). Flush every SHARD_SAMPLES and at source boundaries."""
    if not samples or (len(samples) < SHARD_SAMPLES and not force):
        return
    write_shard(args.out, counters["shard_idx"], list(samples))
    counters["shard_idx"] += 1
    counters["flushed"] += len(samples)
    samples.clear()


def run_source_a(args, samples, csv_w, counters, rng):
    train_seeds, heldout_seeds = load_seeds()
    print(f"seeds: {len(train_seeds)} train, {len(heldout_seeds)} HELD OUT "
          f"(split_seed={SPLIT_SEED}):")
    for sid, _o, _s in heldout_seeds:
        print(f"  heldout: {sid}")
    unit_checks = [] if args.verify_seats else None
    for sid, obs, seat in train_seeds:
        for k in range(args.continuations):
            deviate_at = rng.randint(1, DEVIATE_AT_MAX) if args.deviate_once else None
            res = continue_from(obs, seat, args.eps, rng, unit_checks, deviate_at)
            if res is None:
                counters["capped"] += 1
                continue
            decisions, outcome, plies, deviated = res
            gid = counters["games"]
            counters["games"] += 1
            counters["wins"] += 1 if outcome == 1 else 0
            counters["deviated"] += 1 if deviated else 0
            csv_w.writerow(["A", sid, gid, outcome, len(decisions), plies,
                            int(deviated)])
            for d in decisions:
                d["outcome"] = outcome
                d["game_id"] = gid
                d["seed_id"] = sid
            samples.extend(decisions)
        print(f"  seed {sid}: cum games={counters['games']} "
              f"winrate={counters['wins']/max(1,counters['games']):.2%} "
              f"samples={counters['flushed'] + len(samples)} "
              f"capped={counters['capped']} deviated={counters['deviated']}",
              file=sys.stderr)
        maybe_flush(args, samples, counters)
    if unit_checks is not None:
        print(f"seat-flip unit check: {len(unit_checks)} terminals verified OK")


def run_source_b(args, samples, csv_w, counters):
    from harness import run_matches
    from bc_collect import extract_decisions
    wrapper = os.path.join(_HERE, "regime_explore_agent.py")
    all_opponents = {"main": os.path.join(_REPO_ROOT, "main.py"),
                     "lucario": os.path.join(_REPO_ROOT, "opponents", "lucario_agent.py"),
                     "abomasnow": os.path.join(_REPO_ROOT, "opponents", "abomasnow_agent.py"),
                     "starmie": os.path.join(_REPO_ROOT, "opponents", "starmie_agent.py")}
    names = [n.strip() for n in args.opponents.split(",") if n.strip()]
    opponents = [all_opponents[n] for n in names]
    os.environ["REGIME_EPS"] = str(args.eps)
    per_opp = max(1, args.fresh_games // len(opponents))
    swap_stats = {"checked": 0}
    for opp in opponents:
        label = os.path.basename(opp)
        for net_seat in (0, 1):
            n_seat = per_opp - per_opp // 2 if net_seat == 0 else per_opp // 2
            if n_seat == 0:
                continue
            paths = (wrapper, opp) if net_seat == 0 else (opp, wrapper)
            results = run_matches(paths[0], paths[1], n_seat, workers=None,
                                  keep_steps=True, progress=False)
            for r in results:
                if "error" in r or "steps" not in r:
                    continue
                outcome = r["rewards"][net_seat]
                other = r["rewards"][1 - net_seat]
                if outcome is None or other is None:
                    # a side crashed mid-game — not a legitimate outcome
                    # label (and not a seat bug; night-2 finding: a starmie
                    # crash produced rewards=[1, None] and the strict
                    # assertion killed the run). Skip the game entirely.
                    counters["crashed"] = counters.get("crashed", 0) + 1
                    continue
                # seat-swap verification: label derives from net_seat, and
                # the two rewards must be a zero-sum pair when decisive
                if args.verify_seats and outcome in (1, -1):
                    assert other == -outcome, (
                        f"seat-swap FAILED: rewards={r['rewards']} net_seat={net_seat}")
                    swap_stats["checked"] += 1
                gid = counters["games"]
                counters["games"] += 1
                counters["wins"] += 1 if outcome == 1 else 0
                kept = 0
                for d in extract_decisions(r["steps"], seat=net_seat):
                    cur = d["obs"].get("current") or {}
                    sel = d["obs"].get("select") or {}
                    if _is_multiselect(sel) or len(sel.get("option") or []) <= 1:
                        continue
                    if not regime_fires(cur, net_seat):
                        continue
                    d["outcome"] = outcome or 0
                    d["game_id"] = gid
                    d["seed_id"] = f"fresh:{label}:seat{net_seat}"
                    samples.append(d)
                    kept += 1
                csv_w.writerow(["B", f"{label}:seat{net_seat}", gid, outcome,
                                kept, len(r["steps"]), ""])
        print(f"  fresh vs {label}: cum games={counters['games']} "
              f"samples={counters['flushed'] + len(samples)}", file=sys.stderr)
        maybe_flush(args, samples, counters)
    if args.verify_seats:
        print(f"seat-swap harness check: {swap_stats['checked']} decisive games, "
              f"all reward pairs zero-sum with labels keyed to net_seat OK")


def write_shard(out, idx, samples):
    path = out if idx == 0 else out.replace(".pkl", f".part{idx}.pkl")
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB, {len(samples)} samples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--continuations", type=int, default=600,
                    help="Source A continuations per train seed")
    ap.add_argument("--fresh-games", type=int, default=0,
                    help="Source B whole games (split across --opponents, seats alternated)")
    ap.add_argument("--opponents", default="main,lucario,abomasnow,starmie",
                    help="Source B opponent subset (comma list of main,lucario,abomasnow,starmie)")
    ap.add_argument("--eps", type=float, default=0.25,
                    help="in-regime exploration epsilon (both sources)")
    ap.add_argument("--deviate-once", action="store_true",
                    help="Source A round-2 mode: ONE uniform-random action at "
                         "a random in-regime decision per continuation, pure "
                         "heuristic otherwise (ignores --eps for Source A)")
    ap.add_argument("--verify-seats", action="store_true",
                    help="run the mandatory seat verifications (criterion 2)")
    ap.add_argument("--out", default=os.path.join(_REPO_ROOT, "training", "regime_r1.pkl.gz"))
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.rng_seed)
    csv_path = args.out.replace(".pkl.gz", "").replace(".pkl", "") + "_games.csv"
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["source", "seed_id", "game_id", "outcome",
                    "n_regime_decisions", "plies", "deviated"])

    samples = []
    counters = {"games": 0, "wins": 0, "capped": 0, "shard_idx": 0,
                "flushed": 0, "deviated": 0}

    if args.continuations > 0:
        run_source_a(args, samples, csv_w, counters, rng)
        maybe_flush(args, samples, counters, force=True)  # source boundary
    if args.fresh_games > 0:
        run_source_b(args, samples, csv_w, counters)

    csv_f.close()
    if samples or counters["shard_idx"] == 0:
        write_shard(args.out, counters["shard_idx"], samples)
        counters["flushed"] += len(samples)
    wr = counters["wins"] / max(1, counters["games"])
    print(f"TOTAL games={counters['games']} winrate={wr:.2%} "
          f"samples={counters['flushed']} capped={counters['capped']}")
    if not 0.02 < wr < 0.98:
        print("WARNING: outcome distribution degenerate — Q targets carry "
              "little contrast; inspect before training.")


if __name__ == "__main__":
    main()
