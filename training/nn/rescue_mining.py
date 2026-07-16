"""Rescue-pattern mining (heuristic-fix track, after issue #3 closed).

The round-2 corpus (`training/regime_r2*.pkl.gz`) contains one-step-deviation
continuations from 13 board-thinning failure states: exactly one
uniform-random action at a random in-regime decision, pure v29d otherwise.
This miner recovers each deviated game's deviation point (replaying
main.agent over the recorded decisions and finding the single mismatch) and
contrasts deviations that ended in WINS against ones that ended in LOSSES —
hypothesis generation for heuristic-codifiable rescue rules, to be tested by
the standard A/B gate, never adopted from this analysis alone.

Also prints, per seed, what the LAST in-regime decision state looks like in
wins vs losses (deck/hand/line/board) — the shape of a successful rescue.

Usage: python training/nn/rescue_mining.py [--loss-sample 2000]
"""
import argparse
import collections
import csv
import glob
import gzip
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
from regime_collect import _reset_stateful, _restore_stateful  # noqa: E402
from regime_detector import regime_features  # noqa: E402

OTYPE = ["NUMBER", "YES", "NO", "CARD", "TOOL_CARD", "ENERGY_CARD", "ENERGY",
         "PLAY", "ATTACH", "EVOLVE", "ABILITY", "DISCARD", "RETREAT",
         "ATTACK", "END", "SKILL", "SPECIAL_CONDITION"]
CSV_PATH = os.path.join(_REPO_ROOT, "training", "regime_r2_games.csv")
SHARD_GLOB = os.path.join(_REPO_ROOT, "training", "regime_r2*.pkl.gz")


def oname(opts, i):
    o = opts[i] if 0 <= i < len(opts) else {}
    t = o.get("type")
    return OTYPE[t] if isinstance(t, int) and 0 <= t < len(OTYPE) else f"t{t}"


def load():
    meta = {}
    with open(CSV_PATH, newline="") as f:
        for r in csv.reader(f):
            if r and r[0] == "A":
                meta[int(r[2])] = {"sid": r[1], "win": r[3] in ("1", "1.0"),
                                   "dev": r[6] == "1"}
    games = collections.defaultdict(list)
    for path in sorted(glob.glob(SHARD_GLOB)):
        with gzip.open(path, "rb") as f:
            for s in pickle.load(f):
                if s["game_id"] in meta:
                    games[s["game_id"]].append(s)
    return meta, games


def replay_mismatches(decisions):
    """Replay main.agent over a game's recorded in-regime decisions (fresh
    stall-memo, in order — matching collection) and return indices where the
    recorded action differs from the heuristic's choice."""
    saved = _reset_stateful()
    try:
        out = []
        for i, d in enumerate(decisions):
            pick = heuristic.agent(d["obs"])
            if (list(pick)[:1] or [None]) != list(d["action"])[:1]:
                out.append(i)
        return out
    finally:
        _restore_stateful(saved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss-sample", type=int, default=2000,
                    help="deviated LOSS games to replay (wins: all)")
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()

    meta, games = load()
    dev_wins = [g for g, m in meta.items() if m["dev"] and m["win"] and g in games]
    dev_losses = [g for g, m in meta.items() if m["dev"] and not m["win"] and g in games]
    rng = random.Random(args.rng_seed)
    loss_pick = rng.sample(dev_losses, min(args.loss_sample, len(dev_losses)))
    scale = len(dev_losses) / max(1, len(loss_pick))
    print(f"deviated wins: {len(dev_wins)} (all replayed); deviated losses: "
          f"{len(dev_losses)} (replaying {len(loss_pick)}, scale x{scale:.2f})")

    cls = collections.defaultdict(lambda: [0, 0.0])  # key -> [wins, losses_scaled]
    skipped = collections.Counter()

    def process(gids, is_win):
        for n, gid in enumerate(gids):
            ds = games[gid]  # shard order == play order (games never split)
            mm = replay_mismatches(ds)
            if len(mm) != 1:
                skipped[f"{'win' if is_win else 'loss'}_mm{len(mm)}"] += 1
                continue
            d = ds[mm[0]]
            saved = _reset_stateful()
            try:
                for j in range(mm[0]):          # rebuild memo path to the point
                    heuristic.agent(ds[j]["obs"])
                heur_pick = heuristic.agent(d["obs"])[0]
            finally:
                _restore_stateful(saved)
            obs = d["obs"]
            sel = obs["select"]
            opts = sel.get("option") or []
            r = regime_features(obs["current"], obs["current"]["yourIndex"]) or {}
            key = (sel.get("type"), oname(opts, heur_pick),
                   oname(opts, d["action"][0]))
            if is_win:
                cls[key][0] += 1
            else:
                cls[key][1] += scale
            if n % 200 == 0:
                print(f"  {'wins' if is_win else 'losses'}: {n}/{len(gids)}",
                      file=sys.stderr)

    process(dev_wins, True)
    process(loss_pick, False)
    print(f"skipped (mismatch count != 1): {dict(skipped)}")

    base = len(dev_wins) / max(1, len(dev_wins) + len(dev_losses))
    print(f"\nP(win | deviated) baseline = {base:.1%}")
    print(f"{'stype':>5} {'v29d would':>14} {'deviation did':>14} "
          f"{'wins':>5} {'~losses':>8} {'P(win|class)':>12}")
    rows = sorted(cls.items(), key=lambda kv: -kv[1][0])
    for (stype, h, dv), (w, ls) in rows:
        n = w + ls
        if n < 8:
            continue
        print(f"{stype!s:>5} {h:>14} {dv:>14} {w:>5} {ls:>8.0f} "
              f"{w / n:>11.1%}{'  <<<' if w / n > 2 * base and w >= 5 else ''}")

    # per-seed last-decision snapshot: what a successful rescue looks like
    print("\nlast in-regime decision state, per seed (win vs loss means):")
    feats = ["deck", "hand", "line_in_play", "line_in_hand", "board",
             "opp_armed", "turn"]
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for gid, m in meta.items():
        if gid not in games:
            continue
        last = games[gid][-1]
        r = regime_features(last["obs"]["current"],
                            last["obs"]["current"]["yourIndex"])
        if r:
            for k in feats:
                agg[(m["sid"], m["win"])][k].append(r[k])
    sids = sorted({s for s, _w in agg})
    print(f"{'seed':28} {'n(W/L)':>11} " + " ".join(f"{k:>12}" for k in feats))
    for sid in sids:
        for win in (True, False):
            a = agg.get((sid, win))
            if not a:
                continue
            n = len(a[feats[0]])
            means = " ".join(
                f"{sum(a[k])/n:>12.1f}" for k in feats)
            print(f"{sid:28} {('W' if win else 'L')+str(n):>11} {means}")


if __name__ == "__main__":
    main()
