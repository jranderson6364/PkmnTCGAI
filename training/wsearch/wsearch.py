"""W-space direct policy search — CEM in log-space over main.W's 29
play-relevant constants (pre-registered docs/report-log.md 2026-07-18).

Fitness: seat-alternated mirror win rate of wsearch_agent.py (frozen main.py
code + candidate W) vs. frozen_main.py (same code, stock W) — weights are
the only variable. Errors count as losses; ties 0.5. CEM recipe mirrors
training/nn/cem_tune.py (candidate 0 = current mean each generation;
elite-mean/sigma EMA 0.7/0.3). State + full history checkpoint after every
generation (the 2026-07-07 Kaggle-stall lesson); --resume continues from the
last completed generation with per-generation deterministic RNG
(seed*1000+gen), so a killed run resumes bit-identically.

Adoption candidates per the pre-registration are ONLY the final CEM mean and
the top elite of the final generation — never best-of-noisy-evals (the
Stage 0b winner's-curse lesson). Gates run separately via --gate.

Usage:
  python training/wsearch/wsearch.py --phase0 plumbing
  python training/wsearch/wsearch.py --phase0 anchors --anchors lucario,abomasnow
  python training/wsearch/wsearch.py [--pop 16] [--gens 30] [--games 128] [--resume]
  python training/wsearch/wsearch.py --gate CAND.json --gate-games 600
"""
import argparse
import importlib.util
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO_ROOT, "training"))

import numpy as np  # noqa: E402

from harness import run_matches  # noqa: E402

FROZEN = os.path.join(_HERE, "frozen_main.py")
WRAPPER = os.path.join(_HERE, "wsearch_agent.py")
SCRATCH = os.path.join(_HERE, "scratch")
ANCHORS = {
    "lucario": os.path.join(_REPO_ROOT, "opponents", "lucario_agent.py"),
    "abomasnow": os.path.join(_REPO_ROOT, "opponents", "abomasnow_agent.py"),
    "starmie": os.path.join(_REPO_ROOT, "opponents", "starmie_agent.py"),
    "dragapult": os.path.join(_REPO_ROOT, "opponents", "dragapult_agent.py"),
}


def _defaults():
    """Play-relevant keys + stock values, read from the frozen snapshot."""
    spec = importlib.util.spec_from_file_location("wsearch_defaults_main", FROZEN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wsearch_defaults_main"] = mod
    spec.loader.exec_module(mod)
    keys = [k for k in mod.W if not k.startswith("prior_T")]
    return keys, np.array([float(mod.W[k]) for k in keys])


def eval_candidates(json_paths, games_per_cand, workers):
    """Seat-alternated mirror WR per candidate vs FROZEN. Errors = losses."""
    wins = np.zeros(len(json_paths))
    games = np.zeros(len(json_paths))
    errs = 0
    half = games_per_cand // 2
    for cand_first in (True, False):
        envs = []
        for k in range(len(json_paths)):
            envs += [{"WSEARCH_WEIGHTS": json_paths[k], "WSEARCH_CAND": str(k)}] * half
        p0, p1 = (WRAPPER, FROZEN) if cand_first else (FROZEN, WRAPPER)
        results = run_matches(p0, p1, len(envs), workers=workers,
                              progress=False, extra_envs=envs)
        a = 0 if cand_first else 1
        for r in results:
            k = int(r.get("extra_env", {}).get("WSEARCH_CAND", -1))
            if k < 0:
                continue
            games[k] += 1
            if "error" in r:
                errs += 1
                continue  # counts as loss
            rew = r["rewards"]
            if rew[a] == 1 or rew[1 - a] == -1:
                wins[k] += 1.0
            elif not (rew[a] == -1 or rew[1 - a] == 1):
                wins[k] += 0.5
    return wins / np.maximum(games, 1), games, errs


def _wr_pair(path_a, path_b, n_games, workers):
    """Seat-alternated win rate of path_a vs path_b (plain, untagged)."""
    wins = 0.0
    n = 0
    errs = 0
    half = n_games // 2
    for a_first in (True, False):
        p0, p1 = (path_a, path_b) if a_first else (path_b, path_a)
        results = run_matches(p0, p1, half, workers=workers, progress=False)
        a = 0 if a_first else 1
        for r in results:
            if "error" in r:
                errs += 1
                continue
            n += 1
            rew = r["rewards"]
            if rew[a] == 1 or rew[1 - a] == -1:
                wins += 1.0
            elif not (rew[a] == -1 or rew[1 - a] == 1):
                wins += 0.5
    return (wins / n if n else 0.0), n, errs


def phase0(which, anchor_names, workers):
    os.makedirs(SCRATCH, exist_ok=True)
    out_path = os.path.join(_HERE, "phase0_baselines.json")
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)

    if which in ("plumbing", "all"):
        keys, dv = _defaults()
        dpath = os.path.join(SCRATCH, "defaults.json")
        with open(dpath, "w") as f:
            json.dump(dict(zip(keys, dv.tolist())), f, indent=1)
        fits, games, errs = eval_candidates([dpath], 200, workers)
        out["plumbing"] = {"wr": float(fits[0]), "n": int(games[0]), "errors": errs}
        print(f"phase0 plumbing: wr={fits[0]:.3f} n={int(games[0])} errors={errs}", flush=True)

    if which in ("anchors", "all"):
        for name in anchor_names:
            path = ANCHORS[name]
            if name == "dragapult":
                _, n_ok, errs = _wr_pair(FROZEN, path, 20, workers)
                if errs:
                    out["dragapult"] = {"skipped": True, "trial_errors": errs}
                    print(f"phase0 dragapult: SKIPPED ({errs} errors in 20-game trial)", flush=True)
                    continue
            wr, n, errs = _wr_pair(FROZEN, path, 300, workers)
            out[name] = {"wr": wr, "n": n, "errors": errs}
            print(f"phase0 {name}: frozen_wr={wr:.3f} n={n} errors={errs}", flush=True)

    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)


def gate(cand_json, gate_games, anchor_names, workers):
    """Gate A (mirror n=600 vs frozen) + Gate B panel for one candidate file."""
    res = {"candidate": cand_json}
    envs_path = os.path.abspath(cand_json)
    fits, games, errs = eval_candidates([envs_path], gate_games, workers)
    wr, n = float(fits[0]), int(games[0])
    # Wilson 95% on the win proportion (ties folded as 0.5 -> report raw too)
    z = 1.96
    p = wr
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half_w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    res["gateA"] = {"wr": wr, "n": n, "errors": errs,
                    "wilson95": [center - half_w, center + half_w]}
    print(f"gateA: wr={wr:.4f} n={n} errors={errs} "
          f"wilson95=[{center-half_w:.4f},{center+half_w:.4f}]", flush=True)
    os.environ["WSEARCH_WEIGHTS"] = envs_path
    for name in anchor_names:
        wr_a, n_a, errs_a = _wr_pair(WRAPPER, ANCHORS[name], 300, workers)
        res[f"gateB_{name}"] = {"wr": wr_a, "n": n_a, "errors": errs_a}
        print(f"gateB {name}: wr={wr_a:.4f} n={n_a} errors={errs_a}", flush=True)
    out_path = os.path.splitext(envs_path)[0] + "_gate.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"gate results -> {out_path}", flush=True)


def search(args):
    os.makedirs(SCRATCH, exist_ok=True)
    keys, dv = _defaults()
    state_path = os.path.join(_HERE, args.tag + "_state.json")
    hist_path = os.path.join(_HERE, args.tag + "_history.json")
    mean = np.log(dv)
    sigma = np.full(len(keys), args.sigma0)
    start_gen = 0
    history = []
    if args.resume and os.path.exists(state_path):
        with open(state_path) as f:
            st = json.load(f)
        mean = np.array(st["mean"])
        sigma = np.array(st["sigma"])
        start_gen = st["next_gen"]
        with open(hist_path) as f:
            history = json.load(f)
        print(f"resumed at gen {start_gen}", flush=True)

    for gen in range(start_gen, args.gens):
        t0 = time.time()
        rng = np.random.default_rng(args.seed * 1000 + gen)
        xs = [mean.copy()] + [mean + sigma * rng.standard_normal(len(mean))
                              for _ in range(args.pop - 1)]
        paths = []
        for k, x in enumerate(xs):
            p = os.path.join(SCRATCH, f"{args.tag}_g{gen}_c{k}.json")
            with open(p, "w") as f:
                json.dump(dict(zip(keys, np.exp(x).tolist())), f)
            paths.append(p)
        fits, games, errs = eval_candidates(paths, args.games, args.workers)
        order = np.argsort(-fits)
        elite = np.vstack([xs[i] for i in order[:args.elite]])
        mean = 0.7 * elite.mean(axis=0) + 0.3 * mean
        sigma = np.maximum(0.7 * elite.std(axis=0) + 0.3 * sigma, args.sigma_floor)
        wall = time.time() - t0
        history.append({
            "gen": gen, "cand0_wr": float(fits[0]), "best_wr": float(fits[order[0]]),
            "elite_mean_wr": float(fits[order[:args.elite]].mean()),
            "pop_mean_wr": float(fits.mean()), "errors": errs, "wall_s": round(wall, 1),
            "fits": fits.tolist(),
        })
        # adoption artifacts, refreshed every generation so a killed run still
        # leaves a usable latest state
        with open(os.path.join(_HERE, args.tag + "_mean_weights.json"), "w") as f:
            json.dump(dict(zip(keys, np.exp(mean).tolist())), f, indent=1)
        with open(os.path.join(_HERE, args.tag + "_top_elite.json"), "w") as f:
            json.dump(dict(zip(keys, np.exp(xs[order[0]]).tolist())), f, indent=1)
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=1)
        with open(state_path, "w") as f:
            json.dump({"next_gen": gen + 1, "mean": mean.tolist(),
                       "sigma": sigma.tolist(), "keys": keys}, f, indent=1)
        if errs:
            print(f"gen {gen}: WARNING {errs} errored games (counted as losses)", flush=True)
        print(f"gen {gen}: cand0(mean)={fits[0]:.3f} best={fits[order[0]]:.3f} "
              f"elite_mean={history[-1]['elite_mean_wr']:.3f} "
              f"pop_mean={fits.mean():.3f} wall={wall:.0f}s", flush=True)

    print(f"done — adoption candidates: {args.tag}_mean_weights.json + "
          f"{args.tag}_top_elite.json (gate them per the pre-registration)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0", choices=["plumbing", "anchors", "all"])
    ap.add_argument("--anchors", default="lucario,abomasnow,starmie")
    ap.add_argument("--gate")
    ap.add_argument("--gate-games", type=int, default=600)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--sigma0", type=float, default=0.25)
    ap.add_argument("--sigma-floor", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=18)
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    anchor_names = [a.strip() for a in args.anchors.split(",") if a.strip()]
    if args.phase0:
        phase0(args.phase0, anchor_names, args.workers)
    elif args.gate:
        gate(args.gate, args.gate_games, anchor_names, args.workers)
    else:
        search(args)


if __name__ == "__main__":
    main()
