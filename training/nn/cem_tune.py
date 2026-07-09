"""Phase D (pre-registered docs/report-log.md 2026-07-09): CEM-tune the 11
advisor feature weights by SIMULATION FITNESS — the literature's own method
(Miernik/Santos: GA on win-rate-when-playing), fixing the objective/consumer
mismatch that killed the outcome-fitted advisor (correlational weights
mis-rank causal action deltas, e.g. penalizing draws via deck_clock_diff).

Fitness per candidate: mean win rate of training/nn/advisor_agent.py
(linear scorer, candidate weights) over 8 lucario + 8 abomasnow + 8
mirror-vs-main.py games, seats alternated 4/4 within each block. Errors
count as losses (conservative). Incremental checkpoint after EVERY
generation (the 2026-07-07 Kaggle-stall lesson).

Usage:
  python training/nn/cem_tune.py [--pop 16] [--gens 20] [--elite 4]
                                 [--workers 10] [--out training/advisor_cem]
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "training"))

import numpy as np  # noqa: E402

from harness import run_matches  # noqa: E402

ADVISOR = os.path.join(_REPO_ROOT, "training", "nn", "advisor_agent.py")
MAIN = os.path.join(_REPO_ROOT, "main.py")
OPPONENTS = [
    (os.path.join(_REPO_ROOT, "opponents", "lucario_agent.py"), "opponents.lucario_agent"),
    (os.path.join(_REPO_ROOT, "opponents", "abomasnow_agent.py"), "opponents.abomasnow_agent"),
    (MAIN, "main"),
]
GAMES_PER_OPP = 8  # 4 each seat


def eval_generation(cands, workers, scratch):
    """cands: list of weight vectors. Returns fitness array (mean WR/candidate)."""
    paths = []
    for k, w in enumerate(cands):
        p = os.path.join(scratch, f"cem_w{k}.npy")
        np.save(p, np.asarray(w, dtype=float))
        paths.append(p)

    def make_jobs(advisor_first):
        jobs = []
        for k in range(len(cands)):
            for opp_path, opp_mod in OPPONENTS:
                for _ in range(GAMES_PER_OPP // 2):
                    jobs.append((opp_path, {
                        "ADVISOR_WEIGHTS": paths[k],
                        "ADVISOR_SCORER": "linear",
                        "MCTS_OPPONENT_MODULE": opp_mod,
                        "CEM_CAND": str(k),
                    }))
        return jobs

    wins = np.zeros(len(cands))
    games = np.zeros(len(cands))
    for advisor_first in (True, False):
        jobs = make_jobs(advisor_first)
        envs = [j[1] for j in jobs]
        opp_paths = [j[0] for j in jobs]
        # run_matches takes single path0/path1; batch by opponent path
        for opp_path, _mod in OPPONENTS:
            idx = [i for i, p in enumerate(opp_paths) if p == opp_path]
            if not idx:
                continue
            sub_envs = [envs[i] for i in idx]
            p0, p1 = (ADVISOR, opp_path) if advisor_first else (opp_path, ADVISOR)
            results = run_matches(p0, p1, len(sub_envs), workers=workers,
                                  progress=False, extra_envs=sub_envs)
            for r in results:
                k = int(r.get("extra_env", {}).get("CEM_CAND", -1))
                if k < 0:
                    continue
                games[k] += 1
                if "error" in r:
                    continue  # counts as loss
                rew = r["rewards"]
                a_idx = 0 if advisor_first else 1
                o_idx = 1 - a_idx
                if rew[a_idx] == 1 or rew[o_idx] == -1:
                    wins[k] += 1.0
                elif not (rew[a_idx] == -1 or rew[o_idx] == 1):
                    wins[k] += 0.5  # tie
    return wins / np.maximum(games, 1), games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(_REPO_ROOT, "training", "advisor_cem"))
    ap.add_argument("--sigma", type=float, default=0.6)
    args = ap.parse_args()

    scratch = args.out + "_scratch"
    os.makedirs(scratch, exist_ok=True)

    mean = np.load(os.path.join(_REPO_ROOT, "training", "eval_v4_weights.npy")).astype(float)
    sigma = np.full_like(mean, args.sigma)
    rng = np.random.default_rng(17)
    history = []

    for gen in range(args.gens):
        cands = [mean.copy()] + [mean + sigma * rng.standard_normal(len(mean))
                                 for _ in range(args.pop - 1)]
        fits, games = eval_generation(cands, args.workers, scratch)
        order = np.argsort(-fits)
        elite = [cands[i] for i in order[:args.elite]]
        elite_mat = np.vstack(elite)
        new_mean = elite_mat.mean(axis=0)
        new_sigma = elite_mat.std(axis=0)
        mean = 0.7 * new_mean + 0.3 * mean
        sigma = np.maximum(0.7 * new_sigma + 0.3 * sigma, 0.12)
        rec = {
            "gen": gen,
            "best_fit": float(fits[order[0]]),
            "mean_fit": float(fits.mean()),
            "elite_mean_fit": float(fits[order[:args.elite]].mean()),
            "cur_mean_fit_cand0": float(fits[0]),
            "mean": mean.tolist(),
            "sigma": sigma.tolist(),
            "best": cands[order[0]].tolist(),
        }
        history.append(rec)
        np.save(args.out + "_weights.npy", mean)
        with open(args.out + "_history.json", "w") as f:
            json.dump(history, f, indent=1)
        print(f"gen {gen}: best={rec['best_fit']:.3f} elite_mean={rec['elite_mean_fit']:.3f} "
              f"pop_mean={rec['mean_fit']:.3f} cand0(cur_mean)={rec['cur_mean_fit_cand0']:.3f}",
              flush=True)

    print(f"done — tuned mean weights at {args.out}_weights.npy")


if __name__ == "__main__":
    main()
