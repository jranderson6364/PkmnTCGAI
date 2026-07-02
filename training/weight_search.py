"""SPSA weight search over main.W scoring constants.

The user's idea: many score() constants are hand-guessed. Now that full games run
locally (~0.5s each), tune them empirically. SPSA (simultaneous perturbation)
estimates a gradient from just 2 evaluations per iteration regardless of the
number of parameters — the right tool when each evaluation is a noisy batch of
games.

Evaluation: candidate weights vs the FROZEN v21 baseline (and optionally the
opponent pool), alternating seats. Fitness = win rate.

Usage:
  python training/weight_search.py --iters 30 --games-per-eval 120 [--workers N]
  python training/weight_search.py --resume training/weights_ckpt.json

Output: training/weights_ckpt.json (best weights + history). Apply the result by
pasting the tuned dict into main.W — and confirm on the ladder before trusting
it (offline win rates overrate; this ranks candidates, the ladder decides).

Caveats:
- Win-rate noise: 120 games has ~±9% CI; SPSA tolerates this but don't read
  single iterations as signal. The trend over 20+ iterations is the signal.
- Weights are multiplicative-perturbed (delta = 15% of current value) so scales
  stay sane. Values are clamped to >= 0.1.
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import load_agent, run_matches, summarize

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(REPO_ROOT, "main.py")
BASELINE = os.path.join(REPO_ROOT, "training", "baselines", "v21.py")
CKPT = os.path.join(REPO_ROOT, "training", "weights_ckpt.json")


def make_candidate_file(weights):
    """Write a temp copy of main.py with W overridden (workers import by path,
    so mutating main.W in this process wouldn't reach them)."""
    src = open(MAIN, encoding="utf-8").read()
    override = "\n\n# weight_search override\nW.update(%r)\n" % (weights,)
    # inject right after the W = {...} block: append at end of file is simplest
    # and safe because W is only read inside score() at call time.
    fd, path = tempfile.mkstemp(suffix=".py", prefix="cand_", dir=os.path.dirname(MAIN))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(src + override)
    return path


def evaluate(weights, n_games, workers):
    cand = make_candidate_file(weights)
    try:
        half = n_games // 2
        r1 = run_matches(cand, BASELINE, half, workers=workers, progress=False)
        r2 = run_matches(BASELINE, cand, n_games - half, workers=workers, progress=False)
        s1 = summarize(r1, "C", "B")
        s2 = summarize(r2, "B", "C")
        wins = s1["C_wins"] + s2["C_wins"]
        ties = s1["ties"] + s2["ties"]
        tot = s1["n"] + s2["n"]
        return (wins + 0.5 * ties) / max(tot, 1)
    finally:
        os.unlink(cand)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--games-per-eval", type=int, default=120)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--lr", type=float, default=0.10)      # SPSA a
    ap.add_argument("--delta", type=float, default=0.15)   # SPSA c (relative)
    args = ap.parse_args()

    _, _, mod = load_agent(MAIN)
    theta = dict(mod.W)
    history = []
    if args.resume and os.path.exists(args.resume):
        ck = json.load(open(args.resume))
        theta = ck["theta"]
        history = ck.get("history", [])
        print(f"resumed at iter {len(history)}")

    keys = sorted(theta.keys())
    base_fit = evaluate(theta, args.games_per_eval, args.workers)
    print(f"baseline fitness (current W vs v21): {base_fit:.3f}")
    best = (base_fit, copy.deepcopy(theta))

    for it in range(len(history), args.iters):
        # SPSA: one random ±1 direction per parameter, two evaluations
        signs = {k: random.choice((-1.0, 1.0)) for k in keys}
        plus = {k: max(0.1, theta[k] * (1 + args.delta * signs[k])) for k in keys}
        minus = {k: max(0.1, theta[k] * (1 - args.delta * signs[k])) for k in keys}
        f_plus = evaluate(plus, args.games_per_eval, args.workers)
        f_minus = evaluate(minus, args.games_per_eval, args.workers)
        g = (f_plus - f_minus) / (2 * args.delta)
        for k in keys:
            theta[k] = max(0.1, theta[k] * (1 + args.lr * g * signs[k]))
        fit = max(f_plus, f_minus)
        history.append({"iter": it, "f_plus": f_plus, "f_minus": f_minus})
        if fit > best[0]:
            best = (fit, copy.deepcopy(plus if f_plus >= f_minus else minus))
        json.dump({"theta": theta, "best_fitness": best[0], "best_theta": best[1],
                   "history": history}, open(CKPT, "w"), indent=1)
        print(f"iter {it}: f+={f_plus:.3f} f-={f_minus:.3f} best={best[0]:.3f}")

    print(f"\nbest fitness {best[0]:.3f}; weights in {CKPT} (best_theta)")
    print("Re-validate best_theta with a 600+ game A/B before shipping.")


if __name__ == "__main__":
    main()
