"""Panel-fitness CEM weight search — the one axis the closed W-search never tried.

The 2026-07-18 W-space CEM search CLOSED negative, but its fitness was MIRROR win
rate vs a frozen copy of the same policy, plus weak-anchor non-regression guards.
Its own closure named the failure: mirror-matchup overfit + winner's curse. It
optimized against opponents that could not discriminate real weaknesses (the
anchors read <=6% against us; the mirror just rewards mirror-overfit).

We now have opponents that CAN discriminate: four public agents with published
ladder scores 739.7-933.8, every one of which BEATS or ties our champion. This
search optimizes the 29 tunable W constants against that panel directly, and
gates the winner on a HELD-OUT set (the field anchors) so a panel-overfit
candidate is caught. This is a genuinely new objective, motivated by finding that
the 778.2 same-deck agent's edge is its evolutionarily-tuned weights.

Design against the prior closure's two failure modes:
  * discriminating fitness (real strong opponents, not mirror/weak anchors)
  * winner's curse -> CEM averages over an elite set, and the accepted mean is
    RE-EVALUATED at higher n before it is trusted; the winner is gated on
    held-out opponents it never trained against.

Pre-registered (docs/report-log.md 2026-07-23). Checkpoints every generation to
panel_search_state.json so a detached run can be resumed / inspected.

Run (detached, long):
  python training/wsearch/panel_search.py --gens 12 --pop 20 --elite 6 \
      --games-per-opp 12 --workers 3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "training"))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "training", "local_cg"))

from harness import run_matches  # noqa: E402

PANEL_AGENT = os.path.join(_HERE, "panel_agent.py")

# Fitness opponents: the discriminating public panel (known ladder scores).
FITNESS = {
    "probability_v2":     os.path.join(_REPO, "opponents", "public", "probability_v2.py"),
    "advanced_heuristic": os.path.join(_REPO, "opponents", "public", "advanced_heuristic.py"),
    "alakazam_v9":        os.path.join(_REPO, "opponents", "public", "alakazam_v9.py"),
    "alakazam_v8":        os.path.join(_REPO, "opponents", "public", "alakazam_v8.py"),
}
# Held-out generalization gate: the field anchors the search never optimizes on.
HELDOUT = {
    "abomasnow":  os.path.join(_REPO, "opponents", "abomasnow_agent.py"),
    "dragapult":  os.path.join(_REPO, "opponents", "dragapult_agent.py"),
    "grimmsnarl": os.path.join(_REPO, "opponents", "grimmsnarl_agent.py"),
    "starmie":    os.path.join(_REPO, "opponents", "starmie_agent.py"),
}

STATE = os.path.join(_HERE, "panel_search_state.json")
BEST = os.path.join(_HERE, "panel_search_best.json")


def _stock_W():
    import importlib.util
    spec = importlib.util.spec_from_file_location("fz_read", os.path.join(_HERE, "frozen_main_v30.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["fz_read"] = m
    spec.loader.exec_module(m)
    return {k: float(v) for k, v in m.W.items() if not k.startswith("prior_")}


def eval_weights(w_json_path, opponents, games_per_opp, workers):
    """Mean win rate of the candidate (panel_agent + injected W) across opponents,
    seat-alternated. Errors count as losses."""
    total_w = total_n = 0
    per = {}
    for name, opp_path in opponents.items():
        half = games_per_opp // 2
        # candidate as P0
        envs = [{"PANEL_WEIGHTS": w_json_path}] * half
        r0 = run_matches(PANEL_AGENT, opp_path, half, workers=workers, progress=False, extra_envs=envs)
        # candidate as P1
        envs = [{"PANEL_WEIGHTS": w_json_path}] * (games_per_opp - half)
        r1 = run_matches(opp_path, PANEL_AGENT, games_per_opp - half, workers=workers, progress=False, extra_envs=envs)
        w = n = 0
        for r in r0:
            if "error" in r:
                n += 1; continue
            rew = r["rewards"]; n += 1
            if rew[0] is not None and rew[1] is not None and rew[0] > rew[1]:
                w += 1
        for r in r1:
            if "error" in r:
                n += 1; continue
            rew = r["rewards"]; n += 1
            if rew[0] is not None and rew[1] is not None and rew[1] > rew[0]:
                w += 1
        per[name] = w / max(1, n)
        total_w += w; total_n += n
    return total_w / max(1, total_n), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--elite", type=int, default=6)
    ap.add_argument("--games-per-opp", type=int, default=12)
    ap.add_argument("--sigma0", type=float, default=0.25,
                    help="initial relative std of each weight (fraction of |stock|)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260723)
    args = ap.parse_args()

    random.seed(args.seed)
    stock = _stock_W()
    keys = list(stock.keys())
    mean = {k: stock[k] for k in keys}
    # relative sigma so each weight explores on its own scale (min floor for tiny ones)
    sigma = {k: max(abs(stock[k]) * args.sigma0, 1.0) for k in keys}

    tmpdir = os.path.join(_HERE, "scratch_panel")
    os.makedirs(tmpdir, exist_ok=True)

    # baseline (stock W) fitness for reference
    stock_path = os.path.join(tmpdir, "stock.json")
    json.dump(stock, open(stock_path, "w"))
    base_fit, base_per = eval_weights(stock_path, FITNESS, args.games_per_opp * 2, args.workers)
    print(f"[baseline] stock W panel fitness {base_fit:.3f}  {base_per}", flush=True)

    history = []
    best_fit = base_fit
    best_w = dict(stock)

    for gen in range(args.gens):
        cands = []
        for i in range(args.pop):
            w = {k: mean[k] + random.gauss(0, sigma[k]) for k in keys}
            p = os.path.join(tmpdir, f"g{gen}_c{i}.json")
            json.dump(w, open(p, "w"))
            cands.append((p, w))
        scored = []
        for i, (p, w) in enumerate(cands):
            fit, per = eval_weights(p, FITNESS, args.games_per_opp, args.workers)
            scored.append((fit, w, per))
            print(f"  gen{gen} cand{i:02d} fit {fit:.3f}", flush=True)
        scored.sort(key=lambda x: -x[0])
        elite = scored[: args.elite]

        # CEM update: mean/sigma from elite
        for k in keys:
            vals = [e[1][k] for e in elite]
            mean[k] = sum(vals) / len(vals)
            var = sum((v - mean[k]) ** 2 for v in vals) / len(vals)
            sigma[k] = max(math.sqrt(var), abs(stock[k]) * 0.03, 0.5)

        # re-evaluate the new mean at higher n (winner's-curse guard)
        mp = os.path.join(tmpdir, f"g{gen}_mean.json")
        json.dump(mean, open(mp, "w"))
        mean_fit, mean_per = eval_weights(mp, FITNESS, args.games_per_opp * 2, args.workers)
        top_elite_fit = elite[0][0]
        print(f"[gen{gen}] elite_top {top_elite_fit:.3f}  mean(reeval) {mean_fit:.3f}  "
              f"per {mean_per}", flush=True)

        if mean_fit > best_fit:
            best_fit = mean_fit
            best_w = dict(mean)
            json.dump({"fitness": best_fit, "per": mean_per, "weights": best_w},
                      open(BEST, "w"), indent=2)
            print(f"  ** new best mean fitness {best_fit:.3f} -> {BEST}", flush=True)

        history.append(dict(gen=gen, elite_top=top_elite_fit, mean_fit=mean_fit,
                            mean_per=mean_per, best=best_fit))
        json.dump({"baseline": base_fit, "base_per": base_per, "history": history,
                   "best_fit": best_fit},
                  open(STATE, "w"), indent=2)

    # ---- held-out gate of the best mean ----
    print("\n=== HELD-OUT GATE (field anchors, never optimized on) ===", flush=True)
    bp = os.path.join(tmpdir, "best.json")
    json.dump(best_w, open(bp, "w"))
    stock_ho, stock_ho_per = eval_weights(stock_path, HELDOUT, 40, args.workers)
    best_ho, best_ho_per = eval_weights(bp, HELDOUT, 40, args.workers)
    print(f"stock  held-out {stock_ho:.3f}  {stock_ho_per}", flush=True)
    print(f"best   held-out {best_ho:.3f}  {best_ho_per}", flush=True)
    print(f"\nPANEL fitness: stock {base_fit:.3f} -> best {best_fit:.3f} "
          f"({(best_fit-base_fit)*100:+.1f}pp)", flush=True)
    print(f"HELD-OUT:      stock {stock_ho:.3f} -> best {best_ho:.3f} "
          f"({(best_ho-stock_ho)*100:+.1f}pp)   (must not regress)", flush=True)
    json.dump({"panel_stock": base_fit, "panel_best": best_fit,
               "heldout_stock": stock_ho, "heldout_best": best_ho,
               "heldout_stock_per": stock_ho_per, "heldout_best_per": best_ho_per,
               "weights": best_w}, open(BEST, "w"), indent=2)
    print(f"\nwrote {BEST}", flush=True)


if __name__ == "__main__":
    main()
