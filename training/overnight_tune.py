"""Overnight weight tuning: SPSA search over main.W + Gauntlet finals.

Phase 1 (~ first 2/3 of the time budget): SPSA over the gameplay keys of main.W
(prior_T_* excluded — they don't affect play). Each iteration evaluates two
perturbed candidates vs a FROZEN snapshot of main.py taken at launch (identical
deck + logic, default W), so any win-rate difference is the weights.

Phase 2 (rest of the budget): the top-K distinct candidates from phase 1 are
re-evaluated through the Gauntlet (subset panel: v21, v23, starmie — the
anchors with signal; 94%-matchup anchors and random add ~none). Results append
to training/gauntlet_results.csv under names like "v24tune-i07plus". The
combined-panel winner is written to variants/v24_tuned.py, ready for
`python training/ab_test.py variants/v24_tuned.py main.py 600`.

Crash safety: training/tune_ckpt.json is atomically rewritten after every
evaluation (os.replace) and every evaluation is appended to
training/tune_log.jsonl. If the checkpoint exists at launch the run resumes
from it automatically (--fresh to start over).

Usage:
  python training/overnight_tune.py --hours 5.5 [--games-per-eval 160] [--workers N]
"""
import argparse
import copy
import json
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import load_agent, run_matches, summarize
import gauntlet

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(REPO_ROOT, "main.py")
REF = os.path.join(REPO_ROOT, "training", "_tune_ref.py")   # frozen snapshot
CKPT = os.path.join(REPO_ROOT, "training", "tune_ckpt.json")
LOG = os.path.join(REPO_ROOT, "training", "tune_log.jsonl")
WINNER = os.path.join(REPO_ROOT, "variants", "v24_tuned.py")
SKIP_KEYS = {"prior_T_h_main", "prior_T_h_default", "prior_T_net"}
TOP_K = 3
PHASE2_PANEL = ["v21", "v23", "starmie"]


def log_line(obj):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def save_ckpt(state):
    tmp = CKPT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, CKPT)


def make_candidate_file(weights, tag):
    """Copy of main.py with W overridden (same mechanism as weight_search.py)."""
    src = open(MAIN, encoding="utf-8").read()
    path = os.path.join(REPO_ROOT, "training", f"_tune_cand_{tag}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src + "\n\n# overnight_tune override\nW.update(%r)\n" % (weights,))
    return path


def evaluate_vs_ref(weights, tag, n_games, workers):
    cand = make_candidate_file(weights, tag)
    try:
        half = n_games // 2
        r1 = run_matches(cand, REF, half, workers=workers, progress=False)
        r2 = run_matches(REF, cand, n_games - half, workers=workers, progress=False)
        s1 = summarize(r1, "C", "B")
        s2 = summarize(r2, "B", "C")
        wins = s1["C_wins"] + s2["C_wins"]
        ties = s1["ties"] + s2["ties"]
        tot = s1["n"] + s2["n"]
        errs = s1["errors"] + s2["errors"]
        return (wins + 0.5 * ties) / max(tot, 1), errs
    finally:
        os.unlink(cand)


def phase1(state, args, deadline_p1):
    theta = state["theta"]
    keys = [k for k in sorted(theta) if k not in SKIP_KEYS]
    it = state["iter"]
    while time.time() < deadline_p1:
        signs = {k: random.choice((-1.0, 1.0)) for k in keys}
        plus = dict(theta)
        minus = dict(theta)
        for k in keys:
            plus[k] = max(0.1, theta[k] * (1 + args.delta * signs[k]))
            minus[k] = max(0.1, theta[k] * (1 - args.delta * signs[k]))

        t0 = time.time()
        f_plus, e1 = evaluate_vs_ref(plus, "plus", args.games_per_eval, args.workers)
        f_minus, e2 = evaluate_vs_ref(minus, "minus", args.games_per_eval, args.workers)
        eval_s = time.time() - t0

        g = (f_plus - f_minus) / (2 * args.delta)
        for k in keys:
            theta[k] = max(0.1, theta[k] * (1 + args.lr * g * signs[k]))

        for name, fit, w in ((f"i{it:02d}plus", f_plus, plus),
                             (f"i{it:02d}minus", f_minus, minus)):
            state["candidates"].append({"name": name, "fitness": fit, "weights": w})
        state["candidates"].sort(key=lambda c: -c["fitness"])
        state["candidates"] = state["candidates"][:12]  # keep a healthy shortlist
        state["iter"] = it = it + 1
        state["theta"] = theta
        state["eval_seconds"] = eval_s / 2
        row = {"phase": 1, "iter": it, "f_plus": round(f_plus, 4),
               "f_minus": round(f_minus, 4), "errors": e1 + e2,
               "eval_s": round(eval_s, 1), "t": time.time()}
        log_line(row)
        save_ckpt(state)
        print(f"iter {it}: f+={f_plus:.3f} f-={f_minus:.3f} "
              f"best={state['candidates'][0]['fitness']:.3f} "
              f"({state['candidates'][0]['name']}) [{eval_s:.0f}s]", flush=True)
        # stop early if two more evals plus phase 2 wouldn't fit
        if time.time() + eval_s > deadline_p1:
            break


def phase2(state, args):
    """Gauntlet the top-K distinct candidates; pick by combined panel win rate."""
    finals = []
    seen = set()
    for c in state["candidates"]:
        key = json.dumps(c["weights"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        finals.append(c)
        if len(finals) == TOP_K:
            break

    results = state.setdefault("phase2", [])
    already = {r["name"] for r in results}
    for c in finals:
        if c["name"] in already:
            continue
        tag = "v24tune-" + c["name"]
        cand = make_candidate_file(c["weights"], "final_" + c["name"])
        try:
            rows = gauntlet.run_panel(cand, tag, args.p2_games,
                                      PHASE2_PANEL, args.workers)
        finally:
            os.unlink(cand)
        gauntlet.append_results(rows)
        w = sum(r["cand_wins"] + 0.5 * r["ties"] for r in rows)
        n = sum(r["cand_wins"] + r["anchor_wins"] + r["ties"] for r in rows)
        entry = {"name": c["name"], "panel_wr": round(w / max(n, 1), 4),
                 "n": n, "weights": c["weights"]}
        results.append(entry)
        log_line({"phase": 2, **{k: v for k, v in entry.items() if k != "weights"}})
        save_ckpt(state)
        print(f"phase2 {c['name']}: panel wr {entry['panel_wr']:.3f} over {n}", flush=True)

    # default-W control on the same panel, once (for the accept/reject call)
    if not any(r["name"] == "defaultW" for r in results):
        rows = gauntlet.run_panel(REF, "v24-defaultW", args.p2_games,
                                  PHASE2_PANEL, args.workers)
        gauntlet.append_results(rows)
        w = sum(r["cand_wins"] + 0.5 * r["ties"] for r in rows)
        n = sum(r["cand_wins"] + r["anchor_wins"] + r["ties"] for r in rows)
        results.append({"name": "defaultW", "panel_wr": round(w / max(n, 1), 4),
                        "n": n, "weights": state["default_W"]})
        log_line({"phase": 2, "name": "defaultW", "panel_wr": results[-1]["panel_wr"]})
        save_ckpt(state)
        print(f"phase2 defaultW control: panel wr {results[-1]['panel_wr']:.3f}", flush=True)

    best = max(results, key=lambda r: r["panel_wr"])
    state["winner"] = {k: v for k, v in best.items()}
    save_ckpt(state)
    src = open(MAIN, encoding="utf-8").read()
    with open(WINNER, "w", encoding="utf-8") as f:
        f.write(src + "\n\n# overnight_tune winner: %s (panel wr %.3f)\n"
                      "W.update(%r)\n" % (best["name"], best["panel_wr"], best["weights"]))
    print(f"\nWINNER: {best['name']} panel wr {best['panel_wr']:.3f} "
          f"(defaultW control: "
          f"{next(r['panel_wr'] for r in results if r['name'] == 'defaultW'):.3f})")
    print(f"written to {WINNER} — validate with:\n"
          f"  python training/ab_test.py variants/v24_tuned.py main.py 600\n"
          f"then ladder-confirm before shipping.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=5.5)
    ap.add_argument("--games-per-eval", type=int, default=160)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.15)
    ap.add_argument("--p2-games", type=int, default=100)
    ap.add_argument("--p2-slack", type=float, default=600.0)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    deadline = t_start + args.hours * 3600
    # phase 2 budget: (TOP_K + 1 control) * panel games, ~2.2s/game measured, + slack
    p2_est = (TOP_K + 1) * len(PHASE2_PANEL) * args.p2_games * 2.2 + args.p2_slack
    deadline_p1 = deadline - p2_est

    if os.path.exists(CKPT) and not args.fresh:
        state = json.load(open(CKPT, encoding="utf-8"))
        print(f"resuming: iter {state['iter']}, "
              f"{len(state.get('candidates', []))} shortlisted")
        if not os.path.exists(REF):
            shutil.copyfile(MAIN, REF)  # crash lost the snapshot; re-freeze
    else:
        shutil.copyfile(MAIN, REF)  # freeze tonight's reference
        _, _, mod = load_agent(MAIN)
        state = {"theta": dict(mod.W), "default_W": dict(mod.W),
                 "iter": 0, "candidates": [], "started": t_start}
        save_ckpt(state)

    print(f"budget {args.hours}h; phase 1 until "
          f"{time.strftime('%H:%M', time.localtime(deadline_p1))}, "
          f"phase 2 (gauntlet finals) after; ckpt={CKPT}")
    try:
        phase1(state, args, deadline_p1)
    except KeyboardInterrupt:
        print("interrupted — checkpoint is current; moving to phase 2")
    phase2(state, args)


if __name__ == "__main__":
    main()
