"""The Gauntlet: fixed-panel offline strength indicator with a Bradley-Terry fit.

Every candidate agent plays a fixed panel of anchor opponents (seat-alternating).
Results accumulate in training/gauntlet_results.csv across runs, and a
Bradley-Terry model is fit over ALL accumulated results, so every agent ever
gauntleted sits on one comparable Elo-like scale ("gElo", anchored at
random=0). Pair gElo with realized ladder Elo (training/ladder_history.csv)
to measure how well offline strength predicts online strength — the
offline/online calibration figure for the report.

Usage:
  python training/gauntlet.py --candidate main.py --name v24-deckA --games 100
  python training/gauntlet.py --table                 # just refit + print table
  python training/gauntlet.py --candidate main.py --panel random,v21,v22 --games 40

Notes:
- --name labels the candidate in the results file. ALWAYS give a distinct name
  per agent version/deck variant; reusing a name pools its games.
- --games is per anchor (half as P0, half as P1).
- Ties count as half a win for both sides in the BT fit.
- The panel is all bots-we-wrote, so gElo can drift from the real ladder meta.
  That is exactly what ladder_history.csv is for: never trust gElo un-calibrated.
"""
import argparse
import csv
import datetime
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_matches, summarize

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(REPO_ROOT, "training", "gauntlet_results.csv")
FIELDS = ["timestamp", "candidate", "anchor", "cand_wins", "anchor_wins",
          "ties", "errors", "n_requested",
          # per-seat split (candidate as P0 / as P1) + run id, added 2026-07-03
          # for report figure #6 (robustness panel). Older rows have blank cells.
          # NOTE: "seed" is a run identifier for grouping independent repeat
          # runs (variance measurement), NOT an RNG seed — the engine's shuffle
          # lives in the native cg.dll and exposes no seed API.
          "cand_wins_p0", "cand_wins_p1", "anchor_wins_p0", "anchor_wins_p1",
          "ties_p0", "ties_p1", "seed"]

# Fixed anchor panel: name -> path (relative to repo root).
PANEL = {
    "random":    "training/random_agent.py",
    "lucario":   "opponents/lucario_agent.py",
    "dragapult": "opponents/dragapult_agent.py",
    "abomasnow": "opponents/abomasnow_agent.py",
    "starmie":   "opponents/starmie_agent.py",
    "v21":       "training/baselines/v21.py",
    "v22":       "training/baselines/v22_pre_refactor.py",
    "v23":       "training/baselines/v23.py",
}


def run_panel(cand_path, cand_name, games_per_anchor, panel_names, workers, seed=""):
    rows = []
    for name in panel_names:
        if name == cand_name:
            print(f"[skip] {name}: candidate cannot play itself in the BT fit")
            continue
        anchor_path = os.path.join(REPO_ROOT, PANEL[name])
        half = games_per_anchor // 2
        print(f"[{cand_name} vs {name}] {games_per_anchor} games "
              f"({half}+{games_per_anchor - half} seat-alternated)")
        r1 = run_matches(cand_path, anchor_path, half, workers=workers)
        s1 = summarize(r1, "C", "A")
        r2 = run_matches(anchor_path, cand_path, games_per_anchor - half, workers=workers)
        s2 = summarize(r2, "A", "C")
        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "candidate": cand_name,
            "anchor": name,
            "cand_wins": s1["C_wins"] + s2["C_wins"],
            "anchor_wins": s1["A_wins"] + s2["A_wins"],
            "ties": s1["ties"] + s2["ties"],
            "errors": s1["errors"] + s2["errors"],
            "n_requested": games_per_anchor,
            # s1: candidate seated P0; s2: candidate seated P1
            "cand_wins_p0": s1["C_wins"], "cand_wins_p1": s2["C_wins"],
            "anchor_wins_p0": s1["A_wins"], "anchor_wins_p1": s2["A_wins"],
            "ties_p0": s1["ties"], "ties_p1": s2["ties"],
            "seed": seed,
        }
        n = row["cand_wins"] + row["anchor_wins"] + row["ties"]
        wr = (row["cand_wins"] + 0.5 * row["ties"]) / n if n else 0.0
        print(f"    {row['cand_wins']}W {row['anchor_wins']}L {row['ties']}T "
              f"(wr={wr:.3f}, errors={row['errors']})")
        rows.append(row)
    return rows


def append_results(rows):
    exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        # restval="" keeps rows writable if FIELDS grows again later
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows to {RESULTS_CSV}")


def load_pair_scores():
    """Aggregate all accumulated results into pairwise scores.
    Returns dict[(a, b)] = total 'wins' of a over b (ties = 0.5 each)."""
    scores = defaultdict(float)
    games = defaultdict(float)
    if not os.path.exists(RESULTS_CSV):
        return scores, games
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a, b = row["candidate"], row["anchor"]
            cw, aw, t = int(row["cand_wins"]), int(row["anchor_wins"]), int(row["ties"])
            scores[(a, b)] += cw + 0.5 * t
            scores[(b, a)] += aw + 0.5 * t
            games[(a, b)] += cw + aw + t
            games[(b, a)] += cw + aw + t
    return scores, games


def bradley_terry(scores, games, iters=500, tol=1e-9):
    """Standard MM fit of Bradley-Terry strengths from pairwise win totals."""
    players = sorted({p for pair in games for p in pair})
    if not players:
        return {}
    p = {x: 1.0 for x in players}
    wins = {x: sum(scores[(x, y)] for y in players if (x, y) in scores) for x in players}
    for _ in range(iters):
        newp = {}
        for x in players:
            denom = 0.0
            for y in players:
                n_xy = games.get((x, y), 0.0)
                if n_xy > 0 and y != x:
                    denom += n_xy / (p[x] + p[y])
            # +1 virtual half-win vs a strength-1 ghost regularizes 0% / 100% records
            newp[x] = (wins[x] + 0.5) / (denom + 1.0 / (p[x] + 1.0))
        norm = math.exp(sum(math.log(v) for v in newp.values()) / len(newp))
        delta = 0.0
        for x in players:
            newp[x] /= norm
            delta = max(delta, abs(newp[x] - p[x]))
        p = newp
        if delta < tol:
            break
    return p


def print_table(p, games):
    if not p:
        print("no results yet — run a candidate through the panel first")
        return
    ref = p.get("random", min(p.values()))
    total_games = defaultdict(float)
    for (a, _b), n in games.items():
        total_games[a] += n
    print(f"\n{'agent':<24}{'gElo':>8}{'games':>8}")
    print("-" * 40)
    for name, strength in sorted(p.items(), key=lambda kv: -kv[1]):
        gelo = 400.0 * math.log10(strength / ref)
        print(f"{name:<24}{gelo:>8.0f}{int(total_games[name]):>8}")
    print("\n(gElo anchored at random=0; comparable ONLY within this table.")
    print(" Calibrate against training/ladder_history.csv before trusting.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", help="agent .py to gauntlet (omit with --table)")
    ap.add_argument("--name", help="label for the candidate (REQUIRED with --candidate)")
    ap.add_argument("--games", type=int, default=100, help="games per anchor")
    ap.add_argument("--panel", default=",".join(PANEL),
                    help="comma-separated anchor subset (default: all)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", default=None,
                    help="run identifier recorded per row (groups independent "
                         "repeat runs for variance; NOT an RNG seed — engine "
                         "randomness is in the native cg.dll). Default: "
                         "run-YYYYmmdd-HHMMSS")
    ap.add_argument("--table", action="store_true", help="refit + print table only")
    args = ap.parse_args()

    if args.candidate:
        if not args.name:
            ap.error("--name is required with --candidate (distinct per version/deck)")
        panel_names = [s.strip() for s in args.panel.split(",") if s.strip()]
        unknown = [n for n in panel_names if n not in PANEL]
        if unknown:
            ap.error(f"unknown panel anchors: {unknown} (known: {list(PANEL)})")
        cand_path = os.path.abspath(args.candidate)
        seed = args.seed or datetime.datetime.now().strftime("run-%Y%m%d-%H%M%S")
        rows = run_panel(cand_path, args.name, args.games, panel_names,
                         args.workers, seed=seed)
        append_results(rows)

    scores, games = load_pair_scores()
    print_table(bradley_terry(scores, games), games)


if __name__ == "__main__":
    main()
