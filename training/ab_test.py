"""A/B test two agent files over N local games, alternating seats.

Usage:
  python training/ab_test.py <agentA.py> <agentB.py> [n_games] [--workers N]

Example:
  python training/ab_test.py main.py training/baselines/v21.py 400

Seat order alternates (half the games each way) to cancel first-player advantage.
Prints per-seat and combined win rates, and appends every run to
training/ab_history.csv (per-seat splits persisted for report figure #6).
Remember the project rule: offline win rates systematically overrate — use this
to catch regressions and rank candidate changes, then confirm on the real ladder.
"""
import csv
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_matches, summarize

HISTORY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ab_history.csv")
HISTORY_FIELDS = ["timestamp", "path_a", "path_b", "n_requested",
                  "a_wins", "b_wins", "ties", "errors",
                  "a_wins_p0", "a_wins_p1", "b_wins_p0", "b_wins_p1",
                  "ties_p0", "ties_p1", "a_winrate", "ci95"]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    path_a, path_b = args[0], args[1]
    n = int(args[2]) if len(args) > 2 else 200
    workers = None
    for i, a in enumerate(sys.argv):
        if a == "--workers":
            workers = int(sys.argv[i + 1])

    half = n // 2
    print(f"A={path_a}  B={path_b}  n={n} (alternating seats), workers={workers or 'auto'}")

    print(f"[1/2] A as P0 vs B as P1: {half} games")
    r1 = run_matches(path_a, path_b, half, workers=workers)
    s1 = summarize(r1, "A", "B")
    print("   ", s1)

    print(f"[2/2] B as P0 vs A as P1: {n - half} games")
    r2 = run_matches(path_b, path_a, n - half, workers=workers)
    s2 = summarize(r2, "B", "A")
    print("   ", s2)

    a_wins = s1["A_wins"] + s2["A_wins"]
    b_wins = s1["B_wins"] + s2["B_wins"]
    ties = s1["ties"] + s2["ties"]
    errs = s1["errors"] + s2["errors"]
    tot = a_wins + b_wins + ties
    wr = (a_wins + 0.5 * ties) / tot if tot else 0.0
    # 95% CI half-width for a binomial proportion
    import math
    ci = 1.96 * math.sqrt(max(wr * (1 - wr), 1e-9) / tot) if tot else 0.0
    print(f"\nTOTAL: A {a_wins}W {b_wins}L {ties}T (errors={errs})")
    print(f"A win rate: {wr:.3f} ± {ci:.3f} (95% CI)")

    # persist the run (s1: A seated P0; s2: A seated P1)
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "path_a": path_a, "path_b": path_b, "n_requested": n,
        "a_wins": a_wins, "b_wins": b_wins, "ties": ties, "errors": errs,
        "a_wins_p0": s1["A_wins"], "a_wins_p1": s2["A_wins"],
        "b_wins_p0": s1["B_wins"], "b_wins_p1": s2["B_wins"],
        "ties_p0": s1["ties"], "ties_p1": s2["ties"],
        "a_winrate": round(wr, 4), "ci95": round(ci, 4),
    }
    exists = os.path.exists(HISTORY_CSV)
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS, restval="")
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"logged to {HISTORY_CSV}")


if __name__ == "__main__":
    main()
