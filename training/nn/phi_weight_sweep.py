"""Phi v3 (2026-07-05 user design session, follow-up to threat.py): keep the
PRECISE, deck-specific hand_advantage term for OUR side (it's literally our
Powerful Hand damage stat, exact -- not a generic proxy), and pair it against
threat.threat_against(opp, me) for THEIR side (the best available generic
estimate, since we don't know their deck's exact damage formula the way we
know our own). Asymmetric in METHOD (precise vs. generic proxy), which is
fine and expected -- but the relative WEIGHT between the two terms needs
calibration, not an assumed 1:1 ratio, since they come from techniques of
different precision.

Weight selection uses a SPLIT corpus (first 60% of sorted replay files to
pick the best opp_threat weight, remaining 40% held out to report the final
number) so the reported result isn't just fit to the exact set it's
evaluated on -- the same overfitting risk flagged when phi.py's own weights
were kept hand-set rather than fit.

Usage:
  python training/nn/phi_weight_sweep.py [--replays-dir replays/bulk] [--max-games N]
"""
import argparse
import glob
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

import main as heuristic  # noqa: E402
from phi_baseline import our_seat, bootstrap_ci  # noqa: E402
from threat import threat_against  # noqa: E402


def components(cur, me_idx):
    """Returns (prize_diff, hand_advantage, opp_threat, wall_penalty, line_progress)
    -- all five raw ingredients, computed once per state so the weight sweep
    doesn't recompute the expensive parts (card lookups, census) per weight."""
    players = cur.get("players") or []
    if len(players) != 2:
        return None
    my_p, opp_p = players[me_idx], players[1 - me_idx]
    my_active = heuristic._active(my_p)
    bench = my_p.get("bench") or []
    hand_n = heuristic._hand_size(cur, me_idx)
    my_prizes = len(my_p.get("prize") or [])
    opp_prizes = len(opp_p.get("prize") or [])
    opp_active = heuristic._active(opp_p)
    wall = heuristic._opp_has_blocking_energy(opp_active)
    cen = heuristic._census(my_active, bench)

    prize_diff = (opp_prizes - my_prizes) / 6.0
    opp_hp = (opp_active or {}).get("hp") if opp_active else None
    if opp_hp:
        cards_needed = math.ceil(opp_hp / heuristic.PH_DMG_PER_CARD)
        hand_advantage = max(-1.0, min(1.0, (hand_n - cards_needed) / 10.0))
    else:
        hand_advantage = min(hand_n / 10.0, 1.0)
    opp_threat = threat_against(opp_active, my_active)
    wall_penalty = 1.0 if wall else 0.0
    line_progress = cen["line_count"] / 2.0
    return prize_diff, hand_advantage, opp_threat, wall_penalty, line_progress


def extract_rows(path):
    """Returns list of (components_tuple, outcome, turn)."""
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return []
    info = d.get("info", {})
    you = our_seat(info)
    if you is None:
        return []
    rewards = d.get("rewards")
    if not rewards or len(rewards) != 2 or rewards[you] not in (1, -1):
        return []
    outcome = rewards[you]

    rows = []
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
        turn = cur.get("turn") or 0
        try:
            c = components(cur, you)
        except Exception:
            continue
        if c is None:
            continue
        rows.append((c, outcome, turn))
    return rows


def phi_v3(c, w_hand=1.0, w_threat=1.0):
    prize_diff, hand_advantage, opp_threat, wall_penalty, line_progress = c
    return (2.0 * prize_diff + w_hand * hand_advantage - w_threat * opp_threat
            - 1.5 * wall_penalty + 0.5 * line_progress)


def acc(rows, w_hand, w_threat):
    flat = [(phi_v3(c, w_hand, w_threat), o) for c, o, t in rows]
    return sum(1 for v, o in flat if (v >= 0) == (o >= 0)) / len(flat)


def report(label, game_rows, w_hand, w_threat):
    flat = [r for g in game_rows for r in g]
    if not flat:
        print(f"{label}: no samples")
        return
    n = len(flat)
    a = sum(1 for c, o, t in flat if (phi_v3(c, w_hand, w_threat) >= 0) == (o >= 0)) / n
    scored_games = [[(phi_v3(c, w_hand, w_threat), o, t) for c, o, t in g] for g in game_rows]
    lo, hi = bootstrap_ci(scored_games)
    print(f"{label}: n_decisions={n} n_games={len(game_rows)} sign_acc={a:.3f} "
          f"game_level_95%CI=[{lo:.3f}, {hi:.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays-dir", default=os.path.join(_REPO_ROOT, "replays", "bulk"))
    ap.add_argument("--max-games", type=int, default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.replays_dir, "*.json")))
    if args.max_games:
        paths = paths[: args.max_games]
    split = int(len(paths) * 0.6)
    select_paths, holdout_paths = paths[:split], paths[split:]

    def load(paths):
        games = []
        for p in paths:
            rows = extract_rows(p)
            if rows:
                games.append(rows)
        return games

    print(f"loading {len(select_paths)} selection games...")
    select_games = load(select_paths)
    print(f"loading {len(holdout_paths)} holdout games...")
    holdout_games = load(holdout_paths)

    select_flat = [r for g in select_games for r in g]
    print(f"selection: {len(select_flat)} decisions across {len(select_games)} games")

    grid = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    best_w, best_acc = None, -1.0
    print("weight sweep (w_hand=1.0 fixed, sweeping w_threat) on SELECTION set:")
    for w in grid:
        a = acc(select_flat, 1.0, w)
        print(f"  w_threat={w:.2f}: sign_acc={a:.4f}")
        if a > best_acc:
            best_acc, best_w = a, w

    print(f"\nbest w_threat={best_w} (selection sign_acc={best_acc:.4f})")
    print("\nFinal report on HELD-OUT set (never used for weight selection):")
    early = [[r for r in g if r[2] <= 4] for g in holdout_games]
    mid = [[r for r in g if 5 <= r[2] <= 10] for g in holdout_games]
    late = [[r for r in g if r[2] >= 11] for g in holdout_games]
    early = [g for g in early if g]
    mid = [g for g in mid if g]
    late = [g for g in late if g]
    report("ALL", holdout_games, 1.0, best_w)
    report("EARLY (turn<=4)", early, 1.0, best_w)
    report("MID (5<=turn<=10)", mid, 1.0, best_w)
    report("LATE (turn>=11)", late, 1.0, best_w)

    print("\nFor comparison, w_threat=1.0 (naive equal weighting) on the SAME held-out set:")
    report("ALL (w=1.0)", holdout_games, 1.0, 1.0)


if __name__ == "__main__":
    main()
