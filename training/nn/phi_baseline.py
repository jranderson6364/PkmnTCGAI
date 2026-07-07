"""Phase 0 gate baseline (docs/nn-training.md Phase 0 amendment (a)):
sign-accuracy of the hand-crafted potential function Phi(s) ALONE against
real ladder replays, before any learned n-step value head is trained. Phi is
outcome-correlated by construction (it's built from prize differential and
hand size, i.e. the actual win condition), so this is the real bar the
learned component must clear -- not a flat borrowed number from a different
experiment (the oracle-critic's 62.5%, measured on a different, self-play-
derived holdout, not this replay set).

Phi(s) = 2.0*prize_diff + 1.0*hand_advantage - 1.5*wall_penalty + 0.5*line_progress
  prize_diff      = (opp_prizes_left - my_prizes_left) / 6        in [-1, 1]
  hand_advantage  = clamp((hand_size - cards_needed_for_KO) / 10, -1, 1)
                    in [-1, 1] -- two real bugs found and fixed here in turn,
                    both via isolated-component sign-accuracy checks before
                    trusting the combined formula (a 1361-replay diagnostic
                    run each time):
                      (1) an uncapped raw hand_size/10 swamped prize_diff
                          late-game (hand size grows for BOTH players every
                          turn regardless of who's winning -- an absolute
                          quantity, not a relative one), driving LATE
                          sign-acc to 45.5%, BELOW chance and the opposite of
                          the expected late-game-should-be-most-predictable
                          pattern;
                      (2) capping the raw value at 1.0 fixed the swamping but
                          not the sign: hand_size is only meaningful relative
                          to what's needed for a KO (exactly how main.py's
                          own `at_threshold`/`cards_needed` features use it,
                          `ceil(opp_hp/PH_DMG_PER_CARD)`), not on its own --
                          an isolated-component check on 400 late-game
                          replays showed prize_diff ALONE at 65.8% sign-acc,
                          dropping to 52.2% once the raw hand term was added,
                          confirming the raw term was net-harmful, not just
                          weak. Subtracting cards_needed before normalizing
                          fixes this: now genuinely measures "how close to
                          actually being able to KO," not just "hand is
                          growing." Falls back to hand_size/10 (no
                          subtraction) only when the opponent's Active is
                          empty/HP unknown (cards_needed undefined).
  wall_penalty    = 1.0 if the opponent's Active currently blocks Powerful
                    Hand (Mist/Rock Energy observed), else 0.0
  line_progress   = census()['line_count'] / 2.0                   in [0, 1]
                    (also isolated-checked: 62.6% alone on the same 400-game
                    late slice, a real but weaker signal than prize_diff --
                    kept at low weight per its "setup progress," lowest-
                    priority role.)
Weights are hand-set once from the heuristic's own priority ordering (prize
lead > hand advantage > wall > setup progress), never fit to the outcome
labels -- required for potential-based shaping (Ng/Harada/Russell 1999) to
stay policy-invariant, and it's also what keeps this a fair, ungamed floor
for the learned value head to beat. Every component was sanity-checked in
isolation against real replay outcomes before being trusted in the combined
formula -- see docs/report-log.md 2026-07-05 "Phi-only baseline" entries for
the full diagnostic trail.

Usage:
  python training/nn/phi_baseline.py [--replays-dir replays/bulk] [--max-games N]
"""
import argparse
import glob
import json
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import main as heuristic  # noqa: E402

OUR_TEAM = "Jason Anderson"


def our_seat(info):
    team_names = info.get("TeamNames", [])
    return team_names.index(OUR_TEAM) if OUR_TEAM in team_names else None


def phi(cur, me_idx):
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
    wall_penalty = 1.0 if wall else 0.0
    line_progress = cen["line_count"] / 2.0
    return 2.0 * prize_diff + 1.0 * hand_advantage - 1.5 * wall_penalty + 0.5 * line_progress


def phi_v2(cur, me_idx):
    """Variant (2026-07-05 user design session): swaps the one-sided
    hand_advantage term for `threat.net_threat_diff` -- a genuinely
    antisymmetric (zero-sum: evaluated from the opponent's seat, negates
    exactly) estimate of "my attack readiness against them minus their
    attack readiness against me," built from the real card/attack database
    (damage, energy cost) via the local cg.api shim, rather than a deck-
    specific hand-size heuristic. See threat.py's docstring for the
    known limitations (0-damage/conditional-damage attacks undercounted,
    energy color requirements ignored). Everything else identical to phi().
    """
    players = cur.get("players") or []
    if len(players) != 2:
        return None
    my_p, opp_p = players[me_idx], players[1 - me_idx]
    my_active = heuristic._active(my_p)
    bench = my_p.get("bench") or []
    my_prizes = len(my_p.get("prize") or [])
    opp_prizes = len(opp_p.get("prize") or [])
    opp_active = heuristic._active(opp_p)
    wall = heuristic._opp_has_blocking_energy(opp_active)
    cen = heuristic._census(my_active, bench)

    from threat import net_threat_diff  # local import: needs cg.api on sys.path

    prize_diff = (opp_prizes - my_prizes) / 6.0
    ntd = net_threat_diff(cur, me_idx)
    wall_penalty = 1.0 if wall else 0.0
    line_progress = cen["line_count"] / 2.0
    return 2.0 * prize_diff + 1.0 * ntd - 1.5 * wall_penalty + 0.5 * line_progress


def extract_rows(path, phi_fn=phi):
    """Returns list of (phi_value, outcome, turn) for our-seat decision points
    in one game, or [] if the game is unusable (no our-team seat, no clean
    terminal reward, etc)."""
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
            v = phi_fn(cur, you)
        except Exception:
            continue
        if v is None:
            continue
        rows.append((v, outcome, turn))
    return rows


def bootstrap_ci(game_rows, n_resamples=2000, seed=13):
    """Game-level bootstrap: resample GAMES (not individual decisions) with
    replacement, since decisions within one game are highly correlated --
    per-state CIs would overstate power on a replay corpus with only a few
    hundred distinct games (docs/nn-training.md Phase 0 amendment (e))."""
    rng = random.Random(seed)
    n_games = len(game_rows)
    accs = []
    for _ in range(n_resamples):
        sample_games = [game_rows[rng.randrange(n_games)] for _ in range(n_games)]
        flat = [r for g in sample_games for r in g]
        if not flat:
            continue
        acc = sum(1 for v, o, _ in flat if (v >= 0) == (o >= 0)) / len(flat)
        accs.append(acc)
    accs.sort()
    lo = accs[int(0.025 * len(accs))]
    hi = accs[int(0.975 * len(accs)) - 1]
    return lo, hi


def report(label, game_rows):
    flat = [r for g in game_rows for r in g]
    if not flat:
        print(f"{label}: no samples")
        return
    n = len(flat)
    acc = sum(1 for v, o, _ in flat if (v >= 0) == (o >= 0)) / n
    lo, hi = bootstrap_ci(game_rows) if game_rows else (float("nan"), float("nan"))
    print(f"{label}: n_decisions={n} n_games={len(game_rows)} sign_acc={acc:.3f} "
          f"game_level_95%CI=[{lo:.3f}, {hi:.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays-dir", default=os.path.join(_REPO_ROOT, "replays", "bulk"))
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--version", type=int, default=1, choices=[1, 2],
                     help="1 = original Φ (hand_advantage), 2 = threat.net_threat_diff variant")
    args = ap.parse_args()
    phi_fn = phi if args.version == 1 else phi_v2

    paths = sorted(glob.glob(os.path.join(args.replays_dir, "*.json")))
    if args.max_games:
        paths = paths[: args.max_games]

    all_games, early_games, mid_games, late_games = [], [], [], []
    skipped = 0
    for p in paths:
        rows = extract_rows(p, phi_fn)
        if not rows:
            skipped += 1
            continue
        all_games.append(rows)
        early = [r for r in rows if r[2] <= 4]
        mid = [r for r in rows if 5 <= r[2] <= 10]
        late = [r for r in rows if r[2] >= 11]
        if early:
            early_games.append(early)
        if mid:
            mid_games.append(mid)
        if late:
            late_games.append(late)

    print(f"replay files scanned={len(paths)} usable_games={len(all_games)} skipped={skipped}")
    report("ALL", all_games)
    report("EARLY (turn<=4)", early_games)
    report("MID (5<=turn<=10)", mid_games)
    report("LATE (turn>=11)", late_games)


if __name__ == "__main__":
    main()
