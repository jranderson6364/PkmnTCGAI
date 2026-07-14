"""Phase A of the learn-inside-the-champion pivot (issue #3, 2026-07-13):
data-derived regime detector for the board-thinning/deck-out failure regime.

Fits the tightest conjunctive rule that captures >=90% of decision states in
the intervention window of mined board-thinning LOSSES while firing on <=2%
of depth-matched states from WINS. Every feature is computable from the live
obs_dict main.agent() receives, so the fitted rule can gate a subpolicy at
inference with zero extra information.

Corpora:
  positives: replays/exploiter_wins/*.json (18 games, all board-thinning by
             construction — 2026-07-05 mining) + replays/v29d_ladder losses
             whose final decision state has zero Alakazam-line pieces in play
             (the 2026-07-09 blunder_scan criterion), last WINDOW turns each.
  negatives: replays/v29d_ladder WIN games — both the same last-WINDOW-turns
             slice (depth-matched control) and all-states (reported for
             context).

Usage:
  python training/nn/regime_detector.py [--window 6] [--report]
"""
import argparse
import glob
import itertools
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

from eval_v4 import _active, _in_play, _armed_count, our_seat  # noqa: E402

LINE_IDS = {741, 742, 743}  # Abra, Kadabra, Alakazam
ABRA_ID = 741
ALAKAZAM_ID = 743


def regime_features(cur, me_idx):
    """Obs-derivable absolute features for the regime decision.
    Must stay computable from a live obs_dict (inference-time contract)."""
    players = cur.get("players") or []
    if len(players) != 2:
        return None
    me, opp = players[me_idx], players[1 - me_idx]
    hand = me.get("hand") or []
    hand_ids = [c.get("id") for c in hand if c]
    in_play = _in_play(me)
    line_in_play = sum(1 for pk in in_play if pk.get("id") in LINE_IDS)
    my_active = _active(me)
    opp_active = _active(opp)
    active_ready = 1 if (my_active and my_active.get("id") == ALAKAZAM_ID
                         and len(my_active.get("energies") or [])) else 0
    return {
        "deck": me.get("deckCount") or 0,
        "hand": me.get("handCount") or len(hand_ids),
        "line_in_play": line_in_play,
        "fieldable_line": line_in_play + sum(1 for i in hand_ids if i == ABRA_ID),
        "line_in_hand": sum(1 for i in hand_ids if i in LINE_IDS),
        "board": len(in_play),
        "opp_armed": 1 if (opp_active and len(opp_active.get("energies") or [])) else 0,
        "opp_armed_count": _armed_count(opp),
        "active_ready": active_ready,
        "my_prizes": len(me.get("prize") or []),
        "opp_prizes": len(opp.get("prize") or []),
        "turn": cur.get("turn") or 0,
    }


def walk_game(path):
    """Yields (features_dict, turn) for each of OUR decision states; returns
    (rows, outcome) or None. Mirrors eval_v4.extract_game's walk exactly."""
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    you = our_seat(d.get("info", {}))
    if you is None:
        return None
    rewards = d.get("rewards")
    if not rewards or len(rewards) != 2 or rewards[you] not in (1, -1):
        return None
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
        f = regime_features(cur, you)
        if f is not None:
            rows.append(f)
    if not rows:
        return None
    return rows, rewards[you]


def last_window(rows, window):
    """States within the final `window` turns of the game's decisions."""
    last_turn = rows[-1]["turn"]
    return [r for r in rows if r["turn"] > last_turn - window]


def load_corpora(window):
    """Returns (loss_games, win_games) where each game is the per-decision
    feature-row list; loss_games are pre-sliced to the last-`window`-turns
    intervention window, win_games keep all rows (FP measured per state)."""
    loss_games, win_games = [], []
    other_losses = 0

    for p in sorted(glob.glob(os.path.join(_REPO_ROOT, "replays", "exploiter_wins", "*.json"))):
        g = walk_game(p)
        if not g:
            continue
        rows, outcome = g
        if outcome == -1:
            loss_games.append(last_window(rows, window))

    for p in sorted(glob.glob(os.path.join(_REPO_ROOT, "replays", "v29d_ladder", "*.json"))):
        g = walk_game(p)
        if not g:
            continue
        rows, outcome = g
        if outcome == 1:
            win_games.append(rows)
        elif rows[-1]["line_in_play"] == 0:  # board-thinning loss signature
            loss_games.append(last_window(rows, window))
        else:
            other_losses += 1

    n_pos = sum(len(g) for g in loss_games)
    n_neg = sum(len(g) for g in win_games)
    print(f"games: {len(loss_games)} board-thinning losses (positives), "
          f"{len(win_games)} wins (negatives), {other_losses} other losses (excluded)")
    print(f"states: {n_pos} positive-window, {n_neg} win (FP pool)")
    return loss_games, win_games


# Candidate rule atoms: (name, predicate). Conjunctions of 2-4 atoms searched.
def atoms():
    out = []
    for d in (8, 10, 12, 15, 18):
        out.append((f"deck<={d}", lambda r, d=d: r["deck"] <= d))
    for fl in (0, 1, 2):
        out.append((f"fieldable_line<={fl}", lambda r, fl=fl: r["fieldable_line"] <= fl))
    for lp in (0, 1):
        out.append((f"line_in_play<={lp}", lambda r, lp=lp: r["line_in_play"] <= lp))
    out.append(("opp_armed", lambda r: r["opp_armed"] == 1))
    out.append(("not_active_ready", lambda r: r["active_ready"] == 0))
    for h in (10, 13, 15):
        out.append((f"hand>={h}", lambda r, h=h: r["hand"] >= h))
    for b in (2, 3):
        out.append((f"board<={b}", lambda r, b=b: r["board"] <= b))
    for t in (5, 7, 9):
        out.append((f"turn>={t}", lambda r, t=t: r["turn"] >= t))
    for lh in (1, 2):
        out.append((f"line_in_hand>={lh}", lambda r, lh=lh: r["line_in_hand"] >= lh))
    return out


def _fires(rule, r):
    return all(pred(r) for _n, pred in rule)


# ---------------- FITTED RULE (pre-registered 2026-07-13) ----------------
# Fitted on 28 board-thinning losses (18 exploiter_wins + 10 v29d_ladder,
# window=6 turns) vs 33 v29d_ladder wins (3,514 decision states):
#   game capture 26/28 (92.9%), per-state FP 19/3514 (0.54%),
#   state coverage 62.9%, median headroom 3 turns.
# The 2 uncaptured losses are early setup collapses (turn 3/8, board=1) —
# a different failure class, out of this regime by design.
# A third clause (fieldable_line<=1 AND opp_armed, the report-log fix
# candidate) was tested and REJECTED: FP 3.2-3.7% > the 2% bar — v29d wins
# through those states too often to override there.

def regime_fires(cur, me_idx):
    """Canonical inference-time detector. True => the learned subpolicy
    (not the heuristic) chooses this decision. Cheap: no table lookups."""
    r = regime_features(cur, me_idx)
    if r is None or r["turn"] < 9:
        return False
    return r["line_in_play"] == 0 or (r["deck"] <= 6 and r["hand"] >= 15)


def fit(loss_games, win_games, min_game_capture=0.90, max_fp=0.02,
        min_fires_per_game=1):
    """Per-GAME capture (detector fires >= min_fires times in the loss game's
    intervention window — i.e. the subpolicy gets a chance to act) vs
    per-STATE false-positive rate over every decision in win games."""
    cands = atoms()
    neg_states = [r for g in win_games for r in g]
    results = []
    for k in (2, 3, 4):
        for rule in itertools.combinations(cands, k):
            names = [n for n, _p in rule]
            feats = [n.split("<=")[0].split(">=")[0] for n in names]
            if len(set(feats)) != len(feats):
                continue
            games_hit = sum(1 for g in loss_games
                            if sum(1 for r in g if _fires(rule, r)) >= min_fires_per_game)
            cap = games_hit / max(1, len(loss_games))
            if cap < min_game_capture:
                continue
            fp = sum(1 for r in neg_states if _fires(rule, r)) / max(1, len(neg_states))
            if fp > max_fp:
                continue
            # states covered in loss windows (how much play the subpolicy owns)
            pos_states = [r for g in loss_games for r in g]
            state_cov = sum(1 for r in pos_states if _fires(rule, r)) / max(1, len(pos_states))
            # median turns of headroom before the game ends once it first fires
            headrooms = []
            for g in loss_games:
                fire_turns = [r["turn"] for r in g if _fires(rule, r)]
                if fire_turns:
                    headrooms.append(g[-1]["turn"] - min(fire_turns))
            headrooms.sort()
            med_head = headrooms[len(headrooms) // 2] if headrooms else 0
            results.append((fp, -cap, -state_cov, med_head, names))
    results.sort()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--min-capture", type=float, default=0.90,
                    help="per-GAME capture: rule must fire in this share of loss games")
    ap.add_argument("--max-fp", type=float, default=0.02,
                    help="per-STATE FP over all win-game decisions")
    ap.add_argument("--min-fires", type=int, default=1)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    loss_games, win_games = load_corpora(args.window)
    results = fit(loss_games, win_games, args.min_capture, args.max_fp,
                  args.min_fires)
    if not results:
        print(f"\nNO rule met game-capture>={args.min_capture:.0%} with "
              f"FP<={args.max_fp:.0%} — diagnostic frontier:")
        diag = fit(loss_games, win_games, 0.5, 1.0, args.min_fires)
        for fp, negcap, negcov, head, names in diag[:args.top]:
            print(f"  game_cap={-negcap:.1%} fp={fp:.2%} state_cov={-negcov:.1%} "
                  f"headroom~{head}t  {' AND '.join(names)}")
        return
    print(f"\nrules meeting game-capture>={args.min_capture:.0%}, "
          f"FP<={args.max_fp:.0%} — lowest FP first:")
    for fp, negcap, negcov, head, names in results[:args.top]:
        print(f"  game_cap={-negcap:.1%} fp={fp:.2%} state_cov={-negcov:.1%} "
              f"headroom~{head}t  {' AND '.join(names)}")


if __name__ == "__main__":
    main()
