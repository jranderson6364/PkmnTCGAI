"""Phi v4: literature-driven antisymmetric linear evaluation function
(pre-registered docs/report-log.md 2026-07-09).

Design constraints carried from the 2026-07-05 Phi v3 failure:
  * EVERY feature is difference-form, computed by the SAME method for both
    seats -> the linear combination is antisymmetric by construction (the
    property that made Phi v2 win).
  * Fitting is logistic regression with NO intercept (an intercept breaks
    antisymmetry), L2-regularized, with a game-level 60/40 fit/holdout
    split by sorted file order (same convention as phi_weight_sweep.py)
    and L2 strength chosen by game-level 5-fold CV inside the fit set only.
    The holdout is touched exactly once, for the final pre-registered
    report.

Features (11, all oriented so + = good for me, all roughly [-1, 1]):
  prize_diff      (opp_prizes_left - my_prizes_left)/6          [Phi v2]
  net_threat      threat_against(me->them) - (them->me)         [Phi v2]
  ko_speed_diff   1/turns_to_KO(me->them) - 1/turns_to_KO(them->me)
  energy_dev_diff (my total attached energy - theirs)/8
  board_size_diff (my Pokemon in play - theirs)/6
  armed_diff      (my Pokemon able to pay some damaging attack - theirs)/3
  hand_diff       (my handCount - theirs)/10
  deck_clock_diff (my deckCount - theirs)/20
  wall_diff       block(my active) - block(their active)   (Mist/Rock: my
                  active carrying a blocker hurts THEIR damage -> good)
  stage_dev_diff  (sum of preEvolution depths in play, mine - theirs)/4
  status_diff     (their active condition count - mine)/2

Usage:
  python training/nn/eval_v4.py [--replays-dir replays/bulk] [--max-games N]
                                [--cache training/eval_v4_rows.pkl]
"""
import argparse
import glob
import json
import math
import os
import pickle
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

import main as heuristic  # noqa: E402
from phi_baseline import our_seat, phi_v2  # noqa: E402
from threat import threat_against, _load_tables  # noqa: E402
import threat as threat_mod  # noqa: E402

FEATURES = [
    "prize_diff", "net_threat", "ko_speed_diff", "energy_dev_diff",
    "board_size_diff", "armed_diff", "hand_diff", "deck_clock_diff",
    "wall_diff", "stage_dev_diff", "status_diff",
]

_CONDITIONS = ("asleep", "paralyzed", "confused", "poisoned", "burned")


def _active(p):
    a = p.get("active")
    return a[0] if a and len(a) > 0 and a[0] else None


def _in_play(p):
    out = []
    a = _active(p)
    if a:
        out.append(a)
    for b in p.get("bench") or []:
        if b:
            out.append(b)
    return out


def _ko_speed(attacker, defender):
    """1 / (turns to KO defender with attacker's best static attack),
    0 if no usable damaging attack on record. Same known limitation as
    threat.py: scaling-damage attacks (incl. our own Powerful Hand) list
    static damage 0 and score 0 -- identical treatment both sides."""
    _load_tables()
    if not attacker or not defender:
        return 0.0
    def_hp = defender.get("hp") or 0
    if def_hp <= 0:
        return 0.0
    attack_ids = threat_mod._CARD_ATTACKS.get((attacker or {}).get("id", -1)) or []
    cur_energy = len(attacker.get("energies") or [])
    best = 0.0
    for aid in attack_ids:
        info = threat_mod._ATTACK_INFO.get(aid)
        if not info:
            continue
        damage, needed = info
        if damage <= 0:
            continue
        turns = max(0, needed - cur_energy) + math.ceil(def_hp / damage)
        best = max(best, 1.0 / turns)
    return best


def _armed_count(p):
    """Pokemon in play able to pay the energy cost of at least one
    damaging attack right now (fungible-energy simplification, same as
    threat.py)."""
    _load_tables()
    n = 0
    for pk in _in_play(p):
        cur = len(pk.get("energies") or [])
        for aid in threat_mod._CARD_ATTACKS.get(pk.get("id", -1)) or []:
            info = threat_mod._ATTACK_INFO.get(aid)
            if info and info[0] > 0 and info[1] <= cur:
                n += 1
                break
    return n


def _side_scalars(p):
    return {
        "energy": sum(len(pk.get("energies") or []) for pk in _in_play(p)),
        "board": len(_in_play(p)),
        "armed": _armed_count(p),
        "hand": p.get("handCount") or 0,
        "deck": p.get("deckCount") or 0,
        "wall": 1.0 if heuristic._opp_has_blocking_energy(_active(p)) else 0.0,
        "stage": sum(len(pk.get("preEvolution") or []) for pk in _in_play(p)),
        "cond": sum(1.0 for c in _CONDITIONS if p.get(c)),
    }


def features_v4(cur, me_idx):
    """Returns np.array of the 11 antisymmetric features, or None."""
    players = cur.get("players") or []
    if len(players) != 2:
        return None
    me, opp = players[me_idx], players[1 - me_idx]
    ms, os_ = _side_scalars(me), _side_scalars(opp)
    my_a, opp_a = _active(me), _active(opp)
    return np.array([
        (len(opp.get("prize") or []) - len(me.get("prize") or [])) / 6.0,
        threat_against(my_a, opp_a) - threat_against(opp_a, my_a),
        _ko_speed(my_a, opp_a) - _ko_speed(opp_a, my_a),
        (ms["energy"] - os_["energy"]) / 8.0,
        (ms["board"] - os_["board"]) / 6.0,
        (ms["armed"] - os_["armed"]) / 3.0,
        (ms["hand"] - os_["hand"]) / 10.0,
        (ms["deck"] - os_["deck"]) / 20.0,
        ms["wall"] - os_["wall"],
        (ms["stage"] - os_["stage"]) / 4.0,
        (os_["cond"] - ms["cond"]) / 2.0,
    ])


def eval_v4(cur, me_idx, weights):
    f = features_v4(cur, me_idx)
    return None if f is None else float(f @ weights)


# ---------------- corpus extraction ----------------

def extract_game(path):
    """Returns (list of (features, phi_v2_value, turn), outcome) or None."""
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
        try:
            f = features_v4(cur, you)
            p2 = phi_v2(cur, you)
        except Exception:
            continue
        if f is None or p2 is None:
            continue
        rows.append((f, p2, cur.get("turn") or 0))
    if not rows:
        return None
    return rows, rewards[you]


def load_corpus(replays_dir, max_games=None, cache=None):
    paths = sorted(glob.glob(os.path.join(replays_dir, "*.json")))
    if max_games:
        paths = paths[:max_games]
    key = (len(paths), paths[0] if paths else "", paths[-1] if paths else "")
    if cache and os.path.exists(cache):
        with open(cache, "rb") as fh:
            saved = pickle.load(fh)
        if saved.get("key") == key:
            print(f"loaded cached corpus: {len(saved['games'])} games")
            return saved["games"]
    games = []
    for i, p in enumerate(paths):
        g = extract_game(p)
        if g:
            games.append(g)
        if (i + 1) % 200 == 0:
            print(f"  parsed {i+1}/{len(paths)} files, usable={len(games)}")
    if cache:
        with open(cache, "wb") as fh:
            pickle.dump({"key": key, "games": games}, fh)
    print(f"corpus: {len(games)} usable games of {len(paths)} files")
    return games


# ---------------- logistic fit (no intercept, L2, Newton) ----------------

def fit_logistic(X, y, l2, iters=50):
    """y in {-1,+1}; minimizes mean logloss + l2*||w||^2, no intercept."""
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(iters):
        z = X @ w * y
        p = 1.0 / (1.0 + np.exp(-z))          # P(correct)
        g = -(X * ((1 - p) * y)[:, None]).mean(axis=0) + 2 * l2 * w
        s = p * (1 - p)
        H = (X.T * s) @ X / n + 2 * l2 * np.eye(d)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return w


def sign_acc(values, outcomes):
    return float(np.mean((values >= 0) == (outcomes >= 0)))


def bootstrap_ci_games(per_game, n_resamples=2000, seed=13):
    """per_game: list of (values_array, outcomes_array). Game-level bootstrap
    (same methodology as phi_baseline.py)."""
    rng = random.Random(seed)
    n = len(per_game)
    accs = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        v = np.concatenate([per_game[i][0] for i in idx])
        o = np.concatenate([per_game[i][1] for i in idx])
        accs.append(sign_acc(v, o))
    accs.sort()
    return accs[int(0.025 * len(accs))], accs[int(0.975 * len(accs)) - 1]


def report_arm(label, games, value_fn, turn_filter=None):
    per_game = []
    for rows, outcome in games:
        vals, outs = [], []
        for f, p2, turn in rows:
            if turn_filter and not turn_filter(turn):
                continue
            vals.append(value_fn(f, p2))
            outs.append(outcome)
        if vals:
            per_game.append((np.array(vals), np.array(outs)))
    if not per_game:
        print(f"  {label}: no samples")
        return None
    v = np.concatenate([g[0] for g in per_game])
    o = np.concatenate([g[1] for g in per_game])
    a = sign_acc(v, o)
    lo, hi = bootstrap_ci_games(per_game)
    print(f"  {label}: n={len(v)} games={len(per_game)} sign_acc={a:.3f} CI=[{lo:.3f},{hi:.3f}]")
    return a, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays-dir", default=os.path.join(_REPO_ROOT, "replays", "bulk"))
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--cache", default=os.path.join(_REPO_ROOT, "training", "eval_v4_rows.pkl"))
    args = ap.parse_args()

    games = load_corpus(args.replays_dir, args.max_games, args.cache)
    split = int(len(games) * 0.6)
    fit_games, holdout = games[:split], games[split:]
    print(f"fit={len(fit_games)} games, holdout={len(holdout)} games")

    def stack(gs):
        X = np.vstack([f for rows, _ in gs for f, _, _ in rows])
        y = np.array([out for rows, out in gs for _ in rows], dtype=float)
        return X, y

    # ---- CV over L2 strength on the fit set (game-level folds) ----
    folds = 5
    grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    print("game-level 5-fold CV on fit set (11-feature logistic):")
    best_l2, best_cv = None, -1.0
    for l2 in grid:
        accs = []
        for k in range(folds):
            tr = [g for i, g in enumerate(fit_games) if i % folds != k]
            va = [g for i, g in enumerate(fit_games) if i % folds == k]
            Xtr, ytr = stack(tr)
            Xva, yva = stack(va)
            w = fit_logistic(Xtr, ytr, l2)
            accs.append(sign_acc(Xva @ w, yva))
        cv = float(np.mean(accs))
        print(f"  l2={l2:g}: cv_sign_acc={cv:.4f}")
        if cv > best_cv + 1e-9:
            best_cv, best_l2 = cv, l2
    print(f"selected l2={best_l2:g} (cv={best_cv:.4f})")

    # ---- final fits on the whole fit set ----
    Xf, yf = stack(fit_games)
    w_v4 = fit_logistic(Xf, yf, best_l2)
    print("\nfitted Phi v4 weights:")
    for name, wt in zip(FEATURES, w_v4):
        print(f"  {name:16s} {wt:+.3f}")

    # control: refit of Phi v2's own info content = subset features
    # (prize_diff, net_threat, wall_diff, stage_dev_diff ~ line analog)
    sub_idx = [0, 1, 8, 9]
    w_sub = np.zeros(len(FEATURES))
    w_fit_sub = fit_logistic(Xf[:, sub_idx], yf, best_l2)
    w_sub[sub_idx] = w_fit_sub

    w_equal = np.ones(len(FEATURES))
    w_prize = np.zeros(len(FEATURES)); w_prize[0] = 1.0

    # ---- single holdout evaluation, all arms ----
    arms = [
        ("Phi v2 (champion bar)      ", lambda f, p2: p2),
        ("Phi v4 fitted              ", lambda f, p2: float(f @ w_v4)),
        ("Phi v4 equal weights       ", lambda f, p2: float(f @ w_equal)),
        ("Phi v2-components refit    ", lambda f, p2: float(f @ w_sub)),
        ("prize_diff only            ", lambda f, p2: float(f @ w_prize)),
    ]
    segments = [
        ("ALL", None),
        ("EARLY (turn<=4)", lambda t: t <= 4),
        ("MID (5<=t<=10)", lambda t: 5 <= t <= 10),
        ("LATE (turn>=11)", lambda t: t >= 11),
    ]
    for seg_label, tf in segments:
        print(f"\nHOLDOUT {seg_label}:")
        for label, fn in arms:
            report_arm(label, holdout, fn, tf)

    np.save(os.path.join(_REPO_ROOT, "training", "eval_v4_weights.npy"), w_v4)
    print("\nweights saved to training/eval_v4_weights.npy")


if __name__ == "__main__":
    main()
