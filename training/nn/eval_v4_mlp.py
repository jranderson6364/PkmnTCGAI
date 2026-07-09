"""Phi v4-MLP (pre-registered docs/report-log.md 2026-07-09 overnight):
small MLP over the 11 hand-calculated Phi v4 features (+ turn phase input),
trained on real replay outcomes. Same 60/40 game-level split discipline as
eval_v4.py; model selection by game-level 5-fold CV inside the fit set;
holdout evaluated once with a paired game-level bootstrap vs the Phi v4
linear champion.

Usage:
  python training/nn/eval_v4_mlp.py [--cache training/eval_v4_rows.pkl]
"""
import argparse
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
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


def load_games(cache):
    with open(cache, "rb") as fh:
        saved = pickle.load(fh)
    return saved["games"]


def stack(games):
    """games: list of (rows, outcome); rows: (features11, phi_v2, turn).
    Returns X (features + turn-phase), y in {-1,+1}, game_index."""
    X, y, gi = [], [], []
    for g, (rows, out) in enumerate(games):
        for f, _p2, t in rows:
            X.append(np.concatenate([f, [min(t, 30) / 6.0 / 5.0]]))
            y.append(out)
            gi.append(g)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(gi)


class MLP(nn.Module):
    def __init__(self, d_in, width, depth):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(Xtr, ytr, width, depth, l2, epochs, seed=0):
    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1], width, depth)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=l2)
    X = torch.from_numpy(Xtr)
    y = torch.from_numpy((ytr > 0).astype(np.float32))
    lossf = nn.BCEWithLogitsLoss()
    n = len(X)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        rng.shuffle(idx)
        for s in range(0, n, 4096):
            b = idx[s:s + 4096]
            opt.zero_grad()
            loss = lossf(model(X[b]), y[b])
            loss.backward()
            opt.step()
    model.eval()
    return model


def predict(model, X):
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy()


def sign_acc(v, y):
    return float(np.mean((v >= 0) == (y >= 0)))


def paired_bootstrap(per_game, n_res=4000, seed=13):
    rng = random.Random(seed)
    n = len(per_game)
    diffs = []
    for _ in range(n_res):
        idx = [rng.randrange(n) for _ in range(n)]
        a = np.concatenate([per_game[i][0] for i in idx])
        b = np.concatenate([per_game[i][1] for i in idx])
        o = np.concatenate([per_game[i][2] for i in idx])
        diffs.append(sign_acc(a, o) - sign_acc(b, o))
    diffs.sort()
    return (float(np.mean(diffs)), diffs[int(0.025 * len(diffs))],
            diffs[int(0.975 * len(diffs)) - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(_REPO_ROOT, "training", "eval_v4_rows.pkl"))
    args = ap.parse_args()

    games = load_games(args.cache)
    split = int(len(games) * 0.6)
    fit_games, holdout = games[:split], games[split:]
    print(f"fit={len(fit_games)} games, holdout={len(holdout)} games")

    w_lin = np.load(os.path.join(_REPO_ROOT, "training", "eval_v4_weights.npy"))

    # ---- CV model selection on fit set ----
    folds = 5
    grid = [(32, 1, 1e-4, 6), (32, 2, 1e-4, 6), (64, 2, 1e-4, 6),
            (64, 2, 1e-3, 6), (128, 2, 1e-4, 6), (64, 3, 1e-4, 10)]
    print("game-level 5-fold CV (width, depth, weight_decay, epochs):")
    best_cfg, best_cv = None, -1.0
    for cfg in grid:
        width, depth, l2, epochs = cfg
        accs = []
        for k in range(folds):
            tr = [g for i, g in enumerate(fit_games) if i % folds != k]
            va = [g for i, g in enumerate(fit_games) if i % folds == k]
            Xtr, ytr, _ = stack(tr)
            Xva, yva, _ = stack(va)
            m = train_mlp(Xtr, ytr, width, depth, l2, epochs, seed=k)
            accs.append(sign_acc(predict(m, Xva), yva))
        cv = float(np.mean(accs))
        print(f"  {cfg}: cv_sign_acc={cv:.4f}")
        if cv > best_cv:
            best_cv, best_cfg = cv, cfg
    print(f"selected {best_cfg} (cv={best_cv:.4f})")

    # ---- final train on full fit set, single holdout evaluation ----
    Xf, yf, _ = stack(fit_games)
    width, depth, l2, epochs = best_cfg
    model = train_mlp(Xf, yf, width, depth, l2, epochs, seed=7)

    segments = [("ALL", None), ("EARLY", lambda t: t <= 4),
                ("MID", lambda t: 5 <= t <= 10), ("LATE", lambda t: t >= 11)]
    for label, tf in segments:
        per_game = []
        for rows, out in holdout:
            fs = [(f, t) for f, _p2, t in rows if (tf is None or tf(t))]
            if not fs:
                continue
            Xg = np.array([np.concatenate([f, [min(t, 30) / 6.0 / 5.0]])
                           for f, t in fs], dtype=np.float32)
            v_mlp = predict(model, Xg)
            v_lin = np.array([float(f @ w_lin) for f, t in fs])
            o = np.full(len(fs), out, dtype=float)
            per_game.append((v_mlp, v_lin, o))
        a_mlp = sign_acc(np.concatenate([g[0] for g in per_game]),
                         np.concatenate([g[2] for g in per_game]))
        a_lin = sign_acc(np.concatenate([g[1] for g in per_game]),
                         np.concatenate([g[2] for g in per_game]))
        m, lo, hi = paired_bootstrap(per_game)
        print(f"HOLDOUT {label}: MLP={a_mlp:.3f} linear={a_lin:.3f} "
              f"paired_diff={m:+.4f} CI=[{lo:+.4f},{hi:+.4f}] games={len(per_game)}")

    torch.save(model.state_dict(),
               os.path.join(_REPO_ROOT, "training", "eval_v4_mlp.pth"))
    print("saved training/eval_v4_mlp.pth")


if __name__ == "__main__":
    main()
