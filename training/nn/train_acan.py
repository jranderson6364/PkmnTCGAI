"""Train + gate the Action-Conditioned Advantage Net (S3, pre-registered 2026-07-23).

Reads the merged ACAN corpus (search-decision records from the shipped d2/formula
776 search), trains ACAN to predict each candidate's advantage over the heuristic's
own pick, and gates it on OVERRIDE FIDELITY against held-out GAMES.

Why override-fidelity and not accuracy: a net that never overrides plays the pure
heuristic (=673 on the ladder) and still scores ~90% raw accuracy, because most
candidates are obviously bad and the search agrees with the heuristic a lot. The
only thing that carries the 776-vs-673 gap is the set of MARGIN-CLEARED overrides,
so that set is what we score -- precision/recall against the trivial never-override
baseline, whose recall is 0 by definition.

Thresholding is RATE-MATCHED: the net's regressed advantages are compressed toward
the conditional mean (MSE does that), so applying the search's raw MARGIN to them
under-fires. What transfers is the RANKING, so the deploy threshold is chosen to
reproduce the search's override RATE on held-out games, and reported alongside a
full sweep.

Run:  python training/nn/train_acan.py --corpus-dir training/nn/acan_corpus
"""
import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "training", "local_cg"),
           os.path.join(_REPO, "training", "belief"),
           os.path.join(_REPO, "training", "nn"),
           os.path.join(_REPO, "training"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn

from encode import numeric_feats, NUM_FEATS
from acan_model import ACAN, act_index

MARGIN = 500.0       # the shipped search's override bar (twoply formula leaf)


def load_decisions(corpus_dir):
    decisions, n_bad = [], 0
    for jl in sorted(glob.glob(os.path.join(corpus_dir, "*.jsonl"))):
        for line in open(jl):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                feats = numeric_feats({"current": rec["current"]})
                if feats is None or len(feats) != NUM_FEATS:
                    n_bad += 1
                    continue
                cv, ht = rec["cand_val"], str(rec["heur_top"])
                if ht not in cv:
                    n_bad += 1
                    continue
                base_v, bs = float(cv[ht]), rec.get("base", {})
                cands = [int(i) for i in rec["cand"] if str(i) in rec["acts"]]
                if not cands:
                    n_bad += 1
                    continue
                hb = float(bs.get(ht, 0.0))
                decisions.append({
                    "game": rec.get("game", "?"),
                    "feats": np.asarray(list(feats), dtype=np.float32),
                    "cands": cands,
                    "acts": {i: rec["acts"][str(i)] for i in cands},
                    "adv": {i: float(cv[str(i)]) - base_v for i in cands},
                    "brel": {i: float(bs.get(str(i), hb)) - hb for i in cands},
                    "heur_top": int(rec["heur_top"]),
                })
            except Exception:
                n_bad += 1
    return decisions, n_bad


def build_vocabs(decisions):
    cards, atks = {0: 0}, {0: 0}
    for d in decisions:
        for desc in d["acts"].values():
            c, a = int(desc[1]), int(desc[2])
            if c not in cards:
                cards[c] = len(cards)
            if a not in atks:
                atks[a] = len(atks)
    return cards, atks


def act_floats(d, i, base_scale):
    return [1.0 if i == d["heur_top"] else 0.0, d["brel"][i] / base_scale]


def to_arrays(decisions, cards, atks, clip, base_scale):
    """Decision-GROUPED padded tensors.

    Grouping by decision (rather than a flat list of pairs) is what makes the
    listwise ranking loss possible: the gate scores the ARGMAX over a decision's
    candidates, but MSE only ever sees one candidate at a time and optimises
    pointwise magnitude. Ranking needs the whole candidate set together.
    """
    C = max(len(d["cands"]) for d in decisions)
    D = len(decisions)
    S = np.zeros((D, NUM_FEATS), dtype=np.float32)
    A = np.zeros((D, C, 3), dtype=np.int64)
    F = np.zeros((D, C, 2), dtype=np.float32)
    Y = np.zeros((D, C), dtype=np.float32)
    M = np.zeros((D, C), dtype=np.float32)
    T = np.zeros((D,), dtype=np.int64)
    for k, d in enumerate(decisions):
        S[k] = d["feats"]
        cs = d["cands"]
        for j, i in enumerate(cs):
            A[k, j] = act_index(d["acts"][i], cards, atks)
            F[k, j] = act_floats(d, i, base_scale)
            Y[k, j] = float(np.clip(d["adv"][i], -clip, clip)) / clip
            M[k, j] = 1.0
        T[k] = int(np.argmax([d["adv"][i] for i in cs]))   # search's own pick
    return S, A, F, Y, M, T


def predict(model, decisions, cards, atks, clip, mu, sd, base_scale):
    """-> list of (net_best_cand, net_score_raw); score = -inf when the net's own
    argmax IS the heuristic's pick (i.e. it declines to override)."""
    model.eval()
    out = []
    with torch.no_grad():
        for d in decisions:
            cs = d["cands"]
            s = torch.from_numpy(((np.tile(d["feats"], (len(cs), 1)) - mu) / sd
                                  ).astype(np.float32))
            a = torch.tensor([act_index(d["acts"][i], cards, atks) for i in cs],
                             dtype=torch.long)
            f = torch.tensor([act_floats(d, i, base_scale) for i in cs],
                             dtype=torch.float32)
            p = model(s, a, f).numpy() * clip
            j = int(np.argmax(p))
            out.append((cs[j], -np.inf if cs[j] == d["heur_top"] else float(p[j])))
    return out


def score_at(decisions, preds, th):
    tp = fp = fn = 0
    n_net = n_search = loose = 0
    for d, (nb, sc) in zip(decisions, preds):
        ht = d["heur_top"]
        sb = max(d["cands"], key=lambda i: d["adv"][i])
        search_ov = (sb != ht and d["adv"][sb] >= MARGIN)
        net_ov = sc >= th
        n_net += int(net_ov)
        n_search += int(search_ov)
        if net_ov and search_ov:
            loose += 1
        if net_ov and search_ov and nb == sb:
            tp += 1
        elif net_ov:
            fp += 1
        elif search_ov:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    lprec = loose / n_net if n_net else float("nan")
    return n_net, n_search, prec, rec, lprec


def rate_matched_threshold(decisions, preds):
    """Threshold reproducing the search's override RATE on these decisions."""
    n_search = sum(1 for d in decisions
                   if (lambda sb: sb != d["heur_top"] and d["adv"][sb] >= MARGIN)(
                       max(d["cands"], key=lambda i: d["adv"][i])))
    scores = np.sort(np.array([sc for _, sc in preds if np.isfinite(sc)]))[::-1]
    if n_search <= 0 or scores.size == 0:
        return float("inf"), n_search
    if n_search >= scores.size:
        return float(scores[-1]), n_search
    return float(scores[n_search - 1]), n_search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default=os.path.join(_HERE, "acan_corpus"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--rank-weight", type=float, default=1.0,
                    help="listwise ranking loss weight (0 = MSE-only ablation)")
    ap.add_argument("--rank-temp", type=float, default=0.25,
                    help="softmax temperature over candidate advantages")
    ap.add_argument("--out", default=os.path.join(_HERE, "acan.pth"))
    args = ap.parse_args()

    print("loading corpus ...", flush=True)
    decisions, n_bad = load_decisions(args.corpus_dir)
    if not decisions:
        print("no decisions found -- is the collection finished?")
        return
    games = sorted({d["game"] for d in decisions})
    print(f"decisions {len(decisions)} | games {len(games)} | skipped {n_bad}")

    # SPLIT BY GAME -- records from one game are correlated; a record-level split
    # leaks the same position into train and val and inflates the gate.
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(games))
    n_val = max(1, int(len(games) * args.val_frac))
    val_games = {games[i] for i in perm[:n_val]}
    tr = [d for d in decisions if d["game"] not in val_games]
    va = [d for d in decisions if d["game"] in val_games]
    print(f"train decisions {len(tr)} | val decisions {len(va)} "
          f"({len(val_games)} held-out games)")

    flat = np.array([a for d in tr for a in d["adv"].values()], dtype=np.float64)
    p95 = float(np.percentile(np.abs(flat[np.abs(flat) < 1e6]), 95)) if flat.size else 0.0
    clip = max(4 * MARGIN, p95)
    braw = np.array([abs(b) for d in tr for b in d["brel"].values()], dtype=np.float64)
    base_scale = max(1.0, float(np.percentile(braw, 95)) if braw.size else 1.0)
    print(f"advantage |p95|(non-sentinel) {p95:.1f} -> clip {clip:.1f} "
          f"(MARGIN {MARGIN} = {MARGIN/clip:.3f} normalised) | base_scale {base_scale:.1f}")

    cards, atks = build_vocabs(tr)
    print(f"vocab: cards {len(cards)} | attacks {len(atks)}")
    Str, Atr, Ftr, Ytr, Mtr, Ttr = to_arrays(tr, cards, atks, clip, base_scale)
    mu, sd = Str.mean(0), Str.std(0) + 1e-6
    Xt = torch.from_numpy((Str - mu) / sd)
    At, Ft, Yt, Mt, Tt = (torch.from_numpy(Atr), torch.from_numpy(Ftr),
                          torch.from_numpy(Ytr), torch.from_numpy(Mtr),
                          torch.from_numpy(Ttr))
    npair = int(Mtr.sum())
    print(f"train decisions {len(Ytr)} | pairs {npair} | max cands {Atr.shape[1]} "
          f"| target >0 {100*((Ytr>0)*Mtr).sum()/npair:.1f}%")

    model = ACAN(NUM_FEATS, len(cards), len(atks))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    C = Atr.shape[1]

    def fwd(j):
        """-> [b, C] predicted advantages for a batch of decisions."""
        b = len(j)
        s = Xt[j].unsqueeze(1).expand(b, C, NUM_FEATS).reshape(b * C, NUM_FEATS)
        return model(s, At[j].reshape(b * C, 3),
                     Ft[j].reshape(b * C, 2)).view(b, C)

    for ep in range(args.epochs):
        model.train()
        idx = torch.randperm(len(Yt))
        tot = 0.0
        for b in range(0, len(idx), args.batch):
            j = idx[b:b + args.batch]
            opt.zero_grad()
            out = fwd(j)
            m = Mt[j]
            # magnitude: masked MSE (keeps the scale the margin threshold needs)
            mse = (((out - Yt[j]) ** 2) * m).sum() / m.sum().clamp(min=1)
            loss = mse
            if args.rank_weight > 0:
                # ranking: listwise CE toward the candidate the search ranked top
                logits = out.masked_fill(m < 0.5, -1e9) / args.rank_temp
                loss = loss + args.rank_weight * nn.functional.cross_entropy(
                    logits, Tt[j])
            loss.backward()
            opt.step()
            tot += mse.item() * len(j)
        preds = predict(model, va, cards, atks, clip, mu, sd, base_scale)
        th, _ = rate_matched_threshold(va, preds)
        n_net, n_search, prec, rec, lprec = score_at(va, preds, th)
        print(f"epoch {ep+1}/{args.epochs} | mse {tot/len(Yt):.4f} | "
              f"rate-matched th {th:>8.1f} | prec {prec:.3f} rec {rec:.3f} "
              f"(net {n_net} vs search {n_search})", flush=True)

    preds = predict(model, va, cards, atks, clip, mu, sd, base_scale)
    th_rm, n_search = rate_matched_threshold(va, preds)
    print("\n=== GATE: override fidelity on held-out games ===")
    print(f"search made {n_search} margin-cleared overrides in "
          f"{len(va)} held-out decisions ({100*n_search/max(1,len(va)):.1f}%)")
    print("baseline (never override): recall 0.000 by definition")
    print(f"\n{'thresh':>10} {'net_ovr':>8} {'prec':>7} {'recall':>7} {'loose_p':>8}")
    grid = [th_rm] + [np.quantile([s for _, s in preds if np.isfinite(s)], q)
                      for q in (0.99, 0.95, 0.9, 0.75, 0.5)]
    for t in grid:
        n_net, _, prec, rec, lprec = score_at(va, preds, t)
        tag = "  <-- rate-matched" if t == th_rm else ""
        print(f"{t:>10.1f} {n_net:>8} {prec:>7.3f} {rec:>7.3f} {lprec:>8.3f}{tag}")
    print("\n(prec/recall = net picks the SAME action the search overrode to;"
          "\n loose_p = net overrode where the search also overrode, any action)")

    torch.save({"state_dict": model.state_dict(), "cards": cards, "atks": atks,
                "clip": clip, "mu": mu, "sd": sd, "n_state": NUM_FEATS,
                "margin": MARGIN, "base_scale": base_scale,
                "threshold": th_rm}, args.out)
    print(f"\nsaved {args.out} (deploy threshold {th_rm:.1f})")


if __name__ == "__main__":
    main()
