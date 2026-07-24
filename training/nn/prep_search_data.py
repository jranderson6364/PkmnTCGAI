"""Turn the collected search corpus into training arrays (S3 prep).

Reads <corpus>.jsonl (one search-decision record per line) + <corpus>.outcomes.json
and produces, per record:
  * X_state  : numeric_feats(state) -- encode.py's rich feature vector (belief +
               tempo + census), richer than Phi v4's 11 features (so a value net
               trained on it can, in principle, beat Phi v4 -- the point of the
               value branch).
  * y_value  : the game outcome from the collecting seat (+1/-1/0). The AlphaZero
               value target -- P(win) from the state, to guide the search leaf.
  * v_search : the search's own value of the chosen line (bootstrap/aux target).
  * advantage: per-candidate cand_val[i] - cand_val[heur_top] -- the override
               signal for the action/advantage branch (S1 Q2).

Saves an .npz with the value-branch arrays (X_state, y_value, v_search) and pickles
the per-record advantage lists for the advantage branch. Which branch is trained
is decided by the S1 sweep; this prep serves both.

Run:  python training/nn/prep_search_data.py --corpus training/nn/search_corpus
"""
import argparse
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
from encode import numeric_feats, NUM_FEATS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(_HERE, "search_corpus"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or (args.corpus + ".prep")

    jsonl = args.corpus + ".jsonl"
    outcomes_path = args.corpus + ".outcomes.json"
    outcomes = json.load(open(outcomes_path)) if os.path.exists(outcomes_path) else {}
    if not outcomes:
        print(f"[warn] no outcomes at {outcomes_path} -- value targets will be 0 "
              f"(feature-extraction validation only)")

    X, yv, vs = [], [], []
    adv_records = []
    n_bad = 0
    for line in open(jsonl):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            feats = numeric_feats({"current": rec["current"]})
            if feats is None or len(feats) != NUM_FEATS:
                n_bad += 1
                continue
            gid = rec.get("game", "0")
            outcome = float(outcomes.get(gid, 0))
            # search value of the chosen line (the search's own estimate)
            cv = rec["cand_val"]
            sb = str(rec["search_best"])
            v_search = float(cv.get(sb, cv.get(str(rec["heur_top"]), 0.0)))
            X.append(feats)
            yv.append(outcome)
            vs.append(v_search)
            # advantage over the heuristic's pick, per candidate (robust lookups)
            ht = str(rec["heur_top"])
            base_v = float(cv.get(ht, 0.0))
            adv_records.append({
                "feats": feats.tolist(),
                "cand": rec["cand"],
                "advantage": {int(i): float(cv.get(str(i), base_v)) - base_v
                              for i in rec["cand"]},
                "heur_top": rec["heur_top"], "search_best": rec["search_best"],
                "outcome": outcome,
            })
        except Exception:
            n_bad += 1
            continue

    X = np.asarray(X, dtype=np.float32)
    yv = np.asarray(yv, dtype=np.float32)
    vs = np.asarray(vs, dtype=np.float32)
    np.savez_compressed(out + ".npz", X_state=X, y_value=yv, v_search=vs)
    with open(out + ".adv.json", "w") as f:
        json.dump(adv_records, f)

    print(f"records: {len(X)} (bad/skipped {n_bad}) | NUM_FEATS={NUM_FEATS}")
    if len(X):
        print(f"X_state shape {X.shape} | y_value mean {yv.mean():+.3f} "
              f"(balance: {int((yv>0).sum())} win / {int((yv<0).sum())} loss / "
              f"{int((yv==0).sum())} tie-or-nooutcome)")
        print(f"v_search range [{vs.min():.1f}, {vs.max():.1f}]")
    print(f"wrote {out}.npz + {out}.adv.json")


if __name__ == "__main__":
    main()
