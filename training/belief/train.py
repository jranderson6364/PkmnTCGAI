"""Stage 3 Phase A: fit + calibrate the archetype classifier, produce the
target figure (accuracy by turn + posterior entropy) and confusion matrices
at turns 1/2/3/5, export weights as a plain dict for pure-python inference
(no sklearn at ladder inference time -- timeout = instant loss).

Usage:
  python train.py --data belief_data.pkl.gz --out belief_weights.json
"""
import argparse
import gzip
import json
import os
import pickle

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit


def row_to_dict(row):
    d = {f"card_{cid}": 1.0 for cid in row["card_ids"]}
    for et, cnt in row["energy_types"].items():
        d[f"energy_{et}"] = float(cnt)
    d["turn"] = float(row["turn"])
    d["opp_bench_n"] = float(row["opp_bench_n"])
    d["opp_discard_n"] = float(row["opp_discard_n"])
    d["opp_hand_n"] = float(row["opp_hand_n"])
    d["opp_prizes_taken"] = float(row["opp_prizes_taken"])
    return d


def fit_key_card_baseline(train_rows, labels, max_turn=2):
    """docs/belief-model.md's must-beat ablation floor: a hand-written-style
    lookup, but the "key card" per archetype is DERIVED from the training
    split (highest-precision single card id predicting that label among
    turn<=max_turn observations) rather than hand-guessed — more defensible
    for the report than hardcoding remembered decklist card ids."""
    card_label_counts = {}  # cid -> {label: count}
    for r in train_rows:
        if r["turn"] > max_turn:
            continue
        for cid in r["card_ids"]:
            card_label_counts.setdefault(cid, {}).setdefault(r["label"], 0)
            card_label_counts[cid][r["label"]] += 1

    key_card = {}
    used_cids = set()
    # greedily assign each label its highest-precision available card id
    candidates = []
    for cid, counts in card_label_counts.items():
        total = sum(counts.values())
        for label, cnt in counts.items():
            candidates.append((cnt / total, total, cid, label))
    candidates.sort(reverse=True)
    for precision, total, cid, label in candidates:
        if label in key_card or cid in used_cids:
            continue
        if total < 5:
            continue
        key_card[label] = cid
        used_cids.add(cid)
        if len(key_card) == len(labels):
            break
    return key_card


def predict_key_card_baseline(rows, key_card, fallback_label):
    cid_to_label = {cid: label for label, cid in key_card.items()}
    preds = []
    for r in rows:
        pred = fallback_label
        for cid in r["card_ids"]:
            if cid in cid_to_label:
                pred = cid_to_label[cid]
                break
        preds.append(pred)
    return np.array(preds)


def load_rows(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "belief_data.pkl.gz"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "belief_weights.json"))
    ap.add_argument("--figure", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "belief_accuracy_by_turn.png"))
    args = ap.parse_args()

    rows = load_rows(args.data)
    print(f"loaded {len(rows)} rows")
    labels = sorted(set(r["label"] for r in rows))
    print("labels:", labels)

    # group by (label, game) so a held-out split doesn't leak turns from the
    # same game across train/test -- approximate game grouping via a running
    # counter that increments each time turn resets to 0 within a label.
    groups = []
    gid = -1
    last_label = None
    last_turn = None
    for r in rows:
        if r["label"] != last_label or (last_turn is not None and r["turn"] < last_turn):
            gid += 1
        groups.append(gid)
        last_label = r["label"]
        last_turn = r["turn"]
    groups = np.array(groups)

    X_dicts = [row_to_dict(r) for r in rows]
    y = np.array([r["label"] for r in rows])
    turns = np.array([r["turn"] for r in rows])

    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dicts)
    print(f"feature count: {len(vec.get_feature_names_out())}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X[train_idx], y[train_idx])

    preds = clf.predict(X[test_idx])
    proba = clf.predict_proba(X[test_idx])
    y_test = y[test_idx]
    turns_test = turns[test_idx]

    overall_acc = (preds == y_test).mean()
    print(f"overall held-out accuracy: {overall_acc:.3f}")

    # must-beat baseline (docs/belief-model.md Phase A gate: beat this at
    # turns 1-2, where partial evidence is all a key-card lookup has to go on)
    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]
    fallback_label = max(labels, key=lambda l: sum(1 for r in train_rows if r["label"] == l))
    key_card = fit_key_card_baseline(train_rows, labels)
    print("key-card baseline lookup:", key_card)
    baseline_preds = predict_key_card_baseline(test_rows, key_card, fallback_label)
    baseline_by_turn_acc = {}
    for t in sorted(set(turns_test.tolist())):
        mask = turns_test == t
        if mask.sum() < 5:
            continue
        baseline_by_turn_acc[t] = float((baseline_preds[mask] == y_test[mask]).mean())
    print("baseline accuracy by turn:", baseline_by_turn_acc)

    gate_result = {}
    for target_t in (1, 2):
        if not baseline_by_turn_acc:
            break
        nearest = min(baseline_by_turn_acc, key=lambda x: abs(x - target_t))
        base_acc = baseline_by_turn_acc[nearest]
        mask = turns_test == nearest
        clf_acc = float((preds[mask] == y_test[mask]).mean()) if mask.sum() >= 5 else None
        gate_result[target_t] = {"turn_used": nearest, "baseline_acc": base_acc, "classifier_acc": clf_acc}
        print(f"gate @ turn~{target_t} (actual turn {nearest}): "
              f"baseline={base_acc:.3f} classifier={clf_acc}")

    # accuracy by turn + posterior entropy
    turn_values = sorted(set(turns_test.tolist()))
    by_turn_acc = {}
    by_turn_entropy = {}
    class_order = clf.classes_
    for t in turn_values:
        mask = turns_test == t
        if mask.sum() < 5:
            continue
        by_turn_acc[t] = float((preds[mask] == y_test[mask]).mean())
        p = np.clip(proba[mask], 1e-9, 1.0)
        ent = -(p * np.log(p)).sum(axis=1).mean()
        by_turn_entropy[t] = float(ent)

    print("accuracy by turn:", by_turn_acc)

    # per-class accuracy by turn (for the figure's per-class curves)
    per_class_by_turn = {c: {} for c in class_order}
    for t in turn_values:
        mask = turns_test == t
        if mask.sum() < 5:
            continue
        for c in class_order:
            cmask = mask & (y_test == c)
            if cmask.sum() == 0:
                continue
            per_class_by_turn[c][t] = float((preds[cmask] == y_test[cmask]).mean())

    # confusion matrices at turns 1/2/3/5 (nearest available turn value if exact missing)
    confusions = {}
    for target_t in [1, 2, 3, 5]:
        if not turn_values:
            continue
        nearest = min(turn_values, key=lambda t: abs(t - target_t))
        mask = turns_test == nearest
        if mask.sum() < 5:
            continue
        cm = confusion_matrix(y_test[mask], preds[mask], labels=class_order)
        confusions[target_t] = {"turn_used": nearest, "matrix": cm.tolist(), "labels": list(class_order)}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        ts = sorted(by_turn_acc.keys())
        ax1.plot(ts, [by_turn_acc[t] for t in ts], marker="o", color="#4C78A8", label="overall")
        for c in class_order:
            xs = sorted(per_class_by_turn[c].keys())
            if not xs:
                continue
            ax1.plot(xs, [per_class_by_turn[c][t] for t in xs], marker=".", alpha=0.6, label=c)
        bts = sorted(baseline_by_turn_acc.keys())
        if bts:
            ax1.plot(bts, [baseline_by_turn_acc[t] for t in bts], marker="x", color="black",
                     linestyle=":", label="key-card baseline")
        ax1.axhline(1.0 / len(labels), color="gray", linestyle="--", linewidth=1, label="chance")
        ax1.set_xlabel("turn")
        ax1.set_ylabel("accuracy")
        ax1.set_title("Archetype ID accuracy by turn")
        ax1.legend(fontsize=7)
        ax1.set_ylim(0, 1.02)

        ax2.plot(ts, [by_turn_entropy[t] for t in ts], marker="o", color="#E45756")
        ax2.set_xlabel("turn")
        ax2.set_ylabel("posterior entropy (nats)")
        ax2.set_title("Posterior entropy by turn")

        fig.tight_layout()
        fig.savefig(args.figure, dpi=150)
        print(f"figure saved to {args.figure}")
    except ImportError:
        print("matplotlib not available, skipping figure")

    # export weights as a plain dict for pure-python dot-product inference
    # (Phase C wiring into main.py -- not wired yet, just exported here)
    feature_names = vec.get_feature_names_out().tolist()
    weights = {
        "classes": list(clf.classes_),
        "feature_names": feature_names,
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
    }
    with open(args.out, "w") as f:
        json.dump(weights, f)
    print(f"weights exported to {args.out}")

    summary = {
        "overall_held_out_accuracy": overall_acc,
        "accuracy_by_turn": by_turn_acc,
        "baseline_accuracy_by_turn": baseline_by_turn_acc,
        "key_card_baseline": key_card,
        "gate_vs_baseline_turns_1_2": gate_result,
        "entropy_by_turn": by_turn_entropy,
        "confusions": confusions,
        "n_rows": len(rows),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "labels": labels,
    }
    summary_path = args.out.replace(".json", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary written to {summary_path}")


if __name__ == "__main__":
    main()
