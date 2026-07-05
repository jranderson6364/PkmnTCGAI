"""Convert real ladder replay JSONs into bc_collect.py-format training samples
({"obs": obs_dict, "action": act, "outcome": +-1/0} per decision) for value-head
retraining on the TRUE ladder distribution (real decks, real losses).

Only seats piloted by our own submissions (TeamNames entry == OUR_TEAM) are
extracted — the evaluator is only ever queried from our perspective, and only
our seats' observations match the deck/encoding the net was built around.
Mirror ladder games (both seats ours) yield both seats.

Usage:
  python tools/replay_to_bc.py --out training/vd_ladder.pkl.gz
"""
import argparse
import glob
import gzip
import json
import os
import pickle

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUR_TEAM = "Jason Anderson"


def extract_decisions(steps, seat):
    # same semantics as training/bc_collect.py::extract_decisions
    out = []
    for i in range(1, len(steps)):
        prev = steps[i - 1][seat].get("observation", {})
        act = steps[i][seat].get("action")
        sel = prev.get("select")
        if sel is None or act is None:
            continue
        if not sel.get("option"):
            continue
        out.append({"obs": prev, "action": act})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default=os.path.join(REPO_ROOT, "replays", "**", "*.json"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "training", "vd_ladder.pkl.gz"))
    args = ap.parse_args()

    paths = sorted(set(glob.glob(args.replays, recursive=True)))
    samples = []
    n_games = n_seats = n_wins = n_parse_errors = 0
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            team_names = (d.get("info") or {}).get("TeamNames") or []
            rewards = d.get("rewards") or []
            steps = d.get("steps") or []
            if len(team_names) != 2 or len(rewards) != 2 or not steps:
                continue
        except Exception:
            n_parse_errors += 1
            continue
        n_games += 1
        for seat in (0, 1):
            if team_names[seat] != OUR_TEAM:
                continue
            outcome = rewards[seat]
            if outcome is None:
                continue
            n_seats += 1
            if outcome == 1:
                n_wins += 1
            for dec in extract_decisions(steps, seat):
                dec["outcome"] = outcome
                samples.append(dec)

    opener = gzip.open if args.out.endswith(".gz") else open
    with opener(args.out, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"replays={len(paths)} games_used={n_games} our_seats={n_seats} "
          f"seat_winrate={n_wins/max(n_seats,1):.3f} samples={len(samples)} "
          f"parse_errors={n_parse_errors}")
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
