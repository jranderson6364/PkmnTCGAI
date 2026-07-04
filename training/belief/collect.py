"""Stage 3 Phase A data collection (docs/belief-model.md): labeled per-turn
feature rows for the archetype classifier. Always plays main.py (us) vs a
labeled opponent bot and extracts OUR OWN observation of THEIR public board
state, since that's exactly what's available to main.py at ladder inference
time — no log-window parsing needed, board state (active/bench/discard/
preEvolution/tools) is a complete, monotonically-growing public snapshot.

Usage:
  python collect.py --games 2000 --out belief_data.pkl.gz
"""
import argparse
import gzip
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from harness import run_matches

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAIN_PY = os.path.join(REPO_ROOT, "main.py")

LABELS = {
    "lucario": os.path.join(REPO_ROOT, "opponents", "lucario_agent.py"),
    "dragapult": os.path.join(REPO_ROOT, "opponents", "dragapult_agent.py"),
    "abomasnow": os.path.join(REPO_ROOT, "opponents", "abomasnow_agent.py"),
    "starmie": os.path.join(REPO_ROOT, "opponents", "starmie_agent.py"),
    "alakazam": MAIN_PY,  # mirror match: main.py vs main.py, opponent = alakazam archetype
}


def _revealed_ids(pokemon):
    """All card ids permanently revealed by one in-play Pokemon: itself, its
    pre-evolution chain (revealed the moment it evolves, even if the lower
    stage never separately hits discard), and any attached tool."""
    if not pokemon:
        return []
    ids = [pokemon.get("id")]
    for pre in pokemon.get("preEvolution") or []:
        ids.append((pre or {}).get("id"))
    for tool in pokemon.get("tools") or []:
        ids.append((tool or {}).get("id"))
    return [i for i in ids if i]


def extract_rows(steps, main_seat, label):
    """Walk one game's full step trace; emit one cumulative feature row per
    turn boundary (using the state as of the end of that turn).

    `observed_ids` is the only accumulator (monotonic — a card, once
    revealed, stays revealed). Everything else (bench/discard/hand counts,
    energy-type counts) is read fresh from the CURRENT board each snapshot,
    not summed across steps — the same board is re-observed at every
    decision within a turn, so accumulating would just count re-observations."""
    opp_idx = 1 - main_seat
    observed_ids = set()
    rows = []
    last_turn = None
    last_snapshot = None

    def snapshot(turn, opp, me):
        energy_type_counts = {}
        for pk in [*(opp.get("active") or []), *(opp.get("bench") or [])]:
            for et in (pk or {}).get("energies") or []:
                energy_type_counts[et] = energy_type_counts.get(et, 0) + 1
        return {
            "label": label,
            "turn": turn,
            "card_ids": sorted(observed_ids),
            "energy_types": energy_type_counts,
            "opp_bench_n": len(opp.get("bench") or []),
            "opp_discard_n": len(opp.get("discard") or []),
            "opp_hand_n": opp.get("handCount") or 0,
            "opp_prizes_taken": 6 - len(opp.get("prize") or []),
            "my_prizes_taken": 6 - len(me.get("prize") or []),
        }

    for step in steps:
        obs = (step[main_seat] or {}).get("observation") or {}
        cur = obs.get("current") or {}
        turn = cur.get("turn")
        players = cur.get("players") or []
        if turn is None or len(players) < 2:
            continue
        opp = players[opp_idx]
        me = players[main_seat]

        for pk in [*(opp.get("active") or []), *(opp.get("bench") or [])]:
            for cid in _revealed_ids(pk):
                observed_ids.add(cid)
        for card in opp.get("discard") or []:
            cid = (card or {}).get("id")
            if cid:
                observed_ids.add(cid)

        if last_turn is not None and turn != last_turn:
            rows.append(last_snapshot)
        last_turn = turn
        last_snapshot = snapshot(turn, opp, me)

    if last_snapshot is not None:
        rows.append(last_snapshot)
    return rows


def collect_label(label, bot_path, n_games, workers):
    half = n_games // 2
    rows = []
    # half games: main.py as seat0, bot as seat1
    if label != "alakazam":
        results = run_matches(MAIN_PY, bot_path, half, workers=workers, keep_steps=True, progress=False)
        for r in results:
            if "error" in r or "steps" not in r:
                continue
            rows.extend(extract_rows(r["steps"], main_seat=0, label=label))
        results = run_matches(bot_path, MAIN_PY, n_games - half, workers=workers, keep_steps=True, progress=False)
        for r in results:
            if "error" in r or "steps" not in r:
                continue
            rows.extend(extract_rows(r["steps"], main_seat=1, label=label))
    else:
        # alakazam mirror: main.py vs main.py, opponent (seat 1) IS the label
        results = run_matches(MAIN_PY, MAIN_PY, n_games, workers=workers, keep_steps=True, progress=False)
        for r in results:
            if "error" in r or "steps" not in r:
                continue
            rows.extend(extract_rows(r["steps"], main_seat=0, label=label))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000, help="games per label")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "belief_data.pkl.gz"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--labels", default=None, help="comma-separated subset of labels, default all")
    args = ap.parse_args()

    labels = args.labels.split(",") if args.labels else list(LABELS.keys())
    all_rows = []
    for label in labels:
        bot_path = LABELS[label]
        print(f"collecting {label}: {args.games} games vs {bot_path}", file=sys.stderr)
        rows = collect_label(label, bot_path, args.games, args.workers)
        print(f"  {label}: {len(rows)} rows from {args.games} games", file=sys.stderr)
        all_rows.extend(rows)

    opener = gzip.open if args.out.endswith(".gz") else open
    with opener(args.out, "wb") as f:
        pickle.dump(all_rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {args.out}: {len(all_rows)} total rows", file=sys.stderr)


if __name__ == "__main__":
    main()
