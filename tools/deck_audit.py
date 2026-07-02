#!/usr/bin/env python3
"""Per-card utilization audit over local self-play games.

Answers "which cards earn their slot?" before deck simplification. For each
card ID in our deck, over N mirror games (both seats harvested):

  drawn_games      games where >=1 copy ever appeared in our hand
  plays_per_game   chosen PLAY/ATTACH/EVOLVE options resolving to this card,
                   averaged over games where it was drawn
  end_hand_rate    P(>=1 copy still in hand at game end | drawn) — NOT
                   automatically bad in this deck (hand size IS damage), but
                   high end_hand_rate + near-zero plays = the card is pure
                   Powerful Hand fuel; any card is equally good at that job,
                   so its unique effect is contributing nothing
  wr_played/drawn  win rate of games where the card was played vs merely drawn
                   (correlational, small-N noisy — treat as a hint, not proof)

Usage:
  python tools/deck_audit.py --games 500 [--workers N] [--out training/deck_audit.csv]
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "training"))
from harness import run_matches  # noqa: E402
from bc_collect import extract_decisions  # noqa: E402

MAIN = os.path.join(REPO_ROOT, "main.py")
CARD_CSV = os.path.join(REPO_ROOT, "docs", "EN_Card_Data.csv")
# option types whose `index` points into our hand (see docs/engine-api.md)
PLAY, ATTACH, EVOLVE = 7, 8, 9


def card_names():
    names = {}
    try:
        with open(CARD_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    cid = int(row["Card ID"])
                except (ValueError, KeyError):
                    continue
                names.setdefault(cid, row.get("Card Name", "?"))
    except FileNotFoundError:
        pass
    return names


def hand_ids(obs):
    cur = obs.get("current") or {}
    yi = cur.get("yourIndex", 0)
    pl = cur.get("players") or []
    me = pl[yi] if len(pl) > yi else {}
    return [(c or {}).get("id") for c in (me.get("hand") or [])]


def audit_seat(decisions, outcome, stats):
    """Accumulate one seat's game into per-card stats."""
    drawn, played = set(), defaultdict(int)
    last_hand = []
    for d in decisions:
        obs, act = d["obs"], d["action"]
        hand = hand_ids(obs)
        if hand:
            last_hand = hand
        drawn.update(h for h in hand if h is not None)
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        if not act or not (0 <= act[0] < len(opts)):
            continue
        o = opts[act[0]]
        if o.get("type") in (PLAY, ATTACH, EVOLVE):
            idx = o.get("index")
            if idx is not None and 0 <= idx < len(hand) and hand[idx] is not None:
                played[hand[idx]] += 1
    win = 1 if outcome == 1 else 0
    for cid in drawn:
        s = stats[cid]
        s["drawn_games"] += 1
        s["plays"] += played.get(cid, 0)
        if played.get(cid, 0) == 0:
            s["rot_games"] += 1
            s["wr_drawn_only_n"] += 1
            s["wr_drawn_only_w"] += win
        else:
            s["wr_played_n"] += 1
            s["wr_played_w"] += win
        if cid in last_hand:
            s["end_hand_games"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "training", "deck_audit.csv"))
    args = ap.parse_args()

    names = card_names()
    stats = defaultdict(lambda: defaultdict(int))
    done = 0
    CHUNK = 50  # step traces are heavy — process and discard per chunk
    remaining = args.games
    while remaining > 0:
        n = min(CHUNK, remaining)
        remaining -= n
        results = run_matches(MAIN, MAIN, n, workers=args.workers,
                              keep_steps=True, progress=False)
        for r in results:
            if "error" in r or "steps" not in r:
                continue
            done += 1
            for seat in (0, 1):
                audit_seat(extract_decisions(r["steps"], seat=seat),
                           r["rewards"][seat], stats)
        print(f"  {done}/{args.games} games", file=sys.stderr)

    rows = []
    for cid, s in stats.items():
        dg = s["drawn_games"] or 1
        rows.append({
            "card_id": cid,
            "name": names.get(cid, "?"),
            "drawn_games": s["drawn_games"],
            "plays_per_game": round(s["plays"] / dg, 2),
            "rot_rate": round(s["rot_games"] / dg, 3),
            "end_hand_rate": round(s["end_hand_games"] / dg, 3),
            "wr_played": round(s["wr_played_w"] / s["wr_played_n"], 3) if s["wr_played_n"] else "",
            "wr_drawn_only": round(s["wr_drawn_only_w"] / s["wr_drawn_only_n"], 3) if s["wr_drawn_only_n"] else "",
        })
    rows.sort(key=lambda r: r["plays_per_game"])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'card':<28}{'drawn':>6}{'ppg':>6}{'rot':>7}{'endH':>7}{'wrP':>7}{'wrD':>7}")
    print("-" * 68)
    for r in rows:
        print(f"{r['name'][:26]+'('+str(r['card_id'])+')':<28}{r['drawn_games']:>6}"
              f"{r['plays_per_game']:>6}{r['rot_rate']:>7}{r['end_hand_rate']:>7}"
              f"{str(r['wr_played']):>7}{str(r['wr_drawn_only']):>7}")
    print(f"\n(ppg = plays per game drawn; rot = drawn-never-played rate; "
          f"wrP/wrD = win rate when played / when drawn-only)\nwrote {args.out}")


if __name__ == "__main__":
    main()
