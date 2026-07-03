"""Deck/method bake-off: seat-alternating round-robin over (label, agent, deck) entries.

Unlike ab_test/gauntlet (which bind each agent to its own module DECK), this
exposes the harness's orthogonal (agent, deck) form so any agent can pilot any
deck — the tool behind the Stage 0c deck bake-off (tier 1: specialist agents on
their own decks; tier 2: one generic pilot on every deck) and the method
bake-off. Protocol + decision rules are pre-registered in docs/report-log.md
(2026-07-03) — per-game rows persist exactly the fields named there: seats,
winner, turns, prizes taken per side, first-attack turn per side, end reason,
seed (a run identifier for variance grouping, NOT an RNG seed — the engine's
shuffle lives in the native cg.dll and exposes no seed API).

Usage:
  python training/bakeoff.py --manifest training/manifests/tier1.csv \
      --games 200 --tag tier1 [--seed run-x] [--workers N]
  python training/bakeoff.py --sanity          # main.py mirror, expect ~50%
  python training/bakeoff.py --table --tag tier1   # reprint stats for a tag

Manifest CSV columns: label,agent_path,deck_source
  deck_source: '' → the agent module's own DECK; a .py path → that module's
  DECK; a .csv path → one card id per line (deck.csv format).
Statistical conventions (pre-registered): ties = 0.5 wins, Wilson 95% CIs,
crash games excluded from win rates and reported (>2% → fix and re-run).
"""
import argparse
import csv
import datetime
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import load_agent, play_game
from gauntlet import bradley_terry

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(REPO_ROOT, "training", "bakeoff_results.csv")
FIELDS = ["tag", "timestamp", "seed", "p0_label", "p1_label", "winner_label",
          "reward_p0", "reward_p1", "turns", "p0_prizes_taken", "p1_prizes_taken",
          "p0_first_attack_turn", "p1_first_attack_turn", "end_reason", "error"]

ATTACK_OTYPE = 13  # option type for ATTACK (see tools/analyze_replay.py OT_NAMES)


def load_deck(deck_source, agent_path):
    """Resolve a manifest deck_source to a 60-int list."""
    if not deck_source or deck_source == "agent":
        _, deck, _ = load_agent(agent_path)
        return deck
    path = os.path.join(REPO_ROOT, deck_source) if not os.path.isabs(deck_source) else deck_source
    if path.endswith(".csv"):
        with open(path, encoding="utf-8") as f:
            deck = [int(line.strip()) for line in f if line.strip()]
    else:
        _, deck, _ = load_agent(path)
    if len(deck) != 60:
        raise ValueError(f"deck from {deck_source} has {len(deck)} cards, want 60")
    return deck


def _first_attack_turns(steps):
    """Per seat: current.turn of the first decision whose chosen option is an
    ATTACK (real decision points per tools/analyze_replay.py::real_decisions)."""
    out = [None, None]
    for you in (0, 1):
        for i in range(1, len(steps)):
            prev = steps[i - 1][you]
            if prev.get("status") != "ACTIVE":
                continue
            obs = prev.get("observation") or {}
            sel = obs.get("select")
            if not sel:
                continue
            action = steps[i][you].get("action")
            opts = sel.get("option", [])
            if not action or not (0 <= action[0] < len(opts)):
                continue
            if opts[action[0]].get("type") == ATTACK_OTYPE:
                out[you] = (obs.get("current") or {}).get("turn")
                break
    return out


def _game_stats(result):
    """Extract the pre-registered per-game fields from a keep_steps result."""
    steps = result["steps"]
    turns = 0
    last_players = None
    for s in steps:
        for idx in (0, 1):
            cur = (s[idx].get("observation") or {}).get("current") or {}
            if cur.get("turn") is not None:
                turns = max(turns, cur["turn"])
            pl = cur.get("players") or []
            if len(pl) == 2:
                last_players = pl
    prizes = [None, None]
    if last_players:
        prizes = [6 - len(last_players[k].get("prize") or []) for k in (0, 1)]
    fat = _first_attack_turns(steps)

    r0, r1 = result["rewards"]
    if r0 == 1 or r1 == -1:
        winner = 0
    elif r0 == -1 or r1 == 1:
        winner = 1
    else:
        winner = None
    # end-reason triage from the loser's side (mirrors analyze_replay.classify_terminal)
    reason = "TIE"
    if winner is not None and last_players:
        loser = 1 - winner
        me = last_players[loser]
        if not me.get("active") and not (me.get("bench") or []):
            reason = "NO_POKEMON_IN_PLAY"
        elif me.get("deckCount", 99) == 0:
            reason = "DECK_OUT"
        elif prizes[winner] is not None and prizes[winner] >= 6:
            reason = "PRIZED_OUT"
        else:
            reason = "OTHER"
    return {
        "winner": winner, "turns": turns,
        "p0_prizes_taken": prizes[0], "p1_prizes_taken": prizes[1],
        "p0_first_attack_turn": fat[0], "p1_first_attack_turn": fat[1],
        "end_reason": reason,
    }


def _worker(job):
    """Runs one game of (agent0 piloting deck0) vs (agent1 piloting deck1).
    Loads fresh per process; extracts stats here so steps never cross the pipe."""
    a0_path, deck0, a1_path, deck1, max_steps = job
    a0, _, _ = load_agent(a0_path)
    a1, _, _ = load_agent(a1_path)
    try:
        res = play_game(a0, deck0, a1, deck1, keep_steps=True, max_steps=max_steps)
        stats = _game_stats(res)
        stats["rewards"] = res["rewards"]
        return stats
    except Exception as e:
        return {"error": repr(e)}


def run_pair(e0, e1, n_games, tag, seed, workers, max_steps=3000):
    """n_games of e0-vs-e1, half each seat orientation. e = (label, agent_path, deck).
    max_steps=3000 is ~3x the longest tier-1 game (43 turns); capped games tie."""
    import multiprocessing as mp
    half = n_games // 2
    jobs = ([(e0[1], e0[2], e1[1], e1[2], max_steps)] * half +
            [(e1[1], e1[2], e0[1], e0[2], max_steps)] * (n_games - half))
    seat_labels = ([(e0[0], e1[0])] * half + [(e1[0], e0[0])] * (n_games - half))
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    results = []
    if workers <= 1:
        for j in jobs:
            results.append(_worker(j))
    else:
        with mp.Pool(workers) as pool:
            # imap (ordered) so results stay aligned with seat_labels
            for i, r in enumerate(pool.imap(_worker, jobs)):
                results.append(r)
                if (i + 1) % 25 == 0:
                    print(f"  {i+1}/{n_games}", file=sys.stderr)
    rows = []
    now = datetime.datetime.now().isoformat(timespec="seconds")
    for (p0l, p1l), r in zip(seat_labels, results):
        row = {"tag": tag, "timestamp": now, "seed": seed,
               "p0_label": p0l, "p1_label": p1l}
        if "error" in r:
            row.update({"winner_label": "", "end_reason": "ERROR", "error": r["error"]})
        else:
            w = r["winner"]
            row.update({
                "winner_label": (p0l if w == 0 else p1l) if w is not None else "",
                "reward_p0": r["rewards"][0], "reward_p1": r["rewards"][1],
                "turns": r["turns"],
                "p0_prizes_taken": r["p0_prizes_taken"],
                "p1_prizes_taken": r["p1_prizes_taken"],
                "p0_first_attack_turn": r["p0_first_attack_turn"],
                "p1_first_attack_turn": r["p1_first_attack_turn"],
                "end_reason": r["end_reason"], "error": "",
            })
        rows.append(row)
    return rows


def append_rows(rows):
    exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows to {RESULTS_CSV}")


def wilson_ci(wins, n, z=1.96):
    """Wilson 95% interval for a win proportion (ties folded in as 0.5 by caller)."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def report(tag):
    """Matchup matrix + BT table + pre-registered in-play metrics for one tag."""
    if not os.path.exists(RESULTS_CSV):
        print("no results yet")
        return
    rows = [r for r in csv.DictReader(open(RESULTS_CSV, newline="", encoding="utf-8"))
            if r["tag"] == tag]
    if not rows:
        print(f"no rows for tag {tag!r}")
        return
    n_err = sum(1 for r in rows if r["end_reason"] == "ERROR")
    rows_ok = [r for r in rows if r["end_reason"] != "ERROR"]
    err_pct = 100.0 * n_err / len(rows) if rows else 0.0
    print(f"tag={tag}: {len(rows)} games, {n_err} errors ({err_pct:.1f}%"
          f"{' — EXCEEDS 2% convention, fix and re-run' if err_pct > 2 else ''})")

    # pairwise (unordered) records, ties = 0.5
    scores = defaultdict(float)
    games = defaultdict(float)
    rec = defaultdict(lambda: [0, 0, 0])  # (a,b) -> [a_wins, b_wins, ties]
    for r in rows_ok:
        a, b = sorted([r["p0_label"], r["p1_label"]])
        w = r["winner_label"]
        if w == a:
            rec[(a, b)][0] += 1
        elif w == b:
            rec[(a, b)][1] += 1
        else:
            rec[(a, b)][2] += 1
    for (a, b), (aw, bw, t) in sorted(rec.items()):
        n = aw + bw + t
        wr = (aw + 0.5 * t) / n
        lo, hi = wilson_ci(aw + 0.5 * t, n)
        print(f"  {a} vs {b}: {aw}W {bw}L {t}T  wr({a})={wr:.3f} [{lo:.3f},{hi:.3f}]")
        scores[(a, b)] += aw + 0.5 * t
        scores[(b, a)] += bw + 0.5 * t
        games[(a, b)] += n
        games[(b, a)] += n

    p = bradley_terry(scores, games)
    if p:
        ref = min(p.values())
        print(f"\n{'entry':<24}{'BT-Elo':>8}")
        print("-" * 32)
        for name, s in sorted(p.items(), key=lambda kv: -kv[1]):
            print(f"{name:<24}{400.0 * math.log10(s / ref):>8.0f}")

    # in-play metrics per entry (figure #7: prize-trade efficiency, setup speed)
    taken = defaultdict(int)
    conceded = defaultdict(int)
    fat_sum = defaultdict(int)
    fat_n = defaultdict(int)
    for r in rows_ok:
        for side, opp in (("p0", "p1"), ("p1", "p0")):
            lab = r[f"{side}_label"]
            if r[f"{side}_prizes_taken"] not in ("", None):
                taken[lab] += int(r[f"{side}_prizes_taken"])
                conceded[lab] += int(r[f"{opp}_prizes_taken"])
            v = r[f"{side}_first_attack_turn"]
            if v not in ("", None):
                fat_sum[lab] += int(v)
                fat_n[lab] += 1
    print(f"\n{'entry':<24}{'prize-trade':>12}{'1st-atk turn':>14}")
    print("-" * 50)
    for lab in sorted(taken):
        pt = taken[lab] / conceded[lab] if conceded[lab] else float("inf")
        fa = fat_sum[lab] / fat_n[lab] if fat_n[lab] else None
        print(f"{lab:<24}{pt:>12.2f}{fa if fa is None else round(fa, 1):>14}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="CSV: label,agent_path,deck_source")
    ap.add_argument("--games", type=int, default=200, help="games per pair")
    ap.add_argument("--tag", help="run tag (groups rows; REQUIRED with --manifest)")
    ap.add_argument("--seed", default=None, help="run identifier (variance grouping)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--sanity", action="store_true",
                    help="main.py mirror match (40 games): wr CI must cover 0.5")
    ap.add_argument("--table", action="store_true", help="reprint stats for --tag")
    args = ap.parse_args()

    seed = args.seed or datetime.datetime.now().strftime("run-%Y%m%d-%H%M%S")

    if args.sanity:
        main_py = os.path.join(REPO_ROOT, "main.py")
        deck = load_deck("", main_py)
        e = ("mirror-A", main_py, deck)
        e2 = ("mirror-B", main_py, deck)
        rows = run_pair(e, e2, 40, "sanity", seed, args.workers)
        append_rows(rows)
        report("sanity")
        return

    if args.table:
        if not args.tag:
            ap.error("--table needs --tag")
        report(args.tag)
        return

    if not args.manifest or not args.tag:
        ap.error("--manifest and --tag are required (or use --sanity/--table)")
    entries = []
    with open(args.manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row["label"].strip()
            agent_path = os.path.join(REPO_ROOT, row["agent_path"].strip())
            deck = load_deck(row.get("deck_source", "").strip(), agent_path)
            entries.append((label, agent_path, deck))
    print(f"{len(entries)} entries -> {len(entries) * (len(entries) - 1) // 2} pairs, "
          f"{args.games} games each, tag={args.tag}, seed={seed}")
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            print(f"[{entries[i][0]} vs {entries[j][0]}] {args.games} games")
            rows = run_pair(entries[i], entries[j], args.games, args.tag, seed,
                            args.workers)
            append_rows(rows)
    report(args.tag)


if __name__ == "__main__":
    main()
