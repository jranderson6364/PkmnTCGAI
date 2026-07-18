"""Win rate by archetype across downloaded ladder replays.

Extends tools/meta_survey.py's classification to BOTH seats plus the game
outcome, answering "does any archetype actually dominate the observed field?"
(deck-switch question, 2026-07-18). Games involving our own team are tallied
separately (vs-us) so field rows aren't skewed by our agent's matchups.

Usage:
  python tools/archetype_winrates.py --all
  python tools/archetype_winrates.py --all --matrix   # archetype-vs-archetype
"""
import argparse
import glob
import json
import os
import sys

sys.stdout.reconfigure(errors="replace")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

from meta_survey import OUR_TEAM, classify, load_names  # noqa: E402


def seat_revealed_ids(d, seat):
    """All card ids `seat` revealed on board or discard, from its own records."""
    ids = set()
    for step in d.get("steps", []):
        if len(step) <= seat:
            continue
        obs = step[seat].get("observation") or {}
        cur = obs.get("current") or {}
        pl = cur.get("players") or []
        yidx = cur.get("yourIndex")
        if len(pl) != 2 or yidx is None:
            continue
        me = pl[yidx]
        for zone in ("active", "bench", "discard"):
            for card in (me.get(zone) or []):
                if isinstance(card, dict) and card.get("id") is not None:
                    ids.add(card["id"])
    return ids


def outcome(d):
    """Returns (r0, r1) from the final step, kaggle reward conventions."""
    steps = d.get("steps") or []
    if not steps:
        return None, None
    last = steps[-1]
    if len(last) != 2:
        return None, None
    return last[0].get("reward"), last[1].get("reward")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replays", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--matrix", action="store_true")
    args = ap.parse_args()

    paths = []
    for p in args.replays:
        if os.path.isdir(p):
            paths += glob.glob(os.path.join(p, "**", "*.json"), recursive=True)
        else:
            paths.append(p)
    if args.all:
        paths += glob.glob(os.path.join(REPO_ROOT, "replays", "**", "*.json"),
                           recursive=True)
    paths = sorted(set(paths))
    if not paths:
        ap.error("no replays given")

    names = load_names()
    field = {}   # arch -> [w, l, t]  (games not involving OUR_TEAM)
    vs_us = {}   # arch -> [w, l, t]  (the archetype's record AGAINST us)
    matrix = {}  # (arch_a, arch_b) -> [w_a, n]
    parsed = skipped = 0

    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            r0, r1 = outcome(d)
            if r0 is None and r1 is None:
                skipped += 1
                continue
            team_names = (d.get("info") or {}).get("TeamNames", [])
            arch = [classify(seat_revealed_ids(d, s), names) for s in (0, 1)]
            # w/l/t from seat 0's perspective, crash-asymmetry handled like
            # training/harness.py summarize()
            if r0 == 1 or r1 == -1:
                res0 = "w"
            elif r0 == -1 or r1 == 1:
                res0 = "l"
            else:
                res0 = "t"
            parsed += 1
            ours = [t == OUR_TEAM for t in team_names] if len(team_names) == 2 else [False, False]
            for s in (0, 1):
                res = res0 if s == 0 else {"w": "l", "l": "w", "t": "t"}[res0]
                tgt = None
                if ours[1 - s] and not ours[s]:
                    tgt = vs_us
                elif not ours[0] and not ours[1]:
                    tgt = field
                if tgt is not None:
                    rec = tgt.setdefault(arch[s], [0, 0, 0])
                    rec["wlt".index(res)] += 1
            if not ours[0] and not ours[1]:
                a, b = arch
                for x, y, r in ((a, b, res0), (b, a, {"w": "l", "l": "w", "t": "t"}[res0])):
                    rec = matrix.setdefault((x, y), [0.0, 0])
                    rec[1] += 1
                    rec[0] += 1.0 if r == "w" else (0.5 if r == "t" else 0.0)
        except Exception:
            skipped += 1

    print(f"parsed {parsed} replays ({skipped} skipped)\n")
    for label, tally in (("FIELD (games not involving us)", field),
                         ("VS US (archetype's record against our agent)", vs_us)):
        rows = []
        for arch, (w, l, t) in tally.items():
            n = w + l + t
            wr = (w + 0.5 * t) / n if n else 0.0
            rows.append((wr, n, w, l, t, arch))
        rows.sort(key=lambda r: -r[0])
        print(label)
        print(f"  {'archetype':<18}{'wr':>7}{'n':>6}{'w':>6}{'l':>6}{'t':>5}")
        for wr, n, w, l, t, arch in rows:
            print(f"  {arch:<18}{100*wr:6.1f}%{n:>6}{w:>6}{l:>6}{t:>5}")
        print()

    if args.matrix:
        print("MATRIX (row's wr vs column, field games only, n in parens)")
        archs = sorted({a for a, _ in matrix} | {b for _, b in matrix})
        for a in archs:
            cells = []
            for b in archs:
                wsum, n = matrix.get((a, b), (0.0, 0))
                cells.append(f"{100*wsum/n:5.1f}({n})" if n else "    -")
            print(f"  {a:<18}" + " ".join(cells))
        print("  cols: " + ", ".join(archs))


if __name__ == "__main__":
    main()
