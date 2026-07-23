"""Package the 2-ply search agent as a SELF-CONTAINED Kaggle submission.

Ships a two-file agent (main.py wrapper + heuristic.py = frozen_main_v30) plus
deck.csv. The 2-ply belief search from twoply_agent.py is inlined into main.py
with PLACEHOLDER determinization (filler zones from our own decklist for both
sides) instead of the heavy BeliefDeterminizer dependency tree — this is exactly
alakazam_v9's fallback determinization path (which still scores 778.2), and it
keeps the package small and robust (two prior ships errored on environment gaps:
the __file__ NameError and the missing cg module).

Design guards baked in:
  * __file__-safe: main.py borrows heuristic's __file__ (Kaggle execs main.py
    from a raw string with no __file__).
  * cg-optional: globs for the Kaggle-provided cg-lib; if cg / search_begin is
    unavailable for ANY reason, agent() falls back to pure heuristic play
    (Design Principle #2 — never break agent()).
  * hard wall-clock budget + pure-heuristic fallback on any search failure.

Usage: python training/nn/package_twoply_submission.py [--out-dir DIR]
Then tar DIR's contents (main.py at root) into submission.tar.gz.
"""
import argparse
import os
import shutil

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAIN = r'''"""2-ply belief-determinized minimax OVERRIDE search agent (shipped entrypoint).

Plays heuristic.py (frozen v30) verbatim, except on MAIN decisions where a
bounded 2-ply search (placeholder determinization, greedy-completed turns,
half-prize override margin) finds a clearly better line. Falls back to pure
heuristic play if cg/search is unavailable or anything fails. See
docs/report-log.md 2026-07-23 for why this recipe (conservative override) and
the placebo-confirmed clean Kaggle isolation premise.
"""
import os as _os
import sys as _sys
import glob as _glob
import time as _time
import random as _random
import dataclasses as _dc

for _pat in ('/kaggle/input/**/cg-lib', '/kaggle/input/cg-lib'):
    _m = _glob.glob(_pat, recursive=True)
    if _m:
        _sys.path.insert(0, _m[0]); break

# heuristic.py is imported through the normal import system (real __file__),
# so main.py never needs its own __file__.
from heuristic import DECK, agent as _heur_agent, _choose as _heur_choose, \
    score_options_main as _score_main, _safe_return as _safe

try:
    from cg.api import to_observation_class, search_begin, search_step, search_end
    _SEARCH_OK = True
except Exception:
    _SEARCH_OK = False

MAIN_CTX = 0
_OPT_ATTACK, _OPT_END = 13, 14
BUDGET_S = 0.80
N_DET = 3
K_OPP = 3
MARGIN = 500.0
MAX_OPTS = 24
MAX_SUBSTEPS = 40
HAND_W = 20.0
_FILL = DECK[0] if DECK else 1


def _filler(n):
    if n <= 0:
        return []
    return [DECK[_random.randrange(len(DECK))] for _ in range(n)]


def _obs_dict(o):
    return _dc.asdict(o)


def _leaf(cs, me_i):
    if cs is None:
        return 0.0
    r = cs.result
    if r is not None and r >= 0:
        return 1e7 if r == me_i else (0.0 if r == 2 else -1e7)
    me = cs.players[me_i]; op = cs.players[1 - me_i]
    mf = [p for p in (me.active + me.bench) if p]
    of = [p for p in (op.active + op.bench) if p]
    mhp = sum(p.hp for p in mf); ohp = sum(p.hp for p in of)
    men = sum(len(p.energies) for p in mf); oen = sum(len(p.energies) for p in of)
    noact = 0 if (me.active and me.active[0]) else 1
    hand = getattr(me, "handCount", None)
    if hand is None:
        hand = len(me.hand) if getattr(me, "hand", None) else 0
    return (1000.0 * (len(op.prize) - len(me.prize)) + mhp - ohp
            + 5.0 * (men - oen) - 4000.0 * noact + HAND_W * hand)


def _greedy(o):
    try:
        d = _obs_dict(o)
        sel = d.get("select")
        if not sel or not sel.get("option"):
            return None
        out = _heur_choose(d)
        return out if isinstance(out, list) and out else None
    except Exception:
        return None


def _complete(sid, cur, owner, deadline):
    for _ in range(MAX_SUBSTEPS):
        if _time.monotonic() > deadline:
            return sid, cur
        cs = cur.current
        if cs is None or (cs.result is not None and cs.result >= 0):
            return sid, cur
        if cs.yourIndex != owner or cur.select is None:
            return sid, cur
        ch = _greedy(cur)
        if not ch:
            return sid, cur
        try:
            ss = search_step(sid, ch)
        except Exception:
            return sid, cur
        sid, cur = ss.searchId, ss.observation
    return sid, cur


def _forced(sid, cur, owner, deadline, limit=8):
    for _ in range(limit):
        if _time.monotonic() > deadline:
            break
        cs = cur.current
        if (cs is None or cur.select is None or cs.yourIndex != owner
                or cur.select.context == MAIN_CTX
                or (cs.result is not None and cs.result >= 0)):
            break
        ch = _greedy(cur)
        if not ch:
            break
        try:
            ss = search_step(sid, ch)
        except Exception:
            break
        sid, cur = ss.searchId, ss.observation
    return sid, cur


def _search_decide(obs_dict):
    if not _SEARCH_OK:
        return None
    sel = obs_dict.get("select"); cur = obs_dict.get("current") or {}
    if not sel or sel.get("context") != MAIN_CTX:
        return None
    opts = sel.get("option") or []
    n = len(opts)
    if n < 3 or n > MAX_OPTS or (cur.get("turn") or 0) < 2:
        return None
    if obs_dict.get("search_begin_input") is None:
        return None
    me_i = cur.get("yourIndex", 0)
    base = _score_main(obs_dict, sel)
    if not base or len(base) != n:
        return None
    order = sorted(range(n), key=lambda i: -base[i])
    heur_top = order[0]
    cand = [heur_top]
    for i in order[1:]:
        if opts[i].get("type") in (_OPT_ATTACK, _OPT_END) or base[i] < 0:
            continue
        cand.append(i)
        if len(cand) >= 8:
            break
    if len(cand) < 2:
        return None

    me = cur["players"][me_i]; op = cur["players"][1 - me_i]
    observation = to_observation_class(obs_dict)
    t0 = _time.monotonic(); deadline = t0 + BUDGET_S
    acc = {i: 0.0 for i in cand}; nev = {i: 0 for i in cand}

    for _ in range(N_DET):
        if _time.monotonic() > deadline:
            break
        try:
            oact = _filler(1) if (op.get("active") and op["active"][0] is None) else []
            zones = (_filler(me.get("deckCount", 0)), _filler(len(me.get("prize") or [])),
                     _filler(op.get("deckCount", 0)), _filler(len(op.get("prize") or [])),
                     _filler(op.get("handCount", 0)), oact)
            ss0 = search_begin(observation, *zones, manual_coin=False)
        except Exception:
            try:
                search_end()
            except Exception:
                pass
            return None
        root = ss0.searchId
        try:
            for idx in cand:
                if _time.monotonic() > deadline:
                    break
                try:
                    ss = search_step(root, [idx])
                except Exception:
                    continue
                s1, c1 = ss.searchId, ss.observation
                s1, c1 = _complete(s1, c1, me_i, deadline)
                cs = c1.current
                if (cs is None or (cs.result is not None and cs.result >= 0)
                        or cs.yourIndex == me_i or c1.select is None):
                    acc[idx] += _leaf(cs, me_i); nev[idx] += 1
                    continue
                s1, c1 = _forced(s1, c1, 1 - me_i, deadline)
                cs = c1.current
                if (cs is None or c1.select is None
                        or c1.select.context != MAIN_CTX or cs.yourIndex == me_i):
                    acc[idx] += _leaf(cs, me_i); nev[idx] += 1
                    continue
                d1 = _obs_dict(c1)
                ob = _score_main(d1, d1.get("select") or {})
                oo = (sorted(range(len(ob)), key=lambda i: -ob[i])
                      if ob else list(range(len(c1.select.option))))
                worst = None
                for k in range(min(K_OPP, len(oo))):
                    if _time.monotonic() > deadline:
                        break
                    try:
                        ss2 = search_step(s1, [oo[k]])
                    except Exception:
                        continue
                    s2, c2 = ss2.searchId, ss2.observation
                    s2, c2 = _complete(s2, c2, 1 - me_i, deadline)
                    s2, c2 = _forced(s2, c2, me_i, deadline, limit=6)
                    v = _leaf(c2.current, me_i)
                    worst = v if worst is None else min(worst, v)
                if worst is None:
                    worst = _leaf(cs, me_i)
                acc[idx] += worst; nev[idx] += 1
        finally:
            try:
                search_end()
            except Exception:
                pass

    nt = nev.get(heur_top, 0)
    if nt == 0:
        return None
    ev = [i for i in cand if nev[i] == nt]
    avg = {i: acc[i] / nev[i] + 1e-6 * base[i] for i in ev}
    best = max(ev, key=lambda i: avg[i])
    if best == heur_top or avg[best] < avg[heur_top] + MARGIN:
        return None
    return best


def agent(obs_dict):
    try:
        sel = obs_dict.get("select")
        if sel and sel.get("context") == MAIN_CTX and sel.get("option"):
            ov = _search_decide(obs_dict)
            if ov is not None:
                n = len(sel.get("option") or [])
                return _safe([ov] + [i for i in range(n) if i != ov], sel)
    except Exception:
        pass
    return _heur_agent(obs_dict)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "training", "twoply_submission"))
    args = ap.parse_args()

    out = args.out_dir
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    # heuristic.py = frozen v30 (== current main.py / shipped policy)
    shutil.copy(os.path.join(_REPO, "training", "wsearch", "frozen_main_v30.py"),
                os.path.join(out, "heuristic.py"))
    # main.py wrapper
    with open(os.path.join(out, "main.py"), "w", encoding="utf-8") as f:
        f.write(MAIN)
    # deck.csv from heuristic.DECK
    import importlib.util
    spec = importlib.util.spec_from_file_location("h_read", os.path.join(out, "heuristic.py"))
    h = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, os.path.join(_REPO, "training", "local_cg"))
    sys.modules["h_read"] = h
    spec.loader.exec_module(h)
    with open(os.path.join(out, "deck.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(str(c) for c in h.DECK) + "\n")

    print("staged:", out)
    for fn in sorted(os.listdir(out)):
        print("  ", fn, os.path.getsize(os.path.join(out, fn)), "bytes")


if __name__ == "__main__":
    main()
