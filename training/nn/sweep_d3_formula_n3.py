"""2-ply belief-determinized minimax OVERRIDE on top of our v30 heuristic.

The first search recipe with live-ladder evidence of beating our v29d: it
replicates the mechanism of alakazam_v9 (publicScore 778.2), whose entire edge
over us is search (isolation 2026-07-23: search ON 61% vs OFF 46% vs our v29d) on
a harness the placebo control confirmed is essentially clean (53.2% [48.3,58.1]).

Recipe (understood from mechanism, NOT copied — our heuristic, our determinizer,
our leaf eval):
  * Only on MAIN decisions with 3..24 options and turn >= 2.
  * Candidates = our heuristic's top-K NON-terminal, positive-score options
    (terminal ATTACK/END are reached via the greedy rollout, not branched).
  * For each of N_DET belief-determinizations of the hidden zones:
      - search_begin, then for each candidate:
          ply 1: take it, greedy-complete OUR turn with the heuristic;
          ply 2: MIN over the opponent's top-K greedy-completed replies;
          leaf-eval the resulting state.
  * Average leaf value over determinizations per candidate.
  * OVERRIDE the heuristic's own top pick ONLY when the best candidate beats it
    by a real margin (>= half a prize). Otherwise defer to the heuristic.

The conservative override is the crux: the heuristic drives, search only vetoes
on a clear tactical win. This is structurally different from all five closed
search attempts (endgame-gated / full-PIMC / leaf-value ISMCTS), which replaced
the policy and lost plan coherence. Belief determinization uses our 92.3%
archetype classifier (training/belief/determinize.py), better than alakazam_v9's
template match.

Timeout-safe: hard wall-clock budget, and any failure falls back to the pure
heuristic pick. Env knobs: TWOPLY_BUDGET_S, TWOPLY_NDET, TWOPLY_KOPP,
TWOPLY_MARGIN, TWOPLY_MAXOPTS.
"""
import os
import sys
import glob
import time
import random
import dataclasses
import importlib.util

for _pat in ['/kaggle/input/**/cg-lib', '/kaggle/input/cg-lib']:
    _paths = glob.glob(_pat, recursive=True)
    if _paths:
        sys.path.insert(0, _paths[0]); break
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "training", "local_cg"),
           os.path.join(_REPO, "training", "belief"),
           os.path.join(_REPO, "training"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# our heuristic policy (candidate ordering + greedy picks + fallback)
_spec = importlib.util.spec_from_file_location(
    "twoply_heuristic", os.path.join(_REPO, "training", "wsearch", "frozen_main_v30.py"))
H = importlib.util.module_from_spec(_spec)
sys.modules["twoply_heuristic"] = H
_spec.loader.exec_module(H)

DECK = list(H.DECK)

try:
    from cg.api import to_observation_class, search_begin, search_step, search_end
    from determinize import BeliefDeterminizer
    _SEARCH_OK = True
    _DET = BeliefDeterminizer(conf_threshold=0.97)
except Exception:
    _SEARCH_OK = False
    _DET = None

BUDGET_S = float(os.environ.get("TWOPLY_BUDGET_S", "0.80"))
N_DET    = 3  # baked
K_OPP    = int(os.environ.get("TWOPLY_KOPP", "3"))
MAX_OPTS = int(os.environ.get("TWOPLY_MAXOPTS", "24"))
# DEPTH = number of turn-plies looked ahead including the candidate's own turn.
# 2 = our turn + opp reply + leaf (the shipped 776 behavior). 3 = + our turn.
# Our plies are single greedy lines; only opponent plies branch (top-K), so
# cost is ~linear in our plies and K^(opp plies). (S1, pre-registered 2026-07-23.)
DEPTH    = 3  # baked
MAX_SUBSTEPS = 40
MAIN = 0

# Leaf evaluation mode: "formula" (hand-crafted prize/hp/energy/hand) or "phi4"
# (the project's best fitted state-value, docs/eval-function-research.md). Phi v4
# is a LEARNED linear value function -- the bridge from the search back to the RL
# track: a better value signal at the leaf should sharpen the override decisions.
# Its output scale is ~[-7,7] (normalized features), so the override MARGIN is
# rescaled per mode: half a prize is 1000 in formula units, but weights[0]/6*0.5
# in phi4 units.
LEAF_MODE = "formula"  # baked
_PHI4_W = None
_eval_v4 = None
if LEAF_MODE == "phi4":
    try:
        import numpy as _np
        from eval_v4 import features_v4 as _feat_v4, eval_v4 as _eval_v4
        _PHI4_W = _np.load(os.path.join(_REPO, "training", "eval_v4_weights.npy"))
        _PHI4_HALF_PRIZE = float(_PHI4_W[0]) / 6.0 * 0.5
    except Exception:
        LEAF_MODE = "formula"

if LEAF_MODE == "phi4":
    MARGIN = float(os.environ.get("TWOPLY_MARGIN", str(_PHI4_HALF_PRIZE)))
else:
    MARGIN = float(os.environ.get("TWOPLY_MARGIN", "500"))

_OPT_ATTACK, _OPT_END = 13, 14

# Instrumentation: how often does the search actually fire / override? Dumped to
# stderr at interpreter exit when TWOPLY_STATS=1, so a gate can confirm the
# override is live (not inert like probability_v2's dead beam search).
_STATS = {"main_decisions": 0, "searched": 0, "considered": 0, "overrides": 0}
if os.environ.get("TWOPLY_STATS") == "1":
    import atexit
    atexit.register(lambda: sys.stderr.write(f"[twoply_stats] {_STATS}\n"))


def _obs_dict(observation):
    """SearchState.observation (cg dataclass) -> raw dict our heuristic consumes."""
    return dataclasses.asdict(observation)


HAND_W = float(os.environ.get("TWOPLY_HANDW", "20"))


def _leaf_eval(cs, me_i):
    if cs is None:
        return 0.0
    res = cs.result
    if res is not None and res >= 0:
        if res == me_i:
            return 1e7
        if res == 2:
            return 0.0
        return -1e7
    if LEAF_MODE == "phi4":
        # LEARNED leaf: Phi v4's fitted 11-feature state-value on the leaf state.
        # Terminal handled above; a win/loss still dominates via +/-1e7. Non-
        # terminal values live on the small phi4 scale, compared against the
        # rescaled half-prize margin.
        try:
            v = _eval_v4(_obs_dict(cs), me_i, _PHI4_W)
            if v is not None:
                return v
        except Exception:
            pass
    me = cs.players[me_i]
    op = cs.players[1 - me_i]
    my_field = [p for p in (me.active + me.bench) if p]
    op_field = [p for p in (op.active + op.bench) if p]
    my_hp = sum(p.hp for p in my_field)
    op_hp = sum(p.hp for p in op_field)
    my_en = sum(len(p.energies) for p in my_field)
    op_en = sum(len(p.energies) for p in op_field)
    no_active = 0 if (me.active and me.active[0]) else 1
    # Hand size IS our damage stat: Powerful Hand deals 20 x hand size. The
    # generic prize/hp/energy eval (from alakazam_v9) ignores this, so the search
    # overrode the heuristic's correct tempo/racing lines with board-development
    # plays it wrongly scored as better -- catastrophic vs fast attackers
    # (dragapult 99% -> 50%, 2026-07-23). Value our hand at its damage potential.
    my_hand = getattr(me, "handCount", None)
    if my_hand is None:
        my_hand = len(me.hand) if getattr(me, "hand", None) else 0
    return (1000.0 * (len(op.prize) - len(me.prize))
            + my_hp - op_hp + 5.0 * (my_en - op_en)
            - 4000.0 * no_active
            + HAND_W * my_hand)


def _greedy_choice(observation):
    """Our heuristic's pick for whoever's turn it is in the sim."""
    try:
        d = _obs_dict(observation)
        sel = d.get("select")
        if not sel or not sel.get("option"):
            return None
        out = H._choose(d)
        return out if isinstance(out, list) and out else None
    except Exception:
        return None


def _greedy_complete_turn(sid, cur, owner, deadline):
    for _ in range(MAX_SUBSTEPS):
        if time.monotonic() > deadline:
            return sid, cur
        cs = cur.current
        if cs is None or (cs.result is not None and cs.result >= 0):
            return sid, cur
        if cs.yourIndex != owner or cur.select is None:
            return sid, cur
        ch = _greedy_choice(cur)
        if not ch:
            return sid, cur
        try:
            ss = search_step(sid, ch)
        except Exception:
            return sid, cur
        sid, cur = ss.searchId, ss.observation
    return sid, cur


def _advance_forced(sid, cur, owner, deadline, limit=8):
    for _ in range(limit):
        if time.monotonic() > deadline:
            break
        cs = cur.current
        if (cs is None or cur.select is None or cs.yourIndex != owner
                or cur.select.context == MAIN
                or (cs.result is not None and cs.result >= 0)):
            break
        ch = _greedy_choice(cur)
        if not ch:
            break
        try:
            ss = search_step(sid, ch)
        except Exception:
            break
        sid, cur = ss.searchId, ss.observation
    return sid, cur


def _minimax(sid, cur, me_i, plies_left, deadline):
    """Value of position `cur` looking `plies_left` turn-plies ahead. Our plies
    are single greedy lines; opponent plies branch over top-K and take the MIN
    (worst-case). Leaf = _leaf_eval (formula or Φ v4). Generalizes the shipped
    2-ply logic to arbitrary depth (S1)."""
    cs = cur.current
    if (cs is None or (cs.result is not None and cs.result >= 0)
            or plies_left <= 0 or time.monotonic() > deadline):
        return _leaf_eval(cs, me_i)
    # ensure a MAIN decision for the current owner (resolve forced sub-selects)
    if cur.select is None or cur.select.context != MAIN:
        owner = cs.yourIndex
        sid, cur = _advance_forced(sid, cur, owner, deadline)
        cs = cur.current
        if (cs is None or (cs.result is not None and cs.result >= 0)
                or cur.select is None or cur.select.context != MAIN):
            return _leaf_eval(cs, me_i)
    owner = cs.yourIndex
    if owner == me_i:
        sid2, c2 = _greedy_complete_turn(sid, cur, me_i, deadline)
        return _minimax(sid2, c2, me_i, plies_left - 1, deadline)
    # opponent's turn: branch top-K, minimize
    d = _obs_dict(cur)
    ob = H.score_options_main(d, d.get("select") or {})
    order = (sorted(range(len(ob)), key=lambda i: -ob[i])
             if ob else list(range(len(cur.select.option))))
    worst = None
    for k in range(min(K_OPP, len(order))):
        if time.monotonic() > deadline:
            break
        try:
            ss = search_step(sid, [order[k]])
        except Exception:
            continue
        sid2, c2 = ss.searchId, ss.observation
        sid2, c2 = _greedy_complete_turn(sid2, c2, 1 - me_i, deadline)
        v = _minimax(sid2, c2, me_i, plies_left - 1, deadline)
        worst = v if worst is None else min(worst, v)
    return worst if worst is not None else _leaf_eval(cs, me_i)


def _search_decide(obs_dict):
    if not (_SEARCH_OK and _DET is not None):
        return None
    sel = obs_dict.get("select")
    cur = obs_dict.get("current") or {}
    if not sel or sel.get("context") != MAIN:
        return None
    opts = sel.get("option") or []
    n = len(opts)
    if n < 3 or n > MAX_OPTS or (cur.get("turn") or 0) < 2:
        return None
    if obs_dict.get("search_begin_input") is None:
        return None

    me_i = cur.get("yourIndex", 0)
    base = H.score_options_main(obs_dict, sel)
    if not base or len(base) != n:
        return None
    _STATS["searched"] += 1
    base_order = sorted(range(n), key=lambda i: -base[i])
    heur_top = base_order[0]

    cand = [heur_top]
    for i in base_order[1:]:
        t = opts[i].get("type")
        if t in (_OPT_ATTACK, _OPT_END):
            continue
        if base[i] < 0:
            continue
        cand.append(i)
        if len(cand) >= 8:
            break
    if len(cand) < 2:
        return None

    observation = to_observation_class(obs_dict)
    t0 = time.monotonic()
    deadline = t0 + BUDGET_S
    acc = {i: 0.0 for i in cand}
    n_eval = {i: 0 for i in cand}

    for _ in range(N_DET):
        if time.monotonic() > deadline:
            break
        try:
            z = _DET.sample(obs_dict, me_i, random)
            zones = (z["your_deck"], z["your_prize"], z["opponent_deck"],
                     z["opponent_prize"], z["opponent_hand"], z["opponent_active"])
            ss0 = search_begin(observation, *zones, manual_coin=False)
        except Exception:
            try:
                search_end()
            except Exception:
                pass
            return None
        root_sid = ss0.searchId
        try:
            for idx in cand:
                if time.monotonic() > deadline:
                    break
                try:
                    ss = search_step(root_sid, [idx])
                except Exception:
                    continue
                sid1, c1 = ss.searchId, ss.observation
                # complete our candidate turn (ply 1), then look DEPTH-1 further
                sid1, c1 = _greedy_complete_turn(sid1, c1, me_i, deadline)
                v = _minimax(sid1, c1, me_i, DEPTH - 1, deadline)
                acc[idx] += v; n_eval[idx] += 1
        finally:
            try:
                search_end()
            except Exception:
                pass

    n_top = n_eval.get(heur_top, 0)
    if n_top == 0:
        return None
    _STATS["considered"] += 1
    evaluated = [i for i in cand if n_eval[i] == n_top]
    avg = {i: acc[i] / n_eval[i] + 1e-6 * base[i] for i in evaluated}
    best = max(evaluated, key=lambda i: avg[i])
    if best == heur_top:
        return None
    if avg[best] < avg[heur_top] + MARGIN:
        return None
    _STATS["overrides"] += 1
    return best


def agent(obs_dict: dict):
    try:
        sel = obs_dict.get("select")
        if sel and sel.get("context") == MAIN and (sel.get("option")):
            _STATS["main_decisions"] += 1
            override = _search_decide(obs_dict)
            if override is not None:
                # place the searched action first; keep a legal full ordering
                n = len(sel.get("option") or [])
                rest = [i for i in range(n) if i != override]
                return H._safe_return([override] + rest, sel)
    except Exception:
        pass
    return H.agent(obs_dict)
