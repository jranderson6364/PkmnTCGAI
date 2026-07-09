"""Restricted-authority 1-ply advisor (Phase A/C of the 2026-07-09
overnight /goal — structural redesign after Gate 2 falsified weak-leaf-
signal as the search family's constraint).

Structure, with each choice pinned to a documented failure of the dead
PUCT wrapper (docs/report-log.md 2026-07-08/09):
  * The real `main.agent` picks first; the advisor may only substitute one
    of the heuristic's own near-tied top options ("heuristic proposes,
    eval disposes") — attacks the displacement mechanism directly.
  * 1-ply only: each candidate action is stepped once per determinization
    and the resulting board is scored by the calculated-values eval
    (Φ v4 linear or MLP). NO rollout policy exists → the archetype-
    mismatch, strategy-fusion, and stateful-module-corruption bug classes
    are structurally impossible, not just fixed.
  * Values averaged over N_DET fresh determinizations per candidate.
  * Override only if a candidate beats the teacher's choice by MARGIN in
    tanh-value — hysteresis toward the replay-verified teacher.

Env knobs: ADVISOR_N_DET (default 8), ADVISOR_MARGIN (default 0.10),
ADVISOR_TOPK (default 3), ADVISOR_SCORER ("linear"|"mlp", default linear),
MCTS_OPPONENT_MODULE (deck list for hidden-zone filler only — no agent
calls), ADVISOR_LOG (optional: one JSON line per override for the
disagreement-mining workflow that found the last three search bugs).
"""
import json
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_LOCAL_CG = os.path.join(_REPO_ROOT, "training", "local_cg")
if _LOCAL_CG not in sys.path:
    sys.path.insert(0, _LOCAL_CG)

import importlib  # noqa: E402
import dataclasses  # noqa: E402

import numpy as np  # noqa: E402

import main as heuristic  # noqa: E402
from main import DECK  # noqa: E402
from eval_v4 import features_v4  # noqa: E402

_N_DET = int(os.environ.get("ADVISOR_N_DET", "8"))
_MARGIN = float(os.environ.get("ADVISOR_MARGIN", "0.10"))
_TOPK = int(os.environ.get("ADVISOR_TOPK", "3"))
_SCORER = os.environ.get("ADVISOR_SCORER", "linear")
_LOG = os.environ.get("ADVISOR_LOG")

_OPP_MODULE = importlib.import_module(
    os.environ.get("MCTS_OPPONENT_MODULE", "opponents.lucario_agent"))

_W_LIN = np.load(os.environ.get("ADVISOR_WEIGHTS") or
                 os.path.join(_REPO_ROOT, "training", "eval_v4_weights.npy"))
_MLP = None


def _value(cur, me_idx):
    f = features_v4(cur, me_idx)
    if f is None:
        return 0.0
    if _SCORER == "mlp":
        global _MLP
        if _MLP is None:
            import torch
            from eval_v4_mlp import MLP
            _MLP = MLP(12, 64, 2)  # must match the CV-selected config
            _MLP.load_state_dict(torch.load(
                os.path.join(_REPO_ROOT, "training", "eval_v4_mlp.pth")))
            _MLP.eval()
        import torch
        turn = (cur.get("turn") or 0)
        x = torch.from_numpy(np.concatenate(
            [f, [min(turn, 30) / 6.0 / 5.0]]).astype(np.float32))[None, :]
        with torch.no_grad():
            return math.tanh(float(_MLP(x).item()) / 2.0)
    return math.tanh(float(f @ _W_LIN) / 2.0)


def _filler(n, pool):
    pool = list(pool)
    random.shuffle(pool)
    return pool[:n] if n <= len(pool) else [pool[i % len(pool)] for i in range(n)]


def _is_multiselect(sel):
    return sel is not None and (sel.get("minCount", 0) or 0) >= 2


_fallback_count = 0
_override_count = 0


def agent(obs_dict: dict) -> list:
    global _fallback_count, _override_count
    sel = obs_dict.get("select")
    if sel is None:
        return DECK
    opts = sel.get("option", [])
    n = len(opts)
    teacher = heuristic.agent(obs_dict)
    if n <= 1 or _is_multiselect(sel) or not teacher or len(teacher) != 1:
        return teacher
    try:
        scores = heuristic.score_options(obs_dict, sel)
        if not scores or len(scores) != n:
            return teacher
        t_choice = teacher[0]
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        candidates = set(order[:_TOPK]) | {t_choice}
        if len(candidates) <= 1:
            return teacher

        from cg.api import to_observation_class, search_begin, search_step, search_end

        our_seat = obs_dict["current"]["yourIndex"]
        sums = {a: 0.0 for a in candidates}
        for _ in range(_N_DET):
            observation = to_observation_class(obs_dict)
            state = observation.current
            my_p = state.players[our_seat]
            opp_p = state.players[1 - our_seat]
            your_deck = _filler(my_p.deckCount, heuristic.DECK)
            your_prize = _filler(len(my_p.prize), heuristic.DECK)
            opp_deck = _filler(opp_p.deckCount, _OPP_MODULE.DECK)
            opp_prize = _filler(len(opp_p.prize), _OPP_MODULE.DECK)
            opp_hand = _filler(opp_p.handCount, _OPP_MODULE.DECK)
            opp_active = []
            active = opp_p.active
            if len(active) > 0 and active[0] is None:
                opp_active = _filler(1, _OPP_MODULE.DECK)
            for a in candidates:
                root = search_begin(observation, your_deck, your_prize, opp_deck,
                                    opp_prize, opp_hand, opp_active,
                                    manual_coin=False)
                child = search_step(root.searchId, [a])
                child_cur = dataclasses.asdict(child.observation).get("current") or {}
                sums[a] += _value(child_cur, our_seat)
        search_end()

        best = max(candidates, key=lambda a: sums[a])
        if best != t_choice and (sums[best] - sums[t_choice]) / _N_DET > _MARGIN:
            _override_count += 1
            if _LOG:
                with open(f"{_LOG}.{os.getpid()}", "a") as f:
                    f.write(json.dumps({
                        "turn": (obs_dict.get("current") or {}).get("turn"),
                        "teacher": t_choice, "advisor": best,
                        "v_teacher": sums[t_choice] / _N_DET,
                        "v_advisor": sums[best] / _N_DET,
                        "stype": sel.get("stype"),
                    }) + "\n")
            return [best]
        return teacher
    except Exception as e:
        _fallback_count += 1
        print(f"[advisor_agent] FALLBACK #{_fallback_count} to teacher: {e!r}",
              file=sys.stderr)
        return teacher
