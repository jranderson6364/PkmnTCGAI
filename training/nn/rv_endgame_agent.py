"""RV-leaf endgame agent (2026-07-07, RV-2 promotion — docs/report-log.md
"RV-2 result" entry).

Identical to endgame_agent.py (gate, belief determinizer, PUCT root) except
LEAF EVALUATION: instead of rolling out to a real terminal (which is
signal-free outside closing races — the ISMCTS closure), each leaf advances
real play to our next own decision (opponent turns via the adversarial
rollout policy, bounded by RV_LEAF_DEPTH) and returns
`tanh(Φv2(s) + resid(s))` from the RV-2 replay-trained residual value net —
the control-variate form whose holdout sign-acc is the project's best
on-disk real-replay value signal (ALL 0.640, MID 0.640, LATE 0.718). tanh
squashes the ~[-5, 5] deployed value into the [-1, 1] scale the PUCT
constants were tuned for; it is monotone, so it changes no comparisons.

The point of this agent is to test whether a real value signal lets the
search gate extend EARLIER than rollout terminals can reach — run it with
ENDGAME_PRIZES=4 (or higher) where rollout search has never worked.

Env knobs: ENDGAME_SIMS (60), ENDGAME_PRIZES (2), RV_CKPT
(training/ptcg_rv_r2.pth.ep3), RV_LEAF_DEPTH (40), MCTS_C_PUCT,
MCTS_PRIOR_TEMP.
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

import mcts as _mcts  # noqa: E402
import ismcts_agent as _searchmod  # noqa: E402  (sets up local_cg/belief paths)
from ismcts_agent import BeliefMCTSSearcher  # noqa: E402
from mcts import _obs_to_dict, _is_terminal, _terminal_value  # noqa: E402
from replay_value import TwoHotIQL, batchify  # noqa: E402
from encode import encode_sample  # noqa: E402
from model import PTCGNet  # noqa: E402
from phi_baseline import phi_v2  # noqa: E402
from main import DECK, agent as heuristic_agent  # noqa: E402

_SIMS = int(os.environ.get("ENDGAME_SIMS", "60"))
_PRIZES = int(os.environ.get("ENDGAME_PRIZES", "2"))
_C_PUCT = float(os.environ.get("MCTS_C_PUCT", "1.4"))
_PRIOR_TEMP = float(os.environ.get("MCTS_PRIOR_TEMP", "2.0"))
_RV_CKPT = os.environ.get(
    "RV_CKPT", os.path.join(_REPO_ROOT, "training", "ptcg_rv_r2.pth.ep3"))
_LEAF_DEPTH = int(os.environ.get("RV_LEAF_DEPTH", "40"))

_fallback_count = 0
searched_decisions = 0
_model = None


def _rv_model():
    global _model
    if _model is None:
        import torch
        m = TwoHotIQL(PTCGNet())
        state = torch.load(_RV_CKPT, map_location="cpu")
        m.load_state_dict(state["model"])
        m.eval()
        _model = m
    return _model


def _rv_value(obs_dict, our_seat):
    """tanh(Φv2 + resid) at an our-turn state with live options; 0.0 if
    either component is unavailable (unknown, same convention as the
    parent's depth-cap return)."""
    import torch
    cur = obs_dict.get("current") or {}
    sel = obs_dict.get("select")
    try:
        pv = phi_v2(cur, our_seat)
    except Exception:
        pv = None
    if pv is None:
        return 0.0
    try:
        enc = encode_sample(obs_dict, sel)
        b = batchify([(enc, 0.0, 0, None)])
        with torch.no_grad():
            resid = float(_rv_model().v(b)[0])
    except Exception:
        return 0.0
    return math.tanh(pv + resid)


class RVEndgameSearcher(BeliefMCTSSearcher):
    """Leaf evaluation swapped from rollout-to-terminal to the RV-2 net.
    _rollout is overridden (rather than adding a leaf_eval mode) so the
    parent's choose_with_stats — including the belief-determinization block
    and mirror-rollout policy for opponent turns during the leaf walk —
    is inherited byte-for-byte."""

    def _rollout(self, search_id, obs_dict, our_seat):
        saved = {}
        for module, attrs in _mcts._STATEFUL_MODULES.items():
            saved[module] = {a: getattr(module, a) for a in attrs if hasattr(module, a)}
            for a, default_factory in attrs.items():
                if hasattr(module, a):
                    setattr(module, a, default_factory())
        try:
            cur_id, cur_obs = search_id, obs_dict
            for _ in range(_LEAF_DEPTH):
                if _is_terminal(cur_obs):
                    return _terminal_value(cur_obs, our_seat)
                if cur_obs["current"]["yourIndex"] == our_seat:
                    sel = cur_obs.get("select")
                    if not sel or not sel.get("option"):
                        return _terminal_value(cur_obs, our_seat)
                    return _rv_value(cur_obs, our_seat)
                from cg.api import search_step
                ss = search_step(cur_id, self._action_for(cur_obs, our_seat))
                cur_id, cur_obs = ss.searchId, _obs_to_dict(ss.observation)
            self.depth_cap_hits += 1
            return 0.0
        finally:
            for module, vals in saved.items():
                for a, v in vals.items():
                    setattr(module, a, v)


def _is_endgame(obs_dict):
    try:
        players = obs_dict["current"]["players"]
        lens = [len(p.get("prize") or []) for p in players]
        return 0 < min(lens) <= _PRIZES  # 0 < min: see endgame_agent.py
    except Exception:
        return False


def agent(obs_dict: dict) -> list:
    global _fallback_count, searched_decisions
    sel = obs_dict.get("select")
    if sel is None:
        return DECK
    if not sel.get("option"):
        return []
    if len(sel.get("option", [])) <= 1 or not _is_endgame(obs_dict):
        return heuristic_agent(obs_dict)
    try:
        searcher = RVEndgameSearcher(sims=_SIMS, c_puct=_C_PUCT,
                                     prior_temp=_PRIOR_TEMP, leaf_eval="rollout")
        out = searcher.choose(obs_dict)
        searched_decisions += 1
        return out
    except Exception as e:
        _fallback_count += 1
        print(f"[rv_endgame_agent] FALLBACK #{_fallback_count} to heuristic: {e!r}",
              file=sys.stderr)
        return heuristic_agent(obs_dict)
