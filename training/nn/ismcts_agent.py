"""Belief-weighted ISMCTS agent (2026-07-07 training-methods plan item 4;
docs/belief-model.md Phase C consumer 2).

Identical search to mcts.py's MCTSSearcher rollout mode (root-shared stats,
fresh determinization per simulation — the established anti-strategy-fusion
protocol, adversarial rollout opponent, _STALL_MEMO isolation) with ONE
swapped component: hidden zones come from
training/belief/determinize.py's BeliefDeterminizer (posterior-sampled
archetype decklist minus observed cards, exact own-side unseen multiset)
instead of mirror/opponent-deck filler.

ISMCTS_DETERMINIZER=belief (default) | placeholder — the placeholder setting
reproduces the parent class's filler behavior through the same code path,
giving the pre-registered ablation (belief vs placeholder, same search,
same budget) a single toggle.

Same agent contract as main.py; evaluate with training/ab_test.py.
Pre-registered gate (docs/report-log.md 2026-07-07 sampler entry): >=55%
vs v25c to continue, same bar as the original search line.
"""
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_LOCAL_CG = os.path.join(_REPO_ROOT, "training", "local_cg")
if _LOCAL_CG not in sys.path:
    sys.path.insert(0, _LOCAL_CG)
_BELIEF_DIR = os.path.join(_REPO_ROOT, "training", "belief")
if _BELIEF_DIR not in sys.path:
    sys.path.insert(0, _BELIEF_DIR)

import mcts as _mcts  # noqa: E402
from mcts import MCTSSearcher, _softmax, _is_multiselect, _obs_to_dict  # noqa: E402
from determinize import BeliefDeterminizer  # noqa: E402
from main import DECK  # noqa: E402

_SIMS = int(os.environ.get("ISMCTS_SIMS", os.environ.get("MCTS_SIMS", "150")))
_C_PUCT = float(os.environ.get("MCTS_C_PUCT", "1.4"))
_PRIOR_TEMP = float(os.environ.get("MCTS_PRIOR_TEMP", "2.0"))
# Module-level flag (not a frozen local) so a wrapper agent can flip it
# per-call for the belief-vs-placeholder ablation even though both sides of
# an A/B share one worker process (see ismcts_placeholder_agent.py).
DET_MODE = os.environ.get("ISMCTS_DETERMINIZER", "belief")
_TIMING_LOG = os.environ.get("MCTS_TIMING_LOG")

_determinizer = BeliefDeterminizer()

# Archetype-matched rollout policies (2026-07-08 fix, docs/report-log.md
# "gauntlet regression" entry): the original "bounded fix #1" below only
# special-cased the alakazam read, leaving every OTHER archetype's
# determinized zones piloted by the hardcoded lucario module — nonsense
# rollouts that made the endgame search LOSE 18-29pp vs real
# lucario/abomasnow anchors while winning the mirror. Fix: pilot opponent
# rollout turns with the believed archetype's own real bot when we have
# one; unknown/uncovered archetypes keep the parent's lucario fallback.
# Each imported bot's known mutable globals join _STATEFUL_MODULES so the
# parent's per-rollout save/reset covers them (same _STALL_MEMO bug class;
# dragapult's plan_a/plan_b/log-list globals found by inspection).
import importlib  # noqa: E402

_ARCH_MODULES = {}
for _label, _mod_name in (("lucario", "opponents.lucario_agent"),
                          ("dragapult", "opponents.dragapult_agent"),
                          ("abomasnow", "opponents.abomasnow_agent"),
                          ("starmie", "opponents.starmie_agent")):
    try:
        _m = importlib.import_module(_mod_name)
    except Exception:
        continue
    _ARCH_MODULES[_label] = _m
    if _m not in _mcts._STATEFUL_MODULES:
        _attrs = {}
        if hasattr(_m, "plan"):
            _attrs.update({"plan": type(_m.plan), "pre_turn": lambda: 0,
                           "ability_used": lambda: False})
        for _a in ("plan_a", "plan_b"):
            if hasattr(_m, _a):
                _attrs[_a] = type(getattr(_m, _a))
        for _a in ("pre_turn_log", "current_turn_log", "prize"):
            if hasattr(_m, _a):
                _attrs[_a] = list
        if _attrs:
            _mcts._STATEFUL_MODULES[_m] = _attrs


class BeliefMCTSSearcher(MCTSSearcher):
    """choose_with_stats is a copy of the parent's (mcts.py is under active
    edit by the sign-debug work, so subclass-copy beats refactoring a hook
    into it right now — merge a _determinize() hook upstream once that
    lands) with only the determinization block swapped.

    Bounded fix #1 (after gate 1's 0W-50L, docs/report-log.md 2026-07-07):
    the rollout's opponent POLICY must match the believed archetype the
    determinizer filled the hidden zones with — gate 1 sampled
    alakazam-mirror zones (correct, the opponent was v25c) but piloted them
    with the parent's hardcoded lucario_agent, i.e. a lucario bot playing
    an alakazam hand: nonsense rollouts. When the belief read is
    alakazam(-mirror), opponent rollout turns now use the real heuristic
    (main.agent — its _STALL_MEMO is already isolated per-simulation by the
    parent's _rollout); any other/unknown read keeps the parent behavior."""

    _rollout_arch = None
    # kept for back-compat with subclasses that still set it (gumbel agent)
    _mirror_rollout = False

    def _action_for(self, obs_dict, our_seat):
        yours = obs_dict["current"]["yourIndex"]
        if yours != our_seat:
            arch = self._rollout_arch or ("alakazam" if self._mirror_rollout else None)
            if arch == "alakazam":
                agent_fn = _mcts.heuristic.agent
            else:
                mod = _ARCH_MODULES.get(arch)
                agent_fn = mod.agent if mod is not None else None
            if agent_fn is not None:
                out = agent_fn(obs_dict)
                return list(out) if out else [0]
        return super()._action_for(obs_dict, our_seat)

    def choose_with_stats(self, obs_dict):
        import math
        from cg.api import to_observation_class, search_begin, search_step, search_end

        sel = obs_dict.get("select")
        if not sel or not sel.get("option"):
            return [], None, None
        n_opts = len(sel["option"])
        if n_opts <= 1:
            return ([0] if n_opts == 1 else []), None, None
        if _is_multiselect(sel):
            return _mcts.heuristic.agent(obs_dict), None, None

        our_seat = obs_dict["current"]["yourIndex"]

        scores = _mcts.heuristic.score_options(obs_dict, sel)
        if not scores or len(scores) != n_opts:
            scores = [0.0] * n_opts
        P = _softmax(scores, temp=self.prior_temp)
        N = [0] * n_opts
        W = [0.0] * n_opts

        for _ in range(self.sims):
            observation = to_observation_class(obs_dict)

            if DET_MODE == "belief":
                z = _determinizer.sample(obs_dict, our_seat, random)
                zones = (z["your_deck"], z["your_prize"], z["opponent_deck"],
                         z["opponent_prize"], z["opponent_hand"], z["opponent_active"])
                self._rollout_arch = z["archetype"]
            else:  # "placeholder": parent-class filler behavior, same code path
                state = observation.current
                my_p = state.players[our_seat]
                opp_p = state.players[1 - our_seat]
                opp_deck_pool = _mcts._opponent_module.DECK
                zones = (
                    self._filler(my_p.deckCount, _mcts.heuristic.DECK),
                    self._filler(len(my_p.prize), _mcts.heuristic.DECK),
                    self._filler(opp_p.deckCount, opp_deck_pool),
                    self._filler(len(opp_p.prize), opp_deck_pool),
                    self._filler(opp_p.handCount, opp_deck_pool),
                    self._filler(1, opp_deck_pool)
                    if (opp_p.active and opp_p.active[0] is None) else [],
                )

            root_ss = search_begin(observation, *zones, manual_coin=False)

            total_n = sum(N)
            sqrt_total = math.sqrt(total_n + 1e-8)
            best_score, best_a = -1e18, 0
            for a in range(n_opts):
                q = W[a] / N[a] if N[a] > 0 else 0.0
                u = self.c_puct * P[a] * sqrt_total / (1 + N[a])
                if q + u > best_score:
                    best_score, best_a = q + u, a

            child_ss = search_step(root_ss.searchId, [best_a])
            child_obs = _obs_to_dict(child_ss.observation)
            value = self._rollout(child_ss.searchId, child_obs, our_seat)

            N[best_a] += 1
            W[best_a] += value

        best_a = max(range(n_opts), key=lambda a: N[a])
        search_end()
        total_n = sum(N)
        root_value = (sum(W) / total_n) if total_n > 0 else 0.0
        return [best_a], N, root_value


_fallback_count = 0


def agent(obs_dict: dict) -> list:
    global _fallback_count
    sel = obs_dict.get("select")
    if sel is None:
        return DECK
    if not sel.get("option"):
        return []
    t0 = time.time() if _TIMING_LOG else None
    try:
        searcher = BeliefMCTSSearcher(sims=_SIMS, c_puct=_C_PUCT,
                                      prior_temp=_PRIOR_TEMP, leaf_eval="rollout")
        return searcher.choose(obs_dict)
    except Exception as e:
        _fallback_count += 1
        print(f"[ismcts_agent] FALLBACK #{_fallback_count} to heuristic: {e!r}",
              file=sys.stderr)
        from main import agent as heuristic_agent
        return heuristic_agent(obs_dict)
    finally:
        if t0 is not None:
            path = f"{_TIMING_LOG}.{os.getpid()}"
            with open(path, "a") as f:
                f.write(f"{time.time() - t0:.4f}\n")
