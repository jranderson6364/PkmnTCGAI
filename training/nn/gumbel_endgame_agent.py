"""Gumbel sequential-halving endgame agent (2026-07-07 pre-registration #3,
docs/report-log.md; method-survey pick 3).

Identical to endgame_agent.py (same gate, same belief determinizer, same
rollout leaf eval) except ROOT SELECTION: instead of PUCT (documented
visit-collapse failure mode at low sims), the root runs Gumbel-style
sequential halving (Danihelka et al., ICLR 2022): sample Gumbel noise once,
seed candidate scores with the heuristic prior logits, then run halving
phases where every surviving candidate is evaluated on the SAME
belief-sampled determinizations (paired worlds — determinization variance
cancels out of the elimination comparisons) and the bottom half by
g + logits + sigma(Q) is eliminated each phase.

ENDGAME_SIMS / ENDGAME_PRIZES / MCTS_PRIOR_TEMP as in endgame_agent.py.
GUMBEL_MAX_CANDIDATES (default 16) caps the initial candidate set (top by
g + logits). sigma(q) = (c_visit + max_visits) * c_scale * q with the
paper's c_visit=50, c_scale=1.0.

Pre-registered gate: 400-game A/B vs endgame_agent.py (search-vs-search,
everything else identical); adopt only if the win-rate CI excludes 50%.
"""
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mcts as _mcts  # noqa: E402
import ismcts_agent as _searchmod  # noqa: E402  (sets up local_cg/belief paths)
from ismcts_agent import BeliefMCTSSearcher, _determinizer  # noqa: E402
from mcts import _softmax, _is_multiselect, _obs_to_dict  # noqa: E402
from main import DECK, agent as heuristic_agent  # noqa: E402

_SIMS = int(os.environ.get("ENDGAME_SIMS", "60"))
_PRIZES = int(os.environ.get("ENDGAME_PRIZES", "2"))
_PRIOR_TEMP = float(os.environ.get("MCTS_PRIOR_TEMP", "2.0"))
_MAX_CAND = int(os.environ.get("GUMBEL_MAX_CANDIDATES", "16"))
_C_VISIT = 50.0
_C_SCALE = 1.0

_fallback_count = 0
searched_decisions = 0


class GumbelEndgameSearcher(BeliefMCTSSearcher):
    """Root = Gumbel sequential halving over paired belief determinizations;
    everything below the root (determinize, step, rollout, mirror-rollout
    policy) is inherited unchanged."""

    def choose_with_stats(self, obs_dict):
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
        logits = [math.log(p + 1e-12) for p in P]
        gumbel = [-math.log(-math.log(random.random() + 1e-12) + 1e-12)
                  for _ in range(n_opts)]

        survivors = sorted(range(n_opts), key=lambda a: gumbel[a] + logits[a],
                           reverse=True)[: min(n_opts, _MAX_CAND)]
        N = [0] * n_opts
        W = [0.0] * n_opts

        n_phases = max(1, math.ceil(math.log2(len(survivors))))
        budget_per_phase = max(len(survivors), self.sims // n_phases)

        def sigma_q(a):
            q = W[a] / N[a] if N[a] > 0 else 0.0
            return (_C_VISIT + max(N)) * _C_SCALE * q

        for _ in range(n_phases):
            if len(survivors) == 1:
                break
            worlds = max(1, budget_per_phase // len(survivors))
            for _w in range(worlds):
                z = _determinizer.sample(obs_dict, our_seat, random)
                zones = (z["your_deck"], z["your_prize"], z["opponent_deck"],
                         z["opponent_prize"], z["opponent_hand"],
                         z["opponent_active"])
                self._rollout_arch = z["archetype"]
                # paired: every surviving candidate sees this SAME world
                for a in survivors:
                    observation = to_observation_class(obs_dict)
                    root_ss = search_begin(observation, *zones, manual_coin=False)
                    child_ss = search_step(root_ss.searchId, [a])
                    child_obs = _obs_to_dict(child_ss.observation)
                    value = self._rollout(child_ss.searchId, child_obs, our_seat)
                    N[a] += 1
                    W[a] += value
            survivors = sorted(
                survivors, key=lambda a: gumbel[a] + logits[a] + sigma_q(a),
                reverse=True)[: max(1, math.ceil(len(survivors) / 2))]

        best_a = max(survivors, key=lambda a: gumbel[a] + logits[a] + sigma_q(a))
        search_end()
        total_n = sum(N)
        root_value = (sum(W) / total_n) if total_n > 0 else 0.0
        return [best_a], N, root_value


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
        searcher = GumbelEndgameSearcher(sims=_SIMS, prior_temp=_PRIOR_TEMP,
                                         leaf_eval="rollout")
        out = searcher.choose(obs_dict)
        searched_decisions += 1
        return out
    except Exception as e:
        _fallback_count += 1
        print(f"[gumbel_endgame_agent] FALLBACK #{_fallback_count} to heuristic: {e!r}",
              file=sys.stderr)
        return heuristic_agent(obs_dict)
