"""Heuristic-guided PIMC (perfect-information Monte Carlo) search over the
real cg-lib search API (Stage 5, search-at-inference — see
docs/nn-training.md "Resume Here" 2026-07-04 and docs/engine-api.md "Search
API"). Requires `cg.api` on sys.path (either the live Kaggle environment, or
the local dev shim built by `training/setup_local_search.py`).

Design history (see docs/report-log.md 2026-07-04 "Stage 5 search-at-inference"
for the full story — four gates, three real bugs found and fixed via targeted
diagnostics rather than blind hyperparameter tuning, then a fourth diagnostic
that found the actual limiting mechanism):

1. First two gates came back BELOW 50% — a sign a simple "search adds no
   info" theory can't explain; it means the value signal was misaligned.
2. Bug #1: rollout/opponent-model called `main.score_options` argmax, a
   partial (~85%-agreement) reconstruction of the real teacher — confirmed
   via isolated A/B to win only ~30% alone. Fixed: call the real
   `main.agent` instead.
3. Bug #2: **strategy fusion** — hidden-zone determinization was sampled
   once per real decision and reused across all simulations. Fixed:
   re-determinize (fresh random `filler()` + fresh `search_begin`) every
   simulation, `manual_coin=False`. Structural consequence: tree stats can
   only be shared at the root (single-ply PUCT + full rollout per sim, not
   a deep persistent tree) since different sims are different worlds past
   the root.
4. Bug #3: fixing #1 (routing through the real `main.agent`) reintroduced
   global mutable-state corruption — `main.agent`/`_choose` maintains a
   stall-avoidance cache (`_STALL_MEMO`) safe for one linear real game but
   corrupted by thousands of interleaved simulated-rollout calls (collapsed
   a real gate to 1.7%). Fixed: save/reset/restore `_STALL_MEMO` around
   each simulation's rollout.
5. **The actual limiting mechanism (found via a 2-minute targeted
   diagnostic, not another full gate):** even after all three fixes, PUCT
   visit counts piled entirely on one action every time (zero exploration)
   and every terminating rollout was a WIN (90/90 across 3 test positions,
   0 losses). Root cause: the rollout's simulated "opponent" was our own
   heuristic piloting a random hand of OUR OWN deck — a hapless mirror
   opponent that can't punish a bad root choice, so the rollout carried no
   discriminating signal at all. Per advisor + a Claude Fable consult
   (2026-07-04), fixed as a single time-boxed test: the rollout's opponent
   turns are now played by a real adversarial opponent module
   (`MCTS_OPPONENT_MODULE`, default `opponents/lucario_agent.py`) instead of
   a mirror of our own deck/heuristic, with its own hidden-zone filler
   drawn from THAT opponent's real deck list, and its own module-global
   mutable state (`plan`/`pre_turn`/`ability_used` for lucario_agent) reset
   per simulation the same way `_STALL_MEMO` is.

- **Prior** at the root: `main.score_options` (softmaxed) — weaker than the
  real teacher in isolation, but a weak prior only misguides exploration,
  far less damaging than a weak value signal.
- **Rollout policy:** OUR side uses the real `main.agent`; the OPPONENT
  side uses a real adversarial opponent module's `agent()`, all the way to
  a real terminal result.
- **Multi-select safety:** any node whose `select.minCount >= 2` needs
  multiple simultaneous indices — not modeled combinatorially. The
  top-level real decision defers wholesale to `main.agent` if it's one of
  these; mid-rollout such nodes are handled naturally (rollout just plays
  whichever side's real agent full index list).
- **Leaf evaluator:** the rollout's real terminal result, not the net value
  head (already confirmed saturated bimodally near ±1 during the AWR
  diagnostic — unusable as a leaf evaluator).
"""
import dataclasses
import importlib
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import main as heuristic  # noqa: E402

_OPPONENT_MODULE_NAME = os.environ.get("MCTS_OPPONENT_MODULE", "opponents.lucario_agent")
_opponent_module = importlib.import_module(_OPPONENT_MODULE_NAME)

# Known mutable module-globals that must be isolated per-simulation (found by
# inspection — `main.py`'s `_STALL_MEMO` and `lucario_agent`'s `plan`/
# `pre_turn`/`ability_used` are both real, confirmed-necessary cases; add an
# entry here if a different `MCTS_OPPONENT_MODULE` turns out to need one).
_STATEFUL_MODULES = {
    heuristic: {"_STALL_MEMO": dict},
}
if hasattr(_opponent_module, "plan"):
    _STATEFUL_MODULES[_opponent_module] = {
        "plan": type(_opponent_module.plan),
        "pre_turn": lambda: 0,
        "ability_used": lambda: False,
    }


def _obs_to_dict(observation):
    return dataclasses.asdict(observation)


def _softmax(xs, temp=1.0):
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp((x - m) / max(temp, 1e-6)) for x in xs]
    s = sum(exps)
    return [e / s for e in exps] if s > 0 else [1.0 / len(xs)] * len(xs)


def _is_multiselect(sel):
    return sel is not None and (sel.get("minCount", 0) or 0) >= 2


def _terminal_value(obs_dict, our_seat):
    cur = obs_dict.get("current") or {}
    result = cur.get("result", -1)
    if result == our_seat:
        return 1.0
    if result == (1 - our_seat):
        return -1.0
    return 0.0


def _is_terminal(obs_dict):
    sel = obs_dict.get("select")
    return sel is None or not sel.get("option")


class MCTSSearcher:
    """One instance per real decision. Call `choose(obs_dict)` once."""

    def __init__(self, sims=150, max_rollout_depth=250, c_puct=1.4, prior_temp=2.0):
        self.sims = sims
        self.max_rollout_depth = max_rollout_depth
        self.c_puct = c_puct
        self.prior_temp = prior_temp
        self.depth_cap_hits = 0

    def _action_for(self, obs_dict, our_seat):
        """Real teacher for our own turns, the adversarial opponent module
        for the opponent's turns (see module docstring for why a mirror of
        our own heuristic was a dead end as the rollout opponent)."""
        yours = obs_dict["current"]["yourIndex"]
        agent_fn = heuristic.agent if yours == our_seat else _opponent_module.agent
        out = agent_fn(obs_dict)
        return list(out) if out else [0]

    def _rollout(self, search_id, obs_dict, our_seat):
        saved = {}
        for module, attrs in _STATEFUL_MODULES.items():
            saved[module] = {a: getattr(module, a) for a in attrs if hasattr(module, a)}
            for a, default_factory in attrs.items():
                if hasattr(module, a):
                    setattr(module, a, default_factory())
        try:
            cur_id, cur_obs = search_id, obs_dict
            for _ in range(self.max_rollout_depth):
                if _is_terminal(cur_obs):
                    return _terminal_value(cur_obs, our_seat)
                from cg.api import search_step
                ss = search_step(cur_id, self._action_for(cur_obs, our_seat))
                cur_id, cur_obs = ss.searchId, _obs_to_dict(ss.observation)
            self.depth_cap_hits += 1
            return 0.0
        finally:
            for module, vals in saved.items():
                for a, v in vals.items():
                    setattr(module, a, v)

    @staticmethod
    def _filler(n, pool):
        pool = list(pool)
        random.shuffle(pool)
        return pool[:n] if n <= len(pool) else [pool[i % len(pool)] for i in range(n)]

    def choose(self, obs_dict):
        """Same contract as main.agent(): returns list[int]."""
        from cg.api import to_observation_class, search_begin, search_step, search_end

        sel = obs_dict.get("select")
        if not sel or not sel.get("option"):
            return []
        n_opts = len(sel["option"])
        if n_opts <= 1:
            return [0] if n_opts == 1 else []
        if _is_multiselect(sel):
            # Combinatorial multi-select isn't modeled here — defer to the
            # real teacher rather than emit a truncated/invalid selection.
            return heuristic.agent(obs_dict)

        our_seat = obs_dict["current"]["yourIndex"]

        scores = heuristic.score_options(obs_dict, sel)
        if not scores or len(scores) != n_opts:
            scores = [0.0] * n_opts
        P = _softmax(scores, temp=self.prior_temp)
        N = [0] * n_opts
        W = [0.0] * n_opts

        for _ in range(self.sims):
            # Fresh determinization + fresh search per simulation — the fix
            # for strategy fusion (see module docstring). Only the hidden
            # zones vary between sims; the real observable state (and hence
            # the root's actual select/options) doesn't change.
            observation = to_observation_class(obs_dict)
            state = observation.current
            my_p = state.players[our_seat]
            opp_p = state.players[1 - our_seat]

            your_deck = self._filler(my_p.deckCount, heuristic.DECK)
            your_prize = self._filler(len(my_p.prize), heuristic.DECK)
            opponent_deck = self._filler(opp_p.deckCount, _opponent_module.DECK)
            opponent_prize = self._filler(len(opp_p.prize), _opponent_module.DECK)
            opponent_hand = self._filler(opp_p.handCount, _opponent_module.DECK)
            opponent_active = []
            active = opp_p.active
            if len(active) > 0 and active[0] is None:
                opponent_active = self._filler(1, _opponent_module.DECK)

            root_ss = search_begin(observation, your_deck, your_prize, opponent_deck,
                                    opponent_prize, opponent_hand, opponent_active,
                                    manual_coin=False)

            total_n = sum(N)
            sqrt_total = math.sqrt(total_n + 1e-8)
            best_score, best_a = -1e18, 0
            for a in range(n_opts):
                q = W[a] / N[a] if N[a] > 0 else 0.0
                u = self.c_puct * P[a] * sqrt_total / (1 + N[a])
                score = q + u
                if score > best_score:
                    best_score, best_a = score, a

            child_ss = search_step(root_ss.searchId, [best_a])
            value = self._rollout(child_ss.searchId, _obs_to_dict(child_ss.observation), our_seat)

            N[best_a] += 1
            W[best_a] += value
            if os.environ.get("MCTS_DEBUG"):
                print(f"[mcts debug] a={best_a} value={value} N={N} W={[round(w,2) for w in W]}",
                      file=sys.stderr)

        best_a = max(range(n_opts), key=lambda a: N[a])
        search_end()
        return [best_a]
