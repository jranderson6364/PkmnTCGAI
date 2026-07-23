"""Sham-search PLACEBO agent — the P2 measurement-integrity control.

battlecore-compact-agent (846.8) reported that in a shared process, agent-side
`search_begin`/`search_step` calls perturb the live engine's RNG stream, so a
placebo arm that ran every search and DISCARDED the result read 0.450 vs an
expected 0.480. Our harness runs both agents + the engine in one process
(harness.py), so the same confound may sit under every search result this
project has measured — including the 2026-07-23 isolation finding that
alakazam_v9's edge is "search-associated" (+15pp with search on).

This agent plays our frozen v30 heuristic EXACTLY, but on every MAIN decision it
first runs a throwaway determinized search (search_begin + a short greedy rollout
+ search_end), consumes whatever RNG that consumes, then throws it away and
returns the heuristic's own pick. It is behaviourally identical to plain
frozen_main_v30 BY CONSTRUCTION.

Test: A/B this vs plain frozen_main_v30, seats alternated, n>=400. Expected under
a clean harness: 50%. A reading whose CI excludes 50% proves the search calls
alone move outcomes -> our in-process search measurements are contaminated.

SEARCH_STEPS / N_DET are set to roughly match alakazam_v9's search workload so
the RNG perturbation is comparable in magnitude to the isolation experiment.
"""
import os
import sys
import glob
import importlib.util

for _pat in ['/kaggle/input/**/cg-lib', '/kaggle/input/cg-lib']:
    _paths = glob.glob(_pat, recursive=True)
    if _paths:
        sys.path.insert(0, _paths[0]); break
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "training", "local_cg"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# our shipped heuristic (the actual policy)
_spec = importlib.util.spec_from_file_location(
    "placebo_heuristic", os.path.join(_REPO, "training", "wsearch", "frozen_main_v30.py"))
_H = importlib.util.module_from_spec(_spec)
sys.modules["placebo_heuristic"] = _H
_spec.loader.exec_module(_H)

try:
    from cg.api import to_observation_class, search_begin, search_step, search_end
    _SEARCH_OK = True
except Exception:
    _SEARCH_OK = False

import random as _random

N_DET = 3          # determinizations per decision (matches alakazam_v9's N_DET)
SEARCH_STEPS = 20  # greedy substeps per determinization (a short rollout)
DECK = list(_H.DECK)
_FILL = _H.DECK[0] if _H.DECK else 1


def _filler(n, pool):
    if n <= 0:
        return []
    return [pool[_random.randrange(len(pool))] for _ in range(n)]


def _throwaway_search(obs_dict):
    """Run a determinized search and DISCARD it — pure RNG-consuming placebo."""
    if not _SEARCH_OK:
        return
    try:
        observation = to_observation_class(obs_dict)
        state = observation.current
        our_seat = state.yourIndex
        my_p = state.players[our_seat]
        opp_p = state.players[1 - our_seat]
        for _ in range(N_DET):
            zones = (
                _filler(my_p.deckCount, _H.DECK),
                _filler(len(my_p.prize), _H.DECK),
                _filler(opp_p.deckCount, _H.DECK),
                _filler(len(opp_p.prize), _H.DECK),
                _filler(opp_p.handCount, _H.DECK),
                _filler(1, _H.DECK) if (opp_p.active and opp_p.active[0] is None) else [],
            )
            ss = search_begin(observation, *zones, manual_coin=False)
            sid = ss.searchId
            for _ in range(SEARCH_STEPS):
                o = ss.observation
                if o.current.result is not None and o.current.result >= 0:
                    break
                sel = o.select
                if sel is None or not sel.option:
                    break
                k = max(1, sel.minCount or 1)
                ss = search_step(sid, list(range(min(k, len(sel.option)))))
                if getattr(ss, "state", ss) is None:
                    break
        search_end()
    except Exception:
        try:
            search_end()
        except Exception:
            pass


def agent(obs_dict: dict):
    sel = obs_dict.get("select")
    # Run the throwaway search ONLY on real MAIN decisions (context 0), matching
    # when a search agent would actually search, then discard and play heuristic.
    if sel and sel.get("context") == 0 and sel.get("option"):
        _throwaway_search(obs_dict)
    return _H.agent(obs_dict)
