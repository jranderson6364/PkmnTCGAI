"""Endgame-gated search agent (2026-07-07 training-methods session, item
"endgame search" — docs/report-log.md).

Plays main.agent (v25c heuristic) verbatim EXCEPT in endgame states —
either player <=2 prizes from winning — where it runs the belief-
determinized searcher instead. Rationale: the belief-ISMCTS closure showed
rollout leaf values are uniform("win") in MID-game states (no signal), but
in closing races real terminals sit within rollout reach and a wrong move
loses on the spot — precisely where rollout search CAN discriminate. The
loss mining (2026-07-05/07) also localizes v28's fixable losses to closing
races (desperation/deck-out), so the payoff state-space and the
signal-bearing state-space coincide.

ENDGAME_SIMS (default 60) and ENDGAME_PRIZES (default 2) control the gate.
Pre-registered bar (this is v25c + endgame search vs plain v25c, so any
real edge shows directly): >=55% over the first 50-game A/B to pursue,
kill below 50%'s CI.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ismcts_agent as _searchmod  # noqa: E402  (also sets up local_cg/belief paths)
from ismcts_agent import BeliefMCTSSearcher  # noqa: E402
from main import DECK, agent as heuristic_agent  # noqa: E402

_SIMS = int(os.environ.get("ENDGAME_SIMS", "60"))
_PRIZES = int(os.environ.get("ENDGAME_PRIZES", "2"))
_C_PUCT = float(os.environ.get("MCTS_C_PUCT", "1.4"))
_PRIOR_TEMP = float(os.environ.get("MCTS_PRIOR_TEMP", "2.0"))
# ExIt disagreement mining (2026-07-08, survey pick 4): when set, every
# searched decision where the search choice differs from the heuristic's
# is appended as a JSON line (state + options + both choices + visit
# counts + root value) to <ENDGAME_DISAGREE_LOG>.<pid>, for offline
# clustering into rule candidates — the replay-mining workflow with the
# confirmed-positive searcher as the teacher.
_DISAGREE_LOG = os.environ.get("ENDGAME_DISAGREE_LOG")

_fallback_count = 0
searched_decisions = 0  # introspection for the gate log


def _is_endgame(obs_dict):
    try:
        players = obs_dict["current"]["players"]
        lens = [len(p.get("prize") or []) for p in players]
        # 0 < min: before prizes are dealt (setup phase) both lists are
        # empty, which made the original gate fire on turn-0 placement
        # decisions in EVERY game (found 2026-07-08 via the disagreement
        # log — see report-log "setup-phase gate bug" entry).
        return 0 < min(lens) <= _PRIZES
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
        searcher = BeliefMCTSSearcher(sims=_SIMS, c_puct=_C_PUCT,
                                      prior_temp=_PRIOR_TEMP, leaf_eval="rollout")
        if _DISAGREE_LOG:
            out, visits, root_value = searcher.choose_with_stats(obs_dict)
            import main as _main
            _memo = dict(_main._STALL_MEMO)  # probe call must not perturb
            try:                             # the real game's stall state
                h = heuristic_agent(obs_dict)
            finally:
                _main._STALL_MEMO = _memo
            if out and h and list(out)[:1] != list(h)[:1]:
                import json
                with open(f"{_DISAGREE_LOG}.{os.getpid()}", "a") as f:
                    f.write(json.dumps({
                        "obs": obs_dict, "search": list(out),
                        "heuristic": list(h), "visits": visits,
                        "root_value": root_value}) + "\n")
        else:
            out = searcher.choose(obs_dict)
        searched_decisions += 1
        return out
    except Exception as e:
        _fallback_count += 1
        print(f"[endgame_agent] FALLBACK #{_fallback_count} to heuristic: {e!r}",
              file=sys.stderr)
        return heuristic_agent(obs_dict)
