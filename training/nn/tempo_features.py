"""Tempo (rate-of-change) features -- 2026-07-12, per a user-directed creative
push after the DMC data-scaling study closed negative (docs/report-log.md
2026-07-12 "Strategic reframe" entry): "functions that calculate...tempo or
other stuff" fed alongside game state.

Every existing hand-crafted feature in this project (Phi v2/v4, threat.py,
encode.py's census/belief groups) is a SNAPSHOT: a value computed from the
current board state alone. None of them use `turn` as anything more than a
single normalized scalar (turn/60). Real players' notion of "tempo" is a
RATE: hand size climbing faster than the opponent's, prize race pace, board
development curve -- comparisons that need turn count as a divisor, not just
another feature sitting next to it. A network CAN in principle learn these
ratios from the raw pieces (nonlinear combination of two existing inputs),
but the project's own eval-function research (Phi v2->v4) already found that
handing over a pre-computed calculated value beats making a small MLP
rediscover it from limited data -- the same rationale applies here.

All three features are antisymmetric (mine minus opponent's) so they flip
sign correctly under a seat swap, matching Phi v4's design discipline.
Deliberately just 3, atomic, not pre-combined into a composite "tempo score"
-- per CLAUDE.md's simplicity-first rule, let the network weight them rather
than hand-tuning a combination weight ourselves.

Takes already-extracted scalars (no cg.api/heuristic dependency) so this
stays cheap and unit-testable; encode.py computes them once and passes them
in.
"""
import numpy as np

TEMPO_DIM = 3


def _clip(x):
    return max(-1.0, min(1.0, x))


def tempo_features(my_hand_n, opp_hand_n, my_prizes_left, opp_prizes_left,
                    my_line_progress, opp_line_progress, turn):
    """Returns np.array of 3 antisymmetric rate features:
      0: prize-race pace diff -- who is taking prizes faster (win condition
         is reaching 0 prizes first, margin is irrelevant, so this projects
         race outcome rather than snapshot prize count).
      1: hand-growth pace diff -- hand size accumulated per turn so far;
         directly relevant since Powerful Hand's damage scales with hand
         size, so this approximates how fast each side is building lethal.
      2: setup-pace diff -- evolution-line progress per turn (development
         curve / "on-curve vs behind" -- a genuine tempo concept real
         players track that no existing feature captures).
    All divided by max(turn, 1) to get a per-turn rate, then clipped to
    [-1, 1] (early turns can spike before enough data exists to be
    meaningful; clipping saturates that noise rather than letting it
    dominate)."""
    t = max(turn, 1)
    prize_pace = ((opp_prizes_left - my_prizes_left) / t) / 2.0
    hand_pace = ((my_hand_n - opp_hand_n) / t) / 3.0
    setup_pace = (my_line_progress - opp_line_progress) / t
    return np.array([_clip(prize_pace), _clip(hand_pace), _clip(setup_pace)])
