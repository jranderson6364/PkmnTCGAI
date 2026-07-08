"""Weighted real-meta opponent pool for Phase 2 self-play collection
(mcts_collect.py), replacing the too-weak same-checkpoint-vs-itself mirror
that caused severe win/loss label imbalance and collapsed the round-2 value
head (2026-07-07 — see docs/report-log.md "Root cause of the Phase 2 round 2
regression"). The searching side (mcts_leafeval_agent.py) needs opponents it
can actually lose to sometimes; a diverse, real-meta-weighted pool of the
project's existing rule-based archetype bots is a much stronger and more
representative source of that than the collecting net's own unsearched
policy.

Weights are the real ladder meta share (`tools/meta_survey.py --all`, 1595
replays, 2026-07-07): lucario 21.7%, alakazam-mirror 11.9%, dragapult 10.7%,
starmie 9.8%, crustle 8.7%, archaludon 6.6%, abomasnow 5.3%, grimmsnarl 3.5%,
bellibolt 1.3%, rockets-mewtwo 0.9%, kyogre 0.9%, raging-bolt 0.5%,
gardevoir 0.3%, other/unknown 18.1%. "other/unknown" has no reconstructable
deck/pilot and is dropped rather than fabricated; the rest is renormalized
to sum to 1. "kyogre" is also dropped despite having a real archetype-share
entry: its reconstructed decklist (training/archetype_decks.json, built
from only 13 replays' evidence) has just 30/60 card copies — too sparse to
reconstruct into a legal deck, so playing it would need fabricating the
other half.

Four archetypes (lucario, dragapult, abomasnow, starmie) are official Kaggle
sample bots with their own real decklist AND real piloting logic already in
opponents/*_agent.py — used as-is (deck=None -> the module's own DECK).
"alakazam_mirror" uses the real heuristic (main.py, the actual shipped
piloting logic, not the half-trained net's own weak policy) piloting its
own deck -- both a stronger opponent AND a more faithful proxy for what a
competent human Alakazam mirror opponent on the real ladder looks like.
Everything else has only a reconstructed decklist, no dedicated pilot --
piloted by training/generic_pilot.py (deck-agnostic greedy heuristic),
exactly as the Stage 0c tier-2 bake-off used it.
"""
import json
import os
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_OPPONENTS_DIR = os.path.join(_REPO_ROOT, "opponents")
_GENERIC_PILOT = os.path.join(_REPO_ROOT, "training", "generic_pilot.py")
_MAIN_PY = os.path.join(_REPO_ROOT, "main.py")

with open(os.path.join(_REPO_ROOT, "training", "archetype_decks.json"), encoding="utf-8") as _f:
    _ARCHETYPE_DECKS_RAW = json.load(_f)


def _flatten_deck(cards):
    deck = []
    for c in cards:
        deck.extend([c["cardId"]] * c["copies"])
    return deck


# (label, agent_path, deck_or_None, real-meta-share weight)
_POOL_RAW = [
    ("lucario", os.path.join(_OPPONENTS_DIR, "lucario_agent.py"), None, 21.7),
    ("alakazam_mirror", _MAIN_PY, None, 11.9),
    ("dragapult", os.path.join(_OPPONENTS_DIR, "dragapult_agent.py"), None, 10.7),
    ("starmie", os.path.join(_OPPONENTS_DIR, "starmie_agent.py"), None, 9.8),
    ("crustle", _GENERIC_PILOT, _flatten_deck(_ARCHETYPE_DECKS_RAW["crustle"]), 8.7),
    ("archaludon", _GENERIC_PILOT, _flatten_deck(_ARCHETYPE_DECKS_RAW["archaludon"]), 6.6),
    ("abomasnow", os.path.join(_OPPONENTS_DIR, "abomasnow_agent.py"), None, 5.3),
    ("grimmsnarl", _GENERIC_PILOT, _flatten_deck(_ARCHETYPE_DECKS_RAW["grimmsnarl"]), 3.5),
    ("bellibolt", _GENERIC_PILOT, _flatten_deck(_ARCHETYPE_DECKS_RAW["bellibolt"]), 1.3),
    ("rockets-mewtwo", _GENERIC_PILOT, _flatten_deck(_ARCHETYPE_DECKS_RAW["rockets-mewtwo"]), 0.9),
    ("raging-bolt", _GENERIC_PILOT, _flatten_deck(_ARCHETYPE_DECKS_RAW["raging-bolt"]), 0.5),
    ("gardevoir", _GENERIC_PILOT, _flatten_deck(_ARCHETYPE_DECKS_RAW["gardevoir"]), 0.3),
]
for _label, _path, _deck, _w in _POOL_RAW:
    if _deck is not None:
        assert len(_deck) == 60, f"{_label} deck has {len(_deck)} cards, expected 60"

_TOTAL_WEIGHT = sum(w for _, _, _, w in _POOL_RAW)
POOL = [(label, path, deck, w / _TOTAL_WEIGHT) for label, path, deck, w in _POOL_RAW]


def allocate(n_games, seed=0):
    """Deterministically splits n_games across the pool by weight (largest-
    remainder method, so every archetype gets its exact rounded share and
    the total is always exactly n_games). Returns list of (label, agent_path,
    deck_or_None, n_this_opponent), skipping zero-allocation entries."""
    raw = [(label, path, deck, w * n_games) for label, path, deck, w in POOL]
    base = [(label, path, deck, int(n)) for label, path, deck, n in raw]
    remainder = n_games - sum(n for _, _, _, n in base)
    fracs = sorted(range(len(raw)), key=lambda i: raw[i][3] - base[i][3], reverse=True)
    counts = [n for _, _, _, n in base]
    for i in fracs[:remainder]:
        counts[i] += 1
    return [(POOL[i][0], POOL[i][1], POOL[i][2], counts[i])
            for i in range(len(POOL)) if counts[i] > 0]


def shuffled_assignments(n_games, seed=0):
    """Returns a length-n_games list of (label, agent_path, deck_or_None),
    one per game, shuffled so consecutive games don't cluster by archetype
    (matters for run_matches's progress reporting and for any downstream
    per-chunk analysis, not for correctness)."""
    out = []
    for label, path, deck, n in allocate(n_games, seed):
        out.extend([(label, path, deck)] * n)
    random.Random(seed).shuffle(out)
    return out
