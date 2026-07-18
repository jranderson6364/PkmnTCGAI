"""Scripted Marnie's Grimmsnarl ex opponent — built 2026-07-18 from 77-game
replay evidence (docs/report-log.md 2026-07-18 grimmsnarl entries).

Live ladder grimmsnarl bots beat our champion 74-75% in both eras; no
existing offline opponent wins >6% against it. Purpose: (1) gate opponent
for grimmsnarl-matchup fixes, (2) the first candidate DISCRIMINATING offline
opponent for learned arms. Local training use only, never shipped.

Core loop it expresses: Impidimp -> Morgrem/Rare Candy -> Grimmsnarl ex
(Punk Up: fetch+attach up to 5 Basic {D} on evolve), Shadow Bullet 180+30
snipe (our whole Alakazam line is Dark-weak), Munkidori Adrena-Brain
counter-shuttling, Xerosic's Machinations when our hand grows, Boss's
Orders on KO-able targets. Framework follows opponents/lucario_agent.py
(official sample agent shape).
"""
import glob
import os
import sys
from collections import defaultdict

for _p in ['/kaggle/input/**/cg-lib', '/kaggle/input/cg-lib']:
    _m = glob.glob(_p, recursive=True)
    if _m:
        sys.path.insert(0, _m[0])
        break
else:
    _local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'training', 'local_cg')
    if os.path.isdir(_local):
        sys.path.insert(0, _local)

try:
    from cg.api import (AreaType, CardType, EnergyType, SelectContext, OptionType,
                        Pokemon, all_card_data, to_observation_class)
    card_table = {c.cardId: c for c in all_card_data()}
except Exception:
    card_table = {}

    class _Stub:
        def __getattr__(self, n):
            return None
    AreaType = SelectContext = OptionType = EnergyType = CardType = _Stub()
    to_observation_class = lambda x: x  # noqa: E731

IMPIDIMP, MORGREM, GRIMMSNARL = 646, 647, 648
MUNKIDORI, SNORUNT, FROSLASS = 112, 860, 104
POFFIN, POKE_PAD, RARE_CANDY, NIGHT_STRETCHER = 1086, 1152, 1079, 1097
BOSS, LILLIE, PETREL, XEROSIC, SPIKEMUTH = 1182, 1227, 1219, 1197, 1259
DARK_E = 7
SHADOW_BULLET = 937
LINE = (IMPIDIMP, MORGREM, GRIMMSNARL)

DECK = ([IMPIDIMP] * 4 + [MORGREM] * 3 + [GRIMMSNARL] * 4 + [MUNKIDORI] * 3 +
        [SNORUNT] * 2 + [FROSLASS] * 2 +
        [POFFIN] * 4 + [POKE_PAD] * 4 + [RARE_CANDY] * 3 + [NIGHT_STRETCHER] * 2 +
        [BOSS] * 3 + [LILLIE] * 4 + [PETREL] * 2 + [XEROSIC] * 2 + [SPIKEMUTH] * 3 +
        [DARK_E] * 15)
assert len(DECK) == 60, f"grimmsnarl deck has {len(DECK)} cards"


def get_card(obs, area, index, player_index):
    ps = obs.current.players[player_index]
    if area == AreaType.DECK:
        return obs.select.deck[index]
    if area == AreaType.HAND:
        return ps.hand[index]
    if area == AreaType.DISCARD:
        return ps.discard[index]
    if area == AreaType.ACTIVE:
        return ps.active[index]
    if area == AreaType.BENCH:
        return ps.bench[index]
    if area == AreaType.PRIZE:
        return ps.prize[index]
    if area == AreaType.STADIUM:
        return obs.current.stadium[index]
    if area == AreaType.LOOKING:
        return obs.current.looking[index]
    return None


def _dmg_vs(base, target):
    data = card_table.get(getattr(target, "id", None))
    if data is not None and getattr(data, "weakness", None) == EnergyType.DARKNESS:
        return base * 2
    return base


def _energy_need(pk):
    """How much this Pokemon wants energy (Grimmsnarl line to 2, Munkidori to 1)."""
    n = len(pk.energies)
    if pk.id == GRIMMSNARL:
        return 300 if n < 2 else -50
    if pk.id in (MORGREM, IMPIDIMP):
        return 150 if n < 2 else -50
    if pk.id == MUNKIDORI:
        return 200 if n < 1 else -80
    return -10


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return DECK

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    me = state.players[my_index]
    op = state.players[1 - my_index]

    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    for c in list(me.active) + list(me.bench):
        if c is not None:
            field_counts[c.id] += 1
    for c in me.hand:
        hand_counts[c.id] += 1

    stadium_id = 0
    for c in state.stadium:
        stadium_id = c.id

    # best damage our active could deal this turn (Shadow Bullet only)
    active = me.active[0] if me.active else None
    armed = active is not None and active.id == GRIMMSNARL and len(active.energies) >= 2
    bench_armed = any(c is not None and c.id == GRIMMSNARL and len(c.energies) >= 2
                      for c in me.bench)

    def target_score(pk):
        """Value of pointing our attack at pk (Boss target / their promotion)."""
        if pk is None:
            return -1
        s = len(pk.energies) * 30
        if armed and pk.hp <= _dmg_vs(180, pk):
            s += 500 - pk.hp // 10
        return s

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            score = 1  # accept Punk Up and friends
        elif o.type == OptionType.ABILITY:
            score = 30000  # Adrena-Brain etc.
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table.get(card.id)
            if data is not None and data.cardType == CardType.POKEMON:
                score = 20000
                if card.id in (SNORUNT, FROSLASS) and field_counts[SNORUNT] + field_counts[FROSLASS] >= 2:
                    score = 100
            elif card.id == RARE_CANDY:
                score = 10000 if (field_counts[IMPIDIMP] and hand_counts[GRIMMSNARL]) else -1
            elif card.id == POFFIN:
                score = 8500
            elif card.id == POKE_PAD:
                score = 8000
            elif card.id == NIGHT_STRETCHER:
                score = 4000
            elif card.id == SPIKEMUTH:
                score = -1 if stadium_id == SPIKEMUTH else 5000
            elif card.id == XEROSIC:
                score = 3300 if op.handCount >= 7 else -1
            elif card.id == BOSS:
                score = 3200 if any(target_score(c) >= 400 for c in op.bench) else -1
            elif card.id == LILLIE:
                score = 3100 if len(me.hand) <= 4 else -1
            elif card.id == PETREL:
                score = 3000 if len(me.hand) <= 6 else -1
            else:
                score = 500
        elif o.type == OptionType.EVOLVE:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            score = 9000
            if card.id == GRIMMSNARL:
                score = 9500  # Punk Up
            elif card.id == MORGREM:
                score = 9200
            elif card.id == FROSLASS:
                score = 9050
        elif o.type == OptionType.ATTACH:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if pokemon is not None:
                score = 7000 + _energy_need(pokemon)
                if o.inPlayArea == AreaType.ACTIVE:
                    score += 20
        elif o.type == OptionType.RETREAT:
            score = 2000 if (not armed and bench_armed) else -1
        elif o.type == OptionType.ATTACK:
            score = 1000
            if o.attackId == SHADOW_BULLET:
                score += 100
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is None:
                score = 0
            elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                score = {IMPIDIMP: 5, SNORUNT: 3, MUNKIDORI: 2}.get(card.id, 1)
            elif context == SelectContext.TO_HAND:
                score = 100 - hand_counts[card.id] * 40
                if card.id == GRIMMSNARL:
                    score += 90 if (field_counts[MORGREM] or hand_counts[RARE_CANDY]) else 30
                elif card.id == MORGREM:
                    score += 70 if field_counts[IMPIDIMP] else 10
                elif card.id == IMPIDIMP:
                    score += 60 if field_counts[IMPIDIMP] < 2 else 5
                elif card.id == RARE_CANDY:
                    score += 80
                elif card.id == BOSS:
                    score += 50
                elif card.id == XEROSIC:
                    score += 45
                elif card.id == MUNKIDORI:
                    score += 30 if not field_counts[MUNKIDORI] else -20
                elif card.id == DARK_E:
                    score += 20
            elif o.playerIndex == my_index:
                # own-side pick (promotion, Punk Up attach targets, etc.)
                if isinstance(card, Pokemon):
                    score = 50 + _energy_need(card) + len(card.energies) * 10
                    if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
                        score = len(card.energies) * 20
                        if card.id == GRIMMSNARL and len(card.energies) >= 2:
                            score += 200
                        elif card.id == MUNKIDORI:
                            score += 30
                elif card.id == DARK_E:
                    score = 100  # Punk Up / any energy fetch: take them all
                else:
                    score = 10
            else:
                # opponent-side pick (Boss's Orders target)
                score = target_score(card) if isinstance(card, Pokemon) else 0
        scores.append(score)

    desc = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    return desc[:select.maxCount]
