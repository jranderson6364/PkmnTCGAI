"""
Dragapult ex training opponent — official Kaggle sample agent.
Source: kiyotah/a-sample-rule-based-agent-dragapult-ex-deck

Archetype: Stage 2 spread. Phantom Dive: 200 + 6 damage counters on opponent bench.
Dragapult ex: 320 HP, Stage 2 ex, Tera Dragon (bench protection while benched).
"""
import sys, glob
from collections import defaultdict

for _p in ['/kaggle/input/**/cg-lib', '/kaggle/input/cg-lib']:
    _m = glob.glob(_p, recursive=True)
    if _m: sys.path.insert(0, _m[0]); break

try:
    from cg.api import AreaType, CardType, Log, LogType, Observation, SelectContext, OptionType, Card, Pokemon, State, all_card_data, to_observation_class
    all_card = all_card_data()
    card_table = {c.cardId: c for c in all_card}
except Exception:
    card_table = {}
    class _Stub:
        def __getattr__(self, n): return None
    AreaType = SelectContext = OptionType = CardType = LogType = _Stub()
    to_observation_class = lambda x: x
    # real (empty) classes, not _Stub instances: these names are used in
    # isinstance() checks at runtime (e.g. add_card_count), which need types —
    # their absence crashed 100% of local games as NameError: 'Pokemon'
    class Pokemon: pass
    class Card: pass
    class State: pass
    class Log: pass
    class Observation: pass

# ── Deck (60 cards, embedded) ─────────────────────────────────────────────────
Dreepy               = 119   # ×4
Drakloak             = 120   # ×4
Dragapult_ex         = 121   # ×3
Fezandipiti_ex       = 140   # ×1
Latias_ex            = 184   # ×1
Budew                = 235   # ×2
Meowth_ex            = 1071  # ×1
Rare_Candy           = 1079  # ×2
Unfair_Stamp         = 1080  # ×1
Buddy_Buddy_Poffin   = 1086  # ×4
Night_Stretcher      = 1097  # ×2
Crushing_Hammer      = 1120  # ×4
Ultra_Ball           = 1121  # ×4
Poke_Pad             = 1152  # ×3
Lucky_Helmet         = 1156  # ×1
Boss_Orders          = 1182  # ×3
Crispin              = 1198  # ×4
Brock_Scouting       = 1210  # ×2
Lillie_Determination = 1227  # ×4
Team_Rocket_Watchtower = 1256  # ×2
Basic_Fire_Energy    = 2     # ×4
Basic_Psychic_Energy = 5     # ×4

DECK = (
    [Dreepy] * 4 + [Drakloak] * 4 + [Dragapult_ex] * 3 +
    [Fezandipiti_ex] + [Latias_ex] + [Budew] * 2 + [Meowth_ex] +
    [Rare_Candy] * 2 + [Unfair_Stamp] + [Buddy_Buddy_Poffin] * 4 +
    [Night_Stretcher] * 2 + [Crushing_Hammer] * 4 + [Ultra_Ball] * 4 +
    [Poke_Pad] * 3 + [Lucky_Helmet] + [Boss_Orders] * 3 +
    [Crispin] * 4 + [Brock_Scouting] * 2 + [Lillie_Determination] * 4 +
    [Team_Rocket_Watchtower] * 2 + [Basic_Fire_Energy] * 4 + [Basic_Psychic_Energy] * 4
)
assert len(DECK) == 60, f"Dragapult deck has {len(DECK)} cards, expected 60"

UNNECESSARY = -10000000

class AttackPlan:
    attack: int = 0
    counter: list = []

can_switch = False
can_attack = False
can_main_attack = False
can_energy_attach = False
use_support = 0
bench_attacker = False
pre_turn_log = []
current_turn_log = []

prize = []
card_counts = defaultdict(int)
serial_set = set()
plan_a = AttackPlan()
plan_b = AttackPlan()


def no_damage_dex(pid):
    # Drednaw, Milotic ex, Sylveon, Crustle
    return pid in (158, 207, 330, 345)


def no_damage_counter(pokemon):
    if pokemon.id in (28, 199, 203, 207, 362, 1136):
        return True
    for card in pokemon.energyCards:
        if card.id in (11, 20):  # Mist Energy, Rock Fighting Energy
            return True
    return False


def prize_count(pokemon, is_attack_damage):
    data = card_table.get(pokemon.id)
    if not data: return 1
    count = 3 if data.megaEx else 2 if data.ex else 1
    if is_attack_damage:
        for card in pokemon.energyCards:
            if card.id == 12: count -= 1
        for card in pokemon.tools:
            if card.id == 1172 and "Lillie" in data.name: count -= 1
    return max(0, count)


def pokemon_score(pokemon, is_attack_damage):
    data = card_table.get(pokemon.id)
    if not data: return 0
    score = prize_count(pokemon, is_attack_damage) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:   score += 250
    elif data.stage1: score += 130
    pid = pokemon.id
    if pid in (173, 174, 190, 1071): score -= 200
    if pid == 112 and len(pokemon.energies) >= 1: score += 300
    score += pokemon.hp
    return score


def add_card_count(card, my_index):
    if card is None: return
    if isinstance(card, Pokemon) or card.playerIndex == my_index:
        if card.serial not in serial_set:
            card_counts[card.id] -= 1
            serial_set.add(card.serial)
    if isinstance(card, Pokemon):
        for c in card.energyCards: add_card_count(c, my_index)
        for c in card.tools:       add_card_count(c, my_index)
        for c in card.preEvolution: add_card_count(c, my_index)


def set_card_counts(obs, my_index):
    card_counts.clear()
    serial_set.clear()
    for pid in DECK:
        card_counts[pid] += 1
    state = obs.current
    my_state = state.players[my_index]
    for card in my_state.hand:    add_card_count(card, my_index)
    for card in my_state.discard: add_card_count(card, my_index)
    for card in my_state.bench:   add_card_count(card, my_index)
    for card in my_state.active:  add_card_count(card, my_index)
    for card in state.stadium:    add_card_count(card, my_index)
    if state.looking is not None:
        for card in state.looking: add_card_count(card, my_index)
    add_card_count(obs.select.effect, my_index)


def get_card(obs, area, index, player_index):
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:    return obs.select.deck[index]
        case AreaType.HAND:    return ps.hand[index]
        case AreaType.DISCARD: return ps.discard[index]
        case AreaType.ACTIVE:  return ps.active[index]
        case AreaType.BENCH:   return ps.bench[index]
        case AreaType.PRIZE:   return ps.prize[index]
        case AreaType.STADIUM: return obs.current.stadium[index]
        case AreaType.LOOKING: return obs.current.looking[index]
        case _:                return None


def main_option_proc(obs, damage):
    state = obs.current
    select = obs.select
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    global can_switch, can_attack, can_main_attack, can_energy_attach
    can_switch = False
    can_attack = False
    can_main_attack = False
    can_energy_attach = False
    for o in select.option:
        if o.type == OptionType.RETREAT:     can_switch = True
        elif o.type == OptionType.ATTACK:
            can_attack = True
            if o.attackId == 154: can_main_attack = True  # Phantom Dive

    plan_a.attack = -1
    plan_b.attack = -1
    if not can_main_attack and not (bench_attacker and can_switch):
        return

    cards = [op_state.active[0]] + list(op_state.bench)
    counter_indices = []
    ci = [0]
    remain_damage = 60
    while ci:
        index = ci[-1]
        hp = cards[index].hp
        if remain_damage >= hp:
            counter_indices.append(ci.copy())
            if index < len(cards) - 1:
                remain_damage -= hp
                ci.append(index + 1)
                continue
        if index == len(cards) - 1:
            ci.pop()
            if ci: remain_damage += cards[ci[-1]].hp
        if ci: ci[-1] += 1
    counter_indices.append([])

    remain_prize = len(my_state.prize)
    plan_score = 0
    for i, pokemon in enumerate(cards):
        base_prize_count = 0
        base_score = pokemon_score(pokemon, True)
        active_damage = 0 if no_damage_dex(pokemon.id) else damage
        if pokemon.hp <= active_damage:
            base_prize_count += prize_count(pokemon, True)
        else:
            base_score *= active_damage / pokemon.hp
        ci = []
        max_score = base_score
        if remain_prize <= base_prize_count:
            max_score = 50000
        else:
            for indices in counter_indices:
                if i in indices: continue
                p = base_prize_count
                s = base_score
                for idx in indices:
                    p += prize_count(cards[idx], False)
                    s += pokemon_score(cards[idx], False)
                if remain_prize <= p:
                    s = 50000
                else:
                    if p >= 2:
                        if remain_prize <= 4: s -= 1200
                    elif p == 1:
                        s -= 300
                    else:
                        s += 1200
                if max_score < s:
                    max_score = s
                    ci = indices
        if plan_score < max_score:
            plan_score = max_score
            plan_a.attack = i
            plan_a.counter = ci
        if i == 0:
            plan_b.attack = plan_a.attack
            plan_b.counter = plan_a.counter


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return DECK

    global pre_turn_log, current_turn_log

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    if state.turn == 0:
        prize.clear()
        pre_turn_log.clear()
        current_turn_log.clear()
    else:
        for log in obs.logs:
            current_turn_log.append(log)
            if log.type == LogType.TURN_END:
                pre_turn_log[:] = current_turn_log
                current_turn_log.clear()

    pre_ko = False
    no_item = False
    for log in pre_turn_log:
        if log.type == LogType.ATTACK:
            if log.attackId == 323: no_item = True  # Itchy Pollen
        elif log.type == LogType.MOVE_CARD:
            if (log.playerIndex == my_index
                    and log.fromArea in (AreaType.BENCH, AreaType.ACTIVE)
                    and log.toArea == AreaType.DISCARD):
                pre_ko = True

    if select.deck is not None:
        set_card_counts(obs, my_index)
        for card in select.deck:
            card_counts[card.id] -= 1
        prize.clear()
        for pid in card_counts:
            for _ in range(card_counts[pid]):
                prize.append(pid)

    set_card_counts(obs, my_index)
    for pid in prize:
        card_counts[pid] -= 1
    deck_counts = card_counts

    prize_diff = len(my_state.prize) - len(op_state.prize)

    global bench_attacker
    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    discard_counts = defaultdict(int)

    active_id = 0
    bench_attacker = False
    can_evolve_dreepy = False
    evolve_dreepy_count = 0
    can_evolve_drakloak = False
    damage = 200
    for card in my_state.active:
        if card is None: continue
        active_id = card.id
        field_counts[card.id] += 1
        if not card.appearThisTurn:
            if card.id == Dreepy:
                can_evolve_dreepy = True
                evolve_dreepy_count += 1
            elif card.id == Drakloak:
                can_evolve_drakloak = True
    for card in my_state.bench:
        field_counts[card.id] += 1
        if not card.appearThisTurn:
            if card.id == Dreepy:
                can_evolve_dreepy = True
                evolve_dreepy_count += 1
            elif card.id == Drakloak:
                can_evolve_drakloak = True
        if card.id == Dragapult_ex and len(card.energies) >= 2:
            bench_attacker = True
    main_pokemon_count = field_counts[Dreepy] + field_counts[Drakloak] + field_counts[Dragapult_ex]
    no_more_dex = (field_counts[Dragapult_ex] * 2 >= len(op_state.prize))

    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id

    support_count = 0
    for card in my_state.discard:
        discard_counts[card.id] += 1

    def attach_score(attach_id, pokemon, active):
        energy_count = len(pokemon.energies)
        data = card_table.get(attach_id)
        if data and data.cardType == CardType.TOOL:
            score = 60000
            if active: score += 1000
            return score
        if pokemon.id == Budew: return -1
        elif pokemon.id in (Meowth_ex, Fezandipiti_ex, Latias_ex):
            if active and not can_switch and not my_state.asleep and not my_state.paralyzed:
                return 22000 if bench_attacker or field_counts[Budew] >= 1 else 18000
            else: return -1
        if active and can_main_attack: return -1
        score = 20000
        if energy_count >= 2:
            if active and not can_switch and not my_state.asleep and not my_state.paralyzed:
                score += 200
            else: return -1
        elif energy_count == 1:
            if attach_id == pokemon.energyCards[0].id: return -1
            if pokemon.id == Dragapult_ex:   score += 250
            elif pokemon.id == Dreepy:        score -= 150
            else:                             score -= 200
            if active: score += 200
        else:
            if active:
                if bench_attacker: score += 400
            else:
                if pokemon.id == Dragapult_ex:   score += 150
                elif pokemon.id == Dreepy:        score += 100
                else:                             score += 50
                if bench_attacker: score -= 200
        if no_more_dex and pokemon.id in (Dreepy, Drakloak):
            score -= 500
        return score

    def hand_score(pid, ignore_count):
        score = 0
        if pid == Dreepy:
            score = 1000 if main_pokemon_count >= 3 else 18000
        elif pid == Drakloak:
            score = 20000 if can_evolve_dreepy else 3000
        elif pid == Dragapult_ex:
            if no_more_dex: return UNNECESSARY
            elif can_evolve_dreepy and hand_counts[Rare_Candy] >= 1 and not no_item:
                score = 40000
            elif can_evolve_drakloak:
                if field_counts[pid] == 0:   score = 30000
                elif field_counts[pid] == 1: score = 10000
                else:                        score = 50
            else:
                score = 50 if field_counts[pid] >= 2 else 2000
        elif pid == Fezandipiti_ex:
            if pre_ko:             score = 50000
            elif prize_diff <= -2: score = 5
            elif len(op_state.prize) == 1: return UNNECESSARY
        elif pid == Latias_ex:
            if active_id in (Fezandipiti_ex, Meowth_ex, Dreepy):
                score = 28000 if field_counts[Drakloak] + field_counts[Dragapult_ex] == 0 else 15000
            else:
                score = 10
        elif pid == Budew:
            if field_counts[pid] + field_counts[Drakloak] + field_counts[Dragapult_ex] >= 1:
                return UNNECESSARY
            score = 30000 if state.turn >= 2 else 0
        elif pid == Meowth_ex:
            if support_count > hand_counts[Boss_Orders] or stadium_id == Team_Rocket_Watchtower:
                score = 5
            elif state.supporterPlayed: score = 40
            else:                       score = 35000
        elif pid == Rare_Candy:
            if no_more_dex: return UNNECESSARY
            elif can_evolve_dreepy and hand_counts[Dragapult_ex] >= 1: score = 40000
        elif pid == Unfair_Stamp:
            if pre_ko:                      score = 80000
            elif len(op_state.prize) == 1:  return UNNECESSARY
            else:                           score = 80
        elif pid == Buddy_Buddy_Poffin:
            count = deck_counts[Dreepy]
            if count == 0: return UNNECESSARY
            if state.turn <= 2 and field_counts[Budew] == 0 and deck_counts[Budew] >= 1:
                count += 1
            score = 35000 if count >= 2 else 0
        elif pid == Night_Stretcher:
            for i in discard_counts:
                if discard_counts[i] >= 1:
                    ct = card_table.get(i)
                    if ct and ct.cardType in (CardType.POKEMON, CardType.BASIC_ENERGY):
                        score = max(score, hand_score(i, ignore_count))
        elif pid == Crushing_Hammer:  score = 20
        elif pid == Ultra_Ball:
            score = 70 if main_pokemon_count <= 2 or field_counts[Dreepy] >= 1 else 5
        elif pid == Poke_Pad:
            score = max(hand_score(Dreepy, ignore_count), hand_score(Drakloak, ignore_count))
        elif pid == Lucky_Helmet: score = 15
        elif pid == Boss_Orders:
            if plan_a.attack > 0: score = 60000
        elif pid == Crispin:
            if not ignore_count or support_count == 0:
                if deck_counts[Basic_Fire_Energy] == 0 or deck_counts[Basic_Psychic_Energy] == 0:
                    score = 10
                if not can_main_attack and not bench_attacker and field_counts[Dragapult_ex] >= 1:
                    score = 55000
                else: score = 25000
        elif pid == Brock_Scouting:
            if not ignore_count or support_count == 0:
                if state.turn == 2 and field_counts[Budew] + field_counts[Latias_ex] == 0:
                    score = 50000
                else: score = 30000
        elif pid == Lillie_Determination:
            if not ignore_count or support_count == 0: score = 45000
        elif pid == Team_Rocket_Watchtower:
            if stadium_id != 0 and stadium_id != Team_Rocket_Watchtower: score = 4000
        elif pid in (Basic_Fire_Energy, Basic_Psychic_Energy):
            if can_main_attack and (len(op_state.prize) <= 2
                    or (bench_attacker and len(op_state.prize) <= 4)):
                return UNNECESSARY
            else:
                max_sc = -10000
                for pokemon in my_state.active:
                    if pokemon is None: continue
                    max_sc = max(max_sc, attach_score(pid, pokemon, True))
                for pokemon in my_state.bench:
                    max_sc = max(max_sc, attach_score(pid, pokemon, False))
                score = max_sc - 5000
                if can_main_attack or bench_attacker: score //= 10

        if not ignore_count and hand_counts[pid] > 0:
            if pid == Drakloak and hand_counts[pid] < evolve_dreepy_count: score -= 10
            elif pid == Dreepy: score -= 100
            else:               score -= 100000
        return score

    global use_support
    if context == SelectContext.MAIN:
        main_option_proc(obs, damage)
        use_support = 0
        if not state.supporterPlayed:
            support_score = 0
            for o in select.option:
                if o.type == OptionType.PLAY:
                    card = get_card(obs, AreaType.HAND, o.index, state.yourIndex)
                    ct = card_table.get(card.id)
                    if ct and ct.cardType == CardType.SUPPORTER:
                        s = hand_score(card.id, True)
                        if support_score < s:
                            support_score = s
                            use_support = card.id

    hand_scores = []
    negative_hand_count = 0
    for card in my_state.hand:
        s = hand_score(card.id, False)
        hand_scores.append(s)
        if s < 0: negative_hand_count += 1
        hand_counts[card.id] += 1
        ct = card_table.get(card.id)
        if ct and ct.cardType == CardType.SUPPORTER and card.id != Boss_Orders:
            support_count += 1

    no_draw = (my_state.deckCount <= 8)
    do_switch = (not can_main_attack and (bench_attacker
        or (active_id != Budew and field_counts[Budew] >= 1 and state.turn >= 2)))
    effect_card_id = 0 if select.effect is None else select.effect.id
    context_card_id = 0 if select.contextCard is None else select.contextCard.id

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            score = -1 if context == SelectContext.IS_FIRST else 1
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is not None:
                energy_count = 0
                hp = 0
                if isinstance(card, Pokemon):
                    energy_count = len(card.energies)
                    hp = card.hp
                if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE, SelectContext.SETUP_ACTIVE_POKEMON):
                    if o.playerIndex == my_index:
                        if card.id == Dreepy:          score += 10000
                        elif card.id == Drakloak:
                            score += 20000 if energy_count >= 1 else -10000
                        elif card.id == Dragapult_ex:  score += 50000
                        elif card.id == Budew:
                            score += 100000 if context != SelectContext.SWITCH else (30000 if not bench_attacker else 0)
                        elif card.id == Fezandipiti_ex: score -= 1000
                        elif card.id == Meowth_ex:      score -= 2000
                    else:
                        if plan_a.attack == o.index + 1: score += 100000
                    score += energy_count * 1000 + hp
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    score = -1 if my_index == state.firstPlayer or card.id != Dreepy else 0
                elif context in (SelectContext.TO_BENCH, SelectContext.TO_HAND):
                    score = hand_score(card.id, False)
                    hand_counts[card.id] += 1
                    if effect_card_id == Crispin:
                        score = 100000 - hand_score(card.id, True)
                elif context == SelectContext.DISCARD:
                    hand_counts[card.id] -= 1
                    ct = card_table.get(card.id)
                    if ct and ct.cardType == CardType.SUPPORTER: support_count -= 1
                    score = -hand_score(card.id, False)
                elif context in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
                    if hp > 0:
                        score = 100000 - 10 * hp + pokemon_score(card, False)
                        if context == SelectContext.DAMAGE_COUNTER:
                            if 210 <= hp <= 230:
                                score += 20000 + hp * 20
                                if o.area == AreaType.ACTIVE: score += 10000
                            elif 40 <= hp <= 90:  score += 10000 + hp * 20
                            elif hp <= 30:        score += -10000 + hp * 20
                            if card.id in (133, 351): score += 30000
                        else:
                            index = o.index + 1
                            if index in plan_b.counter: score += 100000
                            else:
                                remain_damage = select.remainDamageCounter * 10
                                if 210 <= hp <= 200 + remain_damage: score += 30000
                                elif 20 <= hp <= 60 + remain_damage: score += 10000
                                elif hp == 10:                       score -= 100000
                            if no_damage_counter(card): score = -1
                elif context == SelectContext.ATTACH_FROM:
                    score = attach_score(context_card_id, card, o.area == AreaType.ACTIVE)
                    if card.id == Dragapult_ex: score += 200
        elif o.type in (OptionType.ENERGY_CARD, OptionType.ENERGY):
            if o.playerIndex != state.yourIndex:
                score = 20 if o.area == AreaType.BENCH else 10
                card = get_card(obs, o.area, o.index, o.playerIndex)
                ct = card_table.get(card.id)
                if ct and ct.cardType == CardType.SPECIAL_ENERGY: score += 1
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            card_score = hand_scores[o.index]
            if card.id == Dreepy:           score = 51000
            elif card.id == Fezandipiti_ex: score = 53000 if card_score > 0 else -1
            elif card.id == Latias_ex:
                score = 51000 if active_id not in (Drakloak, Dragapult_ex) else -1
            elif card.id == Budew:
                score = 52000 if field_counts[Budew] == 0 and field_counts[Dragapult_ex] == 0 else -1
            elif card.id == Meowth_ex:
                if state.supporterPlayed or stadium_id == Team_Rocket_Watchtower: score = -1
                elif support_count == 0:                                            score = 50000
                elif support_count == hand_counts[Boss_Orders] and plan_a.attack > 0: score = 50000
                else:                                                                   score = -1
            elif card.id == Rare_Candy:
                score = -1 if no_more_dex else 75000
            elif card.id == Unfair_Stamp:   score = 15000
            elif card.id == Night_Stretcher:
                score = 42000 if card_score >= 18000 else -1
            elif card.id == Crushing_Hammer: score = 40000
            elif card.id == Boss_Orders:
                score = 35000 if card.id == use_support else -1
            elif card.id == Lillie_Determination:
                score = 14000 if card.id == use_support else -1
            elif card.id == Team_Rocket_Watchtower:
                score = 80000 if stadium_id > 0 or state.turn == 1 else -1
            elif no_draw:
                score = -1
            elif card.id == Buddy_Buddy_Poffin:
                score = 46000 if deck_counts[Dreepy] > 0 else -1
            elif card.id == Ultra_Ball:
                score = 44000 if negative_hand_count >= 2 else -1
            elif card.id == Poke_Pad:
                score = 45000 if deck_counts[Dreepy] + deck_counts[Drakloak] > 0 else -1
            elif card.id in (Crispin, Brock_Scouting):
                score = 35000 if card.id == use_support else -1
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = attach_score(card.id, pokemon, o.inPlayArea == AreaType.ACTIVE)
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score += len(pokemon.energies)
            if pokemon.id == Dreepy:
                score += 30000
            elif field_counts[Dragapult_ex] >= 2 or (field_counts[Dragapult_ex] == 1 and len(op_state.prize) <= 2):
                score = -1
            else:
                score += 70000
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if no_draw:       score = -1
            elif card.id == 1267: score = 1  # Lumiose City
            else:             score = 40000
        elif o.type == OptionType.RETREAT:
            score = 10000 if do_switch else -1
        elif o.type == OptionType.ATTACK:
            score = o.attackId

        scores.append(score)

    output = []
    if scores:
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        for i in range(select.maxCount):
            if (sorted_scores[i][1] >= 0
                    or select.minCount > i
                    or context not in (SelectContext.TO_BENCH, SelectContext.SETUP_BENCH_POKEMON)):
                output.append(sorted_scores[i][0])
    return output
