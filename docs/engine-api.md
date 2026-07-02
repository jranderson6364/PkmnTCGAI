# cabt Engine API — Canonical Reference

*Compiled 2026-07-01 from https://matsuoinstitute.github.io/cabt/ (api.html, game.html)
and verified against live local games. This supersedes the partial/incorrect notes
that previously lived in CLAUDE.md.*

**Last updated:** 2026-07-01

---

## Where the engine lives

The engine ships **inside `kaggle_environments`** (`envs/cabt/cg/` with native
binaries: `cg.dll` Windows, `libcg.so` Linux, `libcg.dylib` macOS, arm64 variant).
`pip install kaggle_environments --no-deps` → `make("cabt", configuration={"decks":[d0,d1]})`
runs full games locally at ~0.5s/game. The richer `cg.api` (dataclasses,
`all_card_data`, `search_begin`) is in the kiyotah/cg-lib Kaggle dataset; the local
package has `cg.game` (battle functions) and `cg.sim`.

---

## Enums (verified numeric values)

**AreaType:** DECK=1, HAND=2, **DISCARD=3**, ACTIVE=4, BENCH=5, **PRIZE=6**, STADIUM=7,
ENERGY=8, TOOL=9, PRE_EVOLUTION=10, PLAYER=11, LOOKING=12
⚠️ Earlier project notes said "6=discard" — wrong. DISCARD=3, PRIZE=6. The v18
"prize-selection stall" select (`stype=1, ctx=7, area=6`) is literally *taking prize
cards* (context 7 = TO_HAND, area 6 = PRIZE).

**EnergyType:** COLORLESS=0, GRASS=1, FIRE=2, WATER=3, LIGHTNING=4, PSYCHIC=5,
FIGHTING=6, DARKNESS=7, METAL=8, DRAGON=9, RAINBOW=10, TEAM_ROCKET=11

**CardType:** POKEMON=0, ITEM=1, TOOL=2, SUPPORTER=3, STADIUM=4, BASIC_ENERGY=5, SPECIAL_ENERGY=6

**SpecialConditionType:** POISON=0, BURN=1, SLEEP=2, PARALYZE=3, CONFUSE=4

**SelectType (stype):** MAIN=0, CARD=1, ATTACHED_CARD=2, CARD_OR_ATTACHED_CARD=3,
ENERGY=4, SKILL=5, ATTACK=6, EVOLVE=7, COUNT=8, YES_NO=9, SPECIAL_CONDITION=10

**SelectContext (ctx):** MAIN=0, SETUP_ACTIVE_POKEMON=1, SETUP_BENCH_POKEMON=2,
SWITCH=3, TO_ACTIVE=4, TO_BENCH=5, TO_FIELD=6, TO_HAND=7, DISCARD=8, TO_DECK=9,
TO_DECK_BOTTOM=10, TO_PRIZE=11, NOT_MOVE=12, DAMAGE_COUNTER=13, DAMAGE_COUNTER_ANY=14,
DAMAGE=15, REMOVE_DAMAGE_COUNTER=16, HEAL=17, EVOLVES_FROM=18, EVOLVES_TO=19,
DEVOLVE=20, ATTACH_FROM=21, ATTACH_TO=22, DETACH_FROM=23, LOOK=24, EFFECT_TARGET=25,
DISCARD_ENERGY_CARD=26, DISCARD_TOOL_CARD=27, SWITCH_ENERGY_CARD=28,
DISCARD_CARD_OR_ATTACHED_CARD=29, DISCARD_ENERGY=30, TO_HAND_ENERGY=31,
TO_DECK_ENERGY=32, SWITCH_ENERGY=33, SKILL_ORDER=34, ATTACK=35, DISABLE_ATTACK=36,
EVOLVE=37, DRAW_COUNT=38, DAMAGE_COUNTER_COUNT=39, REMOVE_DAMAGE_COUNTER_COUNT=40,
IS_FIRST=41, MULLIGAN=42, ACTIVATE=43, FIRST_EFFECT=44, MORE_DEVOLVE=45,
COIN_HEAD=46, AFFECT_SPECIAL_CONDITION=47, RECOVER_SPECIAL_CONDITION=48

**OptionType:** NUMBER=0, YES=1, NO=2, CARD=3, TOOL_CARD=4, ENERGY_CARD=5, ENERGY=6,
PLAY=7, ATTACH=8, EVOLVE=9, ABILITY=10, DISCARD=11, RETREAT=12, ATTACK=13, END=14,
SKILL=15, SPECIAL_CONDITION=16

**LogType:** SHUFFLE=0, HAS_BASIC_POKEMON=1, TURN_START=2, TURN_END=3, DRAW=4,
DRAW_REVERSE=5, MOVE_CARD=6, MOVE_CARD_REVERSE=7, SWITCH=8, CHANGE=9, PLAY=10,
ATTACH=11, EVOLVE=12, DEVOLVE=13, MOVE_ATTACHED=14, ATTACK=15, HP_CHANGE=16,
POISONED=17, BURNED=18, ASLEEP=19, PARALYZED=20, CONFUSED=21, COIN=22, RESULT=23

---

## Data classes

**SelectData:** `type` (SelectType), `context` (SelectContext), `minCount`, `maxCount`,
`remainDamageCounter`, `remainEnergyCost`, `option` (list[Option]),
**`deck`** (list[Card] | None), **`contextCard`** (Card | None), **`effect`** (Card | None)

**Option:** `type`, `number`, `area`, `index`, `playerIndex`, `toolIndex`,
`energyIndex`, `count`, `inPlayArea`, `inPlayIndex`, `attackId`, `cardId`, `serial`,
`specialConditionType` — all optional.

**Pokemon:** `id`, `serial`, `hp` (remaining), `maxHp`, `appearThisTurn`,
`energies` (list[EnergyType]), `energyCards` (list[Card]), `tools` (list[Card]),
`preEvolution` (list[Card])

**PlayerState:** `active` (list of 0-1, None if face-down), `bench`, `benchMax`,
`deckCount`, `discard`, `prize` (face-down = None; first=bottom, last=top),
`handCount`, `hand` (None for opponent), `poisoned`, `burned`, `asleep`,
`paralyzed`, `confused`

**State:** `turn`, `turnActionCount`, `yourIndex`, `firstPlayer`, `supporterPlayed`,
`stadiumPlayed`, `energyAttached`, `retreated`, `result`, `stadium`,
`looking` (list[Card|None] | None), `players`

**Log:** `type`, `playerIndex`, `cardId`, `serial`, `fromArea`, `toArea`,
`cardIdActive/Bench/Before/After/Target` (+serials), `attackId`, `value`,
`putDamageCounter`, `isRecover`, `head`, `result`, `reason`, `hasBasicPokemon`

**Observation:** `select`, `logs`, `current`, `search_begin_input` (str | None —
prefilled input for search_begin)

**CardData** (via `all_card_data()`): `cardId`, `name`, `cardType`, `retreatCost`,
`hp`, `weakness`, `resistance`, `energyType`, `basic`, `stage1`, `stage2`, `ex`,
`megaEx`, `tera`, `aceSpec`, `evolvesFrom`, `skills`, `attacks`

**Attack** (via `all_attack()`): `attackId`, `name`, `text`, `damage`, `energies`

---

## Functions

```python
# game module (available locally in kaggle_environments)
battle_start(deck0: list[int], deck1: list[int]) -> tuple[dict | None, StartData]
battle_select(select_list: list[int]) -> dict
battle_finish() -> None            # frees native resources
visualize_data() -> str

# sim/api module (kiyotah/cg-lib dataset on Kaggle)
all_card_data() -> list[CardData]
all_attack() -> list[Attack]
to_observation_class(obs: dict) -> Observation
search_begin(agent_observation, your_deck, your_prize, opponent_deck,
             opponent_prize, opponent_hand, opponent_active,
             manual_coin: bool = False) -> SearchState
search_step(search_id: int, select: list[int]) -> SearchState
search_end() -> None
search_release(search_id: int) -> None
```

Note `manual_coin=True` — coin flips can be controlled during search (determinize
or enumerate both outcomes in MCTS).

---

## Verified runtime behaviors (from live local games, 2026-07-01)

1. **Deck searches are NOT blind.** For search effects (Poffin ctx=5, Poké Pad/Dawn
   ctx=7, etc.), `select.deck` lists deck cards and each option's `index` points
   into that list: `sel['deck'][o['index']]['id']` = the exact card. (The old
   "deck searches are blind / current.looking is null" note was wrong for these.)
2. **`select.effect` identifies the source card** of nearly every sub-selection
   (Boss=1182 for gust targets ctx=SWITCH, Enhanced Hammer=1081 for
   DISCARD_ENERGY, Rare Candy=1079 for ctx=EVOLVE, Poffin/Poké Pad for searches,
   Wondrous Patch for ATTACH_FROM/ATTACH_TO). `contextCard` gives the card being
   acted on (e.g. ACTIVATE prompts).
3. **`Option.cardId` is never populated in practice** (0 of 1,287 options across a
   full game — confirms the v7 finding). Setup selections (`stype=1, ctx=1/2`) are
   `{type:CARD, area:HAND, index}` → resolve through your own hand.
4. **Enhanced Hammer's energy pick** (stype=4, ctx=30) options carry
   `area/index/playerIndex/energyIndex` → resolve the exact energy card via the
   target pokemon's `energyCards[energyIndex]`.
5. **Rare Candy's target pick** (stype=7, ctx=37) options carry
   `inPlayArea/inPlayIndex` (which Abra) and `index` (hand card to evolve to).
6. First select of a game is `stype=9, ctx=41 (IS_FIRST)` — the go-first choice,
   asked before any hand exists.

---

## Search API — confirmed via a live Kaggle spike (2026-07-01)

`SearchState`'s shape and `search_begin`'s real parameter order were confirmed
directly from `kiyotah/cg-lib`'s `cg/api.py` source (downloaded and inspected
locally), not just this doc's prose:
```python
@dataclass
class SearchState:
    observation: Observation   # New observation. search_begin_input is None.
    searchId: int              # Search state ID — pass to search_step/search_release.
```
A throwaway Kaggle notebook (`training/kaggle_notebook/mcts-spike.ipynb`) then
confirmed the *runtime* behavior against a real mid-game decision (v22 vs
itself, turn 7, a MAIN-phase select with 7 options):

- **`search_begin` succeeds with count-matched filler hidden-zone args.**
  `your_deck`/`your_prize`/`opponent_deck`/`opponent_prize`/`opponent_hand`
  only need to match the real counts (`deckCount`, `len(prize)`, `handCount`)
  — exact card identity doesn't matter for the call to succeed. A mirror
  game's own `DECK` constant is legitimate filler for both sides.
- **`search_begin` re-roots the search at the exact same decision** it was
  passed — `ss.observation.select` came back byte-for-byte identical to the
  original real `select` (same type/context/minCount/maxCount and all 7
  options), not a different or advanced state.
- **The opponent's hand stays `None` in the returned observation**, even
  though a real `opponent_hand` placeholder list was passed in. The hidden-
  zone args feed the engine's internal search/sampling only — they are never
  leaked back into the visible observation. The imperfect-information
  boundary holds inside search exactly as it does in normal play.
- **`search_step` correctly advances through sub-decisions**: observed
  MAIN (stype=0) → a TO_DECK sub-select (stype=1, ctx=9) → back to MAIN with
  a shrunk option count as the board state changed. `State.yourIndex` and
  `State.result` (`-1` = game ongoing) behave exactly as in normal `agent()`
  observations.
- **Not yet exercised:** `manual_coin=True` was passed but no coin-flip
  decision appeared in the sampled steps, so its effect is still unobserved
  in practice (mechanism is documented; revisit when a coin-flip-relevant
  state is sampled, e.g. right after a KO in the actual game/search).

**Net effect:** the MCTS tree (`training/nn/mcts.py`, Kaggle-only, not yet
built) can proceed on the design in `docs/nn-training.md`'s Self-Play Phase
Design section without further spikes — the only open item is confirming
`manual_coin` empirically when a relevant state comes up.
