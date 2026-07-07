# Game Nature — A Detailed Rundown for ML Method Selection

*What kind of game this actually is, at the level of detail needed to choose a
learning algorithm: turn structure, decision granularity, information
structure, stochasticity, card/deck composition, and empirically observed
game-length/branching statistics. Written to be read once by someone deciding
between BC/DAgger/self-play/RL/search — not a strategy doc (see
`docs/competition-strategy.md`) and not the engine API reference (see
`docs/engine-api.md`, which this file draws from and doesn't duplicate).*

**Last updated:** 2026-07-05

---

## 1. What Game This Is

This is the **Pokémon Trading Card Game** (physical-game rules, digital
implementation), played through the `cabt` engine shipped inside
`kaggle_environments`. Two players, each with a private 60-card deck, play
Pokémon as attackers, attach Energy to power attacks, and race to take 6
"prize" cards by knocking out the opponent's Pokémon (or win by opponent
deck-out / opponent having no Pokémon left in play). It is a **turn-based,
two-player, zero-sum, imperfect-information card game** with substantial
built-in stochasticity (deck shuffling, draws, coin flips) — structurally in
the same family as poker or Magic: The Gathering, not the same family as Go,
chess, or Atari. That family membership is the single most important fact for
method selection: **standard AlphaZero-style perfect-information MCTS does
not apply directly** here, because the game tree contains hidden information
(opponent's hand, both decks, both prize piles) that a search tree cannot see
into without either sampling (determinization) or a belief model.

---

## 2. Turn Structure

Each player's turn (after the initial setup) has a fixed shape:

1. **Draw step** — draw 1 card from the deck (deck-out = instant loss if you
   must draw with an empty deck).
2. **Main phase** — an open-ended sequence of actions, in any order/number,
   subject to per-turn limits:
   - Play any number of **Item** cards (unlimited).
   - Play up to **one Supporter** per turn (`state.supporterPlayed` flag).
   - Play up to **one Stadium** per turn (`state.stadiumPlayed` flag).
   - Attach up to **one Energy** card per turn (`state.energyAttached` flag).
   - Evolve any number of eligible Pokémon (each Pokémon only once per turn,
     and never the turn it entered play, with Rare Candy as the sole
     stage-skip exception).
   - Retreat the Active Pokémon **at most once per turn**
     (`state.retreated` flag), paying its retreat Energy cost.
   - Use Abilities (no hard per-turn count in general, but many have their
     own "once per turn" text).
3. **Attack step** — optionally attack once (ends the turn) or explicitly
   **End Turn** with no attack (valid, and sometimes correct — e.g. banking
   hand size for Powerful Hand, or when no legal attack exists).

The turn is **not** "pick one action" — it's a whole shopping-list of
sub-decisions solved in sequence, then a single terminal action (attack or
end). This is reflected directly in the engine's `SelectType`/`SelectContext`
schema (`docs/engine-api.md`): a single logical turn typically involves
several distinct `agent()` calls, one per pending choice — the game does not
hand you the whole turn as one action, it asks a sequence of typed micro-
questions (which card to play, which target, which energy, etc.) and expects
one legal index back each time.

**Setup phase** (before turn 1) has its own selects: an `IS_FIRST` coin-flip
choice, `SETUP_ACTIVE_POKEMON`/`SETUP_BENCH_POKEMON` picks, and (per the
engine enum) an as-yet-unhandled `MULLIGAN` context for redraw-on-no-basic
hands — noted in `docs/version-history.md` as an open gap in `main.py`.

---

## 3. Decisions: Granularity, Types, and Empirical Counts

### 3.1 What counts as "a decision"

Every `agent(obs) -> list[int]` call is one decision: the engine presents a
`SelectData` (a type + context + legal `option` list), and the agent returns
option index/indices. A single physical-game turn decomposes into **many**
engine-level decisions — not one. Concretely, a turn like "attach Psychic to
Kadabra, evolve Abra→Kadabra, play Boss's Orders targeting their benched
Crustle, attack" is at minimum 4 separate `agent()` calls, several of which
(evolve target, Boss's target) are themselves sub-selects requiring their own
resolution logic.

### 3.2 Measured decision volume

- **~158 decisions per game** measured directly from teacher self-play data
  collection (`docs/nn-training.md`). This deck (Alakazam) is called out as
  **high-branching** specifically *because of* search/evolve sub-selections
  — Poffin/Poké Pad/Dawn/Hilda searches all resolve as their own decision
  against the full visible deck list (deck searches are **not blind** — see
  §5), not one flat "play card" click.
- A **10-minute wall clock per match** divided across ~158 decisions ÷ 2
  players ≈ **~4.0 seconds of decision budget per own-side decision**
  (`docs/engine-api.md` "MCTS branching + timing probe") — tight enough that
  **timeout = instant loss** is a first-class engineering constraint, not an
  edge case.
- Game length in turns: tier-1 deck bake-off games (2,000 games, uncapped)
  hit a **max of 43 turns**; a 3,000-engine-step cap (≈3× the longest
  observed tier-1 game) was later added as a safety net for degenerate
  passive-vs-passive matchups, confirming normal games are far shorter than
  that cap in practice.

### 3.3 Types of decisions (by `SelectType`, `docs/engine-api.md`)

| Category | SelectType(s) | What it decides |
|---|---|---|
| Turn-level choice | `MAIN` (0) | Play/attach/evolve/retreat/attack/end — the "what do I do this action" branch point, offering many `OptionType`s at once (PLAY, ATTACH, EVOLVE, ABILITY, RETREAT, ATTACK, END) |
| Card targeting | `CARD` (1), `ATTACHED_CARD` (2), `CARD_OR_ATTACHED_CARD` (3) | Which specific card (hand/deck/discard/attached) an effect acts on |
| Energy targeting | `ENERGY` (4) | Which energy card/type to attach, discard, or move |
| Ability/attack choice | `SKILL` (5), `ATTACK` (6) | Which Ability or which Attack to use |
| Evolution | `EVOLVE` (7) | Which Pokémon evolves, into what, from what hand card |
| Numeric/count | `COUNT` (8) | How many of something (e.g. damage counters, draw count) |
| Binary | `YES_NO` (9) | Optional-effect confirmations, first-player coin choice |
| Status | `SPECIAL_CONDITION` (10) | Poison/Burn/Sleep/Paralyze/Confuse application/removal |

Layered on top, `SelectContext` (48 values) refines *what kind* of MAIN/CARD/
etc. decision this is — e.g. `SETUP_ACTIVE_POKEMON`, `TO_BENCH`,
`DISCARD_ENERGY_CARD`, `EVOLVES_TO`, `SKILL_ORDER`. In practice this means
the action space is **heterogeneous and positional**: options never carry a
semantic card ID directly (`Option.cardId` is populated 0/1,287 times in a
full observed game — verified empirically, `docs/engine-api.md`); every
option is resolved by area/index against the visible board/hand/deck state
(`hand[index].id`, `deck[index]`, `energyCards[energyIndex]`, etc.). This
matters for representation learning: **the same logical action (e.g. "play
Boss's Orders on their Crustle") can appear at a different option index in
every observation**, so a policy has to condition on the enumerated option
list itself (pointer-style), not learn a fixed action embedding table the
way Atari/board-game agents do with a fixed action set.

### 3.4 Effective per-turn branching factor

The MAIN-phase select alone routinely offers **on the order of 5-10+
legal options** (a live example cited in `docs/engine-api.md`: a real
mid-game MAIN select had exactly 7 options), before counting the sub-selects
each of those options can spawn (e.g. a search-card play fans out into one
option per matching card in the full deck list — with a 60-card deck and
several 3-4 copy playsets, a single Poké Pad/Dawn play can offer a double-
digit number of legal deck targets). This is meaningfully higher branching
than classic board games' "one move" turns, though still far below the raw
combinatorial branching of, say, StarCraft; it's closer in spirit to a
card game like Hearthstone or Magic (a "what to play + who to target"
menu each action) than to Go/chess.

---

## 4. Information Structure (Imperfect Information)

This is the second most important classification fact after turn/decision
structure. From either player's own `agent()` observation:

- **Hidden:** the opponent's hand contents (`hand` is `None` for the
  opponent in `PlayerState`), both players' full deck order, and the face-
  down contents of both prize piles (`prize` entries are `None` until
  revealed/taken).
- **Visible:** both players' Active + Bench Pokémon (species, HP, attached
  Energy, attached Tools, `appearThisTurn`), attached-energy history via
  logs, every card either player has *played* (Log type 10 = PLAY reveals
  the opponent's card the instant they play it), deck/discard/hand **counts**
  (not identities) for both sides, and — importantly — **your own deck
  searches are not blind**: search effects (Poffin, Poké Pad, Dawn, Hilda,
  Rare Candy's evolve-target, Enhanced Hammer's energy-pick) all expose the
  *searching player's own* full deck contents via `select.deck`
  (`docs/engine-api.md` §Verified runtime behaviors, item 1). So imperfect
  information here is strictly one-sided per player: you always know your
  own deck/hand fully, you never know the opponent's.
- **Confirmed to hold inside search too:** a live `search_begin` spike
  showed the opponent's hand stays `None` in the returned observation even
  when a placeholder `opponent_hand` was fed in — the hidden-zone args are
  sampling-only, never leaked back (`docs/engine-api.md` §Search API). Any
  search-based method (MCTS, etc.) must **determinize** (sample a plausible
  opponent hand/deck/prize arrangement) rather than assume full state.

This is why the project's belief model exists as a first-class component
(`docs/belief-model.md`): opponent archetype is inferable fast from public
play (a Lucario deck reveals itself by turn 2-3 via Riolu + Fighting energy),
measured at **99.1% classification accuracy by turn 1** and **~100% by turn
2** against the 5-label bot/mirror set, dropping to a realistic **78.7%
honest ceiling** against the full open ladder meta (144 replays form a
genuine long tail of one-off techs, not a few missing signatures away from
full coverage). The practical takeaway for method choice: **cheap,
supervised archetype inference from public information is a solved,
high-value sub-problem here** — it's the natural place to spend effort on
"reading" the hidden state rather than trying to search through it blindly.

---

## 5. Stochasticity

Multiple independent sources of randomness per game:

1. **Deck shuffle** at game start (both decks) and whenever a card is
   shuffled back in (e.g. Dudunsparce's Run Away Draw: draw 3, then shuffle
   those 3 back).
2. **Draw order** — the entire game's card sequence for both players is
   effectively one shuffled draw stream; nothing about future draws is
   observable or controllable beyond deck-thinning (see counting-based
   inference in §4).
3. **Coin flips** — an explicit `COIN` log type and `COIN_HEAD` select
   context; used for the opening first-player decision (`IS_FIRST`) and by
   some card effects. `search_begin(..., manual_coin=True)` exists
   specifically so a search process can control/enumerate coin outcomes
   rather than being at their mercy (confirmed mechanism, not yet exercised
   in a live coin-flip state as of the last spike).
4. **Opponent policy** — on the live ladder, the opponent is an unknown,
   possibly non-deterministic external agent; even offline, opponent bot
   agents (`opponents/*.py`) are not necessarily deterministic turn-to-turn.

Net effect: this is not a game where a single rollout of any given policy
pair reliably reproduces the same trajectory. All evaluation in this project
is therefore done as **statistically-powered A/B testing** (win rate with
Wilson 95% CIs over hundreds of games), never single-game or small-n
comparison — reflected throughout `training/README.md` and
`docs/report-log.md`'s "no claim without a pre-registered trial" discipline.

---

## 6. Deck and Card Population

### 6.1 Universal constraints (standard physical-game rules, enforced by the
engine)

- Exactly **60 cards per deck**, own copy per player (each side's deck is
  independently defined at match config time — `battle_start(deck0, deck1)`).
- **6 prize cards** set aside face-down at game start; a player wins by
  taking all 6 prizes (via knockouts) or if the opponent has no Pokémon left
  in play or must draw from an empty deck.
- **1 Active Pokémon slot + a Bench** (`benchMax` field on `PlayerState` —
  the standard game's bench holds up to several Pokémon; exact size is a
  ruleset constant read from the observation, not hardcoded by us).
- **Per-turn limits** enforced by the engine and exposed as observation
  flags: one Supporter (`supporterPlayed`), one Stadium (`stadiumPlayed`),
  one Energy attach (`energyAttached`), one retreat (`retreated`) — all
  reset each turn.
- **Prize value differs by Pokémon rarity tier**: knocking out a plain
  Pokémon yields 1 prize, an "ex" Pokémon yields 2, a "Mega ex" yields 3
  (`pokemon.ex` / `pokemon.megaEx` booleans; see `prize_value()` in
  `docs/project-reference.md`). This is a first-order strategic axis: most
  of the competitive meta plays 2- or 3-prize attackers for damage output at
  the cost of conceding more per KO, which is exactly the trade-off this
  project's single-prize deck is built to exploit.

### 6.2 Card types (`CardType` enum, `docs/engine-api.md`)

`POKEMON`, `ITEM`, `TOOL`, `SUPPORTER`, `STADIUM`, `BASIC_ENERGY`,
`SPECIAL_ENERGY` — 7 card types, each with distinct play rules (Items:
unlimited per turn; Supporter/Stadium: one per turn; Tools: attach to a
Pokémon, usually one per Pokémon; Energy: one attach per turn, basic vs.
special with different effects, e.g. Mist/Rock energies that block *all
effects* of attacks, not just damage).

### 6.3 Pokémon-side structure

- **Evolution lines**: Basic → Stage 1 → Stage 2, each requiring the prior
  stage in play and (normally) a full turn's wait, with Rare Candy as the
  sole engine-supported stage-skip (Basic → Stage 2 directly,
  `SelectContext.EVOLVE`). A Pokémon **cannot evolve the turn it entered
  play** (an explicit, engine-checked constraint independent of card text).
- **Attacks** cost specific Energy-type combinations (`Attack.energies`) and
  deal damage, sometimes with riders (status conditions, extra draw,
  spread/bench damage, self-damage). Some attacks/abilities key off
  non-obvious state, e.g. this project's centerpiece attack **Powerful
  Hand** (Alakazam, 1 Psychic Energy): `damage = 20 × current hand size`,
  turning "how big is my hand" into the core damage stat instead of raw
  Energy investment — a genuinely different reward-shaping problem than a
  typical fixed-damage attacker (see `docs/piloting-guide.md` for the full
  piloting logic this implies).
- **Weakness/Resistance**: standard type-based damage modifiers, with some
  attacks explicitly bypassing them ("ignores Weakness/Resistance").
- **Abilities**: passive or activated effects independent of the attack
  slot (e.g. Kadabra's evolve-draw, Fezandipiti ex's post-KO draw,
  Shaymin's bench-damage prevention) — a second, parallel decision/effect
  channel alongside attacks.

### 6.4 This project's specific deck (context, not universal)

The active 60-card Alakazam deck (`docs/project-reference.md` §Deck) is one
concrete instantiation of the above: single evolution line (Abra→Kadabra→
Alakazam, all non-ex, 1-prize each), a hand-size-driven win condition, ~13
distinct non-Pokémon card roles (search Items ×4 variants, tool/energy
utility, 2 recovery cards, Boss's Orders for forced targeting), and exactly
2 named "hard counter" special energies in the wider card pool (Mist #11,
Rock Fighting/Rocky #20) that a deck built around this attack must plan
around. This is deck-specific detail, not a game-mechanics universal — see
`docs/project-reference.md` for the full card table if the deck itself
(rather than the game's mechanics) is in scope.

---

## 7. Action-Space / Observation Shape Actually Fed to a Model

For ML architecture purposes (current implementation:
`docs/nn-training.md` §Architecture), the practically relevant shape is:

- **Observation encoder**: 24 discrete "words" per encoder step — both
  players' bench (8 slots × 2), both actives, both player-level states,
  hand, deck, stadium, misc — fed through an `EmbeddingBag(22000 vocab)` +
  a small Transformer. Vocabulary size (~22k) reflects the full card ID
  space plus auxiliary state tokens, not just this deck's cards — the
  observation encoding is deck-agnostic.
- **Decision/decoder side**: a per-decision option list, up to **64
  candidate options** per decision (`decoder_size` padding constant), each
  resolved positionally (index into hand/deck/bench/etc., §3.3) rather than
  by a fixed action-ID vocabulary — i.e. the "action space" is *dynamic and
  pointer-based per decision*, not a fixed discrete action set the way
  Atari/most RL benchmarks assume.
- Practical consequence already hit in this project: **behavior cloning
  alone plateaus** (85.9% held-out action-match accuracy vs. only ~17-22%
  head-to-head win rate against the same teacher it was cloned from) — the
  textbook **compounding-error / distribution-shift** failure mode for
  imitation learning on long-horizon (~150-decision), branching-action
  sequential decision problems, motivating DAgger as the next rung
  (`docs/project-reference.md` §NN Track Summary).

---

## 8. Summary Table — Properties Relevant to Method Choice

| Property | Value | Implication |
|---|---|---|
| Players | 2, zero-sum | Standard adversarial setup |
| Information | Imperfect (opponent hand/deck/prizes hidden; own deck never blind) | Need determinization or belief model for any search method; can't use vanilla perfect-info MCTS/AlphaZero |
| Stochasticity | Shuffle, draw order, coin flips, external opponent policy | Evaluation must be statistical (many-game A/B with CIs), not single-trajectory |
| Game length | Median well under cap; max observed 43 turns (2,000-game bake-off) | Short enough for full-episode Monte Carlo returns to be tractable |
| Decisions/game | ~158 (engine-level `agent()` calls, not physical turns) | High-branching relative to physical turn count; imitation targets are decisions, not turns |
| Decision budget | ~4.0s/own-decision under the 10-min/match clock | Hard latency ceiling — timeout is an instant loss; gates how much search/sims is affordable (≈730 sims/decision on raw engine cost alone, per the timing probe) |
| Action space shape | Positional/pointer per decision, up to 64 options, heterogeneous select types | Action embeddings must condition on the enumerated option list, not a fixed action-ID table |
| Deck size | 60 cards/player, fixed at match start | Deck composition is a designable, fixed input — not learned online |
| Prizes | 6, KO'ing ex/Mega ex Pokémon yields 2/3 at once | Central strategic axis (prize-trade efficiency); single-prize decks trade differently than ex-heavy decks |
| Imitation ceiling | BC plateaus (85.9% action-match, ~17-22% win-rate vs teacher) | Confirmed compounding-error signature → DAgger/on-policy correction needed, not just more BC data |
| Archetype inference | 99.1%/turn-1, ~100%/turn-2 (bot set); 78.7% honest ceiling (open ladder meta) | Cheap, high-value place to "read" hidden info directly rather than search blindly through it |

---

## 9. Sources

All figures and mechanics above are drawn from, and kept in sync with:
`docs/engine-api.md` (canonical engine reference), `docs/project-reference.md`
(deck/architecture reference), `docs/nn-training.md` (architecture + BC/DAgger
results), `docs/belief-model.md` (information-structure findings),
`docs/competition-strategy.md` (bake-off protocol and results), and
`docs/report-log.md` (raw experiment log — bake-off turn-count/game-length
data, timing probe numbers). Update this file's numbers if any of those
change materially (e.g. a new bake-off resets turn-count/branching figures).
