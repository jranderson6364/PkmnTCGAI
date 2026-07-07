# Project Reference — Full Breakdown

*The complete reference layer moved out of `CLAUDE.md` (which stays a slim
orientation file loaded every prompt). Open THIS file whenever a task touches the
engine API, the deck list, `main.py` internals, packaging/shipping, or the NN
architecture. Keep it in sync: any change to code, deck, or plans updates the
matching section here in the same session.*

**Last updated:** 2026-07-06

---

## Repo Structure

```
main.py          ← v23-deck Alakazam heuristic agent (active ladder submission)
deck.csv         ← 60-card deck, one card ID per line (regenerated from main.DECK)
CLAUDE.md        ← slim orientation layer (loaded every prompt)
README.md        ← repo landing page (humans + fresh clones); points here and to CLAUDE.md
docs/
  project-reference.md ← THIS FILE — full reference breakdown
  engine-api.md      ← CANONICAL cabt API reference (full enums, verified behaviors)
  competition-strategy.md ← Stage 0–5 roadmap, writeup strategy, deck thesis
  nn-training.md     ← full NN training log, architecture, roadmap
  belief-model.md    ← belief model design + phase plan (Stage 3 centerpiece)
  report-log.md      ← experiment journal + method glossary + the 5 target figures
                       (the report is assembled from this file — same-day entries)
  piloting-guide.md  ← expert Alakazam piloting logic (NN training target spec)
  matchups.md        ← matchup reference + tech cheat-sheet
  version-history.md ← v1–v24 change log (canonical; CLAUDE.md only summarizes current)
  EN_Card_Data.csv   ← official card text/IDs reference (opponent decks + replay analysis)
training/
  README.md          ← local training rig guide (setup, workflows, opponent pool,
                       reward-shaping notes, discipline)
  harness.py         ← run local games in parallel (kaggle_environments cabt)
  ab_test.py         ← A/B two agent files, alternating seats, 95% CI
  gauntlet.py        ← fixed anchor panel + Bradley-Terry fit → gElo scale;
                       results accumulate in gauntlet_results.csv
  bakeoff.py         ← round-robin over (agent, deck) pairs (Stage 0c/method bake-off)
  generic_pilot.py   ← deck-agnostic greedy pilot (bake-off control/floor)
  manifests/         ← bake-off entry lists (tier1.csv, tier2.csv) + deck manifests
  ladder_history.csv ← realized ladder Elo per shipped version (gElo calibration)
  bc_collect.py      ← BC warmup data collection (teacher self-play → pkl.gz shards)
  setup_local_search.py ← assemble local search-capable cg package (local_cg/, gitignored)
  archetype_decks.json ← reconstructed archetype 60-card lists (belief model, Stage 3)
  random_agent.py    ← uniform-random opponent (baseline gate + gauntlet anchor)
  baselines/         ← frozen v21.py, v22_pre_refactor.py, v23.py (anchors) + v25c.py (A/B ref)
  belief/            ← archetype classifier: collect.py, train.py, exported weights + figure
  nn/                ← NN track: encode/model/dataset, collectors (mcts_collect.py,
                       dmc_collect.py, selfplay_collect.py, dagger_collect.py,
                       exploiter_collect.py), trainers (train_bc/train_sp/train_dmc),
                       gates (dmc_replay_gate.py), search (mcts.py,
                       mcts_leafeval_agent.py), Φ baseline (phi_baseline.py,
                       threat.py) — see docs/nn-training.md
  *.pth              ← local reference checkpoints (gitignored): ptcg_bc_v1/v2,
                       ptcg_dagger_r2, ptcg_dmc_r2, ptcg_dmc_p0_v2_n1_richenc_v2,
                       ptcg_exploiter_r1
  mcts_p2_r3.pkl.gz  ← fixed Phase-2 self-play-with-search corpus (pending retrain)
opponents/
  lucario_agent.py   ← Mega Lucario ex (official Kaggle sample; real deck embedded)
  dragapult_agent.py ← Dragapult ex Stage 2 spread (official Kaggle sample; real deck embedded)
  abomasnow_agent.py ← Mega Abomasnow ex energy mill (official Kaggle sample; real deck embedded)
  starmie_agent.py   ← Mega Starmie ex spread (stub with real IDs: Staryu=1030, Starmie ex=1031)
tools/
  analyze_replay.py  ← kaggle-env replay decoder/auditor (missed lethals, bad retreats,
                       bad Boss targets, wasted energy attaches, timeouts). Usage:
                       `python3 tools/analyze_replay.py <replay.json> [more...]`
                       writes `<name>_summary.txt` next to each input.
  deck_audit.py      ← per-card utilization over local self-play (plays/game-drawn,
                       rot rate, end-hand rate, win-rate deltas) — the deck
                       simplification evidence base. `python tools/deck_audit.py --games 1000`
  deck_math.py       ← analytic consistency panel: hypergeometric mulligan % and
                       combo-by-turn-N odds for any deck list (feeds report figure #7).
                       `python tools/deck_math.py deck.csv --want "label=id:n,...@turn"`
  meta_survey.py     ← classify opponent archetype per replay JSON → ladder meta
                       share table (decision-rule weights, Stage 3 labels, report
                       figure). `python tools/meta_survey.py --all --csv training/meta_survey.csv`
replays/
  bulk/              ← bulk-downloaded ladder replays (tools/download_replays.py);
                       raw JSONs are git-ignored (local only)
  exploiter_wins/    ← 18 mined exploiter-win summaries (board-thinning evidence)
```

---

## Engine API (cabt + kiyotah/cg-lib)

**Full canonical reference: `docs/engine-api.md`** (all enums, dataclasses, verified
runtime behaviors). The engine also ships inside `kaggle_environments` — full games
run LOCALLY (`training/README.md`).

```python
# Install (Kaggle notebook only)
import sys, glob
sys.path.append(glob.glob('/kaggle/input/**/cg-lib', recursive=True)[0])

# Direct play API (A/B harness)
from cg.game import battle_start, battle_select, battle_finish

# Dataclass API (NN track / MCTS)
from cg.api import to_observation_class, all_card_data, all_attack
from cg.api import search_begin, search_step, search_end

# env.run wrapper (recommended for submission validation)
from cg.env import env
```

**Confirmed observation fields:**
- `state.supporterPlayed`, `state.energyAttached`, `state.retreated`, `state.turnActionCount`
- `pokemon.hp` (remaining, not max), `pokemon.maxHp`, `pokemon.appearThisTurn`
- `pokemon.energies` (list of energy type IDs), `pokemon.tools` (list of tool card IDs)
- `pokemon.ex` (bool, 2-prize), `pokemon.megaEx` (bool, 3-prize)

**Option schema — options are positional, never carry cardId (verified: 0/1,287 in a full game):**
- `PLAY {index, type}` → resolve via `hand[o['index']].id`
- `ATTACH {index, inPlayArea, inPlayIndex, type}` → `index`=energy hand pos; `inPlayArea` 4=active/5=bench
- `EVOLVE {index, inPlayArea, inPlayIndex}` → same pattern
- `ABILITY {area, index, type}` → resolve via active(4)/bench(5) pokemon
- `ATTACK {attackId, type}`, `RETREAT {type}`, `END {type}`
- **Area codes (full enum in docs/engine-api.md): 1=deck, 2=hand, 3=DISCARD, 4=active, 5=bench, 6=PRIZE**
  (an earlier note here said 6=discard — wrong)
- **Deck searches are NOT blind** — `select.deck` lists the deck; option `index`
  points into it: `sel['deck'][o['index']]['id']`. (v22's `_deck_search_pick` uses this.)
- `select.effect` = the card causing a sub-selection (Boss=1182, Hammer=1081,
  Rare Candy=1079…); `select.contextCard` = the card being acted on.

**Log types used in v14:**
- Type 10 = PLAY (cardId visible for opponent — use to detect opponent's card plays)
- Type 6 = MOVE_CARD (fromArea in {4,5} → toArea 3 = KO detection)
- Type 16 = HP_CHANGE (`putDamageCounter` flag = bench damage detection)

**Boss/gust target selection:** `stype=1`, options have `area=5`, `playerIndex=opponent`, `index` into `opp_bench`

**Setup contexts:** `ctx=1` = SETUP_ACTIVE_POKEMON, `ctx=2` = SETUP_BENCH_POKEMON

---

## Deck — Alakazam Single-Prize Control/Combo (60 cards, v23 — current after v24 revert)

### Win Condition
**Powerful Hand** (Alakazam 743, attackId 1072, cost 1 Psychic): place 2 damage counters on the opponent's Active for each card **in your hand**. `damage = 20 × hand_size`. Ignores Weakness/Resistance/reduction.

- KO threshold: `ceil(opponent_active_hp / 20)` cards needed
- Hand size IS the damage stat — never discard unnecessarily
- Blocked by: **Mist Energy (#11)** and **Rock Fighting Energy (#20)** — both say "prevent all effects of attacks"

### Card IDs (v23 list — current; v24's Psyduck/Genesect-out swap reverted 2026-07-03)

| Constant | ID | Count | Role |
|----------|----|-------|------|
| ABRA | 741 | 4 | Evolution base |
| KADABRA | 742 | 4 | Evolution middle (+2 draw on evolve) |
| ALAKAZAM | 743 | 3 | Main attacker (+3 draw on evolve) |
| DUNSPARCE | 305 | 3 | Draw engine base |
| DUNSPARCE2 | 65 | — | (alt print, not in deck) |
| DUDUNSPARCE | 66 | 3 | Run Away Draw: draw 3, shuffle back |
| GENESECT | 142 | 1 | ACE Nullifier (Bench + Fan) |
| SHAYMIN | 343 | 1 | Flower Curtain: prevents bench damage |
| PSYDUCK | 858 | 1 | Reactive scoring vs opponent self-damage cards |
| FEZ (Fezandipiti ex) | 140 | 1 | Flip the Script: draw 3 after KO |
| POFFIN (Buddy-Buddy) | 1086 | 4 | Search 2 Basics ≤70 HP to bench |
| POKE_PAD | 1152 | 4 | Search any non-Rule-Box pokemon to hand |
| HANDHELD_FAN | 1161 | 2 | Anti-deck-out tool |
| BOSS (Boss's Orders) | 1182 | 3 | Gust opponent's benched pokemon |
| LANA (Lana's Aid) | 1184 | 1 | Recover pokemon from discard |
| BATTLE_CAGE | 1264 | 4 | Prevents bench damage from opponent attacks |
| DAWN | 1231 | 4 | Search Basic + Stage 1 + Stage 2 to hand |
| WONDROUS_PATCH | 1146 | 1 | Attach Basic Psychic from discard to bench |
| SACRED_ASH | 1129 | 1 | Recover pokemon from discard |
| HILDA | 1225 | 3 | Search Evolution + Energy to hand |
| ENHANCED_HAMMER | 1081 | 2 | Discard Special Energy from opponent |
| RARE_CANDY | 1079 | 3 | Abra → Alakazam (skips Kadabra) |
| BASIC_P | 5 | 2 | Basic Psychic energy (pays Powerful Hand) |
| ENRICHING | 13 | 1 | Draw 4 on attach (Colorless — cannot pay Powerful Hand) |
| TELEPATH_P | 19 | 4 | Psychic energy + bench 2 Abra on attach |

**Special energies that block Powerful Hand:**
- Mist Energy = card #11
- Rock Fighting Energy (Rocky Energy) = card #20
- Detect by: `11 in opp_active.energies` or `20 in opp_active.energies`

**Energy routing rule:** Route Psychic (5, 19) → Alakazam **only if it doesn't already have one** (Powerful Hand costs exactly 1 Psychic; a 2nd does nothing). Once Alakazam is fueled, route further Psychic → Kadabra → Abra, in that order, so the line is pre-loaded before it evolves. **Never proactively attach Psychic to any other support mon** (Dudunsparce/Shaymin/Fez) — they don't attack, and the only legitimate exception is paying a real retreat cost on the Active to switch into an already-ready bench Alakazam. Route Enriching (13) → Dudunsparce (draw + recycle). Never attach Enriching to Alakazam.

### Poffin vs Poké Pad vs Dawn
- **Poffin (1086):** benches Abra(50HP) / Dunsparce(70HP) / Psyduck(60HP) — the full ≤70HP pool in the deck. Cannot grab Shaymin(80HP) or Fez(210HP).
- **Poké Pad (1152):** puts any non-Rule-Box pokemon into hand (includes Shaymin but NOT Fez ex).
- **Dawn (1231):** grabs the full Abra+Kadabra+Alakazam line at once.

---

## Agent Architecture

**File:** `main.py` — currently **v23 deck** (v24's deck-list swap reverted 2026-07-03).

Full version-by-version change narrative (v1 → v24, every bug found, every fix, every
verification run) lives in **`docs/version-history.md`** — that file is canonical;
this section holds only durable reference material (constants, state machine, key
functions) plus a short summary of the current and immediately-prior versions.

### v24 REVERTED (2026-07-03) — back to v23 deck
`DECK` list only: v24 swapped Psyduck + Genesect (deck audit: ~0 plays/game, pure
hand fuel) → 4th Alakazam + 4th Dunsparce; 200-game local A/B favored it (120W–80L,
60.0% ± 6.8%, 0 errors) but live ladder trend (680 Elo at ~7h, 780 at ~24h) prompted
a user call to revert ahead of the documented 48h/50-point decision checkpoint. Note:
v23 itself had decayed to 773 by its own day-2 reading, so this isn't an unambiguous
regression — see `docs/report-log.md` 2026-07-03 entry. `main.py`/`deck.csv` are back
to the exact v23 composition; no scoring-logic changes either direction.

### v23 — Replay Forensics (current)
5 fresh ladder losses reconstructed turn-by-turn from raw JSON board state. Three
fixes: (1) `_score_bench_target` dedicated Kadabra/Abra promotion tiers (previously
tied with support mons in one `-10` catch-all — confirmed losing move in a 1-1
sudden-death game); (2) Enriching → an already-fueled Alakazam always scores -8.0;
(3) `bench_empty` flag — Poffin/Poké Pad jump to top search priority at
`bench_count==0`. **Verified:** 691 replay decisions clean; ladder public score
796.3 (v22: 771.6).

### v22 — Full API Audit → Selection Intelligence
Audited v21 against the official cabt docs (findings: `docs/engine-api.md`). Fixed
deck searches treated as blind, a live `_pick_setup_active` bug (field never
populated, since v7), Enhanced Hammer / Rare Candy target picking; exposed ~20
scoring constants as the tunable `W` dict. **Verified:** 400-game A/B vs frozen
v21: 56.3% ± 4.9%, 0 errors. Baselines: 94% Lucario, 94% Abomasnow, 79% Starmie,
50W–0L–50T Dragapult.

### Constants
```python
ABRA,KADABRA,ALAKAZAM = 741,742,743
DUNSPARCE,DUNSPARCE2,DUDUNSPARCE = 305,65,66
GENESECT,SHAYMIN,PSYDUCK,FEZ = 142,343,858,140
POFFIN,POKE_PAD,HANDHELD_FAN = 1086,1152,1161
BOSS,LANA,BATTLE_CAGE,DAWN = 1182,1184,1264,1231
WONDROUS_PATCH,SACRED_ASH,HILDA = 1146,1129,1225
ENHANCED_HAMMER,RARE_CANDY = 1081,1079
BASIC_P,ENRICHING,TELEPATH_P = 5,13,19
MIST_ENERGY,ROCK_ENERGY = 11,20
PSYCHIC_TYPE = 5
PH_DMG_PER_CARD = 20
```

### 4-Phase State Machine
```python
PHASE_ESTABLISH=1  # Build board: get Alakazam up with Psychic + backup + draw engine
PHASE_CONVERT=2    # Hand conservation, advance setup, energy routing
PHASE_PRESSURE=3   # Can KO or at damage threshold — attack/Boss now
PHASE_CLOSING=4    # ≤2 prizes left — close out

def _detect_phase(cen, can_ko, at_threshold, opp_prizes_left, hand_n):
    if opp_prizes_left <= 2: return PHASE_CLOSING
    not_established = (not cen['has_alakazam'] or not cen['backup_abra'] or
        cen['draw_count'] == 0 or not cen['has_energy_plan'])
    if not_established: return PHASE_ESTABLISH
    if can_ko or at_threshold: return PHASE_PRESSURE
    return PHASE_CONVERT
```

### Key Functions
- `_analyze_logs(obs)` — parses log history for KO detection, bench damage, opponent card plays
- `_census(obs)` — board state snapshot: has_alakazam, backup_abra, draw_count, has_energy_plan, etc.
- `_pick_setup_active(obs, sel)` — handles ctx=1/2 (setup phase pokemon selection)
- `_pick_bench_target(obs, sel)` — which bench pokemon to promote
- `_pick_boss_target(obs, sel)` — which opponent bench to gust (requires prize_value guard)
- `_main_phase(obs, sel)` — main action loop with nested `score()` function
- `score_options(obs, sel)` / `score_options_main(obs, sel)` — standalone per-option
  scoring (DAgger teacher labels + MCTS prior blending)
- `_safe_return(sel)` — always returns a valid legal action (fallback)
- `agent(obs_dict)` — entry point
- `W` — tunable scoring-weight dict (mutated in-process by tuning harnesses; the submission never reads env/files)

### Prize Value Logic
```python
def prize_value(pokemon):
    if pokemon.megaEx: return 3
    if pokemon.ex: return 2
    return 1
```

---

## NN Track Summary

See `docs/nn-training.md` for full details (§Resume Here has the current pipeline).

**Status (as of 2026-07-03):** BC warmup complete **on the pre-v24 deck** — since
v24's deck swap was reverted 2026-07-03, this is now the same deck as the current
`main.py`/`deck.csv`. Re-collect once the deck is ladder-confirmed frozen. Self-play
Phase 1 built but superseded by the DAgger-first plan (direct self-play had no
improvement operator).
- Architecture: EmbeddingBag(22000) + Transformer(128d, 2-head) + actor-critic heads
- Value targets: n-step Monte Carlo with value-head bootstrapping (truncated
  rollouts, not full binary playouts) — see nn-training.md §Value Targets
- BC teacher = heuristic self-play (`training/bc_collect.py`; 547,796 decisions on
  the old deck). Cloning opponent replays is NOT viable (different decks; action
  sequences don't transfer).
- `training/ptcg_bc_v1.pth`: 85.9% held-out action-match accuracy; 86% vs random
  (clears the 65% gate), 22% vs v22 heuristic (compounding error, not a bug — the
  diagnosis that motivates DAgger).

**Phase plan (DAgger first; rationale in `docs/nn-training.md` §Resume Here):**
1. ~~BC warmup~~ **DONE** (old deck — re-collect after the deck freeze)
2. **DAgger** (Stage 1) — net pilots, teacher labels via `score_options`; gate 50%+
   vs teacher → ship to ladder
3. **Advantage-weighted self-play** (Stage 2) — 40/60 BC mix; gate 55–60% vs teacher
4. Belief model (Stage 3, parallel) + curriculum/PFSP hardening (Stage 4) → Jul–early Aug
5. MCTS expert iteration / search-at-inference (Stage 5, Kaggle-gated) → freeze-week go/no-go
6. Report assembly from `docs/report-log.md` → Aug 16–Sep 13

---

## Packaging & Shipping

The Kaggle CLI is configured — ship directly, never ask for a manual upload.
**Ladder** slug: `pokemon-tcg-ai-battle` (agents go here). **Strategy report
track**: `pokemon-tcg-ai-battle-challenge-strategy` (Sep 13).

```python
import tarfile, py_compile

py_compile.compile('main.py', doraise=True)
with open('deck.csv') as f:
    assert len(f.readlines()) == 60, "deck must be exactly 60 cards"
with tarfile.open('submission.tar.gz', 'w:gz') as tar:
    tar.add('main.py')
    tar.add('deck.csv')
```

```bash
kaggle competitions submit -c pokemon-tcg-ai-battle -f submission.tar.gz -m "<version>: <summary>"
kaggle competitions submissions -c pokemon-tcg-ai-battle | head -5   # verify; publicScore = ladder Elo
```

After every ship: add the row to `training/ladder_history.csv` and the entry to
`docs/report-log.md`.

---

## Kaggle File Locations

```
/kaggle/working/main.py              ← v23-deck source (active submission)
/kaggle/working/deck.csv             ← 60-card deck
/kaggle/working/submission.tar.gz    ← packaged submission
/kaggle/working/bc_data.pkl          ← BC warmup data (collected — see training/bc_data*.pkl.gz)
/kaggle/working/checkpoints/         ← NN checkpoints (training/ptcg_bc_v1.pth + self-play iters)
```
