# PTCG AI Battle Challenge — Master Context

Read this file at the start of every session. It is the single source of truth.
Detailed sub-topics live in `docs/`.

---

## Quick Orientation

**Competition:** Pokemon TCG AI Battle Challenge (The Pokemon Company × HEROZ × Matsuo Institute × Kaggle).
**Goal:** Top-8 in the Strategy track → $30k + Tokyo finals.
**Scoring:** 70% model approach / 20% deck concept / 10% report.
**Key insight:** Rule-based bots cap out at ~0% on the 70% axis. A learned piloting agent is the only path to Strategy track.

**Current ladder submission:** v15 Alakazam heuristic agent (`main.py` + `deck.csv`), committed to this repo.
**NN track:** Paused at `sp2_iter2.pth` (~55% vs v11 teacher). See `docs/nn-training.md`.
**Training plan:** Self-play vs diverse opponent pool (Starmie/Lucario/Dragapult) + curriculum. See `docs/training-setup.md`.

---

## Competition Facts

| Item | Detail |
|------|--------|
| Ladder ends | ~Aug 16–17, 2026 |
| Strategy report due | ~Sep 13, 2026 |
| Team merger deadline | ~Aug 9, 2026 |
| Submissions | 5/day; only the latest 2 count for final standing |
| Submission format | `main.py` + `deck.csv` packaged as `submission.tar.gz` |
| Agent contract | `def agent(obs_dict: dict) -> list[int]` returning legal option indices |
| Clock | 10 minutes total per match; timeout = instant loss |
| Rating | Win/loss/tie only; margin irrelevant |

---

## Repo Structure

```
main.py          ← v15 Alakazam heuristic agent (active ladder submission)
deck.csv         ← 60-card deck, one card ID per line
CLAUDE.md        ← this file
docs/
  nn-training.md     ← full NN training log, architecture, roadmap
  piloting-guide.md  ← expert Alakazam piloting logic (BC target spec)
  matchups.md        ← matchup reference + tech cheat-sheet
  version-history.md ← v1–v15 change log
  training-setup.md  ← self-play + curriculum training plan
opponents/
  starmie_agent.py   ← Mega Starmie ex training opponent (DECK IDs TODO)
  lucario_agent.py   ← Mega Lucario ex + Rocky Energy training opponent (DECK IDs TODO)
  dragapult_agent.py ← Dragapult ex Stage 2 spread training opponent (DECK IDs TODO)
```

---

## Engine API (cabt + kiyotah/cg-lib)

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

**Option schema — options are positional, never carry cardId:**
- `PLAY {index, type}` → resolve via `hand[o['index']].id`
- `ATTACH {index, inPlayArea, inPlayIndex, type}` → `index`=energy hand pos; `inPlayArea` 4=active/5=bench
- `EVOLVE {index, inPlayArea, inPlayIndex}` → same pattern
- `ABILITY {area, index, type}` → resolve via active(4)/bench(5) pokemon
- `ATTACK {attackId, type}`, `RETREAT {type}`, `END {type}`
- Selections: area 1=deck (BLIND), 2=hand, 5=bench, 6=discard
- **Deck searches are blind** — `current.looking` is null; don't try to decode them

**Log types used in v14:**
- Type 10 = PLAY (cardId visible for opponent — use to detect opponent's card plays)
- Type 6 = MOVE_CARD (fromArea in {4,5} → toArea 3 = KO detection)
- Type 16 = HP_CHANGE (`putDamageCounter` flag = bench damage detection)

**Boss/gust target selection:** `stype=1`, options have `area=5`, `playerIndex=opponent`, `index` into `opp_bench`

**Setup contexts:** `ctx=1` = SETUP_ACTIVE_POKEMON, `ctx=2` = SETUP_BENCH_POKEMON

---

## Deck — Alakazam Single-Prize Control/Combo (60 cards)

### Win Condition
**Powerful Hand** (Alakazam 743, attackId 1072, cost 1 Psychic): place 2 damage counters on the opponent's Active for each card **in your hand**. `damage = 20 × hand_size`. Ignores Weakness/Resistance/reduction.

- KO threshold: `ceil(opponent_active_hp / 20)` cards needed
- Hand size IS the damage stat — never discard unnecessarily
- Blocked by: **Mist Energy (#11)** and **Rock Fighting Energy (#20)** — both say "prevent all effects of attacks"

### Card IDs

| Constant | ID | Count | Role |
|----------|----|-------|------|
| ABRA | 741 | 4 | Evolution base |
| KADABRA | 742 | 4 | Evolution middle (+2 draw on evolve) |
| ALAKAZAM | 743 | 3 | Main attacker (+3 draw on evolve) |
| DUNSPARCE | 305 | 3 | Draw engine base |
| DUNSPARCE2 | 65 | — | (alt print, not in deck) |
| DUDUNSPARCE | 66 | 3 | Run Away Draw: draw 3, shuffle back |
| GENESECT | 142 | 1 | ACE Nullifier: blocks Rocky Energy |
| SHAYMIN | 343 | 1 | Flower Curtain: prevents bench damage |
| PSYDUCK | 858 | 1 | Damp: strips self-KO abilities |
| FEZ (Fezandipiti ex) | 140 | 1 | Flip the Script: draw 3 after KO |
| POFFIN (Buddy-Buddy) | 1086 | 4 | Search 2 Basics ≤70 HP to bench |
| POKE_PAD | 1152 | 4 | Search any non-Rule-Box pokemon to hand |
| HANDHELD_FAN | 1161 | 2 | Anti-deck-out tool + Genesect ACE blocker |
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

**Energy routing rule:** Route Psychic (5, 19) → Alakazam. Route Enriching (13) → Dudunsparce (draw + recycle). Never attach Enriching to Alakazam.

### Poffin vs Poké Pad vs Dawn
- **Poffin (1086):** benches Abra(50HP) / Dunsparce(70HP) / Psyduck(70HP). Cannot grab Shaymin(80HP) or Genesect(110HP) or Fez(210HP).
- **Poké Pad (1152):** puts any non-Rule-Box pokemon into hand (includes Genesect/Shaymin/Psyduck but NOT Fez ex).
- **Dawn (1231):** grabs the full Abra+Kadabra+Alakazam line at once.

---

## v15 Agent Architecture

**File:** `main.py`

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
- `_safe_return(sel)` — always returns a valid legal action (fallback)
- `agent(obs_dict)` — entry point

### v15 Key Changes (on top of v14)
1. **Bench Alakazam evolution scoring** — bench Kadabra→Alakazam now scores 50 (no Alakazam), 40 (ESTABLISH), 25 (CONVERT), 12 (late). Was 16/10. Second Alakazam on bench is critical for continuity after active KO.
2. **Enhanced Hammer escalation** — scores 45 when opp has blocking energy (was 28). Mist/Rocky Energy means Powerful Hand deals 0; removing it is near-mandatory.
3. **Battle Cage reactive** — scores 22 when bench damage detected in logs (was flat 6). Responds to Dragapult/Starmie spread with appropriate urgency.

### v14 Key Fixes (all carried forward)
1. **Fez suppressed by default** (-1.0 score) — only activates when reactive triggers fire
2. **Sacred Ash deck-out prevention** — scores 35 at deck<5, 25 at deck<10 (was 2-5)
3. **Dudunsparce overdraw guard** — suppressed when deck<10 or hand≥14
4. **Evolution scoring fix** — active Kadabra→Alakazam evolve scores 250-270 (was 13)
5. **Boss prize-value guard** — Boss target must have `prize_value >= prize_value(opp_active)`
6. **Rock Energy detection** — Enhanced Hammer detects Mist (#11) OR Rock (#20)
7. **Genesect role implemented** — Bench Genesect + Handheld Fan scores 11 (ACE Nullifier)
8. **Shaymin reactive** — scores 16 when bench damage detected in logs
9. **Psyduck reactive** — scores 18 when opponent self-damage cards detected (placeholder IDs — see Outstanding Items)

### Prize Value Logic
```python
def prize_value(pokemon):
    if pokemon.megaEx: return 3
    if pokemon.ex: return 2
    return 1
```

---

## Outstanding Items (Priority Order)

1. **Opponent DECK IDs are all placeholder (0)** — fill in `opponents/*.py` using `all_card_data()` on Kaggle before self-play can start. See `docs/training-setup.md`.
2. **v15 A/B harness validation pending** — run `ab_test(v15, random, n=200)` and `ab_test(v15, v11, n=400)` on Kaggle to confirm improvements over v14.
3. **Psyduck threat detection uses guessed placeholder IDs `{109, 110, 111}`** — grep `all_card_data()` for self-damage ability text to find real Dusknoir card ID.
4. **`opp_likely_ace_spec` hardcoded to True** — infer from early-game logs (opponent archetype detection).
5. **NN track decision** — v15 diverges from v11 (the BC teacher). If NN resumes: (a) recollect BC data with v15 as teacher, OR (b) keep v11 as BC teacher (independent tracks).
6. **Verify `battle_finish()` early-exit behavior** — curriculum data generation exits bad-hand games early; confirm this does not count as a ladder loss before running at scale.

---

## Design Principles

1. **Real-ladder A/B is the only honest evaluator.** Offline win rates systematically overrate (v5 went 64% offline, 0-5 live). Never optimize on offline sim alone.
2. **Timeout = instant loss.** Every inference path must have a guaranteed fast fallback (`_safe_return`).
3. **Single-prize + non-ex beats the meta.** Crustle immunity walls all ex attackers. Alakazam hits Crustle normally. Prize trade math: giving up 1, taking 2-3.
4. **Hand size IS damage.** Never discard unnecessarily. No Professor's Research, no Iono, no Ultra Ball.
5. **Rule-based bots cap at ~0% on the 70% Strategy axis.** The neural net track is the competition goal; heuristic v14 is the ladder placeholder while NN training runs.
6. **5 submissions/day, only the latest 2 count.** Use the ladder as an A/B testing rig.

---

## NN Track Summary

See `docs/nn-training.md` for full details.

**Status (as of 2026-06-30):** BC warm-start complete. Self-play paused pending Vivobook access.
- BC data: 113k samples from v11 teacher, at `/kaggle/working/bc_data.pkl`
- Best checkpoint: `sp2_iter2.pth` at ~55% vs v11 teacher
- Architecture: EmbeddingBag(22000) + Transformer(128d, 2-head) + actor-critic heads
- Exit criterion for Phase 1: net beats teacher 55-60%+ over 100 games

**Phase plan:**
1. BC warm-start → done
2. Expert iteration self-play → active (attempt 2)
3. League/PFSP hardening → next (needs meta opponent decks)
4. Report writeup → Aug–Sep

---

## Kaggle File Locations

```
/kaggle/working/main.py              ← v15 source (submitted)
/kaggle/working/deck.csv             ← 60-card deck
/kaggle/working/submission.tar.gz    ← packaged submission
/kaggle/working/bc_data.pkl          ← 113k BC samples (v11 teacher)
/kaggle/working/bc_final.pth         ← current best weights
/kaggle/working/checkpoints/
  sp_iter1.pth                       ← best self-play checkpoint (46% vs teacher, pre-collapse)
  sp2_iter*.pth                      ← attempt 2 checkpoints (current run, ~55% at iter2)
```

---

## Packaging for Submission

```python
import os, tarfile, subprocess, py_compile

# Validate syntax
py_compile.compile('main.py', doraise=True)

# Count deck
with open('deck.csv') as f:
    assert len(f.readlines()) == 60, "deck must be exactly 60 cards"

# Package
with tarfile.open('submission.tar.gz', 'w:gz') as tar:
    tar.add('main.py')
    tar.add('deck.csv')
```
