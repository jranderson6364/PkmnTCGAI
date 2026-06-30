# PTCG AI Battle Challenge — Master Context

Read this file at the start of every session. It is the single source of truth.
Detailed sub-topics live in `docs/`.

---

## Quick Orientation

**Competition:** Pokemon TCG AI Battle Challenge (The Pokemon Company × HEROZ × Matsuo Institute × Kaggle).
**Goal:** Top-8 in the Strategy track → $30k + Tokyo finals.
**Scoring:** 70% model approach / 20% deck concept / 10% report.
**Key insight:** Rule-based bots cap out at ~0% on the 70% axis. A learned piloting agent is the only path to Strategy track.

**Current ladder submission:** v17 Alakazam heuristic agent (`main.py` + `deck.csv`), committed to this repo.
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
main.py          ← v17 Alakazam heuristic agent (active ladder submission)
deck.csv         ← 60-card deck, one card ID per line
CLAUDE.md        ← this file
docs/
  nn-training.md     ← full NN training log, architecture, roadmap
  piloting-guide.md  ← expert Alakazam piloting logic (NN training target spec)
  matchups.md        ← matchup reference + tech cheat-sheet
  version-history.md ← v1–v17 change log
  training-setup.md  ← self-play + curriculum training plan
  EN_Card_Data.csv   ← official card text/IDs reference (for opponent deck building + replay analysis)
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

## v17 Agent Architecture

**File:** `main.py`

### v17 Key Changes — competitive-research alignment (`docs/piloting-guide.md` v3)
Full research of how the deck is actually piloted (Cerys Jones' Indianapolis Regional
win, CL Osaka 2026, Limitless meta lists) confirmed our 60-card backbone matches the
meta exactly, and identified the **#1 documented leak: threshold management / overdraw**
— which is precisely what caused every long-game deck-out loss in the replays.

1. **Threshold discipline (`hand_surplus`).** Once a ready attacker exists (active or
   bench Alakazam) and `hand_n >= cards_needed` (and no Boss-snipe plan / not an
   emergency), all non-essential draw is suppressed: Dudunsparce ability → 0.5, Fez
   ability → -3, Dawn/Hilda/Poké Pad → 2.0. "Hit the threshold, then stop." Replaying
   the deck-out game confirms the agent now ENDs/attacks instead of burning Dawn/Poké
   Pad five separate times, preserving ~4-5 deck cards — the margin between decking out
   and surviving.
2. **Dudunsparce ability hard floor at `deck_danger`** (was only `deck_critical`).
3. **`active_immobile` attach prefers Psychic** (65 vs 55) so a stranded Alakazam gets
   the energy that lets it both retreat *and* attack, and a colorless Enriching isn't
   wasted on the rescue.

Verified: agent runs clean on all 1672 real selections across the 6 replays (0 errors,
0 illegal empties); still takes guaranteed lethal where available.

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

### v16 Key Changes — found via real ladder replay analysis (6 losses replayed turn-by-turn)
A custom replay analyzer (`scratchpad/analyze_replay.py` pattern, not committed) decoded the kaggle-env log format (each step's `action` resolves the *previous* step's option list) and traced 5 losing games. Found two systemic, game-losing bugs that hand-reading the code missed:

1. **ATTACK/Boss score tie at PHASE_CLOSING (the big one).** `ATTACK can_ko` and `BOSS phase==CLOSING` both scored exactly 200.0, so on ties the agent sometimes played Boss instead of taking a guaranteed lethal attack — confirmed in 2 of 4 full games, both times wasting the winning turn and KOing a *smaller*-prize bench target instead of the lethal one. Fixed: `can_ko` attack now scores 500; `boss_ex_snipe` (Boss into a strictly bigger prize KO) scores 600 so it still correctly outranks a same-turn plain KO; generic Boss-in-closing dropped to 199 and is now gated by `not can_ko`.
2. **Energy-starved stuck active → deck-out.** In 2 games, a non-attacker (Fezandipiti ex, Dudunsparce) got promoted to Active with 0 energy attached, leaving it unable to attack *or retreat* (both options absent from `select.option`). The agent then spent 30-40 turns spamming Dawn/Poké Pad searches and Fez's "Flip the Script" draw, burning its own deck to 0 with an 18-card hand it never used, and lost via deck-out at 1-2 prizes remaining. Fixed: new `active_immobile` flag (no attack, no retreat, 0 energy on active) makes attaching energy to Active score 60 (overrides normal Psychic-to-Alakazam routing); Dawn/Poké Pad score -3 while immobile (they can't fix it); Hilda scores 18 while immobile (it *can* fetch energy). Also added a hard `deck_danger` (<5 cards) floor of -8 to Dawn/Hilda/Poké Pad so no search ever fires that close to decking out, and Fez's ability now respects `deck_critical`/`deck_danger` instead of scoring a flat 5.0 regardless of deck size.

One of the 6 replayed games was an unfixable bad-luck loss (lone Abra opening hand, blind Poké Pad search whiffed onto an unplayable Stage-1 card, turn-2 Mega Lucario ex OHKO) — no heuristic bug there, just variance.

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
5. **NN track decision** — v15 diverges from v11 (the self-play warmup teacher). If NN resumes, decide: (a) recollect warmup data with v15 as teacher, OR (b) keep v11 teacher (independent tracks).
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

**Status (as of 2026-06-30):** Imitation warmup complete (self-play on v11 games, NOT opponent BC — opponent decks differ). Self-play paused pending Vivobook access.
- Warmup data: 113k samples from v11 self-play, at `/kaggle/working/bc_data.pkl`
- Best checkpoint: `sp2_iter2.pth` at ~55% vs v11
- Architecture: EmbeddingBag(22000) + Transformer(128d, 2-head) + actor-critic heads
- Exit criterion for Phase 1: net beats v11 teacher 55-60%+ over 100 games

**Note: Behavior cloning from opponent replays is NOT viable** — top bots run different decks/strategies; their action sequences don't transfer to Alakazam.

**Phase plan:**
1. Imitation warmup (self-play on v11 games) → done
2. Expert iteration self-play vs diverse pool → active (attempt 2)
3. League/PFSP hardening → next (needs meta opponent decks + Vivobook)
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
