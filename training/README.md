# training/ — Local Training & Evaluation Rig

*Setup, file map, and standard workflows for the local rig: A/B testing, the
Gauntlet, SPSA weight tuning, BC/self-play data collection, and the opponent
pool. NN architecture and the phased training roadmap live in
`docs/nn-training.md`; the stage plan lives in `docs/competition-strategy.md`
§Master Plan.*

**Last updated:** 2026-07-03

---

## Setup (once per machine — this PC and the Vivobook)

The cabt engine ships **inside `kaggle_environments`** (with native binaries for
Windows/Linux/macOS), so full games run locally at ~0.5s each. No Kaggle session
is needed for self-play, A/B testing, weight tuning, or BC data collection.

```
pip install kaggle_environments --no-deps
```

`--no-deps` matters on Windows: a transitive dependency (orbax, via gymnax) hits
the Windows long-path limit. cabt needs none of those extras. Core deps you may
also need: `flask jsonschema numpy requests` (usually already present).

---

## Files

| File | Purpose |
|------|---------|
| `harness.py` | Load agents by file path, run games (parallel), summarize |
| `ab_test.py` | A/B two agent files, alternating seats, with 95% CI; logs to `ab_history.csv` |
| `ab_history.csv` | Every A/B run persisted (per-seat splits — report figure #6 input) |
| `gauntlet.py` | Fixed 8-anchor panel + Bradley-Terry fit → gElo strength scale |
| `gauntlet_results.csv` | Accumulated panel results (append-only; the BT fit input; per-seat splits + run id since 2026-07-03) |
| `bakeoff.py` | Round-robin over arbitrary (agent, deck) pairs — the Stage 0c/method bake-off tool; per-game rows to `bakeoff_results.csv` |
| `bakeoff_results.csv` | Per-game bake-off rows (seats, winner, turns, prizes, first-attack turn, end reason, run id) |
| `generic_pilot.py` | Deck-agnostic greedy pilot (tier-2 control + method-bake-off floor baseline) |
| `manifests/` | Bake-off entry lists (`tier1.csv`, `tier2.csv`) |
| `meta_survey.csv` | Ladder meta shares from `tools/meta_survey.py` |
| `ladder_history.csv` | Realized ladder Elo per shipped version (gElo calibration) |
| `bc_collect.py` | Teacher self-play → `bc_data.pkl.gz` for NN warmup |
| `weight_search.py` | SPSA tuning of `main.W` scoring constants vs frozen v21 |
| `overnight_tune.py` | Budgeted overnight SPSA + gauntlet finals (crash-safe, auto-resume) |
| `tune_ckpt.json`, `tune_log.jsonl` | 2026-07-02 tune run evidence (kept for the report) |
| `curriculum.py` | Mine bad-hand games + tight mid-game positions |
| `random_agent.py` | Uniform-random opponent (baseline gate + gauntlet anchor) |
| `baselines/` | Frozen `v21.py`, `v22_pre_refactor.py`, `v23.py` (A/B refs + anchors) |
| `nn/` | NN track: encode/model/dataset, BC + self-play training, net agents — see `docs/nn-training.md` |
| `kaggle_notebook/` | Kaggle notebook builders + built notebooks (BC training, MCTS spike) |
| `kaggle_upload/` | Kaggle dataset staging (`dataset-metadata.json`; data files copied in at upload time) |
| `../tools/deck_audit.py` | Per-card utilization stats for deck simplification |

---

## Standard workflows

```bash
# regression / candidate evaluation (run before ANY main.py ship)
python training/ab_test.py main.py training/baselines/v23.py 400

# strength rating vs the fixed panel (distinct --name per version/deck variant!)
python training/gauntlet.py --candidate main.py --name v24-deckA --games 200
python training/gauntlet.py --table          # refit + print current gElo table

# per-card utilization audit (deck simplification evidence)
python tools/deck_audit.py --games 1000

# deck/method bake-off round-robin (see docs/report-log.md pre-registrations)
python training/bakeoff.py --manifest training/manifests/tier1.csv --games 200 --tag tier1
python training/bakeoff.py --sanity            # mirror-match gate: CI must cover 0.5
python training/bakeoff.py --table --tag tier1 # reprint matchup matrix + BT + metrics

# ladder meta survey (archetype share from downloaded replays)
python tools/meta_survey.py --all --csv training/meta_survey.csv

# BC warmup data (~150+ samples/game; both seats harvested in mirror games)
python training/bc_collect.py --games 2000

# weight tuning (leave running overnight; checkpoints to weights_ckpt.json)
python training/weight_search.py --iters 30 --games-per-eval 120

# hard-position mining for curriculum + eval suites
python training/curriculum.py --games 500
```

After every ladder ship: add the realized Elo/rank to `training/ladder_history.csv`
and write the day's `docs/report-log.md` entry.

---

## Opponent Pool

**Directory:** `opponents/` (official Kaggle sample notebooks they were built
from: `opponents/samples/`).

| File | Archetype | Status |
|------|-----------|--------|
| `opponents/lucario_agent.py` | Mega Lucario ex + Rocky Energy lock (340 HP megaEx) | Complete — real deck embedded |
| `opponents/dragapult_agent.py` | Dragapult ex Stage 2 spread (Phantom Dive, 320 HP) | Complete — real deck embedded |
| `opponents/abomasnow_agent.py` | Mega Abomasnow ex energy-discard mill | Complete — real deck embedded |
| `opponents/starmie_agent.py` | Mega Starmie ex spread (330 HP megaEx) | Complete — real deck embedded |

All four agents return their DECK list when `sel is None` (required for
`search_begin` in MCTS). Adding more opponents: any `agent(obs_dict)` function
following this contract can join the pool. Crustle is the next candidate.

---

## Curriculum & Reward Shaping

`curriculum.py` implements two mechanisms (→ `training/hard_positions.pkl.gz`):

1. **Bad-hand filter:** keep only games whose first dealt hand has no Abra and
   no search card (Poffin/Poké Pad/Dawn). Measured incidence ~5% of games —
   generate in bulk locally and discard the rest (local games are free).
2. **Hard-position mining:** every decision point is tagged against tight-spot
   predicates — `mist_walled`, `deck_danger`, `behind_3_prizes`,
   `opp_one_prize_from_win`, `weak_active`, `bad_opening`. Tagged positions
   (with full obs dicts) become (a) fixed evaluation suites for candidate nets
   and (b) alternate game starts for targeted self-play via `search_begin`.

**Reward shaping** (variance reduction for value targets in 100+ decision
games): per-step signals for prizes taken (+1), prizes conceded (-0.5), and
hand-size progress toward the KO threshold (+0.1 max), mixed toward the
terminal outcome (`0.7 * terminal + 0.3 * shaped`). Implemented in
`training/nn/selfplay_collect.py`; design rationale in `docs/nn-training.md`
§Value Targets.

**Checkpoint pool / PFSP** (Stage 4): sparring partners drawn ~70% current
self / ~30% past checkpoints + the meta opponents above, weighted toward
current losing matchups — see `docs/competition-strategy.md` §Master Plan.

---

## Discipline

Offline win rates **systematically overrate** (v5: 64% offline, 0-5 live).
Local A/B is for (a) regression catching, (b) *ranking* candidates cheaply.
The real ladder remains the only honest evaluator — ship the top candidate and
confirm there.
