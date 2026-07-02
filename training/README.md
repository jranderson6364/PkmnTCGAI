# training/ — Local Training & Evaluation Rig

The cabt engine ships **inside `kaggle_environments`** (with native binaries for
Windows/Linux/macOS), so full games run locally at ~0.5s each. No Kaggle session
is needed for self-play, A/B testing, weight tuning, or BC data collection.

## Setup (once per machine — this PC and the Vivobook)

```
pip install kaggle_environments --no-deps
```

`--no-deps` matters on Windows: a transitive dependency (orbax, via gymnax) hits
the Windows long-path limit. cabt needs none of those extras. Core deps you may
also need: `flask jsonschema numpy requests` (usually already present).

## Files

| File | Purpose |
|------|---------|
| `harness.py` | Load agents by file path, run games (parallel), summarize |
| `ab_test.py` | A/B two agent files, alternating seats, with 95% CI |
| `gauntlet.py` | Fixed 8-anchor panel + Bradley-Terry fit → gElo strength scale |
| `gauntlet_results.csv` | Accumulated panel results (append-only; the BT fit input) |
| `ladder_history.csv` | Realized ladder Elo per shipped version (gElo calibration) |
| `bc_collect.py` | Teacher self-play → `bc_data.pkl.gz` for NN warmup |
| `weight_search.py` | SPSA tuning of `main.W` scoring constants vs frozen v21 |
| `overnight_tune.py` | Budgeted overnight SPSA + gauntlet finals (crash-safe, auto-resume) |
| `curriculum.py` | Mine bad-hand games + tight mid-game positions |
| `baselines/v21.py` | Frozen v21 (pre-API-audit) |
| `baselines/v22_pre_refactor.py` | Frozen v22 (pre-score_options refactor) |
| `baselines/v23.py` | Frozen v23 (pre-deck-simplification) |
| `../tools/deck_audit.py` | Per-card utilization stats for deck simplification |

## Standard workflows

```bash
# regression / candidate evaluation (run before ANY main.py ship)
python training/ab_test.py main.py training/baselines/v23.py 400

# strength rating vs the fixed panel (distinct --name per version/deck variant!)
python training/gauntlet.py --candidate main.py --name v24-deckA --games 200
python training/gauntlet.py --table          # refit + print current gElo table

# per-card utilization audit (deck simplification evidence)
python tools/deck_audit.py --games 1000

# BC warmup data (~150+ samples/game; both seats harvested in mirror games)
python training/bc_collect.py --games 2000

# weight tuning (leave running overnight; checkpoints to weights_ckpt.json)
python training/weight_search.py --iters 30 --games-per-eval 120

# hard-position mining for curriculum + eval suites
python training/curriculum.py --games 500
```

After every ladder ship: add the realized Elo/rank to `training/ladder_history.csv`
and write the day's `docs/report-log.md` entry.

## Discipline

Offline win rates **systematically overrate** (v5: 64% offline, 0-5 live).
Local A/B is for (a) regression catching, (b) *ranking* candidates cheaply.
The real ladder remains the only honest evaluator — ship the top candidate and
confirm there.
