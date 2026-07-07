# training/ — Local Training & Evaluation Rig

*Setup, file map, and standard workflows for the local rig: A/B testing, the
Gauntlet, bake-offs, data collection, and the opponent pool. NN architecture
and the phased training roadmap live in `docs/nn-training.md`; the stage plan
lives in `docs/competition-strategy.md` §Master Plan.*

**Last updated:** 2026-07-06

---

## Setup (once per machine — this PC and the Vivobook)

The cabt engine ships **inside `kaggle_environments`** (with native binaries for
Windows/Linux/macOS), so full games run locally at ~0.5s each. No Kaggle session
is needed for self-play, A/B testing, or data collection.

```
pip install kaggle_environments --no-deps
```

`--no-deps` matters on Windows: a transitive dependency (orbax, via gymnax) hits
the Windows long-path limit. cabt needs none of those extras. Core deps you may
also need: `flask jsonschema numpy requests` (usually already present).

For local MCTS development (search-capable `cg.api`), run once:
`python training/setup_local_search.py` — regenerates the gitignored
`training/local_cg/` package.

---

## Files

| File | Purpose |
|------|---------|
| `harness.py` | Load agents by file path, run games (parallel), summarize |
| `ab_test.py` | A/B two agent files, alternating seats, with 95% CI; logs to `ab_history.csv` |
| `ab_history.csv` | Every A/B run persisted (per-seat splits — report figure #6 input) |
| `gauntlet.py` | Fixed anchor panel + Bradley-Terry fit → gElo strength scale |
| `gauntlet_results.csv` | Accumulated panel results (append-only; the BT fit input) |
| `bakeoff.py` | Round-robin over arbitrary (agent, deck) pairs — the Stage 0c/method bake-off tool |
| `bakeoff_results.csv` | Per-game bake-off rows |
| `generic_pilot.py` | Deck-agnostic greedy pilot (tier-2 control + method-bake-off floor baseline) |
| `manifests/` | Bake-off entry lists (`tier1.csv`, `tier2.csv`) |
| `meta_survey.csv` | Ladder meta shares from `tools/meta_survey.py` |
| `ladder_history.csv` | Realized ladder Elo per shipped version (gElo calibration) |
| `phasec_replay_check.csv` | Phase C classifier verification against real replays |
| `archetype_decks.json` | Reconstructed 60-card lists per ladder archetype (belief model) |
| `bc_collect.py` | Teacher self-play → BC data for NN warmup |
| `setup_local_search.py` | Assemble the local search-capable `cg` package (`local_cg/`, gitignored) |
| `random_agent.py` | Uniform-random opponent (baseline gate + gauntlet anchor) |
| `baselines/` | Frozen `v21.py`, `v22_pre_refactor.py`, `v23.py` (anchors) + `v25c.py` (A/B reference) |
| `belief/` | Archetype classifier: `collect.py`, `train.py`, exported weights + accuracy figure |
| `nn/` | NN track: encode/model/dataset, collectors, trainers, gates, net agents — see `docs/nn-training.md` |
| `../tools/deck_audit.py` | Per-card utilization stats for deck simplification |

Bulk data (`*.pkl`, `*.pkl.gz`) and checkpoints (`*.pth`) are gitignored.
Kept local reference checkpoints: `ptcg_bc_v1/v2` (BC baselines),
`ptcg_dagger_r2` (best imitation ckpt), `ptcg_dmc_r2` (value net used by
`nn/mcts.py`), `ptcg_dmc_p0_v2_n1_richenc_v2` (adopted encoding),
`ptcg_exploiter_r1` (exploiter track). Current corpus: `mcts_p2_r3.pkl.gz`
(fixed Phase-2 self-play-with-search corpus, pending retrain).

---

## Standard workflows

```bash
# regression / candidate evaluation (run before ANY main.py ship)
python training/ab_test.py main.py training/baselines/v25c.py 400

# strength rating vs the fixed panel (distinct --name per version/deck variant!)
python training/gauntlet.py --candidate main.py --name v28 --games 200
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

# self-play-with-search collection + retrain + gate (AlphaZero track)
python training/nn/mcts_collect.py --games 300 --workers 15 --out training/mcts_p2_r3.pkl.gz
python training/nn/train_sp.py --data training/mcts_p2_r3.pkl.gz --out training/ptcg_sp_p2_r2.pth
python training/nn/dmc_replay_gate.py --ckpt <ckpt> --value-source head
```

After every ladder ship: add the realized Elo/rank to `training/ladder_history.csv`
and write the day's `docs/report-log.md` entry.

---

## Opponent Pool

**Directory:** `opponents/`

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

## Reward Shaping

**Reward shaping** (variance reduction for value targets in 100+ decision
games): per-step signals for prizes taken (+1), prizes conceded (-0.5), and
hand-size progress toward the KO threshold (+0.1 max), mixed toward the
terminal outcome (`0.7 * terminal + 0.3 * shaped`). Implemented in
`training/nn/selfplay_collect.py`; design rationale in `docs/nn-training.md`
§Value Targets. The zero-sum-consistent Φ potential lives in
`training/nn/phi_baseline.py` (`--version 2`) / `training/nn/threat.py`.

---

## Discipline

Offline win rates **systematically overrate** (v5: 64% offline, 0-5 live).
Local A/B is for (a) regression catching, (b) *ranking* candidates cheaply.
The real ladder remains the only honest evaluator — ship the top candidate and
confirm there.
