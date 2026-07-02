# Training Setup — Self-Play + Curriculum

*Plan for raising Elo from current (~750) to 1000+ using self-play with a diverse
opponent pool and curriculum data generation.*

**Last updated:** 2026-07-01

---

## Overview

**2026-07-01: the engine runs locally** — `training/` contains the working rig
(see `training/README.md`). Everything below runs on this desktop or the Vivobook
via `kaggle_environments` (`pip install kaggle_environments --no-deps`); no Kaggle
sessions needed for data generation. Measured: ~0.5s/game single-thread.

Replay BC from opponents is off the table (different decks, different strategies —
their action sequences don't transfer to Alakazam). The stage roadmap lives in
`docs/competition-strategy.md` §Master Plan; the training-side pieces are:

1. **The Gauntlet** — `training/gauntlet.py`: fixed 8-anchor panel + Bradley-Terry
   fit over all accumulated results (`gauntlet_results.csv`) → one comparable
   offline scale (gElo), calibrated against `training/ladder_history.csv`
2. **DAgger → advantage-weighted self-play** — the NN pipeline (see
   `docs/nn-training.md` §Resume Here)
3. **Curriculum via selective game starts + mined hard positions** — `training/curriculum.py`
4. **Reward shaping / n-step value targets** — intermediate signals to reduce
   variance in long games
5. **Checkpoint pool / PFSP** — train against past versions and worst matchups,
   not just current self
6. **Heuristic weight search** — `training/weight_search.py` (SPSA over `main.W`);
   improves both the ladder agent and the DAgger teacher — run unattended overnight

⚠️ The inline `battle_start(our_agent, opponent_agent, seed=...)` sample code that
used to live in this file had the wrong signature (`battle_start` takes two DECKS
and returns `(obs, StartData)`; `battle_select` takes only the select list). Use
`training/harness.py` instead — it drives games through `kaggle_environments.make("cabt")`,
which handles the loop correctly. The `battle_finish()` early-exit concern is moot:
local games are free, so uninteresting games are simply discarded after the fact.

---

## Opponent Pool

**Directory:** `opponents/`

| File | Archetype | Status |
|------|-----------|--------|
| `opponents/lucario_agent.py` | Mega Lucario ex + Rocky Energy lock (340 HP megaEx) | Complete — real deck embedded |
| `opponents/dragapult_agent.py` | Dragapult ex Stage 2 spread (Phantom Dive, 320 HP) | Complete — real deck embedded |
| `opponents/abomasnow_agent.py` | Mega Abomasnow ex energy-discard mill | Complete — real deck embedded |
| `opponents/starmie_agent.py` | Mega Starmie ex spread (330 HP megaEx) | Complete — real deck embedded |

All four agents return their DECK list when `sel is None` (required for `search_begin`
in MCTS). Adding more opponents: any `agent(obs_dict)` function following this
contract can join the pool. Crustle is the next candidate.

---

## Curriculum: Bad-Hand Games + Mined Hard Positions

Implemented in `training/curriculum.py` (the previous sample code in this section
used a wrong `battle_start` signature and is deleted). Two mechanisms:

1. **Bad-hand filter:** keep only games whose first dealt hand has no Abra and no
   search card (Poffin/Poké Pad/Dawn). Measured incidence ~5% of games — generate
   in bulk locally and discard the rest (games are free now; no early-exit concern).
2. **Hard-position mining:** every decision point is tagged against tight-spot
   predicates — `mist_walled`, `deck_danger`, `behind_3_prizes`,
   `opp_one_prize_from_win`, `weak_active`, `bad_opening`. Tagged positions
   (with full obs dicts) become (a) fixed evaluation suites for candidate nets and
   (b) alternate game starts for targeted self-play via `search_begin` on Kaggle.

```bash
python training/curriculum.py --games 500   # → training/hard_positions.pkl.gz
```

---

## Reward Shaping

Use intermediate rewards instead of terminal win/loss only. Reduces variance in long games.

```python
def shaped_reward(obs_before, obs_after, me_idx):
    """Intermediate reward signal for one turn."""
    reward = 0.0
    cur_b = (obs_before.get('current') or {})
    cur_a = (obs_after.get('current') or {})
    pl_b  = cur_b.get('players', [])
    pl_a  = cur_a.get('players', [])
    if len(pl_b) < 2 or len(pl_a) < 2: return 0.0

    my_b  = pl_b[me_idx];   my_a  = pl_a[me_idx]
    opp_b = pl_b[1-me_idx]; opp_a = pl_a[1-me_idx]

    # Prizes taken this turn (strongest signal)
    opp_prizes_before = len(opp_b.get('prize') or [])
    opp_prizes_after  = len(opp_a.get('prize') or [])
    reward += (opp_prizes_before - opp_prizes_after) * 1.0   # +1 per prize taken

    # Prizes given up this turn
    my_prizes_before = len(my_b.get('prize') or [])
    my_prizes_after  = len(my_a.get('prize') or [])
    reward -= (my_prizes_before - my_prizes_after) * 0.5     # -0.5 per prize given

    # Hand size vs KO threshold (progress signal)
    opp_active = (opp_a.get('active') or [None])[0]
    opp_hp     = (opp_active or {}).get('hp', 99999) or 99999
    my_hand_n  = my_a.get('handCount') or len(my_a.get('hand') or [])
    needed     = -(-opp_hp // 20)  # ceil(opp_hp / 20)
    if needed > 0:
        progress = min(my_hand_n / needed, 1.0)
        reward  += progress * 0.1   # small bonus for being near threshold

    return reward
```

**Weight against terminal:** Mix `0.7 * terminal_reward + 0.3 * sum(shaped_rewards)`. Tune this ratio once baseline self-play is running.

---

## Checkpoint Pool (Population-Based Training)

Training against only the current self leads to overfitting to your own weaknesses. Maintain a checkpoint pool:

```python
import random, torch

CHECKPOINT_DIR = '/kaggle/working/checkpoints/'
POOL_SIZE = 8   # keep last 8 checkpoints

def sample_opponent_from_pool(pool_paths, current_model):
    """70% chance of playing current self, 30% chance of playing a past checkpoint."""
    if random.random() < 0.3 and pool_paths:
        path = random.choice(pool_paths)
        # load checkpoint into a copy of the model, return as opponent
        ...
    return current_model
```

Combine with the meta opponents so the pool covers:
- Current self (Alakazam mirror)
- Past Alakazam checkpoints (diversity)
- Starmie ex (spread matchup)
- Lucario ex (Rocky Energy lock matchup)
- Dragapult ex (Stage 2 spread matchup)
- Abomasnow ex (energy mill matchup)

---

## Launch Checklist

- [x] Local harness (`training/harness.py` via kaggle_environments) — 2026-07-01
- [x] `battle_finish()` early-exit concern — moot (local games, discard freely)
- [x] 400-game A/B v22 vs v21: 56.3% ± 4.9%, 0 errors — 2026-07-01
- [x] Opponent-pool baselines: 94% Lucario, 94% Abomasnow, 79% Starmie,
      50W-0L-50T Dragapult (step-limit ties) — 2026-07-01
- [x] BC warmup collected + trained (`ptcg_bc_v1.pth`, old deck) — 2026-07-01
- [x] Gauntlet + deck audit tools built and smoke-tested — 2026-07-01
- [ ] **Stage 0:** deck audit at scale (`python tools/deck_audit.py --games 1000`)
- [ ] **Stage 0:** deck variant A/Bs (600 games each) → freeze the 60 → regen deck.csv
- [ ] **Stage 0:** Gauntlet baseline on new deck (`python training/gauntlet.py
      --candidate main.py --name v24-<deck> --games 200`)
- [ ] **Stage 0b:** weight search on frozen deck: `python training/weight_search.py --iters 30`
- [ ] **Stage 1:** re-collect BC data on frozen deck (`bc_collect.py --games 2000`) + retrain
- [ ] **Stage 1:** build `training/nn/dagger_collect.py`; 2–3 DAgger rounds; gate 50%+ vs teacher
- [ ] **Stage 2:** advantage weights in `train_sp.py`; AWR iterations; gate 55–60% vs teacher
- [ ] **Stage 3:** belief classifier + accuracy-by-turn figure + `opp_likely_ace_spec` fix
- [ ] **Stage 4:** frozen hard-position/bad-hand eval suites (`curriculum.py --games 2000`);
      PFSP pool
- [ ] Every run → `docs/report-log.md` entry; every ladder ship → `ladder_history.csv` row

---

## Priority Order

| Priority | Action | Impact | Compute |
|----------|--------|--------|---------|
| 1 | Deck audit → simplification → freeze (Stage 0) | Unblocks all training data | Local CPU |
| 2 | Gauntlet baselines + results logging (Stage 0) | Evaluation backbone + report figure | Local CPU |
| 3 | Weight search on frozen deck (Stage 0b, unattended) | Ladder Elo + teacher quality | Local CPU (idle) |
| 4 | BC re-collect + retrain on frozen deck (Stage 1) | Foundation for DAgger | Local CPU + Kaggle GPU |
| 5 | DAgger rounds (Stage 1) | The 70% axis; ship learned agent | Local CPU + Kaggle GPU |
| 6 | Advantage-weighted self-play (Stage 2) | Exceed the teacher | Local CPU + Kaggle GPU |
| 7 | Belief model (Stage 3, parallel B-track) | Report centerpiece | Local CPU |
| 8 | Curriculum suites + PFSP (Stage 4) | Robustness | Local CPU |
