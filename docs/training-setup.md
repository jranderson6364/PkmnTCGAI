# Training Setup — Self-Play + Curriculum

*Plan for raising Elo from 600–700 to 1000+ using self-play with a diverse opponent pool and curriculum data generation.*

---

## Overview

Replay BC from opponents is off the table (different decks, different strategies — their action sequences don't transfer to Alakazam). Instead:

1. **Self-play against a diverse opponent pool** — v15 heuristic + meta opponent bots
2. **Curriculum via selective game starts** — bias training toward hard positions
3. **Reward shaping** — intermediate signals to reduce variance in long games
4. **Checkpoint pool** — train against past versions, not just current self

Needs the Vivobook (16 CPU workers) for self-play throughput. All code below is pre-staged; run it when the machine is available.

---

## Opponent Pool

**Directory:** `opponents/`

| File | Archetype | Status |
|------|-----------|--------|
| `opponents/starmie_agent.py` | Mega Starmie ex spread (330 HP megaEx) | Code complete; DECK IDs TODO |
| `opponents/lucario_agent.py` | Mega Lucario ex + Rocky Energy lock (340 HP megaEx) | Code complete; DECK IDs TODO |
| `opponents/dragapult_agent.py` | Dragapult ex Stage 2 spread (Phantom Dive) | Code complete; DECK IDs TODO |

**Filling in DECK IDs (do this first on Kaggle):**
```python
from cg.api import all_card_data
card_names = {c.cardId: c.name for c in all_card_data()}
# Then search: {k:v for k,v in card_names.items() if 'Starmie' in v}
```

Replace all `0` values in each agent's card ID constants and build a realistic 60-card DECK list. The strategic logic (heuristic) does not depend on specific card IDs — only the DECK list and `_pick_active` need real IDs.

**Adding more opponents later:** Any `agent(obs_dict)` function that returns DECK when `sel is None` and legal indices otherwise can join the pool. Crustle is next candidate.

---

## Curriculum via Selective Game Starts

Generate training data biased toward hard opening hands:

```python
from cg.game import battle_start, battle_select, battle_finish
import random

def has_bad_hand(obs):
    """True if this starting hand has no Abra AND no search card (Poffin/Poké Pad/Dawn)."""
    ABRA = 741
    SEARCH_IDS = {1086, 1152, 1231}  # Poffin, Poké Pad, Dawn
    cur = obs.get('current') or {}
    me  = cur.get('yourIndex', 0)
    pl  = cur.get('players', [])
    hand = (pl[me].get('hand') or []) if len(pl) > me else []
    hand_ids = {(c or {}).get('id', -1) for c in hand}
    return ABRA not in hand_ids and not (hand_ids & SEARCH_IDS)

def generate_curriculum_game(our_agent, opponent_agent, seed=None):
    """
    Start a game, check if the opening hand is bad (curriculum target).
    If not bad, return None (skip this game to preserve curriculum bias).
    If bad, play out and return the full game trajectory.
    Returns: list of (obs_dict, chosen_indices) pairs, or None.
    """
    obs = battle_start(our_agent, opponent_agent, seed=seed)
    if not has_bad_hand(obs):
        battle_finish(obs)   # clean exit — do NOT count as a loss
        return None
    trajectory = []
    while True:
        sel = obs.get('select')
        if sel is None:
            break
        action = our_agent(obs)
        trajectory.append((obs, action))
        obs, done = battle_select(obs, action)
        if done:
            break
    return trajectory

# Usage: generate N curriculum games
# data = [t for _ in range(5000) if (t := generate_curriculum_game(agent, lucario_agent)) is not None]
```

**Note on early exits:** The engine may record an early exit as a loss. Verify with `battle_finish()` semantics before running at scale. If early exits cost rating, run curriculum generation in a local harness (not on the live ladder).

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

Combine with the meta opponents (Starmie, Lucario, Dragapult) so the pool covers:
- Current self (Alakazam mirror)
- Past Alakazam checkpoints (diversity)
- Starmie ex (spread matchup)
- Lucario ex (Rocky Energy lock matchup)
- Dragapult ex (Stage 2 spread matchup)

---

## Launch Checklist (When Vivobook Available)

- [ ] Fill in DECK card IDs in all three opponent agents (Kaggle notebook: `all_card_data()`)
- [ ] Verify `battle_finish()` behavior for early-exit curriculum games
- [ ] Set up local self-play harness (based on `battle_start/battle_select/battle_finish`)
- [ ] Run 200-game A/B: v15 vs random (sanity check)
- [ ] Run 400-game A/B: v15 vs v11 (confirm v15 improvement)
- [ ] Start self-play: v15 vs pool (Starmie + Lucario + Dragapult + self)
- [ ] First curriculum run: 1000 bad-hand games vs Lucario (Rocky Energy lock)
- [ ] Checkpoint pool: save every 50 self-play iterations

---

## Priority Order

| Priority | Action | Elo Impact | Compute |
|----------|--------|-----------|---------|
| 1 | Fill opponent DECK IDs, run A/B harness | Measurement only | Kaggle only |
| 2 | Self-play vs opponent pool | High | Vivobook |
| 3 | Curriculum bad-hand generation | Medium | Vivobook |
| 4 | Reward shaping integration | Medium | Vivobook |
| 5 | Checkpoint pool diversity | Medium | Vivobook |
