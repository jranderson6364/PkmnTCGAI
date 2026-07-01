# Neural Net Training Log & Roadmap

**Last updated:** 2026-07-01  
**Status:** Paused at `sp2_iter2.pth` (~55% vs v11 teacher, right at the Phase 1 exit
threshold) while the heuristic ladder track (now v21, ~750 Elo / 1600-4000) runs.

---

## Resume Here

Concrete next steps, in order, picking this back up:

1. **Run the clean 100-game eval of `sp2_iter2.pth` vs v11** (pending — this is the
   actual blocker on the Phase 1 → Phase 2 decision below). If ≥55-60%, Phase 1 is
   done. If it's borderline/a fluke, run 2-3 more self-play iterations first.
2. **BC-teacher decision — resolved as Option B (keep v11).** Don't recollect BC
   warm-start data from the current heuristic (v21), even though it's now
   meaningfully stronger than v11. Self-play iterations exist specifically to grow
   past the seed teacher's ceiling — recollecting is a real time cost for a benefit
   the training loop should already deliver. Heuristic (v21+) and NN stay
   independently-scored, parallel tracks.
3. **Fill in the opponent `DECK` IDs** in `opponents/starmie_agent.py`,
   `opponents/lucario_agent.py`, `opponents/dragapult_agent.py` — still `[0]*60`
   placeholders. One `all_card_data()` call on Kaggle. Hard blocker on both the
   self-play opponent pool and Phase 2's league/PFSP hardening.
4. **Open/unverified:** whether `kiyotah/cg-lib` can be `kaggle datasets download`'d
   and used standalone outside a Kaggle notebook session — would unlock local
   self-play iteration speed instead of Kaggle's session limits / ~30hr weekly GPU
   quota. See `README.md` § Local Development.

---

## Architecture

**Encoder:** EmbeddingBag(22000 vocab, 128d) + 1-layer TransformerEncoder(128d, 2-head, 256 FFN)  
**Decoder:** EmbeddingBag(decoder_size vocab, 128d) + 1 DecoderLayer (cross-attention) + policy head  
**Value head:** Linear(128,1) → tanh → scalar  
**Policy head:** Linear(128,1) → 64-dim logit vector  

**Encoder constants (must match between data collection and training):**
```python
encoder_size = 22000
card_count = max_card_id + 1          # from all_card_data()
attack_count = max_attack_id + 1      # from all_attack()
decoder_main_feature = 8
decoder_attack_offset = 14
decoder_card_offset = decoder_attack_offset + attack_count
decoder_size = decoder_card_offset + (1 + decoder_main_feature + SelectContext.RECOVER_SPECIAL_CONDITION) * card_count
num_words_encoder = 24
```

**24 encoder words:** bench(8 slots × 2 players) + active(×2) + player_state(×2) + hand + deck + stadium + misc

---

## BC Warm-Start (Phase 0 — Complete)

**Teacher:** v11 agent (83% vs random)  
**Games collected:** 700 teacher-vs-teacher  
**Samples:** 113,119 BCStep objects → `/kaggle/working/bc_data.pkl`  
**Each BCStep:** `sv_enc, sv_dec, n_actions, chosen_idx, outcome`  
**~158 decisions/game** (Alakazam has high branching due to search/evolve sub-selections)

**Training run 1** (200 games, 5 epochs, LR 3e-4, from scratch):
- Policy loss: 0.84 → 0.62 (-27%)
- Eval (30 games): 66% vs random, 43% vs teacher

**Training run 2** (113k samples, 10 epochs, LR 1e-4, resumed):
- Policy loss: 0.62 → 0.48 (-23% further)
- Eval (100 games): 65% vs random | 68% teacher vs random | **52% net vs teacher**
- Conclusion: BC plateau. Self-play required to exceed teacher.

---

## Self-Play Phase (Phase 1 — Active)

### Design
- Net plays vs itself using MCTS (10 sims per decision)
- `search_begin` fills hidden zones: own deck/prizes sampled from DECK; opponent hand/deck/prizes filled with placeholder `[1072]*n` (Snorlax) — belief model replaces this in Phase 2
- Policy targets = advantage-based (child Q - root Q, clamped to [-1, 1])
- Value target = game outcome (1.0 win, -1.0 loss, 0.0 draw)
- UCB: `q + 0.4 * sqrt(parent_visit) * prior / (1 + child_visit)`
- Loss: HuberLoss for value (delta=0.2) + HuberLoss for policy (delta=0.1, masked to valid actions)

### Attempt 1 — Collapsed (do not repeat)
- Config: 3 iters, 100 games each, 5 epochs/iter, LR 1e-4, NO BC mixing
- Results: iter1=46%, iter2=46%, iter3=20% vs teacher
- Cause: small SP dataset (~17k) + too many epochs → catastrophic forgetting
- **Lesson: always mix BC data into SP training batches**

### Attempt 2 — Current (recovering from sp_iter1.pth)
- Config: 5 iters, 100 games each, 2 epochs/iter, LR 3e-5, **40% BC / 60% SP batch mixing**
- SP buffer grows across iterations; BC buffer = full 113k samples always present
- Checkpoints: `/kaggle/working/checkpoints/sp2_iter*.pth`
- **Best so far: sp2_iter2.pth at ~55% vs v11 teacher**

### Exit criterion for Phase 1
Net wins **55-60%+ vs teacher** over 100 games → proceed to Phase 2.

When attempt 2 completes:
1. Run full eval: net vs random, net vs teacher, net vs sp_iter1
2. If improving: run 3 more iterations with SEARCH_COUNT=20
3. If plateau: debug MCTS (check search_begin is called, policy targets have variance, value head learning)
4. Commit best checkpoint to Kaggle dataset for persistence

---

## Critical: Decoder Padding Rule

**SP samples (SPStep):** `sv_dec` is pre-padded to 64 words INSIDE `eval_node` before storing.  
**BC samples (BCStep):** `sv_dec` is NOT pre-padded — must pad in training batch builder:
```python
if not hasattr(s, 'policy_targets'):  # BC sample
    for _ in range(64 - s.n_actions):
        dec.offset.append(len(dec.index))
```
**Violating this causes:** `RuntimeError: shape '[128, -1, 128]' is invalid`

---

## MCTS Implementation Notes

```python
# search_begin kwargs (confirmed)
search_begin(your_deck=..., your_prize=...,
             opponent_deck=..., opponent_hand=...,
             opponent_active=..., opponent_prize=...)

# Step and end
new_state = search_step(searchId, select_list)
search_end()  # release after each decision

# Negation: when state.yourIndex != your_index (opponent's turn), negate value in UCB
# Backprop propagates value up through parent chain
```

---

## Net Agent Inference (single forward pass)

```python
obs = to_observation_class(obs_dict)
actions = enumerate_actions(obs)           # list of action index lists (up to 64)
sv_enc = get_encoder_input(obs, DECK)
sv_dec = get_decoder_input(obs, actions)   # pad to 64 words
mask = [float('-inf') if invalid else 0.0 for each action]
value, policy = model(ie, ve, oe, id_, vd, od)
best = (policy + mask).argmax()
return list(actions[best])
```

---

## Phase 2: League/PFSP Hardening (Next)

**Trigger:** net consistently beats teacher (~55%+ over 100 games)

Actions:
- Get legal 60-card IDs for meta opponents: Mega Starmie ex, Dragapult ex, Bellibolt ex, Crustle
- Run net vs each meta deck (random pilot first; rule-based from wmh/ptcg-abc if available)
- Hall-of-fame: keep past checkpoints as sparring partners
- PFSP weighting: prioritize opponents the net is currently losing to

**Originality contribution — opponent belief model:**  
Replace `search_begin`'s `[1072]*n` placeholder with a real belief distribution:
- Parse `obs["logs"]` to infer opponent archetype (Starmie vs Dragapult vs Crustle etc.)
- Sample opponent hidden zones consistent with inferred archetype
- This is the standout contribution for the 70% report axis

---

## Kaggle Setup

```
Engine: Add Input → kiyotah/cg-lib
GPU: Session → Accelerator → GPU T4 x1 (for training)
CPU: for data collection
Weekly GPU quota: ~30 hrs (resets weekly)
Save Version (Save & Run All) → commits /kaggle/working/ outputs
```

---

## Heuristic vs NN Track Decision — Resolved

The heuristic ladder track (now v21) has diverged significantly from v11 (the fixed
BC teacher). **Decided: Option B** — keep v11 as the BC teacher permanently, treat
the heuristic and NN as independent, parallel tracks rather than recollecting BC
data every time the heuristic improves. See "Resume Here" at the top of this file
for the reasoning.
