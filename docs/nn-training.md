# Neural Net Training Log & Roadmap

*Running log of NN architecture, training status, and the phased roadmap for the
learned-piloting agent (the 70%-weighted "model approach" axis).*

**Last updated:** 2026-07-03
**Evidence rule:** method claims route through the pre-registered method bake-off —
see `docs/competition-strategy.md` → Method Bake-off Protocol and `docs/report-log.md`.
**Status:** RESTARTED. All prior training data was lost, but the engine now runs
**fully locally** (`training/README.md`) — data collection no longer needs Kaggle
or even the Vivobook (the Vivobook multiplies throughput). Teacher is now v25c
(was v22). `ptcg_bc_v2.pth` (frozen-deck BC retrain) is done and gated.
DAgger rounds 1 and 2 are done: both work (fidelity 73%→81%→82% across BC→
r1→r2) but win-rate vs teacher stayed flat (~12-17%) throughout, and round 2
(testing a lower collection temperature) showed the fidelity gain itself
diminishing (+8pp then +0.8pp) — two rounds of evidence that further DAgger
rounds won't move the needle much more. **DAgger paused here**; `ptcg_dagger_r2.pth`
is the best checkpoint but not ship-ready (needs Stage 2 AWR/search to exceed
the teacher, not more imitation). See "Resume Here" below.

---

## Status (2026-07-01, heuristic-blended MCTS — local parts complete)

Per the approved plan (`when-do-we-start-eager-mountain.md`), the three
Kaggle-independent parts are built and verified:

1. **`main.py` scoring is now standalone-callable.** `score_options(obs, sel)`
   (dispatching by select type) and `score_options_main(obs, sel)` expose the
   heuristic's per-option scores without needing the old argmax-and-return
   control flow. Verified behavior-preserving: 210W-190L (52.5%, well within
   the 95% CI of 50%) over 400 real games vs a pre-refactor snapshot, plus a
   1,875-decision fuzz test across all select types in 15 real games (0
   exceptions, correct output shape every time). `_pick_boss_target` was left
   untouched (too game-critical to risk drift); a parallel `_score_boss_target`
   was added instead and verified to agree with the real function's choice in
   50/50 real instances.
2. **`training/nn/prior_blend.py`** — `heuristic_prior`/`net_prior`/`blended_prior`,
   mixing softmaxed heuristic scores and net policy logits as distributions
   (not raw scores — the scales are wildly different). New `main.py` `W`
   entries: `prior_T_h_main` (40.0), `prior_T_h_default` (3.0), `prior_T_net`
   (1.0). `anneal_lambda()` implements the evidence-gated λ schedule (start
   0.8, step down 0.15 only when a checkpoint beats the previous one by more
   than the A/B's 95% CI, floor 0.2). Diagnostic run on 500 real saved samples:
   0 NaNs; heuristic/net/blend argmax agreement with the actual action taken
   at 85.2%/87.8%/91.4% respectively (sane — many select types get a flat/
   uniform heuristic prior by design, so <100% agreement there is expected).
3. **Soft policy-target + MCTS-Q plumbing** — `dataset.py` now carries an
   optional `policy_target` (normalized visit counts, once `mcts_collect.py`
   exists) through `__getitem__`/`collate`, falling back to one-hot(label) for
   plain BC/direct-SP samples. `train_sp.py` trains with soft cross-entropy
   against `policy_targets` (a strict generalization of hard CE — verified:
   loss magnitude with the one-hot fallback matches the pre-change hard-CE
   loss almost exactly). `selfplay_collect.py.compute_value_targets` accepts
   an optional `mcts_root_values` list to use as the bootstrap `V(s_t)` in
   place of the raw value head, verified to actually change targets where the
   bootstrap window engages, and to leave terminal-dominated targets alone.

**Blocked on Kaggle (next step):** the `SearchState` discovery spike — the
actual tree (`training/nn/mcts.py`) and its Kaggle self-play driver
(`training/nn/mcts_collect.py`) can't start until `cg.api.search_begin`'s
real param order and return structure are confirmed in a Kaggle session (see
the plan file for the exact spike steps). The direct self-play loop
(`selfplay_collect.py`/`train_sp.py`, no tree) keeps running and shipping
checkpoints in parallel in the meantime — unaffected by any of the above.

---

## Status (2026-07-01, later same day)

BC warmup complete: `training/ptcg_bc_v1.pth`, trained on all 547,796 v22 self-play
samples (10 epochs, Kaggle T4). Held-out top-1 action-match accuracy 85.9%. Real-game
gates (`training/net_agent.py` via `training/ab_test.py`, 100 games each):
- vs random: **86% (86W-14L)** — clears the 65% target.
- vs v22 heuristic: **22% (22W-78L)** — well below the ~50% parity target, as
  expected for a first BC pass (compounding-error/distributional-shift, not a bug —
  0 errors in both runs). This is the seed for self-play, not a ladder-ready net.

**Self-play Phase 1 is built and locally smoke-tested** (`training/nn/`):
`selfplay_agent.py` (temperature-softmax sampling for exploration, env-configurable
checkpoint/temperature), `selfplay_collect.py` (net-vs-net games via the local
engine; computes n-step bootstrapped value targets — see below), `train_sp.py`
(warm-starts from a checkpoint, 40% BC / 60% SP mixed batches via
`WeightedRandomSampler`). `dataset.py` transparently uses `value_target` when
present, else falls back to terminal `outcome`, so BC and SP shards share one
loader.

**Scope note — MCTS is deferred, this is direct self-play.** True MCTS needs
`cg.api.search_begin`/`search_step` (tree search over hypothetical futures), which
only exists in the `kiyotah/cg-lib` dataset and must run on Kaggle — it cannot run
against the local `kaggle_environments` engine we use for fast iteration. Phase 1
as built plays full real games with policy sampling for exploration and bootstraps
value targets with the net's own value head; it captures the core self-improvement
loop (fresh data from the current policy → retrain → repeat) without the search
tree. Upgrading to real MCTS-in-the-loop is a Kaggle-only follow-up (see Phase 2).

**To run at scale** (Vivobook, no Kaggle needed for collection):
```bash
python training/nn/selfplay_collect.py --games 500 --ckpt training/ptcg_bc_v1.pth \
    --temp 1.0 --workers 14 --out training/sp_data.pkl.gz
python training/nn/train_sp.py --bc-data "training/bc_data*.pkl.gz" \
    --sp-data training/sp_data.pkl.gz --init training/ptcg_bc_v1.pth \
    --out training/ptcg_sp_iter1.pth --epochs 3
python training/ab_test.py training/nn/net_agent.py main.py 200   # set NET_CKPT=.../ptcg_sp_iter1.pth
```
Repeat: collect fresh self-play with the newest checkpoint, retrain, re-evaluate.
Exit criterion unchanged: 55-60%+ vs v22 over 100+ games before shipping.

---

## Resume Here (2026-07-01 roadmap revision — DAgger first)

**Why the plan changed:** direct self-play as previously designed (net imitates
its own temperature-sampled games) has **no improvement operator** — nothing
makes iteration k+1 better than k, so the most likely outcome was hovering at
the BC seed forever. See the glossary in `docs/report-log.md`. The revised
pipeline names an operator at every step: DAgger (teacher supervision on the
net's own state distribution) → advantage weighting (imitate
better-than-expected actions harder) → optionally MCTS expert iteration
(Kaggle-gated). Full roadmap: `docs/competition-strategy.md` §Master Plan.

Concrete next steps, in order:

0. ~~WAIT for the Stage 0 deck freeze~~ DONE — deck frozen at v23 (see
   `CLAUDE.md` Deck Essentials); old-deck `bc_data*.pkl.gz` / `ptcg_bc_v1.pth`
   superseded (kept for reference, no longer the active corpus/checkpoint).
1. ~~Re-run BC warmup on the frozen deck~~ DONE 2026-07-03: `python
   training/bc_collect.py --games 2000` on v25c → 2000 games, 579,169 samples,
   `training/bc_data_v25c*.pkl` (5 shards, uncompressed, ~3.1GB — note for next
   collection: gzip or `--out` with a `.gz` suffix keeps this smaller). Retrain
   **runs locally now** (`encode.py`/`model.py` were rebuilt cg-lib-free per the
   "RESTARTED" note above, no Kaggle GPU needed) — `python training/nn/train_bc.py
   --data "training/bc_data_v25c*.pkl" --epochs 10 --out training/ptcg_bc_v2.pth`.
   Took ~106 min wall (longer than the ~75-90 min smoke-test extrapolation, but
   CPU-time deltas confirmed steady never-stalled compute throughout — just
   real per-epoch cost at this sample count, not a hang; peak ~24GB RSS,
   comfortably within this machine's 39.6GB). `training/ptcg_bc_v2.pth`,
   val_top1_acc 0.8397 → 0.8750 (epoch 8, best) → 0.8740 (epoch 9, saved).
   **Real-game gates** (`training/ab_test.py`, 100 games each): vs random
   **86% (86W-14L)** — matches `ptcg_bc_v1.pth`'s 86% almost exactly; vs the
   v25c heuristic teacher **17% (17W-83L)** — even below v1's 22% vs v22,
   consistent with v25c (gElo 589) being meaningfully stronger than v22 was.
   This is the expected BC-plateau/compounding-error signature, not a red
   flag — it's the reason DAgger exists. See `docs/report-log.md` 2026-07-03
   entry for full numbers.
2. **DAgger rounds (Stage 1) — round 1 done: confirmed working, win-rate gate
   just can't see it yet.** `training/nn/dagger_collect.py` — BUILT
   2026-07-03: the *net* (`selfplay_agent.py`, temperature-sampled for
   exploration) pilots mirror games; at every decision, `main.score_options(obs,
   sel)`'s argmax is recorded as the label (not the net's own action) — same
   shard/writer format as `bc_collect.py`. Round 1 collected: 1000 games,
   `training/dagger_data_r1*.pkl`, 326,240 samples, 0 relabel errors.

   Retrained via `training/nn/train_sp.py` (warm-start + BC/DAgger mixed
   batches, 40/60 ratio). First pass (`ptcg_dagger_r1.pth`, 3 epochs/lr 5e-5)
   gated at 12% vs teacher (nominally below BC's 17%, but not statistically
   distinct at n=100) — diagnosed via a 96.8% label-agreement check as
   undertraining, not bad labels. Retrained harder (`ptcg_dagger_r1b.pth`, 10
   epochs, lr 2e-4, comparable total steps to the BC warmup) — gated at
   **15% vs teacher, still statistically flat.**

   **The decisive measurement (100-game win-rate can't resolve what DAgger
   targets — per-decision fidelity, not aggregate outcome):** collected 60
   fresh argmax/temp≈0 deployment-realistic games (not training states) and
   compared BOTH checkpoints' argmax against the teacher's on the SAME 3000
   sampled states: `ptcg_bc_v2.pth` **74.9%** agreement vs
   `ptcg_dagger_r1b.pth` **79.7%** — a real +4.8pp gain. **DAgger round 1 IS
   working**; the flat win-rate is a measurement-resolution problem (a
   single-decision accuracy gain compounds over ~150 decisions/game, but
   detecting that in head-to-head win-rate needs more rounds or a much
   larger n than 100 to clear the ~±7% noise floor), not a broken pipeline.
   Paranoia-checked first (per advisor, given this session's infra
   gremlins): confirmed the two checkpoints load as genuinely distinct
   weights (max abs diff 0.144), ruling out a silent-fallback bug.

   **Infra bug found + fixed at the root (not just papered over):**
   `--bc-limit`/`--sp-limit` alone didn't prevent the earlier OOM kills
   because `training/nn/dataset.py::load_shards` always read every
   glob-matched shard FULLY before any caller-side slicing — a "capped" load
   still momentarily needed the entire ~37GB combined corpus in RAM. Fixed
   `load_shards` to accept a `limit` param and stop reading shards once
   enough samples accumulate (shuffles shard READ ORDER for randomness
   instead of shuffling post-load). Verified: 30k-sample capped load against
   the full 5-shard glob dropped from ~24GB peak to ~2GB. Removed the
   now-redundant post-load shuffle+slice (and unused `random` imports) from
   `train_bc.py`/`train_sp.py`.

   **Round 2 — DONE, diminishing returns confirmed, DAgger track paused.**
   User chose to spend a round 2 testing the temperature lever: collected
   1000 games at temp 0.2 (vs round 1's 1.0) piloted by `ptcg_dagger_r1b.pth`
   — `training/dagger_data_r2*.pkl`, 314,081 samples, 0 relabel errors.
   Retrained on BC + round-1 + round-2 combined (the comma-separated
   multi-pattern `load_shards` support added earlier made this trivial) →
   `training/ptcg_dagger_r2.pth`. Gated: 81% vs random, **16% vs teacher —
   still flat**, same ~12-17% range as every prior checkpoint.

   Fresh-state fidelity re-check (same method as round 1, new rollout
   states): `ptcg_bc_v2.pth` 73.1% → `ptcg_dagger_r1b.pth` 81.1% (round 1's
   ~8pp jump, confirmed again on an independent sample) → `ptcg_dagger_r2.pth`
   **81.9%** (round 2 added only +0.8pp). **The temperature lever helped a
   little but wasn't the dominant factor — fidelity is flattening near
   80-82%, not accelerating.** Two rounds of evidence now agree: further
   DAgger rounds are unlikely to produce another large jump. Per the
   advisor's original framing, imitation asymptotes toward — never above —
   teacher parity regardless of rounds; this is a natural stopping point
   for the DAgger track, not a case for a round 3 on the same premise.

   **Status:** DAgger paused here. `ptcg_dagger_r2.pth` is the best net
   checkpoint (81.9% teacher-fidelity on deployment states, still ~16%
   head-to-head — not ship-ready, but the strongest evidence yet for the
   report's compounding-error narrative: BC 73-75% fidelity / 12-17%
   win-rate → DAgger 82% fidelity / still 12-17% win-rate shows *fidelity
   gains alone don't linearly convert to win-rate* at this deck's difficulty,
   a finding in itself). Advancing past ~50% vs teacher needs the Stage 2
   improvement operator (AWR/search), not more imitation rounds.
   **Gate: ~50%+ vs the teacher (Gauntlet + 400-game A/B) → ship to ladder**
   (single forward pass, no timeout risk) and log its bracket results.
3. **Advantage-weighted self-play (Stage 2) — infra BUILT 2026-07-03, not yet
   trained/gated.** `selfplay_collect.py::compute_value_targets` now stores
   `v_pred` (the collecting net's own value-head estimate at s_t) per decision
   alongside `value_target`. `dataset.py` computes `advantage = value_target -
   v_pred` per sample (`None`/weight-1.0 for BC samples, which carry no
   `v_pred`) and threads `advantages`/`has_advantage` through `collate()`.
   `train_sp.py` weights the policy loss `exp(advantage/β)` per SP sample
   (`--awr-beta`, default 1.0), rescaled by a corpus-level normalizer
   (`awr_normalizer()`) so the SP portion's mean weight stays ~1.0 — protects
   the non-negotiable 40/60 BC/SP mix from silently drifting — and clipped to
   `[1/awr-clip, awr-clip]` (default 20) so a few high-advantage samples can't
   dominate the gradient. `--winner-only` is the dumb-baseline ablation
   (filters SP samples to `outcome>0`, weight 1.0 uniformly, no AWR term).
   **Pre-training value-head diagnostic** (per advisor, before committing to a
   full collect): 30 self-play games with `ptcg_dagger_r2.pth`, temp 1.0 →
   9,536 decisions. `v_pred` std 0.935 (not collapsed to ~0 — the value head
   discriminates, though it saturates toward ±1 for most states rather than a
   smooth spread: p25 -0.999, p75 +0.9997). `advantage` mean 0.15, std 0.97,
   range clipped to [-2,2] as expected — confirms AWR weighting is
   distinguishable from the winner-only ablation before spending hours on a
   full collect+train+gate cycle. Both code paths smoke-tested end-to-end (1
   epoch, 5 steps, on the 30-game shard + a 2k-sample BC slice) — no errors,
   `awr_norm` computed sanely (1.85 at β=1.0).
   **Not yet done:** fresh large-scale SP collection with `ptcg_dagger_r2.pth`
   (the existing `sp_data.pkl.gz` is stale, old-deck, pre-freeze — see the
   RESTARTED note above), the real AWR training run, and the gate. Per
   advisor: also run AWR-checkpoint vs its own seed (`ptcg_dagger_r2.pth`),
   not just vs the teacher — vs-teacher alone can't distinguish "AWR improved
   nothing" from "the seed was already near parity," and unlike DAgger's
   imitation ceiling, AWR is supposed to exceed parity, so a plateau at ~50%
   vs teacher here would be a real negative result, not a measurement-
   resolution artifact. Exit: 55-60%+ vs the teacher over 400 local games.
4. **Gauntlet + ladder A/B each meaningful checkpoint** (`training/gauntlet.py`
   with a distinct `--name` per checkpoint). Real ladder is the only honest
   evaluator; gElo is the cheap ranking proxy being calibrated against it.
5. **Every run gets a `docs/report-log.md` entry the same day.**

---

## Value Targets: n-step Monte Carlo with Bootstrapping

Full-game binary win/loss targets are high-variance in 100+ decision games, and
full playouts inside search are expensive. Instead:

- **Training value target:** n-step TD — `G_t = Σ_{k<n} γ^k r_{t+k} + γ^n V(s_{t+n})`
  with shaped intermediate rewards r (prizes taken/conceded, threshold progress —
  see `training/README.md` §Curriculum & Reward Shaping) mixed toward the terminal outcome:
  `target = 0.7 * outcome + 0.3 * G_t^(n)`, n ≈ 8-12 decisions, γ ≈ 0.997.
- **Search evaluation:** truncated rollouts — expand ~n steps with the policy net,
  evaluate the leaf with the value head instead of playing to termination
  (exactly the AlphaZero leaf-evaluation trick, applied to determinized rollouts).
  This is what makes 10-20 sims/decision affordable under the 10-minute clock.
- **Ablation for the report:** binary-terminal vs n-step-bootstrapped value targets
  on the same BC base — variance reduction is measurable and write-up-worthy.

`search_begin(..., manual_coin=True)` lets the search control coin flips —
determinize per-rollout (sample) rather than letting hidden randomness leak variance.

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

## BC Warm-Start Design

**Teacher:** v21 agent (current best heuristic)  
**Collection:** v21 vs v21 self-play, 700+ games  
**Each BCStep:** `sv_enc, sv_dec, n_actions, chosen_idx, outcome`  
**~158 decisions/game** (Alakazam has high branching due to search/evolve sub-selections)

**Training config:**
- 10 epochs, LR 1e-4, batch 128
- Policy loss target: converge to ~0.50 range
- Eval after each epoch: net vs random (want 65%+), net vs teacher (want ~50%)

**Decoder padding rule (critical — violating this crashes training):**  
BC samples (`BCStep`): `sv_dec` is NOT pre-padded — must pad in training batch builder:
```python
if not hasattr(s, 'policy_targets'):  # BC sample
    for _ in range(64 - s.n_actions):
        dec.offset.append(len(dec.index))
```
SP samples (`SPStep`): `sv_dec` IS pre-padded to 64 words inside `eval_node` before storing.  
Violating this causes: `RuntimeError: shape '[128, -1, 128]' is invalid`

---

## Self-Play Phase Design (Phase 1)

- Net plays vs itself using MCTS (10 sims per decision; increase to 20 in later iters)
- `search_begin` fills hidden zones: own deck/prizes sampled from DECK; opponent
  hand/deck/prizes filled with placeholder `[1072]*n` (belief model replaces this in Phase 2)
- Policy targets = advantage-based (child Q - root Q, clamped to [-1, 1])
- Value target = game outcome (1.0 win, -1.0 loss, 0.0 draw)
- UCB: `q + 0.4 * sqrt(parent_visit) * prior / (1 + child_visit)`
- Loss: HuberLoss for value (delta=0.2) + HuberLoss for policy (delta=0.1, masked to valid actions)
- **Batch mix: 40% BC / 60% SP — non-negotiable.** BC buffer = full warmup dataset always
  present. SP buffer grows across iterations.

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

## Phase 2: League/PFSP Hardening

**Trigger:** net consistently beats v21 teacher (~55%+ over 100 games)

Actions:
- Run net vs each meta opponent (Lucario, Dragapult, Abomasnow, Starmie agents in `opponents/`)
- Hall-of-fame: keep past checkpoints as sparring partners
- PFSP weighting: prioritize opponents the net is currently losing to

**Originality contribution — opponent belief model:**  
Replace `search_begin`'s placeholder with a real belief distribution:
- Parse `obs["logs"]` to infer opponent archetype (Starmie vs Dragapult vs Lucario etc.)
- Sample opponent hidden zones consistent with inferred archetype
- This is the standout contribution for the 70% report axis

---

## Kaggle Setup

```
Engine: Add Input → kiyotah/cg-lib
GPU: Session → Accelerator → GPU T4 x1 (for training)
CPU: for data collection (no GPU needed)
Weekly GPU quota: ~30 hrs (resets weekly)
Save Version (Save & Run All) → commits /kaggle/working/ outputs
```

Consider using the Kaggle MCP server (docs.kaggle.com/docs/mcp) to run
notebook cells directly from Claude Code without manual copy-paste.

---

## Heuristic vs NN Track

The heuristic (v21+) and NN are independent parallel tracks. The heuristic serves
as the live ladder submission and the BC teacher for the NN warmup. They don't share
checkpoints or training data — each optimizes on its own axis.
