# Report Log — Experiment Journal & Method Glossary

*Every experiment gets a dated entry here THE DAY IT RUNS: hypothesis, method in
plain English, result with numbers, decision, report relevance. In September the
final report is assembled from this file — nothing gets retrofitted. Newest first.*

**Last updated:** 2026-07-04 (Stage 2 AWR self-play: infra built, first β=1.0 result a real negative vs teacher and tied vs seed)

---

## How to Use This File

Every entry has five fields. The "method in plain English" paragraph is written
*at implementation time* — that's the mechanism by which we actually understand
the methods we use, not just run them. Target figures the report needs (log
material for these continuously, don't reconstruct later):

1. **Archetype-inference accuracy by turn** (belief model calibration)
2. **Gauntlet gElo vs. realized ladder Elo scatter** (offline/online calibration
   — `training/gauntlet_results.csv` + `training/ladder_history.csv`)
3. **Win rate vs. teacher across training stages** (BC → DAgger → AWR self-play,
   with 95% CI bars)
4. **Ablation table** (BC mix ratio, value target type, belief vs. placeholder
   determinization, DAgger on/off, prior-blend λ)
5. **Latency budget curve** (search sims/decision vs. p99 move time vs. win rate)
6. **Robustness panel** (per-anchor matchup table, per-seat win split, cross-run
   seed-to-seed variance, opening-hand-quality conditional win rate)
7. **Pro-metrics panel** (consistency table via `tools/deck_math.py` — mulligan %,
   key-combo-by-turn odds per deck; prize-trade efficiency; setup speed = mean
   first-attack turn; meta-weighted expected win rate)

Target tables: **Table A — deck bake-off** (5 decks × tier-1 BT rating, tier-2 BT
rating, head-to-head vs Alakazam ±CI, meta-weighted win rate, mulligan %,
prize-trade efficiency); **Table B — method comparison** (one row per method —
random, generic-greedy, heuristic, bc, dagger-r2, + future — same gauntlet
protocol, one-line keep/reject verdict each).

---

## Glossary (plain-English, for the report author = us)

| Term | What it actually means here |
|------|------------------------------|
| **Behavior cloning (BC)** | Supervised learning: show the net board states and the teacher's (heuristic's) chosen action; train it to predict that action. Cheap, but the net only sees states the *teacher* reaches. |
| **Compounding error / distribution shift** | Why 85.9% action-match ≠ 50% head-to-head: one wrong move per ~7 puts the net in states the teacher never reached, where it was never trained, so it errs again — errors snowball. The diagnosis that motivates DAgger. |
| **DAgger** ("Dataset Aggregation") | Fix for compounding error: let the *net* drive games, but at every decision also record what the *teacher* (`score_options`) would have done, and train on those labels. Now the net is trained exactly on the states it actually reaches. |
| **Improvement operator** | Whatever makes iteration k+1 better than k. BC's operator is "teacher is better than student." Plain self-play imitation has *none* (net imitates itself). AWR's is "imitate your lucky/good actions harder." MCTS's is "search output beats raw policy." Every training loop must name its operator or it won't improve. |
| **Advantage-weighted regression (AWR)** | Self-play training where each action's imitation weight is `exp(advantage/β)` — actions that led to better-than-expected outcomes (per the value head) get imitated harder, bad ones softer. How the net *exceeds* the teacher without full RL machinery. |
| **Value head / value function** | The part of the net that predicts win probability from the current state. This IS the "who will win" predictor. Used for n-step bootstrapped targets, advantage computation, and search leaf evaluation. |
| **Belief model** | NOT a win predictor. Predicts *hidden information*: P(opponent archetype \| their observed plays, turn), then samples plausible contents for their hand/deck/prizes. Feeds determinization; the report's originality centerpiece. |
| **Determinization** | Imperfect-information search trick: sample ONE concrete assignment of all hidden cards (from the belief model), search that perfect-information game, repeat with fresh samples, aggregate. |
| **Bradley-Terry model** | Statistical model behind the Gauntlet: each agent gets a strength number; P(A beats B) = strength_A / (strength_A + strength_B). Fit from all pairwise results; log-scaled it becomes an Elo-like scale (gElo). |
| **PFSP** (prioritized fictitious self-play) | Choose sparring partners you currently *lose to* more often, instead of uniformly. Prevents overfitting to your own latest self. |
| **SPSA** | Gradient-free tuning: nudge all ~20 heuristic weights randomly up/down together, measure win-rate difference, step toward whichever perturbation won. Cheap unattended weight search. |
| **Expert iteration** | The AlphaZero loop: search (MCTS) produces a policy better than the raw net; train the net toward the search output; repeat. Our Stage 5 aspiration, gated on Kaggle's `search_begin` API. |

---

## 2026-07-04 — Stage 2 AWR self-play: infra built, first result is a real negative

**Hypothesis:** advantage-weighted self-play (reweight the imitation-style
policy loss by `exp(advantage/β)`, advantage = n-step bootstrapped value
target minus the value head's own V(s)) gives the net an improvement
operator DAgger couldn't — DAgger only asymptotes to teacher parity by
construction, AWR is supposed to be able to exceed it.

**Method (plain English):** `selfplay_collect.py` now stores `v_pred` (the
collecting net's own value-head estimate) per decision alongside the
existing bootstrapped `value_target`. `dataset.py` computes
`advantage = value_target - v_pred` per sample (BC samples get `None` →
flat weight 1.0, since they carry no value estimate). `train_sp.py` weights
each self-play sample's policy loss by `exp(advantage/β)`, rescaled by a
corpus-level normalizer so the SP portion's mean weight stays ~1.0
(protects the 40/60 BC/SP mix) and clipped to `[1/20, 20]`. `--winner-only`
was added as the dumb-baseline ablation (filter to `outcome>0`, uniform
weight, no AWR term) but not yet run.

**Pre-registered check before the expensive run** (per advisor): dumped
`v_pred`/advantage stats over 30 self-play games (9,536 decisions) with
`ptcg_dagger_r2.pth` first. `v_pred` std 0.935 (not collapsed toward 0 —
saturates bimodally toward ±1 for most states instead of a smooth spread),
advantage mean 0.15/std 0.97. Confirmed AWR weighting would be
distinguishable from winner-only before committing hours to collection.

**Run:** collected 1000 fresh self-play games with `ptcg_dagger_r2.pth`
(temp 1.0) → 317,143 samples (`training/sp_data_awr*.pkl.gz` — the old
`sp_data.pkl.gz` was stale/pre-freeze, per the RESTARTED note). First
training attempt (uncapped `--bc-data`/`--sp-data`, no limit) got
OOM-killed — reproduced the exact bug `--bc-limit`/`--sp-limit` were built
to fix during DAgger, just forgot to pass them this time. Retrained with
`--bc-limit 250000 --sp-limit 250000`, `--awr-beta 1.0`: 10 epochs, loss
0.51→0.41, `awr_norm=1.83` → `training/ptcg_awr1.pth`.

**Gate (400 games each, per the pre-registered Stage 2 exit criterion):**
- **vs v25c teacher:** 15.8% ± 3.6% (63W-337L). Same flat 12-17% range as
  every BC/DAgger checkpoint before it — no improvement.
- **vs its own seed (`ptcg_dagger_r2.pth`)** — added per advisor's specific
  warning that vs-teacher alone can't distinguish "AWR improved nothing"
  from "the seed was already near parity": 47.7% ± 4.9% (191W-209L), a
  clean statistical tie.

**Decision:** this is a **real negative result, not a measurement-resolution
artifact** — unlike DAgger's excused imitation-ceiling plateau, AWR is
supposed to be able to exceed teacher parity, and the tied vs-seed result
at n=400 (tight enough CI to resolve a real effect) confirms training
produced no detectable improvement, not just an invisible one. Direct
self-play (no MCTS/search tree) + a value head that saturates toward ±1 for
most states may simply not carry much per-decision advantage signal beyond
the terminal outcome, making AWR reweighting behave close to (though not
identical to) the winner-only ablation. One follow-up (different β) queued
before concluding the direct-self-play-AWR line — see next entry / status
at top of `docs/nn-training.md` for the outcome.

**Report relevance:** target figure #3 (win rate vs teacher across BC →
DAgger → AWR) gains a real data point either way — this is evidence for the
report's "imitation-family methods plateau on this deck without search"
narrative, not a throwaway failed run.

---

## 2026-07-03 — Analytic consistency panel for all 5 bake-off decks (figure #7)

**Hypothesis:** the hypergeometric consistency stats (`tools/deck_math.py`)
differ meaningfully across the bake-off decks and contextualize the tier-1
result.

**Method (plain English):** exported all 5 deck lists to
`training/manifests/decks/*.csv` (from each agent module's DECK) and ran the
deck_math panel on each: mulligan probability (no Basic in opening 7),
P(ace seen by turn 3), P(key evolution line assembled by turn 3).

**Result:**

| deck | Basics | P(mulligan) | ace ≥1 by T3 | key line by T3 |
|------|-------:|------------:|-------------:|----------------|
| **alakazam** | 11 | **22.2%** | 42.7% (Alakazam) | natural 10.1% / Candy 8.1% |
| lucario | 10 | 25.9% | 52.8% (M-Lucario, Stage 1) | — |
| dragapult | 9 | 30.0% | 42.7% (Dragapult ex) | natural 10.1% / Candy 5.7% |
| abomasnow | 6 | 45.9% | 52.8% | — (Basic ace) |
| starmie | 4 | **60.1%** | 52.8% | — (Basic ace) |

**Findings:** (1) Alakazam has the *lowest* mulligan rate of the pool — the
"deliberately simplified deck" is also the most consistent one, a point the
deck section can make analytically rather than anecdotally. (2) The sample
decks' mulligan rates are catastrophic (abomasnow 45.9%, starmie 60.1% —
4-6 Basics in 60), which partly mechanizes their tier-1 losses and their
weak anchor gElos. (3) Stage-2 line-by-T3 is ~10% for both Stage-2 decks —
piloting around that scarcity (fetch priorities, Candy racing) is where the
pilot earns its margin, consistent with tier 2 flattening when no pilot does.

**Report relevance:** figure #7 consistency column complete for Table A;
finding (1) is a §2 sentence.

---

## 2026-07-03 — Method bake-off complete: Table B + keep/reject verdicts

**Hypothesis:** per the method-bake-off pre-registration (below): the
imitation-first ladder dominates cheaper alternatives under one fixed
protocol, and the known negative results survive re-measurement.

**Method (plain English):** all five rows through the same `gauntlet.py`,
full 8-anchor panel (first run ever with the repaired dragapult), 200
games/anchor (1,600 games/row, 8,000 total), seed `method-run1`, per-seat
splits persisted. Net rows via `net_agent.py` with `NET_CKPT`.

**Result — Table B (gElo on the shared BT scale, anchored random=0):**

| method | gElo | vs sample bots | vs frozen heuristics (v21/v22/v23) | verdict |
|--------|-----:|----------------|-------------------------------------|---------|
| method-heuristic-v25c | **568** | 93-99.5% | 67% / 65.5% / 61% | **KEPT** — ladder submission + DAgger teacher; beats every cheaper row decisively |
| method-dagger-r2 | 246 | 66-92% | 21.5% / 19% / 18.5% | kept as **Stage 2 initialization only** — see verdict note |
| method-bc-v2 | 239 | 65-92% | 21% / 22% / 20% | rejected as pilot, **kept as initialization** (pre-stated non-Elo justification) |
| method-generic-greedy | 57 | 42-86% | 5.5-9.5% | floor baseline only (that's its job) |
| method-random | 5 | 38-61% | 1.5-3% | floor reference |

**Verdict notes (rule applied exactly):** dagger-r2 beats bc-v2 by only +7
gElo — NOT CI-separable at this n, so under the pre-registered rule ("beats
the best cheaper row with CI excluding 0") DAgger does **not** earn a
keep-as-pilot on win-rate. This is the third independent measurement agreeing
with the fidelity-vs-win-rate story: +8pp state fidelity (74.9→81.9%) has not
yet converted to head-to-head Elo. Roadmap unchanged — exceeding the teacher
is Stage 2 AWR's job, with `ptcg_dagger_r2.pth` as the starting checkpoint.

**Bonus replications:** (1) method-heuristic-v25c re-gauntleted at 568 vs the
earlier v25c run's 575 — a Δ7 seed-to-seed replication across independent
200-game runs (first datapoint for figure #6's variance panel); (2)
method-random landed at gElo 5 vs the anchor random's 0 — the scale
self-checks. (3) The sample-notebook bots are weak enough that even random
goes ~50/50 with them — context that the tier-1 deck margins are largely
*pilot* margins, which is the co-design thesis said out loud.

**Report relevance:** Table B complete; the method ladder
(random 5 → generic-greedy 57 → BC 239 → DAgger 246 → tuned heuristic 568)
is the §3 method-narrative figure; the DAgger verdict is the honest-negative
§5 material.

---

## 2026-07-03 — Tier-2 deck bake-off + Stage 0c DECISION: freeze re-closed on Alakazam

**Hypothesis:** per the Stage 0c pre-registration: with one fixed generic
pilot on every deck, deck strength can be compared free of pilot quality.

**Method (plain English):** built `training/generic_pilot.py` — a deliberately
simple deck-agnostic greedy policy (evolve > ability > play > attach > attack
> end; never retreats; uses only option-type codes, no card knowledge). All 5
decks piloted by it, same round-robin protocol as tier 1 (200 games/pair,
seat-alternating, 3,000-step cap ≈ 3× the longest tier-1 game).

**Result (tier 2, 2,000 games, 0 errors, 0 cap-ties, seed tier2-run1):**
**every pairing is statistically 50/50.** All 10 Wilson CIs cover 0.5;
BT-Elo spread is 24 points (vs 1,010 in tier 1); prize-trade efficiency
0.94-1.03 for all decks. End reasons tell the story: **80% of tier-2 games
end in DECK_OUT** (vs 22% in tier 1) and zero in PRIZED_OUT — below a pilot
competence floor, matches degenerate into deck-out races and deck identity
never expresses. The pre-registered "tier-1/tier-2 disagreement is a finding"
clause triggers in its most extreme form: **deck value is entirely
pilot-dependent in this pool** — our own harder-edged version of the
wmh/ptcg-abc observation that simple decks piloted cleanly beat strong decks
piloted badly.

**DECISION (per the pre-registered rule, applied exactly):** no challenger
beats Alakazam by ≥10pp with CI excluding 0 in *either* tier — tier 1 is not
close (best challenger: abomasnow at **7.0%** vs Alakazam; Alakazam ≥93% vs
everything), and tier 2 discriminates nothing. Condition (a) fails for every
challenger → **the Alakazam freeze re-closes, now quantitatively justified.**
Condition (b) (meta-weighted win rate) was not needed, but qualitatively:
Alakazam beats all four available meta-representative decks ≥93% as-piloted,
and the meta survey puts those four at ~43% of the observed field.

**Caveats (stated before anyone asks):** tier 2's null is about the *pilot
floor*, not proof the decks are equal; the 4 challenger decks are the ones
with in-repo lists (Bellibolt/Crustle stretch decks not yet built); tier-1
"as-piloted" advantage includes our 25-version pilot vs sample-notebook
pilots — which is exactly the thesis (co-design of deck + pilot), stated as
such in the report rather than hidden.

**Report relevance:** Table A complete (both tiers); the tier-1/tier-2
contrast is a headline finding for §2 (deck as thesis) and §5 (findings);
DECK_OUT-rate shift (22%→80%) is the mechanism figure.

---

## 2026-07-03 — Bake-off rig built + tier-1 deck bake-off results

**Hypothesis:** per the Stage 0c pre-registration (below): Alakazam is the
strongest available deck as-piloted.

**Method (plain English):** built `training/bakeoff.py` — the first tool that
runs any (agent, deck) pairing on either seat (the gauntlet/ab_test CLIs bind
each agent to its own module deck). Every game logs the pre-registered fields
(seats, winner, turns, prizes per side, first-attack turn per side, end
reason, run-id) to `training/bakeoff_results.csv`. Also upgraded the rig's
logging: `gauntlet_results.csv` now persists per-seat splits + a run id
(columns added, old rows padded), `ab_test.py` appends every run to
`training/ab_history.csv`, and `harness.play_game` accepts a step cap.
Sanity gate passed first: a main.py mirror match came out 17W-23L
(wr 0.425, Wilson CI [0.285, 0.578] — covers 0.5), 0 errors.

**Protocol notes (deviations from pre-registration, both logged before
interpreting results):**
- "seed" is a **run identifier** for grouping independent repeat runs, not an
  RNG seed — the engine's shuffle lives in the native `cg.dll`, which exposes
  no seed API. Variance measurement is unaffected (runs are independent).
- Tier 1 ran uncapped; a 3,000-step cap (≈3× the longest observed tier-1 game)
  was added before tier 2 because two passive pilots are otherwise bounded
  only by deck-out (`episodeSteps` defaults to 10M). Capped games score as
  ties; tier 1 hit no cap (max 43 turns).
- Fixed `opponents/dragapult_agent.py` first: its cg-lib fallback stubs were
  missing the `Pokemon`/`Card`/`State` classes used in `isinstance()` checks —
  the actual root cause of its 100% local crash rate (the import guard itself
  was fine). Post-fix: 3W-1L vs random, real games. The dragapult anchor is
  now usable (closes the CLAUDE.md item-6 follow-up).

**Result (tier 1, as-piloted, 200 games/pair, 10 pairs, 2,000 games, 0 errors,
seed tier1-run1):**

| pair | record | wr (Wilson 95%) |
|------|--------|------------------|
| alakazam vs dragapult | 198W-2L | .990 [.964,.997] |
| alakazam vs lucario | 191W-9L | .955 [.917,.976] |
| alakazam vs starmie | 195W-5L | .975 [.943,.989] |
| alakazam vs abomasnow | 186W-14L | .930 [.886,.958] |
| abomasnow vs dragapult | 150W-50L | .750 |
| abomasnow vs lucario | 123W-77L | .615 |
| lucario vs dragapult | 111W-89L | .555 |
| lucario vs starmie | 186W-14L | .930 |
| dragapult vs starmie | 186W-14L | .930 |
| abomasnow vs starmie | 196W-4L | .980 |

BT-Elo: **alakazam 1010** > abomasnow 577 > lucario 469 > dragapult 410 >
starmie 0. In-play metrics (figure #7): alakazam prize-trade efficiency
**2.35** (the single-prize thesis, measured — takes 2.35 prizes per prize
conceded) with the *slowest* setup (mean first attack turn 6.0 vs 2.4-4.5
for the others) — it wins through the trade, not the race.

**Decision:** tier-1 half of the pre-registered rule is unambiguous — no
challenger comes near the ≥10pp-over-Alakazam bar (best challenger result:
abomasnow's 7.0% vs alakazam). Await tier 2 before closing Stage 0c.

**Report relevance:** Table A tier-1 column complete; figure #7 prize-trade
and setup-speed numbers; the tier-1-vs-tier-2 comparison is the "deck value
is pilot-dependent" finding.

---

## 2026-07-03 — Ladder meta survey tool + first meta share table

**Hypothesis:** the opponent's archetype is identifiable from cards revealed
in play/discard, giving (a) meta weights for the deck decision rule,
(b) belief-model labels (Stage 3), (c) a report figure.

**Method (plain English):** `tools/meta_survey.py` collects every card id the
opponent revealed across a replay, maps ids→names via `EN_Card_Data.csv`, and
matches names against an archetype signature list (ace attackers first).
Unmatched games stay "other/unknown" rather than being guessed.

**Result (28 local replays, v22-v26c bands):** archaludon **17.9%**, alakazam
14.3%, dragapult 14.3%, lucario 14.3%, starmie 10.7%, crustle 10.7%,
grimmsnarl/abomasnow/rockets-mewtwo 3.6% each, 1 unknown, 1 corrupt file.
Two findings: (1) **Archaludon ex is the most-seen opponent and is not in our
anchor panel or ratings table at all** — candidate for a vendored anchor and a
matchups.md entry; (2) initial signature list missed 4 archetypes now added
(archaludon, grimmsnarl, rockets-mewtwo, snorlax-stall).

**Caveats:** n=28 and the sample is conditioned on our Elo band (~880) —
meta weights for the decision rule should be refreshed from a bulk replay
download before any deck-switch call rests on them.

**Report relevance:** meta-share table (figure candidate), decision-rule
weights, Stage 3 label pipeline validated.

---

## 2026-07-03 — PRE-REGISTRATION: Deck bake-off (Stage 0c)

**Date:** 2026-07-03 (registered before any bake-off games)

**Hypothesis:** Alakazam is the strongest available deck for this project both
as-piloted (tier 1) and intrinsically (tier 2, fixed generic pilot), and is not
disadvantaged against the observed ladder meta. Null outcome (freeze stands) is
the expected result; the trial exists to make the freeze quantitative.

**Protocol:** 5 decks — Alakazam (`deck.csv`) + the 4 opponent anchor decks.
Tier 1: each deck with its specialist agent; tier 2: all decks piloted by
`training/generic_pilot.py`. Full seat-alternating round-robin per tier,
200 games/pair, via `training/bakeoff.py` (per-game rows: seats, winner, turns,
prizes taken per side, first-attack turn per side, end reason, seed). Outputs:
Bradley-Terry ranking, matchup matrix with Wilson 95% CIs, figure #7 metrics.
Stretch (does not gate the decision): Bellibolt and Crustle lists in tier 2 only.

**Statistical conventions (verbatim from competition-strategy.md):** ties count
0.5; crash games excluded from win rate and reported (>2% → fix and re-run);
Wilson 95% CIs; recorded seeds, ≥2 independent-seed runs per headline pairing;
per-seat splits persisted.

**Decision rule:** replace Alakazam only if a challenger (a) beats it
head-to-head by ≥10pp with 95% CI excluding 0 in BOTH tiers, and (b) has
meta-weighted expected win rate (weights from the ladder replay meta survey) ≥
Alakazam's. Tier-1/tier-2 rank disagreement is a finding, not a trigger.
Switch cost acknowledged: invalidates BC/DAgger corpora, new heuristic work,
~6 weeks of ladder remaining. Optional 24–48h ladder probe only if tiers
disagree or a challenger is within threshold.

**Report relevance:** deck-concept axis (20%) — turns the deck section from a
story into a controlled comparison; feeds Table A and figures #6/#7.

---

## 2026-07-03 — PRE-REGISTRATION: Method bake-off

**Date:** 2026-07-03 (registered before any method-comparison games)

**Hypothesis:** the imitation-first ladder (BC → DAgger → planned AWR self-play)
strictly dominates the cheaper alternatives (random, generic-greedy heuristic,
hand-tuned specialist heuristic) under one fixed protocol, and each rung's
existing negative result (BC compounding error, SP-only collapse, DAgger plateau)
survives re-measurement under that protocol.

**Protocol:** same `training/gauntlet.py` version for every row, current anchor
panel (incl. repaired dragapult), 200 games/anchor, recorded seed, conventions
above. Rows: `random`, `generic-greedy` (tier-2 pilot on the Alakazam deck),
`heuristic` (`main.py`), `bc`, `dagger-r2`; future methods (AWR self-play,
search+belief) append to the same table — no re-runs of old rows without a
protocol version bump.

**Decision rule:** a method is "kept" (stays on the roadmap) only if it beats
the best cheaper row with 95% CI excluding 0, or has a pre-stated non-Elo
justification written in its keep/reject line (e.g. BC kept as initialization,
not as a pilot). Explicitly out of scope: a from-scratch RL baseline — its
compute competes with Stage 2; imitation-first is justified by the SP-only
collapse result and literature precedent, and the report states this rather than
hiding it.

**Report relevance:** model-approach axis (70%) — the "hypotheses tested"
rubric line; feeds Table B.

---

---

## 2026-07-03 — BC retrain on v25c corpus + dagger_collect.py built

**Hypothesis:** the BC corpus collected overnight (`bc_data_v25c*.pkl`, 579,169
samples) can be retrained locally without a Kaggle GPU session, since
`encode.py`/`model.py` were rebuilt cg-lib-free (see "RESTARTED" note in
`docs/nn-training.md`); and DAgger's collector (`training/nn/dagger_collect.py`,
previously just a TODO in the roadmap) is a small enough delta over
`bc_collect.py` to build and smoke-test now rather than waiting.

**Method:**
- Timed the retrain before committing to a full run: single-shard
  (114k-sample) load + 1 epoch took ~50s; loading all 5 v25c shards at once
  (what the full run needs) peaked at ~24GB RSS (system has 39.6GB, so
  workable but sizable — worth knowing before running anything else
  memory-heavy in parallel). Extrapolated full 579k-sample/10-epoch retrain to
  ~75-90 min CPU. Launched `python training/nn/train_bc.py --data
  "training/bc_data_v25c*.pkl" --epochs 10 --out training/ptcg_bc_v2.pth`,
  log at `training/overnight_logs/bc_retrain_v2.log`.
- Built `training/nn/dagger_collect.py`: mirrors `bc_collect.py`'s
  shard/writer structure exactly, but both seats are piloted by
  `selfplay_agent.py` (temperature-sampled net, for exploration) instead of
  the heuristic, and every recorded decision is *relabeled* with
  `main.score_options(obs, sel)`'s argmax rather than the action the net
  actually took — the core DAgger mechanism (train on the net's own state
  distribution, but with teacher labels). Verified end-to-end against the
  existing `ptcg_bc_v1.pth`: 4-game and 2-game smoke runs, 0 relabel errors,
  and the resulting samples round-tripped cleanly through
  `BCDataset`/`collate`/`PTCGNet.forward` with no shape errors.

**Result:** Retrain finished — took ~106 min wall (longer than the ~75-90 min
smoke-test extrapolation; CPU-time deltas confirmed steady, never-stalled
compute throughout, so the gap is just real per-epoch cost at 579k samples ×
10 epochs, not a hang). `training/ptcg_bc_v2.pth`, 10 epochs, val_top1_acc
0.8397 → 0.8750 (epoch 8, best) → 0.8740 (epoch 9, saved). Real-game gates
(`training/ab_test.py`, `NET_CKPT=training/ptcg_bc_v2.pth`, 100 games each,
same recipe as the original `ptcg_bc_v1.pth` gate):
- vs random: **86% (86W-14L, 0 errors)** — matches v1's 86% almost exactly.
- vs v25c heuristic (the new, stronger teacher): **17% (17W-83L, 0 errors)** —
  even lower than v1's 22% vs v22, consistent with v25c being a meaningfully
  stronger teacher than v22 was (gElo 589 vs whatever v22 scored back then).
  This is the expected BC-plateau/compounding-error signature the whole
  DAgger stage exists to fix, not a red flag.

`dagger_collect.py` (built earlier this session) is verified mechanically
correct end-to-end but not yet run at scale.

**Decision:** `ptcg_bc_v2.pth` is a healthy BC seed (clears the vs-random gate,
plateaus vs teacher exactly as the architecture predicts) — proceed to DAgger
round 1.

**DAgger round 1 collection (same day, following this decision):** ran
`python training/nn/dagger_collect.py --games 1000 --ckpt training/ptcg_bc_v2.pth
--out training/dagger_data_r1.pkl` — 1000 net-piloted mirror games, 326,240
samples (3 shards), 0 relabel errors, net_p0_winrate 0.502 (expected ~50% for
a mirror match). Discovered `training/nn/dataset.py::load_shards` only took a
single glob pattern, not a combinable list — added comma-separated pattern
support (small, needed now and for every future DAgger round that pools
multiple rounds' data) rather than routing around it.

Retrained via `training/nn/train_sp.py` (built for exactly this: warm-start
from a checkpoint + BC/SP mixed batches — DAgger-round data is the same
`{obs, action, outcome}` schema as SP data, so it drops straight in as
`--sp-data`), 40% BC / 60% DAgger-round mix (non-negotiable ratio per this
file — SP-only collapsed the prior project's attempt 46%→20% vs teacher in 3
iterations).

**Infra snag, fixed:** the first two attempts at this retrain died silently
with no error (Windows killed the process outright, no Python traceback
possible) — root cause: `train_sp.py` loaded the FULL BC corpus (579k
samples) AND the full DAgger corpus (326k samples) into RAM simultaneously,
pushing free system memory to ~1.9GB out of 39.6GB. Since the training loop
only resamples ~768k times with replacement from these pools anyway, a full
corpus in RAM bought nothing but memory pressure. Added `--bc-limit`/
`--sp-limit` to `train_sp.py` (mirrors `train_bc.py`'s existing `--limit`
pattern) and reran with `--bc-limit 100000` (kept the full 326k-sample DAgger
corpus, capped only the much-larger BC pool) — comfortably stable.

**Result — round 1, first pass (3 epochs, lr 5e-5, `ptcg_dagger_r1.pth`):**
vs random 85% (85W-15L, matches BC), vs v25c heuristic teacher **12%
(12W-88L)** — nominally *worse* than plain BC's 17%, not better.

**Diagnosis (per advisor consult before assuming a regression or spinning up
round 2 blind):** 12% vs 17% at n=100 is not a statistically meaningful
difference (~±7% CI on each, fully overlapping) — the honest read is "round 1
moved nothing measurable," not "DAgger made it worse." The discriminating
check: does the DAgger label (`argmax(main.score_options(obs,sel))`) actually
agree with what `main.agent` (the real teacher we gate against) would pick on
the same states? Sampled 3000 `dagger_data_r1.pkl` observations and compared
— **96.8% agreement (2903/3000)**, with mismatches concentrated in select
types `score_options` documents as flat/uniform-prior by design (bench
placement etc.) — not a label bug. That points to **undertraining**, not bad
labels: the fine-tune pass was only 6000 steps at lr 5e-5, roughly 1/40th the
gradient work of the original 10-epoch BC warmup.

**Decision:** retrain harder on the SAME round-1 data before collecting
another (expensive) DAgger round — `--epochs 10 --steps-per-epoch 4000 --lr
2e-4` (`training/ptcg_dagger_r1b.pth`, comparable total step count to the BC
warmup), log at `training/overnight_logs/dagger_r1b_retrain.log`.

**Infra snag #2, fixed at the root:** this retrain also died silently, twice,
even with `--bc-limit`/`--sp-limit` caps in place (RSS ballooned to ~31GB
regardless of how small the final capped sample count was — even a
30k-sample cap reproduced it). Root cause, found via an isolated instrumented
repro (`psutil`-tracked RSS every 200 training steps on a tiny 10k-sample
set — training-loop RSS growth was modest and bounded, ruling out a
per-step leak): `training/nn/dataset.py::load_shards` always read every
glob-matched shard FULLY into memory first, and only *then* did the caller
slice it down to the limit — so a "capped" load still momentarily
materialized the entire ~579k+326k corpus (~37GB) before ever trimming it.
Fixed `load_shards` to accept a `limit` param and stop reading further shards
once enough samples are collected (shuffling shard READ ORDER for
randomness, instead of shuffling samples after a full load — avoids ever
needing the full corpus in memory). Verified: a 30k-sample capped load
against the full 5-shard glob dropped from ~24GB peak to ~2GB. Also removed
the now-redundant post-load shuffle+slice in `train_bc.py`/`train_sp.py`
(dead code after routing the limit into `load_shards`) and their now-unused
`random` imports.

**Result — round 1, retrained hard (10 epochs, lr 2e-4, `ptcg_dagger_r1b.pth`,
memory stable throughout, ~150 min wall — longer than estimated but steady
CPU throughout, no stalls):** vs random 85% (matches BC/first attempt), vs
v25c heuristic teacher **15% (15W-85L)** — statistically indistinguishable
from BC's 17% and the first (undertrained) attempt's 12%. Heavy retraining
did NOT move the win-rate gate.

**Paranoia check (per advisor, given this session's infra gremlins):**
confirmed `ptcg_bc_v2.pth` and `ptcg_dagger_r1b.pth` load as genuinely
distinct model weights (max abs diff 0.144 on the card-embedding table) —
the three near-identical win rates are not a silent-fallback bug.

**The decisive measurement (per advisor — a 100-game win-rate physically
cannot resolve what DAgger targets; measure teacher-agreement on FRESH
deployment-realistic states instead of the training states):** collected 60
argmax (temp≈0, matching real deployment) mirror games with
`ptcg_dagger_r1b.pth` via `net_agent.py`, yielding 18,374 decision states
never seen in training. Sampled 3000, compared `main.score_options` argmax
(teacher) against both checkpoints' argmax on these SAME fresh states:
- `ptcg_bc_v2.pth`: **74.9% agreement** (2246/3000)
- `ptcg_dagger_r1b.pth`: **79.7% agreement** (2390/3000)

**A real +4.8pp improvement.** DAgger round 1 IS working — it moved the
metric it actually targets (fidelity on states the deployed net reaches).
The flat 100-game win-rate gate was a resolution problem, not evidence of a
broken pipeline: a single-decision accuracy gain compounds over ~150
decisions/game, but converting that into a detectable head-to-head win-rate
delta needs either more rounds (compounding the accuracy gain further) or a
much larger n than 100 games to clear the ~±7% noise floor.

**Decision:** per the advisor's stop-loss framing — imitation (BC/DAgger)
asymptotes toward parity with the teacher (~50%), never above; that requires
the later improvement-operator stages (AWR/search). The "50%+ vs teacher"
ship gate is really "match the heuristic exactly on a hard single-prize
combo deck," not a milestone DAgger breezes past, and each round costs
~2.5h wall-clock. This round's finding (real fidelity gain, not yet
detectable in win-rate) is itself report-worthy for the compounding-error
narrative regardless of what happens next. Presented this tradeoff to the
user directly (round 2 at lower temperature vs. banking the finding and
moving on) — **user chose round 2.**

**DAgger round 2 (same day, following user decision):** collecting at
temperature 0.2 instead of round 1's 1.0 — the advisor's flagged-but-untested
lever: round 1's temp-1.0 collection teaches the teacher's moves on
near-random exploration states a temp≈0 deployed net rarely actually visits,
which may be exactly what capped how much round 1's fidelity gain (74.9%→
79.7%) transferred into win-rate. Piloted by `ptcg_dagger_r1b.pth` (the best
checkpoint so far, not the original BC net) via `python
training/nn/dagger_collect.py --games 1000 --ckpt training/ptcg_dagger_r1b.pth
--temp 0.2 --out training/dagger_data_r2.pkl` — 1000 games, 314,081 samples,
0 relabel errors, net_p0_winrate 0.535.

Retrained on BC + round-1 + round-2 data combined (`--bc-data
"training/bc_data_v25c*.pkl" --sp-data "training/dagger_data_r1*.pkl,
training/dagger_data_r2*.pkl" --bc-limit 100000 --sp-limit 300000 --init
training/ptcg_bc_v2.pth --epochs 10 --steps-per-epoch 4000 --lr 2e-4` — the
comma-separated multi-pattern `load_shards` support added earlier this
session made this a one-line change) → `training/ptcg_dagger_r2.pth`, ~130
min wall, memory stable throughout (the root-cause fix holds under a bigger
combined corpus too). Gated: 81% vs random, **16% vs teacher — still flat**,
consistent with the same ~12-17% range every checkpoint has landed in.

**Fidelity re-check (fresh states, collected via `ptcg_dagger_r2`-piloted
argmax rollouts this time — same method as round 1's decisive check):**
- `ptcg_bc_v2.pth`: 73.1% (2192/3000 — consistent with the earlier
  measurement's 74.9% on a different fresh sample, as expected)
- `ptcg_dagger_r1b.pth`: 81.1% (2433/3000)
- `ptcg_dagger_r2.pth`: **81.9%** (2457/3000)

**Round 1 gave a real ~8pp fidelity jump (BC→r1b); round 2 (lower
temperature) added only +0.8pp on top (r1b→r2) — diminishing returns, not
the hoped-for unlock.** The temperature lever helped a little but wasn't the
dominant factor capping win-rate transfer; the fidelity curve looks like it's
flattening near 80-82%, not accelerating. Two rounds of evidence now point
the same direction: further DAgger rounds are unlikely to produce a large
additional jump, and per the advisor's original framing, imitation learning
asymptotes toward — never above — teacher parity regardless. This is a
natural stopping point for the DAgger track rather than a case for a round 3
on the same premise.

**Report relevance:** Directly builds the win-rate-vs-teacher-across-stages
figure (#3) — this is the BC anchor point the DAgger rounds will be compared
against.

---

## 2026-07-03 — v25c ladder result, full Gauntlet baseline (top of table), BC re-collect on frozen deck

**Hypothesis:** v25c's desperation-mode + lone-active-opportunity heuristics (see
2026-07-02 entry below) should beat v25b both on the ladder and in the Gauntlet,
and the frozen v23 deck means the stale old-deck BC corpus can finally be
replaced with real v25c self-play data for Stage 1 (DAgger).

**Method:** Ran two unattended overnight jobs, detached from the Claude session
(Windows `Start-Process` + `.bat`, logs in `training/overnight_logs/`) so they'd
survive the session ending:
1. `python training/gauntlet.py --candidate main.py --name v25c --panel
   random,lucario,abomasnow,starmie,v21,v22,v23 --games 200` (dragapult excluded
   — it crashes 100% of local games, per the 2026-07-03 harness-bug entry below;
   the anchor itself is still unfixed).
2. `python training/bc_collect.py --games 2000 --out bc_data_v25c.pkl` — v25c
   mirror self-play on the current frozen (v23) deck, replacing the old-deck
   corpus flagged stale in `docs/nn-training.md` §Resume Here.

**Result:**
- User-reported ladder: v25c peaked ~900, settled ~880 (submission 54282648),
  up from v25b's 861.8 publicScore.
- Gauntlet: v25c gElo **589**, now top of the whole table (v25b 559, v24 516,
  v23 499, v25 490, v22 479, v21 414). Record win rates: 98.5%/random,
  96.0%/lucario, 97.0%/abomasnow, 96.5%/starmie, 68.5%/v21, 63.0%/v22, 63.0%/v23
  — all 0 errors. The offline gElo gain (589 vs 559) points the same direction
  as the realized ladder gain (880 vs 861.8): another offline/online calibration
  data point for the report figure.
- BC re-collect: 2000 self-play games, 579,169 samples (5 shards, ~3.1GB
  uncompressed — larger/slower than the old `.pkl.gz` corpus since this run
  wasn't gzipped; files moved from repo root to `training/bc_data_v25c*.pkl`
  after the fact, script writes relative to cwd not to its own directory),
  self-play win rate 0.551 (near-50% as expected for a mirror match). Total
  wall time for both jobs: ~25 minutes, far under the 6h budget — done in one
  session, not by morning as anticipated.

**Decision:** Both feed direct next steps: fill the v25c gElo cell in
`training/ladder_history.csv` (done) with this table; `training/bc_data_v25c*.pkl`
is the new BC corpus to retrain the warmup net on (same recipe as `ptcg_bc_v1.pth`)
before building `training/nn/dagger_collect.py` for Stage 1. Full 6-8h overnight
window remains available for a heavier job (e.g. `weight_search.py` SPSA re-tune,
or a bigger BC/DAgger pass) since this one finished in minutes.

**Report relevance:** Gauntlet gElo vs. ladder Elo scatter (target figure #2);
BC corpus is the input for the win-rate-vs-teacher-across-stages figure (#3)
once DAgger exists.

---

## 2026-07-02 — Desperation phase, deck-search Alakazam-priority bug, Boss's Orders PHASE_CLOSING bug (v25.2)

- **Hypothesis 1 (user-reported):** replay `83429870` — at 1-1 prizes (sudden
  death), retreated a mist-walled Alakazam into a squishy Kadabra (80 HP)
  instead of a 210-HP Fezandipiti ex on the bench; Kadabra got one-shot next
  turn and we lost the game.
- **Method:** traced the replay's final turns via raw JSON (`energyCards` id
  11 = Mist Energy, confirmed attached to the opponent's Snorlax the entire
  endgame). Confirmed the retreat destination picker (`_score_bench_target`)
  scores purely by attack-readiness with no HP/survivability term.
- **Result/decision:** per user direction, did NOT add HP-survivability
  weighting to retreat targeting — even the tankiest bench option wouldn't
  have survived the actual lethal hit, so out-surviving wasn't the real fix.
  Instead added a `desperation` flag (opponent needs ≤1 more prize, or ≤2 with
  one of our ex Pokémon exposed) that overrides the normal deck-out/threshold-
  discipline guards: once desperate, keep drawing past the normal KO threshold
  (Poffin/Dawn/Hilda/Poké Pad/Dudunsparce's Run Away Draw/Enriching all keep
  firing), force `racing_for_alakazam` (Candy straight to Alakazam over the
  slower manual climb), and stop declining draw abilities on the stype==9
  "may use ability?" prompt. Boss's Orders targeting already preferred
  highest-prize-value-among-KOable, so no change needed there for this case.
- **Hypothesis 2 (user-reported):** replay `83458785` — Poké Pad fetched
  Alakazam from the deck on turn 1 with zero Abra anywhere (play or hand).
- **Method:** read `_score_deck_search`'s `card_score()` — used by every deck
  search (Poké Pad, Dawn, Hilda, Lana's Aid, Sacred Ash).
- **Result:** confirmed, and it's a real bug, not bad luck. Alakazam scored a
  flat 95 (top priority) whenever `not has_alakazam`, with zero check for
  whether an Abra/Kadabra existed anywhere to evolve it from — Abra itself
  only scored 90. Fixed: Alakazam only keeps the 95 priority if a line piece
  exists in play or hand (`have_line_piece`); otherwise it drops to 20, well
  under Abra's 90.
- **Hypothesis 3 (user-reported):** same replay, step 94 — ready to KO a
  110/430-HP Mega Starmie ex (already damaged, best target on the opponent's
  board) but instead played Boss's Orders and gusted in a fresh 70-HP Staryu
  for a 1-prize KO, undoing our own damage progress.
- **Method:** loaded `steps[93]` directly — confirmed our Alakazam had 0
  energy that turn (`attack_available` False, so `can_ko`/`boss_target_exists`
  were both False), and the only branch that could have fired was the
  standalone `phase==PHASE_CLOSING: return 199.0` line, which had no target-
  quality gate at all.
- **Result:** confirmed. `PHASE_CLOSING` (opponent ≤2 prizes) gave Boss's
  Orders a flat 199 regardless of whether a target existed or whether the
  current active was already the best one to leave alone — directly violates
  `docs/piloting-guide.md` §7's own rule ("Never Boss when you already have
  lethal on the Active for the same prize count"). Folded the phase bump into
  `boss_target_exists` (which already verifies a real, worthwhile bench target
  and that we can act this turn) instead of firing unconditionally.
- **Feature (user-requested, scoped down from full simulation):** added
  `lone_active_opportunity` — a rough (not simulated) headroom estimate
  (current hand + untapped Dudunsparce Run Away Draws + best unplayed
  supporter + unattached Enriching) compared against the KO threshold, when
  the opponent's bench is completely empty. When it clears the threshold,
  same override behavior as `desperation` (stop banking draw sources, go for
  the kill this turn) rather than reserving cards for a hypothetical future
  hand-disruption effect.
- **Open lead, not yet confirmed or fixed:** `main.py` has zero handling for
  `sel['context']==42` (MULLIGAN per `docs/engine-api.md`) — one raw-JSON
  example found it arrives as a `stype==9` YES/NO prompt, which currently
  falls through to logic written entirely for the unrelated "may use ability?"
  draw prompt (checks deck count / cards needed, not hand contents). Found
  zero examples of OUR agent actually facing this prompt across the 6 replays
  checked this session, so this is unconfirmed as a cause of any real loss —
  but if it ever fires, the current code answers it with logic that has
  nothing to do with "does this hand have a Basic Pokémon," which could
  silently invert keep/mulligan. Needs a confirmed firing example before
  writing a fix (same lesson as hypothesis 2 in the prior 07-03 entry: don't
  guess at select semantics without a real replay showing them).
- **Hypothesis 4 (user-reported):** replay `83461698` — turn 3, played Hilda
  instead of Dawn with zero Abra anywhere (active was a non-attacker
  Genesect, no bench); Hilda's search fetched an unplayable Alakazam.
- **Method:** dumped the raw `sel['deck']`/`sel['option']` for both of
  Hilda's search picks. `docs/piloting-guide.md` line 149 already documents
  Dawn as "Basic + Stage 1 + Stage 2" vs Hilda's narrower pool.
- **Result:** confirmed on two levels. (1) Hilda's search options in this
  exact state were only {Alakazam, Kadabra, Dudunsparce} — Abra was never a
  legal choice, confirming Hilda's pool structurally excludes Basics. (2) The
  hypothesis-2 fix's `have_line_piece` check incorrectly counted a Kadabra
  *card sitting in hand* as "having a line piece" — but a Kadabra in hand with
  no Abra anywhere can't be played at all (it isn't a Basic, and Rare Candy
  also needs a Basic already in play), so it's exactly as dead as an
  Alakazam. Removed `kadabra_in_hand` from `have_line_piece`. Separately, Dawn
  vs Hilda's flat ESTABLISH-phase weights (22 vs 24) don't know about this
  context — Hilda's is tuned slightly higher and wins by default even when it
  structurally can't fix the actual problem. Added `need_basic_abra` (no line
  pieces in play, no Abra in hand) that drops Hilda to 3.0 in that state,
  letting Dawn's unaffected 22 win instead.
- **Hypothesis 5 (user-reported):** replay `83462350` — Boss's Orders gusted
  a 340-HP Mega Lucario ex away for a 1-prize Solrock/Lunatone KO, "when we
  could have done ~440 damage to the Lucario directly."
- **Method:** replayed the exact game state (raw step 131, our side down to
  a lone Abra + Dunsparce after our Alakazam line got repeatedly KO'd, 24-card
  hand, deck at 11) through `score_options_main` directly, pre- and post-fix.
- **Result:** same root cause as hypothesis 3, confirmed via direct replay
  rather than inference this time — pre-fix this state hits the exact
  unconditional `PHASE_CLOSING: 199.0` branch (active was Abra, so
  `active_can_attack` was False and the 440-damage attack the user describes
  was never actually available no matter what we played — Powerful Hand only
  works from Alakazam). Post-fix, Boss's Orders scores 4.0 in this state, well
  below Dawn/Poffin/Poké Pad's desperation-override 30.0. Rare Candy/manual
  evolve weren't legal options here either (the Abra had `appearThisTurn` —
  can't evolve the same turn it entered play), so this specific turn was
  likely lost regardless once we were down to Abra+Dunsparce against a
  repeat-KOing Lucario ex; the fix stops the wasted Boss's Orders play but
  doesn't address the deeper pattern (see report relevance below).
- **Report relevance:** five replay-verified correctness fixes (retreat
  desperation logic, deck-search priority x2, Boss's Orders phase override,
  Hilda-vs-Dawn context) plus one new scoped heuristic feature for the
  heuristic-agent section; the mulligan gap is worth a mention in the "known
  limitations" framing even unconfirmed, since paper-TCG mulligan handling is
  a natural reviewer question for a heuristic Pokémon TCG agent. Also worth
  flagging as a limitation: hypothesis 5's replay shows a "board-thinning"
  pattern not fixed this session — ending up with only 1-2 Pokémon in play
  and a bloated, mostly-dead hand after our attacker line gets repeatedly
  KO'd by a hard-hitting opponent. Card-search/draw sequencing fixes can't
  address this; it's a bench-development-under-pressure question (do we
  over-invest in card advantage relative to board presence when facing a
  attacker that keeps trading through our line?) worth a dedicated look.
- **Shipped 2026-07-03 as v25c, submission `54282648`** (status PENDING at
  ship time). See `docs/version-history.md` v25c entry and
  `training/ladder_history.csv` for the ladder result once it converges.

---

## 2026-07-03 — v10 "fluke" check + full gauntlet baseline for v10/v25/v25b

- **Hypothesis (user-reported):** Kaggle kernel `notebook9655c0145f` version 10
  scored 680.7 on the ladder on 2026-06-24, a big same-day jump from version
  9's 499.7 — was this a fluke (ladder noise) or a real strength jump?
- **Method:** `kaggle kernels pull` 403'd on a specific historical version for
  this kernel; `kaggle kernels output jander6364/notebook9655c0145f/10` worked
  instead (fetches the notebook's saved `/kaggle/working` files — i.e. the
  actual built `main.py`+`deck.csv` submission, not just source). Copied to
  `training/baselines/v10_kaggle.py`, gauntleted 200 games/anchor. Also took
  the opportunity to gauntlet the current `main.py` fresh (200/anchor, name
  `v25b`) now that `training/harness.py::summarize()` is fixed.
- **Result:** not a fluke, and not a strength outlier either — v10 is the
  **weakest** agent in the whole gauntlet table (gElo 292, below v21's 402),
  losing 22.5%/29.0%/25.0% to v21/v22/v23 while beating the simple sample bots
  solidly (69-94.5%). The same-day jump has a mundane explanation sitting in
  the pulled code's own docstring: "v6 critical fix: REMOVED false-positive
  wall detection that was suppressing attacks against ALL Fighting/Colorless
  decks" — v9 was very plausibly barely attacking at all; v10 fixed that. A
  real bug fix producing a real jump, not ladder noise. Side note: the
  project's internal `vN` docstring counter and Kaggle's autoincrementing
  kernel "Version N" counter are two different numbering schemes that don't
  correspond 1:1 (confirmed by diffing this pull against the notebook's
  current/latest version, which is internally labeled "v6" despite being a
  much later kernel version) — don't conflate them when reading old
  submission history.
- **Bonus finding:** `v25b`'s fresh gauntlet gElo is **551 — top of the whole
  table**, above `v24` (507) for the first time offline, and the real ladder
  score confirms the same direction: submission `54279766` (v25b) scored
  **861.8**, up from `54277762`'s (v25) 732.0 and above v24's 698.1. First
  time this session's fixes show a clear, non-noisy signal rather than a
  within-CI wobble.
- **Report relevance:** methodology note (kernel-version retrieval quirk,
  internal-vs-platform version numbering mismatch) plus a genuinely positive
  result for the heuristic-fixes section — worth leading with 861.8 over the
  more equivocal local A/B percentages.

---

## 2026-07-03 — Mist-wall retreat + Candy-racing fixes; Dragapult ties were a harness bug, not the game (v25b)

- **Hypothesis 1 (user-reported):** watching the first v25 replay live, a fully
  fueled, full-HP Alakazam retreated into a Kadabra with no attack available —
  looked like a bad heuristic call.
- **Method:** traced replay `83429870.json` turn 20 by hand (raw step/option
  indexing — the option list an action applies to is `steps[i-1]`, not
  `steps[i]`, easy to get off-by-one on). Reproduced the exact game state
  through `score_options_main` directly.
- **Result:** confirmed. `main.py`'s RETREAT scoring had `if opp_mist and
  active_is_alak: return 9.0` — an unconditional bonus for retreating Alakazam
  whenever the opponent's active carries Mist/Rocky Energy, with no check for
  whether we actually had Hammer/Boss in hand to exploit the escape, or any
  better bench target to promote into. In this deck (single-attacker,
  Genesect/Fez/Dudunsparce are all non-attackers), retreating a mist-walled
  Alakazam never helps — Hammer and Boss both target the *opponent's* side and
  work regardless of who's active, and re-promoting Alakazam later costs a
  second wasted retreat. Removed the branch; falls through to the existing
  `active_can_attack: return -2.0`, which discourages the retreat correctly.
- **Hypothesis 2 (user-reported):** Boss's Orders skipped a lethal fueled
  Trevenant to gust in a weak unevolved Pokémon "with no wall present."
- **Method:** traced all 3 Boss's Orders plays in the same replay, checking
  energy card IDs (not just displayed energy type — Mist Energy shows as
  colorless type but carries card id 11).
- **Result:** could not reproduce. All 3 plays targeted an active that
  genuinely held Mist Energy (verified by id), so Powerful Hand was actually
  blocked each time and Boss was the only legal way to progress; the bench
  target chosen was the best legal one (highest-HP guaranteed KO among
  non-walled targets). No fix applied — need a specific replay ID if this
  shows up again, since the one available game doesn't support the report.
- **Hypothesis 3 (user-reported):** Rare Candy is too slow to rush Alakazam
  online early, costing free KOs on undeveloped opponent attackers.
- **Method:** traced Candy timing in the same replay (no delay found — fired
  the first legal turn both times) and read `racing_for_alakazam`'s gate.
- **Result:** no bug in this replay, but a real, independent design gap: the
  gate only fires on defense (`active_below_half`) or late phase — it has no
  notion of "Candying now sets up a near-term KO," so it can't rush Alakazam
  purely for tempo/damage reasons early game. Added `candy_lethal_soon`
  (`active_abra_can_evolve and not has_alakazam and hand_n*20>=opp_hp and not
  opp_mist`) as a third OR-branch.
- **Hypothesis 4 (user-reported):** "there shouldn't be any way to tie a
  game" — re: the 50/100-one-seat-direction Dragapult ties noted in
  `CLAUDE.md` Outstanding Items.
- **Method:** reproduced `opponents/dragapult_agent.py` directly via
  `training/harness.py`, both seats, with `debug=True` to surface the
  traceback instead of swallowing it.
- **Result:** confirmed — and it's not the game. `dragapult_agent.py` (the
  Kaggle sample bot used as a local gauntlet anchor) imports `cg.api` for
  rich dataclasses; that library only exists in the Kaggle-hosted
  `kiyotah/cg-lib` dataset, not locally. The `try/except Exception` fallback
  stubs `AreaType`/`SelectContext`/`OptionType`/`CardType`/`LogType` but never
  defines `Pokemon`, so `isinstance(card, Pokemon)` throws `NameError` almost
  immediately — **100% of local games vs this anchor crash, in both seats.**
  Separately, `training/harness.py::summarize()` only ever read
  `rewards[0]` to classify win/loss/tie; when the crash landed in slot 0
  (reward `None`), it fell through to the tie branch instead of recognizing
  slot 1's reward of `1` as a win. Same event (opponent crash), miscounted
  as a tie in one seat and correctly counted as a win in the other — exactly
  the "ties in one seat direction" symptom. Fixed `summarize()` to check both
  reward slots. Verified: post-fix, 20/20 games in both seat orders vs
  `dragapult_agent.py` now correctly show `errors=0, ties=0`, 100% win.
  `lucario`/`abomasnow`/`starmie` anchors do NOT have this crash (0/15 each
  spot-checked) — this is specific to `dragapult_agent.py`. Left the anchor's
  underlying crash unfixed (would require vendoring `cg/api.py` from the
  Kaggle dataset — confirmed downloadable, plain ~26KB Python, not urgent);
  flagging that the entire `dragapult` column in `gauntlet_results.csv`
  across every prior version has never reflected real play, only "does the
  sample bot crash."
- **Decision:** shipped both `main.py` fixes as submission `54279766`. 400-game
  A/B vs frozen v23: 53.0% ± 4.9% (up from 52.0% pre-fix, CIs overlap — not a
  confident signal). Gauntlet gElo 767 (200 games/anchor, post-harness-fix):
  below v24 (791), above v23 (753)/v22/v21.
- **Report relevance:** two more replay-verified correctness fixes for the
  heuristic-agent section, plus a methodology note worth keeping: a
  measurement-harness bug can look exactly like a game-engine anomaly
  ("impossible" ties) — always reproduce anchor crashes with `debug=True`
  before trusting an aggregate stat.

---

## 2026-07-03 — Replay analyzer rebuild + confirmed deck-out root cause (v25)

- **Hypothesis:** `tools/analyze_replay.py`'s existing flags (missed-lethal, bad
  retreat, wasted energy) were reading 0 across all prior forensics passes because
  the underlying leaks are in categories the predicates don't check — not because
  the tool's decision extraction was wrong.
- **Method:** Cross-checked the tool's output against raw replay JSON by hand for
  two games. Found the tool never gates on each step's `status` field: the `select`
  object echoes into the opponent's INACTIVE steps (where we correctly return
  `[]`), and the tool read every echo as a fresh "timeout" — 28 phantom timeouts on
  a 40-step game, 102 on a 169-step game. Rebuilt the decision extraction (real
  decision = `steps[i-1][you].status=='ACTIVE'` + a `select`; action read from
  `steps[i]` regardless of that step's own status), added a terminal-cause triage
  classifier, and enriched each logged decision with hand/deck/active context.
  Verified decision counts now equal true ACTIVE-select counts (8 and 63 on the
  two spot-checked games, vs. 4 and 0-useful-signal before). Re-ran on all 17
  banked loss replays (`replays/v22/`, `replays/v23/`) and triaged by terminal
  cause before reading any turn-by-turn.
- **Result:** Triage: 1 `NO_POKEMON_IN_PLAY`, 2 `DECK_OUT`, 14 `OTHER` (mostly
  ordinary prize-race losses; one, `83166796`, is a suspected engine stall — see
  below). Turn-by-turn read of `DECK_OUT` replay `83348630` found a confirmed,
  fixable root cause: evolving a bench Kadabra into Alakazam triggers a separate
  Yes/No "use this Ability?" prompt for Psychic Draw (select `stype==9`), and
  `_choose`'s handler for it answered YES unconditionally — no deck-count check,
  unlike every other draw source in `main.py`. It fired at deck=3 with hand
  already at 17 (`cards_needed`=7) and a 5-2 prize lead, drew 3 more cards, and
  emptied the deck the same turn — losing a game that was otherwise winning.
- **Decision:** Shipped as v25: stype==9 now declines when `deck_count<5` and
  `hand_n >= cards_needed+3`. Verified end-to-end by replaying the exact obs from
  `83348630` step 160 through the patched `main._choose` directly (not just the
  aggregate A/B, which is a null result — 52.2% ± 4.9% only shows the fix didn't
  break the common case, it can't show the fix fires): confirmed `deck_count=3`,
  `hand_n=16`, `cards_needed=7` (opp HP correctly populated, not the 99999
  default) and the patched function returns `[1]` (NO), where the old code
  returned `[0]` (YES) unconditionally. This prevents *one* deck-out path in that
  game — a separate leak (sitting on non-attacking Dudunsparce with the Psychic
  energy misrouted there instead of onto the developing Alakazam/Kadabra line,
  so the `hand_surplus` stop-drawing gate's `ready_attacker_exists` precondition
  never engaged) means 83348630 was mislost several ways, not fixed outright by
  this change alone.
- **Triage is incomplete — flag, don't overclaim:** of 17 banked losses, only 3
  were read turn-by-turn (the confirmed `DECK_OUT` fix above, plus two spot
  checks). Both spot checks (`83166796`, `83168738`) turned up the **same**
  anomaly: healthy full-HP boards, opponent having taken only 2-4 of their 6
  prizes, the game stalling in `INACTIVE` for 7-14 steps, then an abrupt `DONE`
  loss with an empty action — i.e. **2 of the first 3 `OTHER`-bucketed losses
  inspected are this stall, not a normal prize race.** This is fresh, recurring
  evidence for the previously-unconfirmed "prize-selection engine stall" gap in
  `docs/piloting-guide.md` §13, but with only 2 data points it's not yet
  root-caused, and the remaining ~11 `OTHER` losses are still unread — the triage
  classifier itself may be under-detecting this stall pattern (it only catches
  `NO_POKEMON_IN_PLAY`/`DECK_OUT`/`PRIZED_OUT`==6, not "stalled with a healthy
  board"). Next session: add a stall detector to the triage classifier (last N
  steps have unchanged board state + empty/near-empty action) and read the
  remaining `OTHER` games before considering the loss-mining pass complete.
  `83344386` (`NO_POKEMON_IN_PLAY`) is a bad-opening-hand instant loss, consistent
  with the already-documented mulligan gap; no new fix.
- **Report relevance:** A concrete case study for the methods section on why
  tooling correctness must be verified against ground truth before trusting its
  output for analysis — the old tool's predicates looked like "no bugs found"
  when the actual defect was in the tool's own event extraction. Also a clean
  before/after pair (heuristic ceiling vs. DAgger teacher quality) for the
  ablation table.

---

## 2026-07-02 — v24 full Gauntlet baseline: the gElo scale reproduces known orderings

- **Hypothesis:** the Bradley-Terry gauntlet produces a strength scale consistent
  with independently-known results (version ordering, meta-bot win rates).
- **Method:** `gauntlet.py --candidate main.py --name v24 --games 200`, all 8
  anchors (1,600 games), pooled with the tune's phase-2 rows in the BT fit.
- **Result:** v24 **762** > v23 719 > v22 661 > v21 599 > dragapult 571 >
  starmie 432 > abomasnow 329 ≈ lucario 316 > random 0. Per-anchor: 99% random,
  93% Lucario, 92.5% Abomasnow, 86% Starmie, 55% v23 (200 games; pooled with the
  earlier 200-game A/B ≈ 57.5%). **Flags:** Dragapult 100W-0L-100T (step-limit
  ties persist, one seat direction); Starmie now 150W-6L-44T — the tie problem
  may extend to spread decks generally, not just Dragapult.
- **Decision:** gauntlet adopted as the standard candidate gate. Tie diagnosis
  (outstanding item 6) upgraded in importance — ties are half-losses on ladder.
- **Report relevance:** Figure 2's offline axis is now live (gElo column in
  `ladder_history.csv`); the scale-reproduces-known-orderings result is the
  validity argument for the evaluation section.

---

## 2026-07-02 — v24 deck simplification: +10% vs the pro list, logic untouched

- **Hypothesis:** Psyduck and Genesect (fine-grained human-meta tech, ~0 plays/game
  and ~100% rot in the audit) can be swapped for consistency copies (4th Alakazam,
  4th Dunsparce) at a measurable win-rate gain, with zero code changes beyond the
  deck list.
- **Method:** Edit `DECK` in `main.py` only (all Psyduck/Genesect code paths are
  play-conditional and simply go dead); regenerate `deck.csv`; 200-game
  seat-alternating A/B vs frozen v23 (identical logic, old deck) so the deck is
  the only variable.
- **Result:** **120W–80L (60.0% ± 6.8%), 0 errors.** CI excludes parity.
- **Decision:** v24 adopted. Ladder confirm pending; the 60 freezes after that.
- **Report relevance:** The §2 deck-thesis centerpiece: a pro list is tuned for
  human pilots; instrumented utilization + a controlled A/B adapted it for a
  machine pilot. Also a clean example of the "same logic, one variable" A/B
  methodology for §4.

---

## 2026-07-02 — Overnight SPSA weight tune launched (5.5h budget)

- **Hypothesis:** the ~25 hand-guessed `main.W` scoring constants leave win rate
  on the table; SPSA can recover some of it in one unattended night.
- **Method:** `training/overnight_tune.py` — phase 1: SPSA (±15% multiplicative
  perturbations, lr 0.10, 160-game evals) vs a frozen launch snapshot of v24, so
  weights are the only variable; every eval checkpointed atomically
  (`tune_ckpt.json`, auto-resume) and appended to `tune_log.jsonl`. Phase 2
  (last ~45 min): top-3 shortlisted candidates + a default-W control each play a
  gauntlet subset panel (v21, v23, starmie, 100 games/anchor, appended to
  `gauntlet_results.csv`); best panel win rate wins and is written to
  `variants/v24_tuned.py`.
- **Result (2026-07-02, run cut at 63% of budget to free the machine):** 172 SPSA
  iterations (~55,000 games). Best search-time candidate `i122plus` showed 61.9%
  vs the frozen snapshot — but the gauntlet finals read 71.0% vs the default-W
  control's 70.2% (Δ0.8% over 300 games, CI ±5.2%), and the 600-game A/B gate came
  in at **52.0% ± 4.0%** — CI includes 50%. The search-time 61.9% was winner's-curse
  selection (best of 344 noisy 160-game evals) plus mirror-matchup overfit.
- **Decision:** gate not cleared → **default weights kept**, `variants/v24_tuned.py`
  deleted per the variants policy. Tune logs (`tune_ckpt.json`, `tune_log.jsonl`)
  retained as report evidence.
- **Report relevance:** §4/§5 twofer: (a) ablation row "hand-tuned weights survived
  ~55k games of SPSA — the heuristic's constants were not the binding constraint";
  (b) a live demonstration of why pre-registered gates matter (a 61.9% search-time
  number would have shipped without one).

---

## 2026-07-01 — Roadmap locked; Gauntlet + deck audit built

- **Hypothesis:** n/a (infrastructure). A fixed-panel Bradley-Terry rating
  ("gElo") over accumulated results gives a single offline strength scale, and
  pairing it with realized ladder Elo quantifies how much offline results
  overrate — turning a war story into a calibration figure.
- **Method:** `training/gauntlet.py` — candidate plays a fixed 8-anchor panel
  (random, 4 meta bots, frozen v21/v22/v23), seat-alternating; all results
  accumulate in `gauntlet_results.csv`; BT fit with a half-win ghost-game
  regularizer; gElo = 400·log10(p/p_random). `tools/deck_audit.py` — per-card
  utilization (plays/game-drawn, rot rate, end-hand rate, win-rate deltas) over
  local mirror games, both seats harvested.
- **Result:** Both smoke-tested clean (4-game runs). Even at 4 games the audit
  fingered the expected passengers: Genesect and Psyduck at 0.0 plays/game, 100% rot.
- **Decision:** Deck simplification happens NOW, before any rigorous training
  (deck changes invalidate collected teacher data). Order: deck audit at scale →
  variant A/B → freeze deck → full gauntlet baseline → heuristic tuning → DAgger.
- **Report relevance:** Figure 2 (calibration scatter) is now being logged from
  day one. The deck audit is the evidence base for the "adapted a pro list for
  machine pilots" deck-concept story (the 20% axis).

---

## 2026-07-01 — Retroactive: BC warmup result and its diagnosis

- **Hypothesis:** A policy/value net behavior-cloned from v22 self-play reaches
  ~parity with the teacher.
- **Method:** 547,796 decisions from v22 mirror self-play; EmbeddingBag +
  1-layer Transformer (128d); 10 epochs on Kaggle T4 (`training/ptcg_bc_v1.pth`).
- **Result:** 85.9% held-out action match; **86% vs random** (clears 65% gate);
  **22% vs v22** (far from parity). 0 runtime errors in both evals.
- **Decision:** The 85.9%-vs-22% gap is the textbook compounding-error signature,
  not a bug. Cure is DAgger (train on the *net's* state distribution with
  teacher labels), not more BC epochs.
- **Report relevance:** This exact pair of numbers, plus the explanation of why
  they coexist, is the opening of the method section's failure→fix narrative.

---

## 2026-07-03 — Full retreat/energy/evolution logic audit (v25, second fix)

- **Hypothesis:** requested directly — a systematic, player's-eye read through
  every RETREAT/ATTACH/EVOLVE/ABILITY branch of `main.py`'s `score()` would turn
  up additional bugs beyond the stype==9 deck-out fix already shipped today.
- **Method:** Manual branch-by-branch read of `score()`, cross-checked against
  `docs/piloting-guide.md` §4-10 and the real card text (`docs/EN_Card_Data.csv`).
  Every hypothesis was checked against either replay evidence or a direct call
  into the real scoring function on a constructed state — not left as code-
  reading speculation.
- **Result:** One confirmed, high-confidence bug: `KADABRA` was absent from
  `NON_ATTACKER_IDS`, so a stuck Kadabra active (Super Psy Bolt is permanently
  suppressed to -5 elsewhere in the same file, making Kadabra a de facto
  non-attacker) fell through every retreat-priority tier. Verified directly:
  constructing a Kadabra-active/fueled-bench-Alakazam state and calling
  `score_options_main`, RETREAT scored 0.5 — *below* END TURN's 1.0 — before the
  fix, and 22.0 (correctly dominant) after. Two more findings surfaced at lower
  confidence, held for a decision rather than auto-fixed: (a) Wondrous Patch's
  energy-recipient select is routed through `_score_bench_target`, whose
  fuel-status tiebreak is correct for retreat/promotion but backwards for
  "who should receive this energy" — confirmed reachable via replay (same
  `area==5` option shape), not yet observed to misfire since the one instance
  checked had no already-fueled candidate; (b) Enriching Energy attached to a
  non-Dudunsparce support mon scores a mild +1.0 instead of a discouraging
  score, inconsistent with the Alakazam case's -8.0 for an equally wasteful
  attach.
- **Decision:** Shipped the Kadabra fix (400-game A/B vs frozen v23: 53.2% ±
  4.9%, CI includes parity as expected for a narrow-state fix, 0 errors, no
  regression) bundled with the stype==9 fix — same v25, not yet on the ladder.
  User said to keep auditing and fixing rather than pause for review; the two
  follow-up findings got a second look before touching more code: the Enriching
  "target preference" framing was wrong on closer reading (the draw-4 fires
  regardless of target, so a support-mon attach is real value, not waste — the
  actual bug was a *missing deck-safety gate*, fixed instead, and traced to a
  real contributing factor in the other `DECK_OUT` replay, `83156504`: attach at
  deck=5/hand=18 dropped the deck to 1 in one action). The Wondrous Patch fix
  was verified feasible before writing it — checked the actual select object
  across replays and found `effect.id==1146` cleanly distinguishes it from
  plain retreat/promotion selects, so it's a clean dispatch branch, not a
  design compromise. Both fixed, both verified against the real functions on
  reconstructed/constructed failure states, both A/B-gated (0 errors each,
  53.2% and 54.0%, both parity-range as expected for rare-state fixes). All
  four v25 fixes now: stype==9 deck-out, Kadabra retreat, Wondrous Patch
  targeting, Enriching deck-safety gate.
- **Report relevance:** A second, cleaner instance of the same methodological
  point as the first v25 entry (fix, then verify against the real function on
  the real failure state — not just an aggregate A/B, which is often a null
  result for narrow-state fixes). Also good ablation-table material: two
  distinct, root-caused heuristic bugs found and fixed in one audit pass before
  any DAgger data collection, directly raising the teacher's ceiling.

**Follow-up same day — closed the evolution gap, caught two bugs in the fix
itself by testing:** user asked to fix the manual-evolve-vs-Rare-Candy gap
(`piloting-guide.md` §13's long-standing "⚠️ approximate") rather than leave it
open. Added `racing_for_alakazam` (no Alakazam yet + (behind-and-hurt or
opponent ≤2 prizes)) to gate the Candy-vs-manual scores. First draft was wrong
twice, both caught by direct synthetic testing before trusting it: (1) reused
`active_vulnerable`, whose `active_hp<60` absolute clause is always true for
Abra (50 max HP) — made the gate a permanent no-op; switched to relative
`active_below_half`. (2) included `emergency_draw` (hand≤4), spuriously true
turn 1-2 before any drawing has happened — exactly when there's the *most* time
to climb manually; dropped it. Final version verified correct on three
synthetic states (neutral → manual, hurt-and-behind → Candy, opponent-closing →
Candy). 400-game A/B vs frozen v23: 52.7% ± 4.9%, 0 errors. This is the
methodological throughline for the whole day: every fix, including this one's
own two false starts, was checked against the real scoring function on a
concrete state before being trusted — code that "looks right" was wrong twice
in a row here.
- **Shipped 2026-07-03** — bundled all five `main.py` logic fixes with the other
  agent's v23-deck revert into one Kaggle submission. See
  `training/ladder_history.csv`.

---

## 2026-07-03 — v24 deck reverted to v23 on early ladder trend

- **Hypothesis:** v24's deck swap (4th Alakazam + 4th Dunsparce in, Genesect +
  Psyduck out) would hold or improve on v23's ladder rating, per the 200-game
  local A/B (60.0% ± 6.8% favoring v24).
- **Method:** Watched `training/ladder_history.csv`-tracked public Elo across
  the first day of v24 live: 680 at ~7h, 780 at ~24h.
- **Result:** User judgment call to revert to v23 now, ahead of the documented
  48h/50-point decision-rule checkpoint (`CLAUDE.md` Outstanding Items #1).
  Note for the record: v23 itself had decayed to 773 by its own day-2 reading,
  so v24's 780-at-24h is not obviously a regression on an apples-to-apples
  basis — the local A/B result favoring v24 stands unexplained by this data.
  Reverted anyway per explicit user instruction.
- **Decision:** `main.py`'s `DECK` list and `deck.csv` reverted to the v23
  composition (3× Alakazam, 3× Dunsparce, Genesect + Psyduck back in). No
  scoring-logic changes — v24's change was deck-list-only, so this is a clean
  revert. Ship via Kaggle CLI as a new submission.
- **Report relevance:** Honest negative/inconclusive result for the deck
  simplification experiment — local A/B favored the change, live ladder data
  was ambiguous, and the team chose to revert on a provisional read rather than
  wait out the full evaluation window. Useful methodology-section material on
  the tension between fast local A/B and slow, noisy ladder confirmation.

---

## Pre-2026-07 — Retroactive: findings already banked

| Finding | Numbers | Report use |
|---------|---------|-----------|
| Offline sims systematically overrate | v5: 64% offline → 0-5 on ladder | Motivates the Gauntlet + ladder-only evaluation discipline; Figure 2 |
| SP-only training collapses without BC mixing | 46% → 20% vs teacher over 3 iterations | Honest negative result; motivates the 40/60 BC/SP batch-mix rule |
| Engine reverse-engineering | Options are positional (0/1,287 carried cardId); deck searches are NOT blind; area enum 6=PRIZE not discard; setup-active field never populated (live bug since v7) | Credibility material — engine understood at a depth most entrants won't reach |
| Heuristic version arc | v22 beat v21 56.3%±4.9% (400 games); baselines 94% Lucario / 94% Abomasnow / 79% Starmie / 50W-0L-50T Dragapult | Teacher-quality context for the BC/DAgger sections |
| Dragapult step-limit ties | 50/100 local games tie in one seat direction | Open question; ties are half-losses on ladder rating |