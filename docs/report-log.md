# Report Log — Experiment Journal & Method Glossary

*Every experiment gets a dated entry here THE DAY IT RUNS: hypothesis, method in
plain English, result with numbers, decision, report relevance. In September the
final report is assembled from this file — nothing gets retrofitted. Newest first.*

**Last updated:** 2026-07-02

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

## Pre-2026-07 — Retroactive: findings already banked

| Finding | Numbers | Report use |
|---------|---------|-----------|
| Offline sims systematically overrate | v5: 64% offline → 0-5 on ladder | Motivates the Gauntlet + ladder-only evaluation discipline; Figure 2 |
| SP-only training collapses without BC mixing | 46% → 20% vs teacher over 3 iterations | Honest negative result; motivates the 40/60 BC/SP batch-mix rule |
| Engine reverse-engineering | Options are positional (0/1,287 carried cardId); deck searches are NOT blind; area enum 6=PRIZE not discard; setup-active field never populated (live bug since v7) | Credibility material — engine understood at a depth most entrants won't reach |
| Heuristic version arc | v22 beat v21 56.3%±4.9% (400 games); baselines 94% Lucario / 94% Abomasnow / 79% Starmie / 50W-0L-50T Dragapult | Teacher-quality context for the BC/DAgger sections |
| Dragapult step-limit ties | 50/100 local games tie in one seat direction | Open question; ties are half-losses on ladder rating |
