# Neural Net Training Log & Roadmap

*Running log of NN architecture, training status, and the phased roadmap for the
learned-piloting agent (the 70%-weighted "model approach" axis).*

**Last updated:** 2026-07-07 (Phase 2 round 3 result: retrain on the
redesigned-opponent-pool corpus does NOT clear the pre-training baseline —
CLOSED, negative. See "AlphaZero-Style Push" → Resume Here.)
**Evidence rule:** method claims route through the pre-registered method bake-off —
see `docs/competition-strategy.md` → Method Bake-off Protocol and `docs/report-log.md`.
**>>> ACTIVE PLAN: AlphaZero-style push (user-directed 2026-07-05) — Phase 1
(encoding enrichment) RESOLVED 2026-07-06: a warm-start confound explained
the earlier regression; fixed, richer encoding now ties the plain baseline
(harmless, not yet proven better) and is ADOPTED. Phase 2
(`mcts_collect.py` self-play-with-search infra) is next. See
"AlphaZero-Style Push" section below. Supersedes/extends the Phase 0
framing (still active in parallel — see "Phase 0" section).**
**Status:** RESTARTED. All prior training data was lost, but the engine now runs
**fully locally** (`training/README.md`) — data collection no longer needs Kaggle
or even the Vivobook (the Vivobook multiplies throughput). Teacher is now v25c
(was v22). `ptcg_bc_v2.pth` (frozen-deck BC retrain) is done and gated.
DAgger rounds 1 and 2 are done: both work (fidelity 73%→81%→82% across BC→
r1→r2) but win-rate vs teacher stayed flat (~12-17%) throughout, and round 2
(testing a lower collection temperature) showed the fidelity gain itself
diminishing (+8pp then +0.8pp) — two rounds of evidence that further DAgger
rounds won't move the needle much more. **DAgger paused here**; `ptcg_dagger_r2.pth`
is the best checkpoint but not ship-ready. **Stage 2 direct-self-play AWR CLOSED,
negative (2026-07-04):** two β values tested (1.0, 0.5), both flat-to-worse
vs teacher (15.8%, 11.5%) and not significantly better than the seed either — more
aggressive weighting made it worse, ruling out a simple β fix.

**Stage 5 (search-at-inference) STARTED same day, ahead of schedule:** per
advisor guidance (given AWR's value-head-saturation finding below), inference-
time search over a heuristic-dominant prior is very likely a shorter path to
beating the teacher than another training campaign — search over a good prior
improves on it almost by construction, no retraining needed. Two cheap probes
before committing engineering: (1) branching semantics — confirmed, one
`search_begin` root branches into independent children via repeated
`search_step` calls; (2) timing — confirmed, ~730 sims/decision affordable
under the 10-min clock on engine cost alone. Both green-lit the approach; see
`docs/engine-api.md` "MCTS branching + timing probe" for full numbers. Also
found: a local dev shim (`training/setup_local_search.py`) pairs the
locally-installed native engine binary with the fuller cg-lib dataset source,
so the whole MCTS build/test loop now runs with **zero Kaggle round-trips** —
see `docs/engine-api.md` "Local search dev shim". Built `training/nn/mcts.py`
(`MCTSSearcher`, heuristic-guided PUCT selection + heuristic-vs-heuristic
rollout-to-terminal leaf evaluation — deliberately NOT the value head, since
the AWR diagnostic already confirmed it saturates bimodally near ±1 and would
carry little per-decision signal as a leaf evaluator) and
`training/nn/mcts_agent.py` (agent-contract wrapper, plugs into
`training/ab_test.py` directly). Smoke-tested: 0 errors across 10 games at
sims=30 (all decision types exercised, no crashes).

**CLOSED, negative (2026-07-04) — five gates, three real bugs fixed, one
architectural fix tried, still no signal.** Sequence: 41.7% (untuned) →
32.5% (sharper prior + more sims — worse) → 1.7% (after fixing the rollout
to use the real `main.agent` instead of a weaker reconstruction, AND fixing
strategy fusion via per-simulation re-determinization — a `_STALL_MEMO`
global-state corruption bug made this catastrophic) → 8.3% (after fixing
the `_STALL_MEMO` leak). **The limiting mechanism**, found via a 2-minute
targeted diagnostic (3 fixed positions, 30 sims each) rather than a 5th
full gate: PUCT visits piled entirely on one action every time (zero
exploration — MCTS collapsed to `argmax(score_options)`, the weak ~30%
prior), and every terminating rollout across all 3 positions was a win
(90/90, zero losses) — the simulated "opponent" (a random hand of our own
deck piloted by our own Alakazam heuristic) is too weak to ever punish a
bad root choice, so the rollout carries zero discriminating signal.
**Escalated to the user; a Claude Fable consult recommended one time-boxed
fix**: swap the rollout's opponent to a real adversarial bot
(`opponents/lucario_agent.py`) instead of a mirror of our own
deck/heuristic (also required isolating a second `_STALL_MEMO`-style
global-state hazard in that module). Diagnostic re-check showed real, if
partial, improvement (losses appeared, one position's visits split
15/15). **Pre-registered gate (≥55% to continue): 0W-50L (0%)** —
decisively below bar, though at this low true rate not statistically
distinguishable from "still the same ~2-8% broken regime," i.e. not a new
regression, just confirmation the fix wasn't enough. **CLOSED per the
pre-registration** as a third load-bearing negative result alongside
DAgger's imitation-ceiling and AWR's saturated-value-head findings — all
three converge on the same theme: a value/policy signal built without
genuinely external information can't supply what's needed to exceed the
teacher it's built from. Exceeding v25c via search would need real
Kaggle-gated MCTS expert iteration with a trained (non-degenerate)
value/policy pair — out of scope for now. Per Fable's sequencing, pivoted
to Stage 3 Phase B next regardless of outcome. Full writeup:
`docs/report-log.md` 2026-07-04 "Stage 5 search-at-inference" entries
(multiple, same day). See "Resume Here" for the full AWR writeup this
decision responds to.

---

## AlphaZero-Style Push (2026-07-07 — Phase 1 RESOLVED+adopted; Phase 2 infra DONE; round 2 regression ROOT-CAUSED as collection label imbalance and FIXED via a redesigned opponent pool; round 3 run to completion — CLOSED, negative, see Resume Here below)

**>>> RESUME HERE:** round 3's redesigned-opponent-pool retrain is DONE and
gated — result: **does NOT clear the pre-training baseline** (post-training
0.566 vs pre-training 0.584 ALL sign-acc, CIs heavily overlapping — flat,
not a regression, but no real signal either). Full numbers, per-segment
breakdown, and the two `mcts_collect.py` bugs found/fixed while getting
this result (a stalled Kaggle notebook that lost 9-11 hours of compute to
zero recoverable output, then a local re-run that hit a second, distinct
crash): `docs/report-log.md` 2026-07-07 "Phase 2 round 3 (redesigned
opponent pool)" entry. **This closes the "more self-play data alone" line
started by round 2** — consistent with every other arm this project has
tried (DAgger, AWR, PIMC search, oracle-critic, IQL): fixing the collection
pipeline's bugs (label imbalance, then these two) stops active harm but
doesn't manufacture a real learning signal from data generated by the same
checkpoint it's training. The corpus (`training/mcts_p3_r1.pkl.gz`, 300
games, 24116 samples) and checkpoint (`training/ptcg_sp_p3_r1.pth`) are
kept for reference but not a basis for further iteration without a
genuinely new information source (real replays, a stronger opponent, or
search-derived targets that reach further than the value net's own
priors). The one thing that DID clearly work this session, orthogonal to
whether round 3 itself is used further: the endgame-gated search line
(`docs/report-log.md` 2026-07-07 "Endgame-gated search" entries) — a real,
CI-confirmed 59.0%±4.8% win rate over v28 itself, now shipped as v29.

**Original plan/setup, for context:** `training/nn/opponent_pool.py`
replaces the same-checkpoint mirror opponent (root cause of round 2's
regression — see below) with a real-ladder-meta-weighted pool of the
project's existing rule-based archetype bots (lucario/dragapult/starmie/
abomasnow with their own real decks+logic; several more archetypes'
reconstructed decks piloted by `generic_pilot.py`; a mirror slice piloted
by the real heuristic `main.py` rather than the net's own weak policy).
Validated at n=20 games: win rate 96.7%→45.0%, corpus outcome balance
96.7%→43.1% positive (real ladder: 47.1%), `value_target` mean
+0.716→-0.034 (collapsed→well-centered). Full data:
`docs/report-log.md` 2026-07-07 "Collection-opponent redesign" entry.
`training/harness.py`'s `run_matches` gained an optional `decks` param
(mirrors the existing `extra_envs` pattern) to support this. **The 300-game
run itself was originally launched on Kaggle
(`jander6364/ptcg-phase2-round3-collect-retrain-gate`) but stalled 9-11
hours and was killed with zero recoverable output — re-run locally instead
once two real `mcts_collect.py` bugs were found and fixed (no incremental
checkpointing; a fatal crash on a None-outcome game), completing in about
1 hour.** This project's local search engine (confirmed working since
2026-07-04) sidesteps Kaggle's opaque session cap entirely for this kind of
collection job going forward.

**Root cause of round 2's regression (for context, now resolved by the
above):** severe win/loss label imbalance in `mcts_collect.py`'s corpus
(`mcts_p2_r3.pkl.gz` was 96.7% positive outcomes — the 40-sim
search-augmented collecting side crushed its no-search, same-checkpoint
opponent almost every game — vs. the real ladder replay gate set's 47.1%
positive), which collapsed the trained value head toward unconditionally
predicting "winning" (confirmed via output histogram on 300 real replays:
54% of ALL outputs, wins and losses alike, landed near +1). That collapsed
predictor scored close to the replay set's true positive rate on the
sign-accuracy gate — almost exactly the measured 0.446. **The initial
instinct to blame the sign-flip framing from the buggy run, or a
newly-found leaf-eval bug, for this regression was wrong** — see
`docs/report-log.md` 2026-07-07 "Root cause of the Phase 2 round 2
regression" entry for the full diagnostic chain (3 cheap on-disk checks,
no recollection needed) and why the initial diagnoses didn't reconcile with
the facts.

**A separate, real bug WAS found and fixed during the same audit, but
confirmed NOT to be the round-2 regression's cause:** `mcts.py`'s
`MCTSSearcher._net_leaf_value` used the DMC "max policy logit as Q"
convention (correct for the old Phase 0 probe checkpoint it was built for)
instead of the train_sp.py-lineage checkpoint's actual `value_head` output.
Fixed via a new `net_value_source` parameter ("qmax" default preserves old
behavior; `mcts_collect.py` now explicitly sets "head"). This fix is
correct and stays in the code regardless of round 3's outcome.

**Retracts the earlier (2026-07-06/morning-2026-07-07) hope that "the
corrected/negated number (0.630 ALL) would already be the best real-replay
value signal ... if it holds up post-fix"** — it did not hold up, and per
the diagnosis above, the reason isn't a further sign-convention issue; the
buggy run's own 0.630 was very likely a coincidental artifact of negating a
bug-confounded number, not a real recoverable signal, and the underlying
label-imbalance problem was present in both the buggy and fixed corpora
(the `me_idx` collection bug and the label-imbalance problem are
independent issues in the same pipeline).

**Superseded diagnostic speculation, kept for context only — do NOT chase
this without rereading the resolution above:** a logical tension had
suggested a possible fourth instance of the seat/sign bug class in the
`dataset.py`/`collate()` load path. That path was fully audited this
session (both `dataset.py`'s `collate`/`BCDataset.__getitem__` and every
seat-dependent function in `encode.py`/`threat.py`/`phi_baseline.py`) and
found clean — no sign bug exists there. The tension is fully explained by
label imbalance instead: a collapsed near-constant-positive predictor is
not "anti-predictive," it's just uninformative, and it produces a below-
baseline sign_acc on a balanced set for reasons that have nothing to do
with sign conventions. Any further work in this area should start from the
collection-design fix above, not from re-auditing the load path again.

**Prior (superseded) framing, for context only:** a third instance of the
"hardcoded seat index" bug (same class as `dmc_nstep.py`'s `_phi_at`) was
found and fixed in `selfplay_collect.py::compute_value_targets`'s
`shaped_reward` call (was hardcoded `me_idx=0`, now reads the real seat
from each decision's own obs) — this fix is real and correctly applied;
it's the ROUND 2 RESULT above (not the fix itself) that's now retracted.

**Why:** the user directed a genuine push toward AlphaZero-style training —
self-play generated via real search with MCTS visit-count policy targets
fed back into training — rather than the current setup, where a DMC-trained
value net and an inference-only search wrapper (`mcts_leafeval_agent.py`)
never actually talk to each other during training. The user specifically
named two levers: richer situational/belief information into the network,
and substantially more training, not just inference-time search.

**Plan (per an ml-engineer consult, sequenced to keep results
interpretable):**
1. **Encoding enrichment first, standalone-gated** — DONE 2026-07-05,
   result flat-to-slightly-worse (see below). Isolated so a later Phase 2
   result isn't confounded between "features helped" and "search helped."
2. **Phase 2 — DONE 2026-07-06: `mcts_collect.py`**, the infrastructure-
   validation milestone — asymmetric self-play (search on our side only via
   `mcts_leafeval_agent.py`, reduced sims=40 not eval-time's 100; opponent
   side via `selfplay_agent.py`'s plain temperature-sampled net policy,
   same checkpoint — true self-play), extracting real MCTS visit-count
   policy targets + search-backed root values into the ALREADY-EXISTING but
   previously-unused plumbing in `train_sp.py`/`dataset.py`/
   `selfplay_collect.py` (`policy_target` soft cross-entropy,
   `mcts_root_values` override, `compute_value_targets`). Built:
   `MCTSSearcher.choose_with_stats()` (exposes visit counts N + root value,
   `choose()`'s existing contract unchanged), `mcts_leafeval_agent.py`'s
   new `MCTS_COLLECT_LOG` hook (logs one record per real search decision),
   `mcts_collect.py` itself (runs serially, `workers=1`, for correct
   game/outcome correlation on this first validation — a wrong correlation
   would be a silent, hard-to-detect training-data bug). **A real bug found
   and fixed along the way:** matching collected records to game decisions
   by raw obs content initially gave 0% matches — traced to
   `remainingOverageTime`/`step` being inconsistently populated between the
   live agent-facing obs and the same state reconstructed from `env.steps`
   afterward; fixed by keying matches on only `current`+`select` (the
   stable game-state/options parts) via `json.dumps(sort_keys=True)`, not
   raw pickle bytes (which is itself unsafe — dict insertion order can
   differ between `==`-equal dicts). **Validation: 30 games, 0 relabel
   errors, 2336 samples, match_rate=0.847** (the ~15% gap is trivial/
   multiselect decisions that never involve real search, by design, not a
   bug) — clears the success bar this milestone was scoped to (the plumbing
   works, zero relabel errors, matching the bar DAgger's own collector was
   held to). Full data: `docs/report-log.md` 2026-07-06 "Phase 2 built and
   validated" entry.
   **PARALLELIZED same day (2026-07-06, per a Claude Fable sequencing
   consult):** `harness.py`'s `run_matches`/`_worker` extended with
   optional per-job `extra_envs` (backward compatible — verified existing
   callers unaffected), `mcts_leafeval_agent.py` embeds a new
   `MCTS_GAME_ID` per record, and `mcts_collect.py` now correlates records
   to games by that ID instead of serial submission order — enabling real
   `workers>1`. **Validated at 300 games/15 workers: 0 relabel errors,
   21,143 samples, match_rate=0.838** — consistent with every smaller-scale
   run this session (0.794-0.847), no degradation from parallelism, and at
   the actual scale (300 games) this project's trustworthy results have
   historically needed. Full data: `docs/report-log.md` 2026-07-06
   "Parallel game_id collection mechanism" entry. **A second real latent
   bug found via the smoke-test discipline while first exercising this
   pipeline through `train_sp.py`:** `dataset.py`'s `collate()` hardcoded
   `numeric = torch.zeros(B, 13, ...)` instead of importing `NUM_FEATS`
   from `encode.py` — harmless before Phase 1's 13→25 feature change, a
   hard crash once exercised. Fixed. Next: an actual (not just plumbing)
   training pass on the 21,143-sample corpus.
3. **Gating between generations** (once real training is attempted): 400+
   games minimum, given this project's repeated history of 30-game gates
   being later retracted (see the 40%→19% correction above).

**Phase 1 result: flat-to-slightly-worse, not the hoped-for improvement.**
Confirmed the hypothesis was well-motivated first: `encode.py::numeric_feats`
had only 13 raw counters (HP ratios, hand/deck/prize counts, one hardcoded
Mist flag, turn) — no archetype belief posterior, no opponent-threat
estimate, no evolution-line progress, despite all three already existing
elsewhere in the project. Extended to 25 features (belief posterior + wall/
crustle flags via `main.py`'s own `_belief_posterior`, `threat.net_threat_diff`,
line progress + `has_alakazam` via `_census`, hand-vs-KO-threshold
`hand_advantage`). Retrained on the IDENTICAL data/targets as the existing
round-2 n=1 checkpoint (isolating the encoding as the only variable), gated
on the same 1356-replay corpus: **ALL 0.586 [0.558,0.612], LATE 0.646
[0.611,0.678] — WORSE than the 13-feature checkpoint's 0.609/0.700** on
both metrics (CIs overlap, not a clean statistical loss, but clearly not an
improvement either). In-distribution training accuracy was actually HIGHER
with richer features (0.9896 vs 0.9789) — consistent with this project's
repeated in-distribution-fit-doesn't-predict-generalization pattern.
Plausible, unconfirmed explanations: more parameters (13→25 input dims)
without more training data/epochs to constrain them; redundancy with what
the transformer already extracts from raw board-slot card IDs; belief
posterior is known near-coinflip early-game (Phase A/B work) and folding it
in at ALL turns may inject noise in exactly the early states this project
has repeatedly found near-chance anyway. Single run, not yet replicated.

**Decision (superseded, see below): PAUSED for user input before Phase 2**,
per the plan's own contingency — proceeding to the much more expensive
self-play-with-search infrastructure work while carrying an unresolved
features question forward would make Phase 2's own eventual gate
uninterpretable. Full data: `docs/report-log.md` 2026-07-05 "AlphaZero-style
push, Phase 1" entry.

**RESOLVED 2026-07-06 — the regression was a training-harness confound, not
the features.** Isolated-component ablation (per the same discipline that
found the two Φ bugs) tested each of the three added feature groups alone
(`ENCODE_FEATURE_SET=base+threat/base+census/base+belief`): all three landed
at essentially the SAME regressed accuracy (0.583-0.585 ALL, 0.640-0.643
LATE) rather than spreading out by feature quality — a strong signal of a
structural cause, not feature-specific harm. Root cause: any width change to
`numeric_proj`'s input caused `train_dmc.py`'s shape-mismatch filter to
discard that whole layer's pretrained weights (fresh random init), while the
comparison checkpoint (`base13`) kept its already-trained version of that
same layer (no shape change there) — an unfair, confounded comparison. Fixed
`train_dmc.py`'s warm-start logic to copy the old weights into the
original 13 columns and leave only the genuinely new columns at fresh
init. Retrained the full 25-feature encoding with this fix
(`ptcg_dmc_p0_v2_n1_richenc_v2.pth`): **ALL 0.608 [0.582,0.634], LATE 0.700
[0.670,0.728] — essentially IDENTICAL to the 13-feature baseline
(0.609/0.700)**, closing the gap entirely. **Verdict: the user's hypothesis
is NOT falsified (the apparent harm was purely a measurement artifact), but
also not yet PROVEN as an improvement — a clean tie, not a win.** The 12 new
feature columns still only had 5 epochs, same as everything else, to learn
from scratch while riding an otherwise-converged network; more
epochs/data might let them pull ahead, not yet tested given time
constraints. **Decision: adopt the richer, warm-start-fixed encoding
going forward (confirmed harmless, philosophically aligned with the
user's direction, removes a future confound) and PROCEED TO PHASE 2**
(`mcts_collect.py` self-play-with-search infrastructure milestone) — the
original blocking condition (an unresolved features question) is now
resolved. Full data: `docs/report-log.md` 2026-07-06 "Feature-ablation
isolates the real cause" and "Warm-start fix confirmed" entries.

---

## Phase 0 (2026-07-05 — ACTIVE): n-step value targets + Φ-shaping + diverse self-play

**Status of the line this replaces — DMC (Deep Monte Carlo, DouZero recipe):**
`training/nn/dmc_agent.py` (epsilon-greedy Q-value agent) + `training/nn/train_dmc.py`
(regresses the taken action's logit, treated as Q(s,a), to the full-episode
Monte Carlo outcome — no BC/imitation mixing at all, the one operator in the
2026-07-04 literature review never tried). Three rounds, all vs. frozen v25c
only: round 1 `ptcg_dmc_r1.pth` 1.0%, round 2 (fresh temp-0.2 data) 1.7%,
round 3 (`ptcg_dmc_r2.pth`, retrained on rounds 1+2 combined) 2.5% (5/200,
greedy). A `--oversample-wins 6x` variant got 2.0% (ruled out class
imbalance as the bottleneck). Monotone-but-slow climb — genuine learning
signal (rules out a degenerate fixed-action bug) but at this rate the
25-30% pre-registered target needs an estimated 30+ more rounds. Full
numbers: `docs/report-log.md` 2026-07-05 DMC entries.

**Re-architected 2026-07-05** (full reasoning: `docs/report-log.md`
2026-07-05 "Strategy re-architecture" entry) rather than just running more
DMC rounds. Diagnosis: DMC/AWR/oracle-critic all share one root cause — a
single sparse terminal ±1 label backed up uniformly to every one of a
game's ~158 decisions, most of which are optional sub-plays (unlike chess/
Go's one-forced-move-per-ply) that didn't actually matter. This is a
credit-assignment problem, not proof mid-game value is unlearnable.

**Plan amended 2026-07-05 per ml-engineer review** (full transcript reasoning
below the plan) before implementation started. Five changes from the
original plan: (a) the calibration gate must beat **Φ-only sign-accuracy on
the 725 replays**, not the flat 62.5% oracle-critic figure — Φ (hand size +
prize differential) is outcome-correlated by construction and could clear
62.5% on its own, telling us nothing about the learned component; also
verify whether 62.5% was even measured on this same replay set before using
it as a reference point at all. (b) n-step bootstrapping is a bet *against*
DouZero's own design rationale (pure MC specifically avoids the
function-approximation + bootstrapping + off-policy deadly triad) — treat as
a real hypothesis under tension, not a safe default. (c) run an ablation
grid (n-step alone / Φ-only / n-step+Φ) so a pass/fail on the combined gate
is attributable to a specific piece. (d) **before the full week**, run a
cheap probe: plug the existing `ptcg_dmc_r2.pth` value net as-is into
leaf-eval search (no new value net needed) and check for any life over raw
argmax — the value net's main payoff is Phase 1 search, which is an
explicitly non-automatic stretch goal, so confirm the consumer has legs
before spending the week polishing its input. (e) use game-level (not
per-state) bootstrapped CIs for the replay gate — states within one game are
correlated, and per-state CIs would overstate power on only 725 games.

**Plan (target ~1 week, extends the existing DMC pipeline, no new search
infra):**
0. **Cheap probe — DONE 2026-07-05, result: promising vs a weak baseline,
   decisively negative vs the real teacher.** PUCT/leaf-eval search using
   `ptcg_dmc_r2.pth`'s existing Q-head as-is (no retraining) —
   `training/nn/mcts_leafeval_agent.py`, `MCTSSearcher(leaf_eval="net")` in
   `mcts.py` (plays forward via the real adversarial opponent only until our
   own next decision, capped at 40 ply, then evaluates max_a Q(s,a) — never
   queries the net on an opponent-turn state, matching its training
   distribution). First result: 38W-2L (95%) vs. raw DMC argmax
   (`training/nn/dmc_agent.py`), 0 errors, 40 games — but the decisive
   follow-up gate vs. v25c (`main.py`) came back **2W-28L (6.7%)**, 0 errors,
   30 games, avg ~270-325s/game. Correct read: search reliably amplifies a
   weak value net's signal (the 95% vs. weak-baseline effect is real and
   large), but the DMC net itself is too weak (2.5% vs v25c on its own) for
   search to close the gap to the real teacher. **Conclusion: Phase 1
   infrastructure is validated and worth keeping, but is a consumer of
   Phase 0's output, not a shortcut past it** — re-gate once a Phase-0-
   improved value net exists. Full writeup + correction:
   `docs/report-log.md` 2026-07-05 "Phase 0 step-0 probe" entry and its
   same-day correction. Compute-budget flag carried forward: ~270-325s/game
   at MCTS_SIMS=100 is within the 10-minute match clock on average but needs
   a real per-move budget check, not just an average, before any ladder use.
1. **n-step bootstrapped value targets** — **full sweep DONE 2026-07-05
   (round 1: 300 games/n=5 only; round 2: 1500 games/n∈{1,5,15}).**
   `training/nn/dmc_nstep.py` (`compute_nstep_targets`, bootstrap =
   `max_a Q(s,a)` from the DMC convention, not the untrained `value_head`)
   wired additively into `dmc_collect.py --n-step N --bootstrap-ckpt ...`
   (default unchanged); `dmc_relabel.py` relabels one shared raw corpus
   offline into multiple n-step arms (avoids re-playing the same expensive
   games once per arm), supports multi-shard corpora via glob.
   **Round-2 result (1500-game corpus, gated against the full 1356-replay
   corpus):**

   | Arm | ALL | LATE |
   |---|---|---|
   | Φ v1 baseline | 0.563 [0.543,0.583] | 0.606 [0.576,0.635] |
   | Φ v2 baseline (improved, see item 2 below) | 0.604 [0.587,0.620] | 0.696 [0.670,0.723] |
   | full-MC baseline (matched retrain) | 0.555 [0.535,0.575] | 0.621 [0.594,0.649] |
   | n=1 | **0.609 [0.583,0.634]** | **0.700 [0.671,0.728]** |
   | n=5 | 0.607 [0.581,0.631] | 0.698 [0.668,0.726] |
   | n=15 | 0.603 [0.578,0.626] | 0.686 [0.658,0.713] |

   **Confirmed, robust: n-step beats full-MC across the whole sweep** (not
   just n=5, as round 1 alone showed) — replicates at 5x scale. n=1/5/15 are
   statistically indistinguishable from each other (n=1 marginally ahead).
   **Sobering: none of the n-step arms clearly beats the IMPROVED Φ v2
   baseline anymore** (heavily overlapping CIs) — round 1's "beats Φ-only"
   gate pass was against the since-superseded, needlessly-weak Φ v1. The
   original Phase 0 gate criterion is not currently satisfied against the
   honest current baseline; n-step training itself remains a real, validated
   improvement over full-MC, just not (yet) proven to beat a good fixed
   heuristic. Full data: `docs/report-log.md` 2026-07-05 "Round-2 n-sweep"
   entry. Caveat unchanged: all n warm-start from a full-MC checkpoint
   (`ptcg_dmc_r2.pth`), so this doesn't rule out that starting from a
   genuinely from-scratch or n-step-native pretrain could do better.
2. **Hand-crafted potential-shaping Φ(s)** — **mechanism DONE 2026-07-05,
   ablation grid not yet run.** Φ itself (`training/nn/phi_baseline.py`,
   already gated against 1356 real replays — ALL sign_acc 0.563, LATE
   0.606, two real bugs found and fixed via isolated-component checks along
   the way) is now wired into the actual training target via
   `dmc_nstep.py`'s `compute_nstep_targets(..., use_phi_shaping=True)` —
   orthogonal to n-step (can be used alone with full-MC, or combined with
   any n_step value), reusing the SAME `phi()` function rather than a second
   implementation. `dmc_collect.py --phi-shaping` (combinable with
   `--n-step`). Formula: `F_k = γΦ(s_{k+1}) - Φ(s_k)`, accumulated over the
   same window as the n-step target — Ng/Harada/Russell 1999
   potential-based shaping, provably policy-preserving and can't saturate
   the way AWR's *learned* value head did, since Φ is never fit to the
   noisy sparse label. Verified: telescoping sum matches a manual recursive
   computation exactly; produces genuinely distinct per-decision targets
   from one flat game outcome (the actual credit-assignment fix); reduces
   exactly to unchanged full-MC when the flag is off; combines correctly
   with n-step; three live end-to-end CLI runs (shaping alone, n-step alone,
   both combined) all completed cleanly. One real bug caught before it could
   contaminate data: `_phi_at` initially hardcoded the acting seat to 0 —
   `extract_decisions` does NOT renormalize seats, so this would have
   silently computed Φ from the wrong player's perspective in half of all
   collected games; fixed to read `yourIndex` from each sample's own obs.
   Full writeup: `docs/report-log.md` 2026-07-05 "Potential-shaping Φ built
   as a training-time target arm" entry. Φ is built from features `main.py`
   already computes internally: `_hand_size` relative to KO threshold
   (Powerful Hand damage proxy), prize differential, `_census()`
   line-progress (Abra→Kadabra→Alakazam), Mist/Rock wall detection.
3. **Diverse self-play opponent mix from batch one — DONE 2026-07-05.**
   `dmc_collect.py` now mixes frozen v25c (`main.py`) + real coded bots
   (`opponents/lucario_agent.py`, `dragapult_agent.py`, `abomasnow_agent.py`,
   `starmie_agent.py`, via new `--bots`/`--bots-frac`, opt-in like the
   existing `--pool`/`--pool-frac`, default off so a bare invocation is
   unchanged) + a rolling checkpoint pool (`dmc_agent_pool.py`, second DMC
   agent slot reading `NET_CKPT_POOL`). Never mirror-only (mirror-only
   self-play is exactly what gave PIMC's rollout zero discriminating
   signal — 90/90 vs mirror-self). Bot games round-robin across the given
   bot list per chunk the same way the checkpoint pool already does; the
   per-chunk mix is `n_bots = round(n·bots_frac)`,
   `n_pool = round(n·pool_frac)`, `n_main = max(0, n - n_pool - n_bots)`.
   Verified: 3 live end-to-end runs (bots alone with correct per-game CSV
   opponent attribution; bots + n-step; bots + Φ-shaping), 0 errors. Full
   writeup: `docs/report-log.md` 2026-07-05 "Diverse opponent pool" entry.
   `archetype_decks.json` decklists are still explicitly NOT used as
   opponents (would need `generic_pilot.py`, already shown to go ~80%
   deck-out in the Stage 0c bake-off — too weak).
4. **Gate: real-replay value calibration, NOT win-rate.** Value-head
   sign-accuracy/calibration against the 1361 downloaded real ladder
   replays' (state, eventual outcome) pairs (1356 usable games after
   filtering to a valid seat + clean ±1 reward), bucketed by game phase,
   must beat **the Φ-only baseline computed 2026-07-05 on this same corpus**
   (`training/nn/phi_baseline.py`, two real bugs found and fixed via
   isolated-component checks — see report-log) by a statistically
   meaningful margin, using game-level bootstrapped CIs (amendment (e)).
   **Φ v1 baseline: ALL sign_acc=0.563 [0.543, 0.583], EARLY (turn≤4)
   0.507 [0.485, 0.530], MID (5-10) 0.581 [0.559, 0.604], LATE (turn≥11)
   0.606 [0.576, 0.635]** — this replaces the flat borrowed 62.5%
   oracle-critic figure (confirmed measured on a different, self-play-
   derived holdout, not real replays — never actually comparable).
   **UPDATED, better baseline — Φ v2, 2026-07-05 (`training/nn/threat.py` +
   `phi_baseline.py --version 2`):** a user design session identified Φ v1
   was not actually zero-sum (only `prize_diff` is antisymmetric;
   `hand_advantage`/`wall_penalty`/`line_progress` are one-sided). Rebuilt
   with `net_threat_diff` (a genuinely antisymmetric term — verified to sum
   to exactly 0 across both seats on the same state — using the real
   card/attack database, `min(1, damage/defender_hp)/(1+turns_to_afford)`,
   maxed over the attacker's known attacks) replacing `hand_advantage`.
   **Φ v2: ALL sign_acc=0.604 [0.587, 0.620], LATE 0.696 [0.670, 0.723]** —
   clean, non-overlapping CIs vs v1 on both metrics. A follow-up "v3" hybrid
   (keep the precise deck-specific `hand_advantage` for our own side, use
   the generic threat estimate only for the opponent, weight-tuned on a
   held-out split) did NOT beat v2, and its own weight-tuning failed a
   real generalization check (best-on-selection-set weight scored WORSE on
   held-out than no tuning at all — a caught overfitting-to-selection-set
   result). **Φ v2 is now the gate baseline any learned value head must
   beat** — this raises the bar (round-1's n=5 result, 0.602/0.690, is now
   roughly at parity with Φ v2 rather than clearly ahead of Φ v1). Full
   data: `docs/report-log.md` 2026-07-05 "Φ redesigned for genuine
   zero-sum consistency" entry. Win-rate at feasible n (100-400 games) has
   repeatedly failed to resolve effects at this project's scale (DMC's own
   climb, AWR's saturated value head,
   oracle-critic's 62.5%) — this gate is designed to not inherit that same
   measurement-resolution failure. A pass here is necessary but not
   sufficient — treat as an OOD-generalization check (self-play-trained net
   evaluated on real ladder trajectories), not a full substitute for
   eventual win-rate confirmation once measurement resolution allows it.
   **Ablation grid run 2026-07-05 — GATE CLEARED for n-step, Φ-shaping-in-
   training is a clean NEGATIVE:** 4 arms (full-MC baseline / n_step=5 /
   Φ-shaping alone / n_step=5+Φ), all trained identically (3 epochs,
   warm-started from `ptcg_dmc_r2.pth`) on ONE shared 300-game diverse
   corpus (bots+main.py), relabeled offline per-arm via new
   `dmc_relabel.py`, gated via new `dmc_replay_gate.py` on the same 1356-game
   corpus/methodology as the Φ-only baseline:

   | Arm | ALL | LATE |
   |---|---|---|
   | Φ-only baseline | 0.563 [0.543,0.583] | 0.606 [0.576,0.635] |
   | full-MC (matched retrain) | 0.582 [0.560,0.604] | 0.647 [0.619,0.676] |
   | **n_step=5** | **0.602 [0.575,0.628]** | **0.690 [0.658,0.719]** |
   | Φ-shaping alone | 0.488 [0.465,0.510] | 0.483 [0.452,0.514] |
   | n_step=5 + Φ-shaping | 0.597 [0.570,0.623] | 0.680 [0.649,0.710] |

   n_step=5 beats the matched full-MC baseline (+0.020 ALL, +0.043 LATE,
   isolating the n-step effect specifically) and clears the gate outright on
   LATE (non-overlapping CIs vs. Φ-only). Φ-shaping folded into the training
   target is a clean negative — worst of all 4 arms, at/below chance — even
   though the same Φ function scores a real 0.563/0.606 as a fixed,
   unlearned baseline; the failure is specific to the shaping-into-
   regression-target mechanism, not Φ itself. Combining Φ-shaping with
   n-step adds no value over n-step alone. **Recommendation: n-step is the
   priority lever; do not pursue Φ-shaping-as-implemented further without a
   redesign** (e.g. as a fixed leaf/warm-start prior instead of folded into
   the loss). Caveats: single run per arm, single n value (5, not the full
   {1,15} sweep), modest 300-game/3-epoch scale — a real, direction-
   consistent signal worth more compute, not yet a fully powered result.
   Full data + discussion: `docs/report-log.md` 2026-07-05 "Phase 0
   ablation grid" entry.
5. **Phase 1 (search + value net composition) — CORRECTED 2026-07-05: real
   ~3x win-rate improvement vs v25c (6.7%→~20-24%), NOT the ~6x "40%"
   briefly believed from an underpowered check.** Real AlphaZero-style
   guided search — `training/nn/mcts_leafeval_agent.py` (limited-sim PUCT,
   leaf evaluation via the value net's `max_a Q(s,a)` directly, no
   rollout-to-terminal, eliminating PIMC's too-weak-rollout-opponent failure
   mode since no rollout opponent is needed). Same protocol each time
   (`mcts_leafeval_agent.py` vs. `main.py`, `MCTS_SIMS=100`, seats
   alternated), swapping only `NET_CKPT` and (once) game count:
   - old weak checkpoint (`ptcg_dmc_r2.pth`, pre-Phase-0): **6.7%** (2W-28L,
     n=30)
   - round-1 n=5 (300-game corpus, 3 epochs): **20.0%** (6W-24L, n=30) — per
     a user-directed ml-engineer sequencing consult (re-gate the validated
     search wrapper with a new value net BEFORE spending compute on a full
     n-sweep, since this number is the one that tells you if the composed
     line can plausibly beat v25c at all)
   - round-2 n=1 (1500-game corpus, 5 epochs), n=30: **40.0%** (12W-18L) —
     **RETRACTED as a real improvement.** A user-requested larger-n follow-up
     (n=100) came back **19.0%** (19W-81L) — essentially identical to
     round-1, not a further jump. Pooled across both runs (130 games, same
     checkpoint/settings): **23.8% [16.5%, 31.2%]**, comfortably consistent
     with round-1's 20.0%. The 40.0% reading was an underpowered-sample
     fluctuation, not a real effect — exactly the failure mode this
     project's own history warns about (see the several prior "CORRECTION"
     report-log entries); this session should have run the larger-n check
     before declaring an improvement, not after.

   **Honest standing result: a real, now twice-independently-replicated ~3x
   win-rate improvement over the pre-Phase-0 baseline (6.7%→~20-24%)** —
   round 2's larger corpus and full n-sweep did not produce a FURTHER
   improvement over round 1, despite appearances. Separately, round-2's n=1
   checkpoint only tied (didn't clearly beat) the improved Φ v2 baseline on
   the standalone replay-calibration gate — consistent with the corrected
   win-rate picture (no further real gain from round 2's larger-scale
   training on THIS metric either).

   **Compute-budget check: SAFE at MCTS_SIMS=100.** 6,897 real per-decision
   timings collected during the n=100 gate (via a new `MCTS_TIMING_LOG` env
   var hook in `mcts_leafeval_agent.py`): mean 3.14s, median 2.90s, p95
   7.34s, max 21.12s (a real, heavier-than-normal tail on individual
   decisions). ~69 of our own decisions/game on average; CLT-estimated
   per-game total think time ≈216s mean, ~292s even at a pessimistic ~5σ
   tail — comfortably under half the 600s match clock. This was the one
   open technical risk flagged since the very first step-0 probe; now
   resolved. (A first attempt at this check via a hand-rolled direct-in-
   process timing script hit a real engine-state issue — running
   `kaggle_environments.make("cabt")` + native `cg.api` search repeatedly
   in one process broke immediately, likely leftover native search-handle
   state from the local shim; abandoned in favor of reusing the
   already-proven `ab_test.py`/`harness.py` multiprocess runner instead.)

   **Decision:** the search+value-net line has a real, replicated ~3x
   win-rate improvement that is compute-safe to run — a meaningful result,
   just more modest than the retracted headline suggested. Whether ~20-24%
   vs. v25c is itself worth shipping, vs. continuing to improve the value
   net or increasing `MCTS_SIMS` now that budget headroom is confirmed, is
   an open decision, not resolved this session. Full data:
   `docs/report-log.md` 2026-07-05 "n5 value net re-gated...", "Round-2 n=1
   checkpoint re-gated...", and "CORRECTION: the 40.0% result was noise..."
   entries. Re-audit rather than reuse the closed `training/nn/mcts.py`
   PIMC rollout code path (confirmed `_STALL_MEMO` global-state corruption
   bug) if extending this further — the `leaf_eval="net"` mode added this
   session is a separate, simpler code path already free of that bug.

**Engine source note:** the competition's real C++ engine source
(`ptcg_engine`) was obtained 2026-07-05 from the Kaggle Data page
(`training/engine_src/`, gitignored — competition-use-only license, do not
redistribute). Confirms `SearchBegin`/`SearchStep`'s in-memory `State` clone
is the exact same native path PIMC already used (doesn't change Phase 0/1
feasibility) and confirms the terminal win-condition logic matches what was
already understood empirically. Useful going forward for verifying ambiguous
rules directly rather than empirical probing; a custom native self-play
harness (bypassing Python/JSON per-step overhead) is a lever available later
IF self-play throughput becomes an actual bottleneck, not before Phase 0 runs.

Belief-model consumers in `main.py` and exploiter-replay mining continue in
parallel, unconditionally, regardless of Phase 0's outcome.

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
   **Collection + first training run — DONE 2026-07-04:** 1000 fresh
   self-play games with `ptcg_dagger_r2.pth` (temp 1.0) → 317,143 samples,
   `training/sp_data_awr*.pkl.gz` (the old `sp_data.pkl.gz` was stale,
   old-deck, pre-freeze). First train attempt (uncapped) got OOM-killed —
   the exact bug `--bc-limit`/`--sp-limit` were built for during DAgger,
   just omitted this run. Retrained with `--bc-limit 250000 --sp-limit
   250000 --awr-beta 1.0`: 10 epochs, loss 0.51→0.41, `awr_norm=1.83` →
   `training/ptcg_awr1.pth`.

   **Gate (400 games each) — REAL NEGATIVE, not a measurement artifact:**
   vs v25c teacher: **15.8% ± 3.6%** (63W-337L) — same flat 12-17% band as
   every BC/DAgger checkpoint, no movement. vs its own seed
   (`ptcg_dagger_r2.pth`, added per advisor's warning that vs-teacher alone
   can't tell "AWR improved nothing" from "seed already near parity"):
   **47.7% ± 4.9%** (191W-209L) — a clean statistical tie. Unlike DAgger's
   excused imitation-ceiling plateau, AWR is supposed to be able to exceed
   teacher parity, so this tied-vs-seed result at a CI tight enough to
   resolve a real effect is a genuine null result, not a resolution
   problem. Full writeup: `docs/report-log.md` 2026-07-04 entry.

   **β=0.5 follow-up (the single pre-registered follow-up) — DONE
   2026-07-04, confirms the null result:** retrained on the SAME
   `sp_data_awr*.pkl.gz` corpus with `--awr-beta 0.5` (more aggressive
   reweighting — `awr_norm=6.95` vs β=1.0's 1.83) → `training/ptcg_awr_beta0.5.pth`.
   Gate: vs v25c teacher **11.5% ± 3.1%** (46W-354L) — *worse* than β=1.0's
   15.8%, not better. vs its own seed (`ptcg_dagger_r2.pth`): **53.7% ± 4.9%**
   (215W-185L) — CI still spans 50%, not a statistically significant win,
   though nominally above β=1.0's 47.7%.

   **Table (both β values, 400-game gates):**

   | β | vs teacher | vs seed (`dagger_r2`) |
   |---|---|---|
   | 1.0 | 15.8% ± 3.6% | 47.7% ± 4.9% (tied) |
   | 0.5 | 11.5% ± 3.1% | 53.7% ± 4.9% (not significant) |

   **Conclusion — Stage 2 direct-self-play AWR line CLOSED, negative result
   (2026-07-04):** more aggressive advantage weighting made things *worse*
   against the teacher while producing no statistically significant gain
   against the seed either — ruling out "just needed a stronger β" as the
   fix. Working theory: the value head saturates bimodally toward ±1 for
   most states (see the pre-collection diagnostic below), so its
   `advantage = value_target - v_pred` signal carries little per-decision
   nuance beyond the terminal outcome; AWR reweighting under these
   conditions concentrates training on a narrow winner-biased slice of the
   self-play corpus rather than learning genuinely better-than-expected
   actions, and the more aggressively it does so (lower β), the more it
   overfits away from the teacher's distribution without a compensating
   quality gain. Per the pre-registered "1-2 follow-ups then stop" rule,
   this closes the direct-self-play-AWR line without a search tree — it
   does not exceed teacher parity on this deck at either β tested. Full
   writeup: `docs/report-log.md` 2026-07-04 entries.

   **What this means for Stage 2 going forward:** advancing past the
   teacher on the 70%-axis via self-play needs either (a) Kaggle-gated
   MCTS/search-in-the-loop (the `SearchState`/`cg.api.search_begin` spike
   already scoped earlier in this file — the "real" Stage 2/5 upgrade this
   whole direct-self-play phase was always a stand-in for), or (b) accepting
   `ptcg_dagger_r2.pth` (81.9% teacher-fidelity, not ship-ready) as the
   ceiling of the imitation-family approach and shifting effort to Stage 3
   (belief model, parallel track, doesn't depend on this result) while
   `main.py` (v25c heuristic) remains the ladder submission. Either way,
   **both β results are real, citable report material** for the ablation
   table (target figure #4) and the "imitation-family methods plateau
   without search" narrative (target figure #3) — this was not a wasted
   night, it's a load-bearing negative result.

   **Pre-training value-head diagnostic** (run before the collection, per
   advisor): 30 self-play games with `ptcg_dagger_r2.pth`, temp 1.0 → 9,536
   decisions. `v_pred` std 0.935 (not collapsed toward ~0 — the value head
   discriminates, though it saturates toward ±1 for most states rather than
   a smooth spread: p25 -0.999, p75 +0.9997). `advantage` mean 0.15, std
   0.97, range clipped to [-2,2] as expected — confirmed AWR weighting
   would be distinguishable from the winner-only ablation before spending
   hours on a full collect+train+gate cycle (it was; both null results
   above are real, not degenerate-into-baseline).
   Exit: 55-60%+ vs the teacher over 400 local games — **not cleared;
   Stage 2 direct-self-play line concluded negative, see above.**
4. **Gauntlet + ladder A/B each meaningful checkpoint** (`training/gauntlet.py`
   with a distinct `--name` per checkpoint). Real ladder is the only honest
   evaluator; gElo is the cheap ranking proxy being calibrated against it.
5. **Every run gets a `docs/report-log.md` entry the same day.**

---

## Value Targets: n-step Monte Carlo with Bootstrapping

**Superseded by "Phase 0" above (2026-07-05) — that section is the active,
scoped plan (concrete n sweep, real-replay calibration gate). This section is
kept as the original pre-DMC design sketch it grew out of.**

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
