Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.
2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.




# PTCG AI Battle Challenge — Orientation

This file is the slim always-loaded layer: current state, working rules, and
pointers. **The full breakdown — repo tree, engine API summary, complete deck
tables, `main.py` architecture, NN details, packaging/shipping — lives in
`docs/project-reference.md`. READ IT before any task that touches the engine,
the deck, `main.py` internals, training, or shipping.** It is the reference
layer; this file only orients.

---

## Quick Orientation

**Competition:** Pokemon TCG AI Battle Challenge (The Pokemon Company × HEROZ × Matsuo Institute × Kaggle).
**Goal:** Top-8 in the Strategy track → $30k + Tokyo finals.
**Scoring:** 70% model approach / 20% deck concept / 10% report — but the report
is the *delivery channel* for the 70%: report-driven development, every experiment
logged same-day in `docs/report-log.md`.
**Key insight:** Rule-based bots cap out at ~0% on the 70% axis. A learned piloting
agent is the only path; the heuristic is the ladder placeholder and DAgger teacher.

**Mission** (canonical: `docs/competition-strategy.md` → "Mission"): deck and learning
problem are co-designed; every major choice is quantitatively defended against named
alternatives; **no report claim without a pre-registered trial** in `docs/report-log.md`.
The Alakazam deck freeze was re-opened 2026-07-03 for the Stage 0c bake-off and
**re-closed the same day on the pre-registered rule** (tier 1: ≥93% vs all
challengers; tier 2: pilot floor flattens everything — see report-log).
**Current agent:** v29d — plain heuristic revert (submission 54481189,
shipped 2026-07-08): `main.py` + `deck.csv`, i.e. v29c's two retreat fixes
WITHOUT the endgame search wrapper. **The search line is CLOSED, negative:**
the first diverse-anchor gauntlet since v25c found the shipped v29b/v29c
search stack loses 16-29pp vs lucario/abomasnow (~90% of ladder opponents
are non-alakazam) despite its +9pp mirror result. Three real bugs were
found behaviorally and fixed (archetype-mismatched rollout policies;
`_is_endgame` firing on setup-phase decisions, undealt prize lists reading
as ≤2; the either-side prize gate handing the search every losing mid-game
vs aggro — per-game disagreement logging found each one) and the anchors
STILL did not recover (final pre-registered gate: lucario 73.0%±6.2%,
abomasnow 75.0%±6.0% vs a ≥88% bar) → the pre-registered revert rule
fired. Sharpest mirror-blindness exhibit the project has for the report.
Full sequence: `docs/report-log.md` 2026-07-08 entries (GAUNTLET FINDS →
SECOND real bug → THIRD gate bug → GATE FAILED), `docs/version-history.md`
v29c/v29d entries. Historical context for the v29 ship below.

**Prior agent (v29b, for context)** — endgame-gated belief-determinized
rollout search (submission 54440211, shipped 2026-07-07,
`SubmissionStatus.COMPLETE`).
Plays the v28+Enriching-fix heuristic verbatim except when either player is
≤2 prizes from winning, where it switches to MCTS rollout search over
belief-sampled determinizations (`training/nn/endgame_agent.py`,
`ENDGAME_SIMS=60`) — the first search-based component ever shipped to this
ladder, after five prior full-game search configs came back negative.
**Offline: confirmed 59.0% ±4.8% (236W-164L, n=400, seats alternated) vs
the plain heuristic itself** — an exact isolation of the search
contribution. First multi-file submission this project has shipped
(main.py + heuristic.py + opponents/* + training/nn/* + training/belief/*).
**The first ship (v29, submission 54439688) errored on its validation
episode** (`NameError: __file__` — Kaggle execs the submitted main.py from
a raw string with no `__file__`, a gap the first clean-room test's
`importlib`-based loader didn't reproduce); fixed by deriving the base
path from an already-imported sibling module's `__file__` instead, and
reshipped as v29b same day after re-validating directly against Kaggle's
actual loader function (`get_last_callable`) and a full `env.run` with real
file paths. **No live ladder read beyond the 600.0 fresh-submission floor
at ship time** — per Design Principle #1 the ladder is still the final
evaluator. Full data: `docs/report-log.md` 2026-07-07 "Endgame-gated
search" and "`__file__` NameError" entries, `docs/version-history.md` v29b
entry.
**Two fresh v29b ladder losses (episodes 84709203, 84710776) mined and
fixed same day:** both were deck-out losses caused by Psyduck getting
stuck active with 0 energy (unable to pay its own retreat cost) after a
voluntary MAIN-phase retreat swapped out a fully-fueled, ready Alakazam for
it. Root cause: `main.py`'s `RETREAT` option scoring never looked at WHICH
bench Pokemon a retreat targeted — every retreat option in a decision tied
on score and the tie broke on array order, not board value (the same bug
class as an earlier `_score_bench_target` fix, just never applied to the
voluntary-retreat action itself). Fixed via a shared `_bench_target_priority`
helper used both by forced-promotion scoring and as a small tiebreak on
retreat options. **A second, related "setup-race" pattern found and fixed
same day** (episodes 84710513, 84712093; 84711188 turned out to be a draw
drought, not a targeting bug): a healthy non-attacker (e.g. a full-HP Fez)
was voluntarily retreated OUT to feed an unevolved, unfueled Abra/Kadabra IN
right as the opponent powered up, and it got one-shot for nothing. Fixed
with a new `_opp_threatening()` proxy (opponent's active already carries
energy) that overrides the retreat-target scoring — retreating Abra/Kadabra
into an armed opponent now scores below just holding position, so the
heuristic sacrifices fodder or stays put instead. Deliberately a shallow
heuristic gate, not an endgame-style rollout search — this project's five
prior full-game search attempts all failed, and ISMCTS's closure pinned
that on weak leaf-value signal in exactly this early-game regime, so a
1-ply threat heuristic was chosen over a new search subsystem. Sanity-
checked (not a confirmatory gate) at 200 games vs. the pre-session
baseline: 55.0%±6.9%, 0 errors — directionally positive, no crashes, but
below the project's usual n=400 confirmatory bar. **Shipped same day as
v29c** (submission `54441561`, repackaged via
`training/nn/package_endgame_submission.py`, re-validated against Kaggle's
actual loader plus 5 full local `env.run` games before shipping —
`SubmissionStatus.COMPLETE`, validation episode `84717810` clean). No
`publicScore` signal yet beyond the standard 600.0 fresh-submission floor.
Full data: `docs/report-log.md` 2026-07-07 "v29b ladder losses" and
"setup-race losses" entries, `training/ladder_history.csv` v29c row.

**Prior agent (v28, for context) — board-thinning fix + Phase C +
confidence recalibration** (submission 54356683, shipped 2026-07-05).
**Honest state at the time (2026-07-05): no confirmed evidence v28 beats
v25c.** Live
`publicScore` reads bounced repeatedly (an earlier "confirmed settled at
829.8" claim was retracted after a subsequent read showed 724.4, a
116-point swing) and never converged after 4+ hours — this ladder's score
may be a genuinely live metric, not one that settles like a fixed offline
benchmark. **Pivoted to a trustworthy measurement instead: a 400-game
offline mirror A/B, v28's exact code vs. pure v25c, came back 50.2%±4.9%
— statistically indistinguishable from parity.** The board-thinning fix
alone had shown a real edge in its own isolated tests (54.3%, 56.0%), but
that signal isn't observable in the combined v28-vs-v25c mirror test,
plausibly because Phase C's payoff needs non-mirror wall/unknown-archetype
opponents a mirror match doesn't surface. **Takeaway for future sessions:
do not trust any single or double `publicScore` read as confirmation of
anything on this ladder — it has been wrong twice this session.** A real
ranking needs either much longer live observation or a properly-powered
offline test against non-mirror opponents (not yet done). v28 remains
shipped (harmless, reasoned basis at ship time) with no confirmed
improvement or regression established.

**Context (why this took
several ships to (partially) untangle):** v26 and v27's `publicScore` reads
initially appeared to decline same-day (818.3→726.2→695.1), prompting a
panic revert — a follow-up pull then showed a plain v25c revert scoring
only 600.0 while v27 simultaneously read 735.9, revealing the "decline"
was **age-confounded** (`publicScore` climbs/settles over hours, not a
stable snapshot), not a real regression. Two same-time-shipped isolation
submissions (v25c-revert `54354862` and v25c+boardthin `54354935`) were
tracked over ~3h and appeared to narrow (isolation candidate settling
~720-750 for several consecutive reads) before the isolation candidate
also later swung again — informing the v28 ship decision at the time,
though per the retraction above that narrowing did not turn out to be
true convergence either. Separately, Phase C's classifier + wall-anticipation were
confirmed correct against 725 real replays (82.4% archetype agreement,
0.9% wall false-positive rate), and a real confidence miscalibration
(only 39.3% of true-unknown decks correctly read as low-confidence) was
found and fixed via a data-driven threshold change (0.8→0.97). Full
detail: `docs/report-log.md` 2026-07-05 entries in order (CORRECTION →
v28 shipped → CORRECTION #2, which retracts the settled claim and lands
on the 50.2%±4.9% offline result as the trustworthy number), and
`docs/version-history.md` v27/v28 entries.
Before that, v25c
(`main.py` + `deck.csv`, submission 54282648, shipped
2026-07-03, user-reported ladder Elo peaked ~900, settled ~880; gauntlet gElo
589, top of the whole table) — see the v25c paragraph below. Before that,
v25b — v23 deck (reverted from v24 on
2026-07-03; v24's ladder reading (680) prompted a user call to revert ahead of
the 48h decision checkpoint) + 5 heuristic logic fixes from a full retreat/
energy/evolution audit (stype==9 deck-out, Kadabra retreat, Wondrous Patch
targeting, Enriching deck-safety gate, manual-evolve-vs-Candy racing gate,
shipped as submission 54277762) + 2 more replay-verified fixes: retreat
scoring no longer force-retreats a fully-fueled Alakazam whenever the
opponent has a Mist/Rock wall regardless of whether we can exploit it, and
Rare Candy racing gained an offensive trigger (races to Alakazam when it sets
up a near-term KO, not just when already in danger). Shipped 2026-07-03,
submission 54279766, **publicScore 861.8** — up from v25's 732.0 and above
v24's 698.1, and gauntlet gElo 551 (top of the whole table, first version to
clear v24 offline too). Also fixed a `training/harness.py` scoring bug that
miscounted opponent-crash games as ties in one seat direction (root cause of
the "Dragapult ties" item below — not a game anomaly). See
`docs/version-history.md` and `docs/report-log.md` 2026-07-03 entries for the
full data and caveats.
**v25c (shipped 2026-07-03, submission 54282648):** replay-verified fixes from
5 fresh ladder losses — a `desperation` mode (opponent ≤1-2 prizes from
winning) that overrides deck-out caution and forces racing to Alakazam
instead of trying to out-survive a lethal hit; a `lone_active_opportunity`
heuristic (opponent's bench is empty + a rough max-hand estimate clears the
KO threshold → stop banking draw and go for the kill); fixed `_score_deck_search`
preferring a dead-weight Alakazam/Kadabra fetch over Abra when no line piece
exists anywhere; fixed Hilda's search (Stage-1/2 + energy only, can't fetch
Basics) beating Dawn (Basic+Stage1+Stage2) by raw weight even with zero Abra
in play; and fixed Boss's Orders' `PHASE_CLOSING` branch firing an
unconditional flat score regardless of target quality, confirmed wasting Boss
plays in 2 separate replays. User-reported ladder Elo peaked ~900, settled
~880 as of 2026-07-03 evening. Gauntlet re-run 2026-07-03 overnight (200
games/anchor, dragapult excluded — crashes locally): gElo 589, top of the
whole table (above v25b's 559, v24's 516, v23's 499). See
`training/ladder_history.csv`, `docs/report-log.md` 2026-07-03 entry, and
`docs/version-history.md` v25c entry for full replay evidence. Open,
unconfirmed lead: `main.py` has no handling for the MULLIGAN select context.
Open, unfixed pattern: "board-thinning" (ending up with 1-2 Pokémon in play
and a bloated dead hand after the attacker line gets repeatedly KO'd).
**Quantified 2026-07-05:** mined 18 games where the v25c exploiter (a weak,
mostly-imitating net) beat frozen `main.py` — **18/18 end the same way**:
`main.py` at deck=0-10 cards (or a fully empty active slot) stuck on a
non-attacker tech Pokémon, opponent sitting on a fully-fueled mirror
Alakazam needing 1 prize. This is now the top-priority next heuristic-fix
target (evidence: `docs/report-log.md` 2026-07-05 "Exploiter-win replay
mining" entry, replays at `replays/exploiter_wins/`).
**Roadmap (canonical: `docs/competition-strategy.md` §Master Plan):**
Stage 0 deck freeze + Gauntlet baseline → Stage 0b heuristic tuning → **Stage 0c
deck bake-off (DONE — freeze re-closed) + method bake-off (DONE)** → Stage 1
DAgger → Stage 2
advantage-weighted self-play → Stage 3 belief model (parallel) → Stage 4
hardening → Stage 5 search-at-inference (Kaggle-gated).
**Engine runs locally:** `pip install kaggle_environments --no-deps` → ~0.5s/game.
Rig + workflows: `training/README.md`.
**NN track:** BC re-collected + retrained on v25c (17% vs teacher); DAgger r1-r2
done — fidelity 73→82% but win-rate flat, **paused** pending Stage 2 AWR.
`ptcg_dagger_r2.pth` is the best checkpoint. `docs/nn-training.md` §Resume Here.

---

## Competition Facts

| Item | Detail |
|------|--------|
| Ladder ends | ~Aug 16–17, 2026 (slug: `pokemon-tcg-ai-battle`) |
| Strategy report due | ~Sep 13, 2026 (slug: `pokemon-tcg-ai-battle-challenge-strategy`) |
| Team merger deadline | ~Aug 9, 2026 |
| Submissions | 5/day; only the latest 2 count for final standing |
| Submission format | `main.py` + `deck.csv` → `submission.tar.gz` (ship via Kaggle CLI — see project-reference §Packaging) |
| Agent contract | `def agent(obs_dict: dict) -> list[int]` returning legal option indices |
| Clock | 10 minutes total per match; timeout = instant loss |
| Rating | Win/loss/tie only; margin irrelevant |

---

## Deck Essentials (full tables: `docs/project-reference.md` §Deck)

**Alakazam single-prize control (v23 deck, 60 cards).** Win condition: **Powerful Hand**
(Alakazam 743, cost 1 Psychic) — `damage = 20 × hand_size`. Hand size IS the damage
stat; never discard unnecessarily. KO threshold: `ceil(opp_active_hp / 20)` cards.
**Blocked by** Mist Energy (#11) and Rock Fighting Energy (#20) on the opponent's
Active. Energy routing: exactly one Psychic on Alakazam, then Kadabra → Abra;
Enriching (13) → Dudunsparce only, never Alakazam.

---

## Where Things Live

| Need | Go to |
|------|-------|
| **Full reference breakdown** | `docs/project-reference.md` |
| Game mechanics/decision-structure rundown (for ML method choice) | `docs/game-nature.md` |
| Roadmap / writeup strategy | `docs/competition-strategy.md` |
| Experiment journal + glossary + target figures | `docs/report-log.md` |
| 120-method ML/RL survey + shortlist w/ alterations | `docs/method-survey.md` |
| Eval-function literature research (Φ v3 design basis) | `docs/eval-function-research.md` |
| Engine API (canonical) | `docs/engine-api.md` |
| Version change log (v1–v24) | `docs/version-history.md` |
| NN training log + pipeline | `docs/nn-training.md` |
| Belief model design + phases (Stage 3) | `docs/belief-model.md` |
| Training rig workflows (A/B, gauntlet, tuning) | `training/README.md` |
| Piloting logic spec / matchups | `docs/piloting-guide.md`, `docs/matchups.md` |

---

## Outstanding Items (Priority Order)

**>>> HARD DEADLINE Aug 14: re-ship v29d (submission `54481189` packaging,
`training/nn/package_endgame_submission.py`-era known-good) unless a gated
better agent is already the 2 newest submissions — only the latest 2 count
at ladder close (~Aug 16). Do not let an experimental read be final.**

**>>> NEXT STEP (as of 2026-07-15): PIVOT — "learn inside the champion"
(GitHub issue #3). The 2026-07-12 interrupted jobs are SKIPPED at user
direction — do NOT relaunch them. Phase A DONE: detector codified
(`training/nn/regime_detector.py::regime_fires` — `turn>=9 AND
(line_in_play==0 OR (deck<=6 AND hand>=15))`, game capture 92.9%, FP 0.54%).
Phase B DONE: ~187k seat-verified in-regime samples. **Phase C ROUND 1
GATED 2026-07-15: gate 1 FAIL (CI-separably negative, hybrid rescues 5/1500
held-out continuations vs plain v29d's 28), gate 2 PASS (no anchor
regression). Root cause found: the round-1 corpus contained ZERO wins —
eps=0.25 per-decision exploration destroys the ~30-decision clean sequences
a rescue needs (verified: eps=0 through the same machinery rescues 10.6%).**
ROUND 2 (the ONE iteration issue #3 allows) pre-registered + chain running
detached as of 2026-07-15 night: `--deviate-once` collection (one random
action at a random in-regime decision, pure v29d otherwise — the
one-step-deviation Q^v29d estimator; smoke: 18.3% wins), contrast
precondition ≥3% enforced before training, identical recipe →
`training/regime_qnet_r2.pth`, unchanged gate → results in
`training/nn/regime_gate_r2.log`. **Gate 1 fails again → issue #3 CLOSES
per its own rule (closure entry + v29d re-ship path).** Kill date Aug 6.
Full spec: issue #3. Separately: v-dmc1b (54740723, COMPLETE) is the live
learned-model read; any future learned-model ship MUST use the
extracted-tarball clean-room protocol + bundled card_tables.json (see
report-log 2026-07-15 no-cg-module entry).
Prior 2026-07-12 next-step block below kept for context — superseded.**

Prior next-step block (superseded 2026-07-13, kept for context): three local
training jobs stopped mid-run at user request (machine needed for other use).
Context: after the DMC round-6 closure further below, the user rejected
"closed, move to report" and directed continued RL/ML pursuit: *"We need to
figure SOMETHING rl/AI/ML out. Coming up empty handed is not going to land
me top 8. I don't care how long it takes, we still have a month for leader
submissions and another month after that for the report."* An `advisor`
consult found every closed method (BC/DAgger/AWR/4 search variants/
AlphaZero/sequence-transformer/DMC) was tested identically — offline, vs
v29d only, on a 50%-v29d curriculum, CPU, ≤4.5M params — nine algorithm
variations, four other axes never varied. User chose three fronts:
**(1) ship a live-ladder read — CORRECTED 2026-07-15: 54624481 ERRORED at
validation and never played an episode** (`ModuleNotFoundError: No module
named 'cg'` — Kaggle's agent sandbox has no cg module; the encode→threat
top-level import chain died; local validation missed it because
training/local_cg is always importable locally — same environment-gap class
as v29's `__file__`). Fixed (threat.py falls back to bundled
`card_tables.json`; tempo_features.py added to the package) and re-shipped
same day as **v-dmc1b, submission `54740723`, `SubmissionStatus.COMPLETE`**
— the learned-model live read is now actually running; clean-room-validated
from the extracted tarball with cg unimportable. See report-log 2026-07-15.
v29d (`54481189`, 685.4 at the 07-15 read) remains on record — **must be
re-shipped before ladder close if this experimental read is still the most
recent submission**, since only the latest 2 count. **(2) Kaggle GPU scale-up — SCOPED, not executed.**
Collection is CPU-bound (the local game engine, no GPU-acceleratable step),
so GPU's honest value is training-step throughput / batched inference, not
raw data volume. The concrete, already-partially-validated use: GPU-batch-
relabel the corpus with n-step=5 targets (the 2026-07-10 retest found an
encouraging 7.0%±2.5% on 1/8th data, deferred only because CPU relabeling
doesn't scale) using the existing reusable pattern in
`scratch_kaggle_collect_notebook/ptcg-p2-round3-collect-retrain-gate.ipynb`
(a real working `%%writefile`-cells + cg-lib-assembly notebook from the
2026-07-07 AlphaZero push — needs a DMC-specific fork, not yet built). See
`docs/report-log.md` 2026-07-12 "Kaggle GPU scale-up" entry. **(3)
hand-crafted tempo features fed as DMC input — BUILT, TESTED, ABLATION
INTERRUPTED before any result.** `training/nn/tempo_features.py` (3 new
antisymmetric rate-of-turn features: prize-race pace, hand-growth pace,
setup pace — genuinely new since every existing feature is a snapshot, none
divide by turn count) wired into `encode.py` as a `full+tempo` feature-set
(env-gated via `ENCODE_FEATURE_SET`, every other `encode.py` consumer
untouched). Validated against 2000 real states, 0 errors. Checked first for
overlap with the closed 2026-07-09 sequence-policy aux-feature experiment
(that was imitation-only; no DMC checkpoint has ever received calculated
features as input) — genuinely untested combination. **A paired ablation
(control vs. control+tempo, identical corpus/seed/epochs) was launched and
running cleanly when the user asked to stop all jobs to free up the
machine — NOT a negative result, just incomplete.** Exact resume commands
(re-run as-is, no changes needed):
```
powershell -File training/nn/tempo_control_detached.ps1
powershell -File training/nn/tempo_arm_detached.ps1
powershell -File training/nn/r6_checkpoint2_moreepochs_detached.ps1   # 8-epoch undertraining diagnostic, also interrupted, also unresolved
```
See `docs/report-log.md` 2026-07-12 entries (Kaggle GPU scoping, tempo
ablation pre-registration + interruption note, strategic reframe) for full
detail and the exact user quotes. Prior (2026-07-11/12) entry, for context,
unchanged below: DMC round-6 local data-scaling study CLOSED, decisively
negative.
Overnight autonomous session (user-authorized `/loop`, Fable-consulted on
critical decisions): pre-registered a scaling study on the confirmed
7.5%-win-rate DMC recipe (fresh-init + more epochs + `model_big.py`, full-MC
targets). Checkpoint-1 (831,643-sample baseline, re-verified): **8.25% ±
2.7%**. Checkpoint-2 (1,373,553 samples, ~1.65x, same recipe): **3.0% ±
1.67%** — CI-separably BELOW checkpoint-1 (95% CIs [5.55,10.95] vs
[1.33,4.67], no overlap). **This is a real negative slope, not a flat
one — more data made the checkpoint measurably worse**, decisively firing
the pre-registered kill rule. **DMC local data-scaling is closed; do NOT
proceed to the Kaggle-GPU-scale follow-up** (its only justification, a
positive trend, is now the opposite of what was found). Separately,
Φ-shaping was retested with a corrected win-rate gate (the original
2026-07-05 closure had used the wrong metric) — still a clean negative
(3.3%), and n-step=5 showed an encouraging-but-not-clean 7.0% on 1/8th
the data, deferred (not adopted) because relabeling doesn't scale
cheaply. **Also resolved: a multi-hour infrastructure fight** — dozens
of background job interruptions were first (wrongly) attributed to
Windows Modern Standby sleep, then correctly isolated to the CLI
harness's own `run_in_background` task lifecycle (confirmed via a
control process launched outside that system surviving untouched for
1hr+ while everything inside it kept dying). **Fix, now reusable infra:
launch long local jobs as detached Windows processes via PowerShell
`Start-Process`** (see `training/nn/keep_awake.ps1` and the
`*_detached.ps1` pattern in `training/nn/`), monitored via `Get-Process`
+ log files instead of the Bash tool's background-task tracking. Full
blow-by-blow, every correction included: `docs/report-log.md` 2026-07-11/12
entries. Round-6 raw data (541,910+ samples across 8 chunks) remains on
disk for any future DMC work. Open question for the user: whether to
pursue a genuinely different DMC recipe (the deferred n-step lever, once
GPU-batched relabeling makes it cheap) or treat the full round 3→6 DMC
arc (2.5%→8.25%, closed) as concluded report material and move to
another line. Prior 2026-07-10 next-step entry, superseded but kept for
context: a large-scale DMC push, run at the user's explicit direction
after they pushed back on "heuristic + report only," reached a real,
well-evidenced conclusion. User's stated goal: a genuinely
learned/RL model competitive at "top 5%" (both the report-judged Strategy
track, which structurally requires a learned artifact, AND ideally the
live ladder). Resumed this project's own standing DMC pre-registration
(originally paused to a 2026-07-19 checkpoint) early, at real scale.
**Found and fixed a real bug**: DMC's imitation-derived warm-start was
actively hurting training (`train_dmc.py --no-init` roughly doubles win
rate on identical data). Then systematically tested every other cheap,
real lever: closing the DouZero-style iterative self-play loop, more
training epochs, and a ~2.9x-bigger Q-network (`model_big.py`). **Full
trajectory: round-3 baseline 2.5% → fresh-init ~5% (3 independent large-
sample reads, tight convergence) → +more epochs 6.5% → +bigger capacity
7.5%.** Every lever gave a small, real, additive gain; none produced a
breakthrough or inflection. Cumulative effect: a genuine, hard-won ~3x
improvement (2.5%→7.5%) — real progress, but not remotely competitive for
a "top 5%" ladder goal. **Recommendation, stated to the user: this is a
natural stopping point for actively pursuing DMC as a ladder-
competitiveness effort** — the only untested lever left is a genuine
order-of-magnitude data/compute jump (Kaggle GPU, millions of samples),
which is a real gamble (no accelerating trend in tonight's data predicts
success at 10-100x scale, though nothing rules it out either) and would
need explicit user buy-in given the size of that commitment. This
session's real bug-find + 4 clean ablations are strong, rigorous report
material for the Strategy track's 70% axis regardless. Full numbers,
every gate, every correction: `docs/report-log.md` 2026-07-09 entries,
"PRE-REGISTRATION: DMC round 4 at real scale" through "TRUE FINAL
SUMMARY — after testing the capacity lever too." New reusable infra:
`training/nn/dmc_collect.py` (round-4 curriculum collector, now
validated), `training/nn/model_big.py`, `--no-init`/`--big`/`--seed`
flags on `train_dmc.py`, `NET_BIG` env var on `dmc_agent.py`. Prior
same-day item, for context, still relevant and unaffected by the above:
the scaled
sequence-policy experiment from `docs/next-session-plan.md` was EXECUTED
to a real, gated conclusion.** Its first "CLOSED, information ceiling
confirmed" verdict was WRONG (a premature 5-epoch checkpoint stopped
mid-descent, 76.6% fidelity) and was RETRACTED same day via an `advisor`
consult, before any Kaggle GPU compute was spent. Resumed training with
fresh-game fidelity as the stopping signal (no held-out val split exists)
and found the REAL plateau at epoch 7: **83.0% fidelity — genuinely above
both reference points (BC-MLP 74.9%, DAgger-r2 81.9%)**, a real if modest
capacity signal once trained to convergence. Because that cleared the
82% threshold, ran the pre-registered win-rate gate too (built
`training/nn/seq_agent.py` for live inference, n=400 vs. main.py, seats
alternated): **12.2% ± 3.2% win rate — BELOW the 35% threshold, at the
LOW end of the historical 12-17% BC/DAgger plateau, not above it.**
**Real conclusion: fidelity and win-rate decouple once fidelity clears
roughly the mid-70s%** — a second independent architecture (after
DAgger's own MLP) shows the same disconnect, meaning the ~12-17%
win-rate ceiling is not primarily a per-decision-accuracy problem.
**Recommendation to the user: do not spend Kaggle GPU quota scaling this
architecture further on the same objective** — a bigger version buys more
fidelity, not more wins. If a GPU push is still wanted, the one lever
with real precedent (per `advisor`) is DAgger-style on-policy correction,
not raw capacity — though even DAgger's own historical fidelity gains
(73%→82%) never moved win-rate either, so temper expectations there too.
Two real OOM bugs were found and fixed along the way regardless (raw obs
dicts held in memory instead of encoded-and-discarded, both in the
training loop and at the raw-shard level; see `training/nn/reshard.py`)
— those fixes stand as reusable infra. Full numbers, both gates:
`docs/report-log.md` 2026-07-09 "PRE-REGISTRATION: scaled sequence-policy
experiment" entry, full thread from "CORRECTION" through the Gate 2
result. **Still unresolved, flagged by `advisor`:** every tested method
(imitation now included) can only approach v29d's own performance, never
exceed it, by construction (imitation) or by empirical closure (search,
AWR, IQL, DMC, AlphaZero-style self-play — all closed negative). Whether
the goal is report-grade documentation of a learned agent or an agent
that actually beats v29d is an open question for the user; no tested
method gets to the latter. Prior same-day state, for context: the
overnight run's last active item was the live deck-search audit —
board-thinning confirmed as the
dominant live failure (10/27 fresh v29d losses end with zero Alakazam-line
pieces in play; case study episode 84955813: 15-card lethal hand + 2 Boss
with Psyduck active). Replays can't audit fetch targeting (ids stripped);
a live instrumented wrapper can. Overnight results so far: Φ v4-MLP is the
new champion STATE eval (0.675/0.724/0.752 — Φ v2 0.610 → v4 0.650 → MLP
0.675, each step paired-CI-verified); the ENTIRE
calculated-values-as-action-ranker family is CLOSED (four override
mechanisms — PUCT+rollouts, PUCT+Φv4, 1-ply outcome-fitted advisor, 1-ply
CEM-tuned advisor — all land 62-75% vs anchors while plain heuristic reads
94-96%: per-decision value override breaks plan coherence at any
achievable eval quality); `training/nn/blunder_scan.py` (eval-guided loss
mining, self-validated on the v29c fix episodes) is the eval's proven
non-override consumer. Full data: `docs/report-log.md` 2026-07-09 entries.
Earlier same-day state, for context: the Φ v4 line (literature-driven
Φ v4 evaluation function, `docs/eval-function-research.md`): **Gate 1
PASSED** — `training/nn/eval_v4.py`'s 11-feature antisymmetric fitted eval
is the project's new best state-value signal (0.650 ALL / 0.700 MID
holdout sign-acc on 642 games; +4.0pp ALL / +6.2pp MID over Φ v2, paired
bootstrap CI excluding 0 — weights `training/eval_v4_weights.npy`).
**Gate 2 KILLED at the pre-registered kill-check** — Φ v4 as a
depth-limited leaf eval in the PIMC search (`mcts.py leaf_eval="phi4"`,
`phi4_agent.py`) scored 74.0%/62.0% vs lucario/abomasnow (plain heuristic
same-day: 94.0%/96.0%), i.e. a verified-better eval transfers ZERO
improvement through the search wrapper. **This falsifies weak-leaf-signal
as a sufficient explanation for the search failures — the wrapper itself
is the bottleneck; do not put more compute into this search family
without a structural redesign.** Φ v4 stands for other consumers
(value-net training target, gate baseline, possible direct heuristic
scoring). Full data: `docs/report-log.md` 2026-07-09 entries. Prior state,
for context: both 2026-07-08 lines closed:
the endgame search was reverted off the ladder (v29d — see Current agent
above) and the winner-BC replay-imitation family closed negative (RP-1
77%/5%, RP-2 89%/8% vs the 25% bar — see report-log 2026-07-08). The
heuristic (v29d) is the ladder agent; watch its publicScore vs v29c's
774-784 reads over the next day. Remaining candidate directions for the
user: more ladder-loss replay mining (the one reliably positive workflow),
board-thinning follow-ups, MULLIGAN select-context handling, or report
assembly work — every learned/search arm tried so far is a documented
negative. Prior next-step history, for context:
the AlphaZero-style push's Phase 2 round
3 (redesigned real-meta opponent pool, replacing the collapsed
same-checkpoint mirror) ran to completion and is CLOSED, negative — retrain
does NOT clear the pre-training baseline (0.566 vs 0.584 ALL sign-acc,
CIs heavily overlapping). Two real `mcts_collect.py` bugs were found and
fixed along the way: no incremental checkpointing (a Kaggle run stalled 9-11
hours and lost everything; fixed to checkpoint after every opponent batch)
and a fatal crash on a None-outcome game (now skipped). This closes the
"more self-play data alone" line — consistent with every other arm this
project has tried (DAgger, AWR, PIMC search, oracle-critic, IQL). Full
data: `docs/report-log.md` 2026-07-07 "Phase 2 round 3 (redesigned
opponent pool)" entry; `docs/nn-training.md` "AlphaZero-Style Push" §Resume
Here. Separately and unrelated to this line: the endgame-gated search
result (below, v29) is this session's real positive — a decision for the
user is whether to pursue further search-based lines instead of more
self-play data collection. Prior next-step history, for context: Phase 1
(encoding) RESOLVED+adopted; Phase 2 (`mcts_collect.py`
self-play-with-search infrastructure) DONE, validated, and parallelized;
Phase 1 briefly looked negative (richer 25-feature encoding scored
worse than plain 13-feature) but an isolated-component ablation traced
this to a training-harness confound (fixed; now a clean tie with the
baseline, adopted). Phase 2 built, validated, and parallelized same day
(`MCTSSearcher.choose_with_stats()`, `mcts_leafeval_agent.py`'s
`MCTS_COLLECT_LOG`/`MCTS_GAME_ID` hooks, `harness.py`'s `extra_envs` for
`workers>1`) — validated at 300 games/15 workers, 0 relabel errors, 21,143
samples. **The first real (not plumbing) training pass surfaced a THIRD
instance of a recurring bug class** (a function silently assuming seat=0
instead of reading the real seat — same pattern already caught twice
elsewhere this session): `selfplay_collect.py`'s `shaped_reward` call had
`me_idx` hardcoded to 0, computing the reward from the wrong side's
perspective for half of all collected games. Diagnosed via a below-chance,
backwards-phase-ordering result that was suspicious enough to check rather
than accept (0.428 ALL/0.375 LATE as-is; negating gave a genuinely
promising 0.630/0.639 — better than Φ v2's 0.604/0.696 on ALL — and the
pre-training checkpoint already had a real 0.606 baseline, confirming
training added real value, just sign-inverted). Fixed the bug and
re-collected a corrected 300-game corpus
(`training/mcts_p2_r3.pkl.gz`, 22,167 samples, 0 relabel errors,
match_rate=0.831). **Explicitly PAUSED here at the user's request before
spending the retrain compute** — a training attempt with the fix was
started and deliberately stopped mid-startup (before any epoch ran or
checkpoint saved, confirmed nothing lost). **Next step, not yet run:**
retrain via `train_sp.py` on the fixed corpus and gate via
`dmc_replay_gate.py --value-source head`, comparing against the
pre-training/buggy/Φ-v2/DMC reference points above. Full data:
`docs/report-log.md` 2026-07-06 "Phase 2 built and validated," "Fable
consult on Phase 2 sequencing," "Parallel game_id collection mechanism,"
and "Real training gate reveals a THIRD instance..." entries;
`docs/nn-training.md` "AlphaZero-Style Push" §Resume Here. Prior Phase 0
next-step summary (superseded but still relevant background), for context:
the composed search+value-net system has a real, twice-replicated ~3x
win-rate improvement vs the real teacher (6.7%→~20-24%) — an earlier "40%,
roughly doubling again" reading was an underpowered-sample fluctuation,
RETRACTED. Full Phase 0 pipeline built
this session (n-step targets, Φ-shaping, diverse opponent pool, plus a
user-driven Φ redesign for genuine zero-sum consistency — see
`threat.py`/`phi_baseline.py --version 2`). Same protocol each time
(`mcts_leafeval_agent.py` vs `main.py`, `MCTS_SIMS=100`): old weak
checkpoint 6.7% (n=30) → round-1 n=5 (300-game corpus) 20.0% (n=30) →
round-2 n=1 (1500-game corpus) 40.0% (n=30, 12W-18L). **A user-requested
n=100 follow-up on that same round-2 checkpoint came back 19.0% (19W-81L)
— essentially identical to round 1, not a further jump.** Pooled across
both round-2 runs (130 games): 23.8% [16.5%, 31.2%], comfortably consistent
with round-1's 20.0%. Round 2's larger corpus/full n-sweep did NOT produce
a further win-rate improvement over round 1, despite appearances — exactly
the failure mode this project's own history warns about (several prior
report-log "CORRECTION" entries); the larger-n check should have run
before declaring an improvement, not after. Consistent with this, round-2's
n=1 checkpoint also only TIED (didn't clearly beat) the improved Φ v2
baseline on the standalone replay-calibration gate. Separately, the
n-step-vs-Φ ablation itself was fully swept this session (n∈{1,5,15} all
clearly beat full-MC training, replicating at 5x scale; Φ-shaping-folded-
into-training remains a closed, clean negative — see `dmc_nstep.py`'s
docstring for the confirmed root cause: it trains toward `outcome − Φ(s)`,
not `outcome`, which is not comparable in sign across different states).
**Also resolved: a compute-budget check (the other open item) confirms
`MCTS_SIMS=100` is SAFE against the 10-minute match clock** — ~69
decisions/game for our own side, CLT-estimated per-game total think time
≈216s mean, ~292s even at a pessimistic tail estimate, well under the 600s
budget (measured via a new `MCTS_TIMING_LOG` hook in
`mcts_leafeval_agent.py` during the n=100 run). Full data:
`docs/report-log.md` 2026-07-05 "Round-2 n-sweep," "Φ redesigned for
genuine zero-sum consistency," "Round-2 n=1 checkpoint re-gated," and
"CORRECTION: the 40.0% result was noise" entries. **Honest current state:**
a real, replicated ~3x win-rate improvement over the pre-Phase-0 baseline
that is compute-safe to run — worthwhile, but more modest than briefly
believed. Whether ~20-24% vs. v25c is worth shipping as-is, vs. continuing
to improve the value net or increasing `MCTS_SIMS` now that budget headroom
is confirmed, is an open decision for the user. Belief-model consumers in
`main.py` and exploiter-replay mining continue in parallel,
unconditionally. The competition's real C++ engine source (`ptcg_engine`)
was also obtained this session (`training/engine_src/`, gitignored,
competition-use-only license). Prior next-step history below, for context:

**(2026-07-04) Stage 5 search-at-inference CLOSED, negative; Stage 3 Phase B
DONE; pivoted to Phase C.** `training/nn/mcts.py`
(heuristic-guided PIMC, 100% local dev via `training/setup_local_search.py`)
went through five gates and three real bug fixes (weak-teacher rollout,
strategy fusion, `_STALL_MEMO` global-state corruption in two different
modules) plus one architectural fix (adversarial `lucario_agent` rollout
opponent, per a user-directed Claude Fable consult) — pre-registered gate
(≥55% to continue) came back **0W-50L (0%)**, decisively closing the line
as a third load-bearing negative result alongside DAgger and AWR. See
`docs/nn-training.md` and `docs/report-log.md` 2026-07-04 "Stage 5
search-at-inference" entries for the full sequence. Per Fable's
sequencing, pivoted to Stage 3 Phase B same day: signature extension
(unknown 25.9%→21.3%, honest 78.7% recognition ceiling confirmed via
re-clustering — genuine long tail, not more signatures away) and
archetype library (`training/archetype_decks.json`, reconstructed from
real replay evidence) both DONE — see item 5 below and
`docs/belief-model.md` §Phase B. **Phase C consumer 1 DONE + SHIPPED
2026-07-04 (v26, submission 54346817, ladder pending):** belief posterior
wired into `main.py` — `opp_likely_ace_spec` now belief-driven, Mist/Rock
wall anticipation keyed on crustle/unknown reads per a new 679-replay tech
survey (walls: crustle 35.8%, unknown tail 29.0%, the 5 classifier
archetypes 0-3%). Gate passed: 400-game A/B 50.7%±4.9%, 0 errors. Next:
ladder confirm for v26. **Oracle-critic value head (PerfectDou-style)
CLOSED, negative, 2026-07-04:** privileged opponent-hand critic built,
retrained (161k samples after an OOM fix — see report-log), pre-registered
gate `value_holdout_eval` came back **62.5% sign-acc, below the 65%
threshold** (mid-game 56.5%, barely above coinflip — the segment privilege
should help most). Root-cause ON/OFF diagnostic confirmed the critic does
learn a real but modest signal from oracle info (0.870 vs 0.8425 sign-acc)
that doesn't survive dropout to the oracle-free deployment path well
enough to clear the bar. **Fourth load-bearing negative alongside DAgger,
AWR, PIMC search** — not integrated into `mcts.py`. Full data:
`docs/report-log.md` 2026-07-04 oracle-critic entry. **Now running:** v25c
exploiter collection (`training/nn/exploiter_collect.py --games 1000
--temp 1.0`, launched 2026-07-05, background) — orthogonal track, trains
directly against a frozen v25c teacher rather than extracting more signal
from the existing self-play distribution. Determinization sampler (Phase C
item 2) BUILT + validated 2026-07-07 (`training/belief/determinize.py`,
442-decision validation). Its first consumer, belief-weighted ISMCTS,
CLOSED same day (kill rule fired: 0W-50L, then 0W-20L after fixing a named
rollout-policy confound — leaf-value signal, not determinization quality,
is the binding constraint on this search family; see `docs/report-log.md`
2026-07-07 ISMCTS entry). Sampler remains available for future consumers.

Stage numbers refer to `docs/competition-strategy.md` §Master Plan.

0a. ~~Stage 0c deck bake-off~~ DONE 2026-07-03 (same day as pre-registration):
    tier 1 (as-piloted) Alakazam ≥93% vs all 4 challengers, BT-Elo 1010 vs 577
    next-best; tier 2 (fixed generic pilot) all pairings ~50/50 with 80%
    DECK_OUT — deck value is pilot-dependent. **Freeze re-closed on Alakazam
    per the pre-registered rule.** See `docs/report-log.md` 2026-07-03 entries.
    Rig gains: `training/bakeoff.py` (orthogonal agent×deck round-robins),
    `training/generic_pilot.py`, dragapult anchor fixed (stub-class root cause),
    per-seat+seed logging in gauntlet/ab_test, `tools/meta_survey.py` (meta
    share from replays — Archaludon ~18%, not in our anchor pool; candidate
    future anchor).
0b. ~~Method bake-off~~ DONE 2026-07-03: all 5 rows through one protocol
    (8 anchors incl. fixed dragapult, 200 games/anchor, seed method-run1).
    gElo ladder: random 5 → generic-greedy 57 → bc-v2 239 → dagger-r2 246 →
    heuristic-v25c 568. DAgger's +7 over BC is not CI-separable → per the
    pre-registered rule it is kept as Stage 2 initialization, not as a pilot
    (third independent confirmation of the fidelity-plateau story). v25c
    replicated 568 vs 575 across independent runs (figure-#6 variance
    datapoint). Table B + verdicts: `docs/report-log.md` 2026-07-03 entry.

1. ~~Ship the v23-deck revert + v25 logic fixes~~ DONE 2026-07-03: submission
   54277762 (v23-deck revert + 5 heuristic fixes — stype==9 deck-out, Kadabra
   retreat, Wondrous Patch targeting, Enriching deck-safety gate, manual-evolve-
   vs-Candy racing gate); ~~ship v25b~~ DONE 2026-07-03: submission 54279766
   (Mist-wall retreat fix + Candy-racing offensive trigger, publicScore 861.8,
   gElo 551); ~~ship v25c~~ DONE 2026-07-03: submission 54282648 (desperation
   mode + lone_active_opportunity + 3 replay-verified fixes, user-reported
   ladder ~880 settled, gElo 589 — top of the table). v24 deck swap had been
   reverted 2026-07-03 on early ladder trend, ahead of the documented 48h/50-point
   decision rule — see `docs/report-log.md` 2026-07-03 entries (the local A/B
   favoring v24 stands unexplained; revisit the deck-simplification question
   with more ladder data before retrying it). ~~Stage 0b weight tune~~ DONE
   2026-07-02: gate not cleared (52.0% ± 4.0% over 600 games) → default `W`
   kept; see `docs/report-log.md`.
2. ~~Stage 0: full Gauntlet baseline~~ DONE 2026-07-03 (overnight run): v25c
   gElo 589 vs the 7 working anchors (dragapult excluded, see item 6), top of
   the table. See `training/ladder_history.csv` and `docs/report-log.md`
   2026-07-03 entry.
3. **Stage 1: BC re-collect + retrain on the frozen deck, then DAgger.**
   ~~BC re-collect~~ DONE 2026-07-03: `training/bc_data_v25c*.pkl`, 2000 v25c
   self-play games, 579,169 samples. ~~Build `dagger_collect.py`~~ DONE
   2026-07-03: `training/nn/dagger_collect.py`, smoke-tested end-to-end (0
   relabel errors). ~~Retrain on frozen deck~~ DONE 2026-07-03:
   `training/ptcg_bc_v2.pth`, 10 epochs (~106 min CPU, confirmed runs locally,
   no Kaggle GPU needed), val_top1_acc 0.875 peak. Gated: 86% vs random
   (matches v1), 17% vs v25c heuristic (expected BC-plateau signature — v25c
   is a stronger teacher than v22 was). ~~DAgger round 1 collection~~ DONE:
   1000 games, `training/dagger_data_r1*.pkl`, 326,240 samples, 0 relabel
   errors. ~~Retrain + gate round 1~~ DONE: both a 3-epoch and a heavier
   10-epoch retrain (`ptcg_dagger_r1.pth`/`ptcg_dagger_r1b.pth`) gated flat
   at 12%/15% vs teacher (statistically tied with BC's 17% at n=100). **The
   decisive check (per advisor): 100-game win-rate can't resolve DAgger's
   actual target.** Measured teacher-agreement on 3000 FRESH deployment-
   realistic (argmax) states instead: BC 74.9% → DAgger round 1 79.7% — a
   real +4.8pp gain. **DAgger is confirmed working**; the flat win-rate is a
   measurement-resolution artifact (single-decision gains compound over
   ~150 decisions/game but need more rounds or larger n to show up in
   head-to-head win-rate), not a broken pipeline. Also fixed the REAL root
   cause of two silent OOM kills (not just papered over with `--bc-limit`):
   `dataset.py::load_shards` always read every glob-matched shard fully
   before slicing, so even a small final cap needed the whole ~37GB corpus
   in RAM transiently — fixed to cap during read (shuffles shard order, not
   post-load samples). See `docs/report-log.md` 2026-07-03 entries for the
   full story. ~~DAgger round 2 (user-directed, testing lower collection
   temperature)~~ DONE: collected 1000 games at temp 0.2 (vs round 1's 1.0),
   retrained on BC+round-1+round-2 combined → `ptcg_dagger_r2.pth`, gated 81%
   vs random, **16% vs teacher — still flat**. Fresh-state fidelity re-check:
   BC 73.1% → r1b 81.1% (+8pp, confirmed) → r2 **81.9%** (only +0.8pp) —
   **diminishing returns; the temperature lever helped a little but fidelity
   is flattening near 80-82%, not accelerating.** Two rounds of evidence now
   agree further DAgger rounds are unlikely to move the needle much more.
   **DAgger paused here** — `ptcg_dagger_r2.pth` is the best checkpoint
   (strongest report evidence yet: fidelity gains don't linearly convert to
   win-rate on this deck) but not ship-ready. Gate 50%+ vs teacher → ship
   (per advisor: imitation asymptotes to parity, never above — exceeding it
   needs Stage 2 AWR/search, not more imitation rounds).
4. ~~Stage 2: advantage-weighted self-play (direct, no search)~~ **CLOSED,
   NEGATIVE, 2026-07-04.** Two β values gated over 400 games each: β=1.0 →
   15.8% vs teacher / 47.7% vs seed (tied); β=0.5 (more aggressive) → 11.5%
   vs teacher (worse) / 53.7% vs seed (not significant). More aggressive
   weighting made it worse, ruling out a simple β fix — per the
   pre-registered "1-2 follow-ups then stop" rule this line is concluded.
   Neither exceeds teacher parity; working theory is the value head
   saturates near ±1 on most states, so AWR's advantage signal carries
   little per-decision nuance beyond terminal outcome here. **Exceeding the
   teacher needs Kaggle-gated MCTS/search-in-the-loop, not more direct
   self-play.** Both results are real report material (ablation table +
   plateau-without-search narrative). See `docs/nn-training.md` §Resume Here
   item 3 and `docs/report-log.md` 2026-07-04 entries for full numbers.
   **Next-step options for the user:** (a) scope the Kaggle
   `search_begin`/MCTS spike, (b) start Stage 3 belief model in parallel
   (doesn't depend on this result), or (c) keep v25c heuristic as the ladder
   submission and treat tonight's result as report material for now.
   ~~(a)~~ **DONE + CLOSED same day, per user /goal directive (2026-07-04):**
   two cheap probes (branching + timing) both green-lit inference-time
   search — see `docs/engine-api.md` "MCTS branching + timing probe". Built
   `training/nn/mcts.py`/`mcts_agent.py` (heuristic-guided PIMC) and
   `training/setup_local_search.py` (local `cg.api` search, zero Kaggle
   round-trips for dev). Five gates, three real bugs fixed (weak-teacher
   rollout, strategy fusion, `_STALL_MEMO` global-state corruption in two
   modules), one architectural fix tried (adversarial `lucario_agent`
   rollout opponent, per a user-directed Claude Fable consult) —
   pre-registered gate (≥55% to continue) came back **0W-50L (0%)**.
   **CLOSED, negative** — third load-bearing negative result alongside
   DAgger and AWR: all three converge on "a value/policy signal built
   without genuinely external information can't exceed the teacher it's
   built from." See `docs/nn-training.md` and `docs/report-log.md`
   2026-07-04 "Stage 5 search-at-inference" entries (multiple) for the full
   sequence. Per Fable's sequencing, pivoted to Stage 3 Phase B next
   regardless of outcome (see item 5).
5. **Stage 3 (parallel): belief model** — archetype classifier + accuracy-by-turn
   figure. **Phase A DONE 2026-07-04:** 92.3% held-out accuracy, 99.1% by
   turn 1 (clears the >90%-by-turn-3 target two turns early), beats the
   data-derived key-card baseline at both gate turns (99.1% vs 80.2% at
   turn 1; 100% vs 85.7% at turn 2). 93,006 rows across 5 labels
   (lucario/dragapult/abomasnow/starmie/alakazam-mirror), weights exported
   to `training/belief/belief_weights.json`. **Caveat:** this is the "easy"
   5-fixed-bot classification, not the real ladder — Phase B (real
   archetype library + `unknown` mass) is the honest test. **Phase B replay
   blocker resolved 2026-07-04:** `tools/download_replays.py` bulk-downloads
   via previously-undiscovered Kaggle CLI subcommands; 680 ladder replays
   now on disk (up from 28, stopped short of the full 1,282-episode set on
   API rate-limiting + user call that top-percentile coverage suffices).
   Real meta share: lucario 16.3%, starmie 12.1%, dragapult 10.9%, alakazam
   10.4%, abomasnow 7.8%, crustle 7.8%, archaludon 4.0% (down from an
   apparent 17.9% on the old 28-replay sample), plus 25.9% `other/unknown`.
   **Signature extension + archetype library DONE 2026-07-04:** clustered
   the unknown slice's non-generic revealed cards, added 2 pre-evolution
   aliases (`snover`→abomasnow, `dreepy`/`drakloak`→dragapult) + 1 new
   archetype (`kyogre`, 13 replays) to `tools/meta_survey.py`'s
   `SIGNATURES` — unknown dropped 25.9%→21.3%, confirmed via re-clustering
   to be a genuine long tail (144 replays, each a unique/scattered minor
   combo) not a few signatures away from full coverage — **78.7% honest
   recognition ceiling**, reported as-is. `tools/build_archetype_decks.py`
   reconstructs representative 60-card lists from real replay card-reveal
   frequency; `training/archetype_decks.json` covers crustle/archaludon/
   bellibolt/kyogre/raging-bolt/rockets-mewtwo/gardevoir/grimmsnarl
   (lucario/dragapult/abomasnow/starmie already have exact decklists via
   `opponents/*_agent.py`, skipped). `opp_likely_ace_spec` still hardcoded
   `True` — not wired yet (Phase C, next up, not started this session). See
   `docs/belief-model.md` and `docs/report-log.md` 2026-07-04 entries.
6. ~~Dragapult step-limit ties~~ DIAGNOSED 2026-07-03, **partially corrected
   2026-07-04:** the "`opponents/dragapult_agent.py` crashes 100% of local
   games" claim below was WRONG — confirmed 0/20 errors in a direct
   spot-check and 22,137 clean rows collected from it during Stage 3 data
   collection; it has a working try/except fallback around its Kaggle-only
   `cg.api` import. The `training/harness.py` `summarize()` opponent-crash-
   as-tie miscount is still a real, separately-fixed bug (2026-07-03), it
   just wasn't the reason for prior dragapult anomalies. **The `dragapult`
   column in `training/gauntlet_results.csv` may be trustworthy again** —
   not re-verified via a full gauntlet re-run; user should decide whether to
   re-run the gauntlet with dragapult included.

---

## Design Principles

1. **Real-ladder A/B is the only honest evaluator.** Offline win rates systematically overrate (v5: 64% offline, 0-5 live). Gauntlet gElo ranks candidates; the ladder decides.
2. **Timeout = instant loss.** Every inference path needs a guaranteed fast fallback (`_safe_return`).
3. **Single-prize + non-ex beats the meta.** Crustle immunity walls ex attackers; Alakazam hits it normally. Prize trade: give 1, take 2-3.
4. **Hand size IS damage.** No Professor's Research, no Iono, no Ultra Ball.
5. **Rule-based caps at ~0% on the 70% axis.** The NN track is the goal; the heuristic (v23) is placeholder + teacher.
6. **5 submissions/day, latest 2 count.** The ladder is an A/B rig; ship via the Kaggle CLI without asking.

---

## Documentation Standards

**MANDATORY — after ANY change (code, deck, plans, results), update the docs in
the same session:** the canonical `docs/*.md` home for that fact, the matching
section of `docs/project-reference.md`, and the summary lines here (Quick
Orientation / Outstanding Items) if the current state changed. A change without
its doc update is an unfinished change — this is how the v24 deck swap briefly
went missing from this file.

Every doc file (`CLAUDE.md`, every `docs/*.md`) follows the same shape:

```
# Title

*One- to three-line italic description of what this file is for.*

**Last updated:** YYYY-MM-DD

---

## Section
content
---
## Next Section
content
```

Rules:
1. **`---` separates every top-level (`##`) section**; never stack two `---`.
2. **`**Last updated:**` is mandatory** — bump on material changes, real date.
3. **One canonical home per fact** — link, don't copy-paste. `CLAUDE.md` stays a
   slim orientation layer; durable reference lives in `docs/project-reference.md`;
   narrative history, API dumps, and matchup detail live in their dedicated docs.
4. **Version/change-log entries newest-first**, each under its own `##` heading.
5. **Tables over prose** for anything enumerable.
6. **When a file's status changes**, update the home doc AND the pointer/summary
   here in the same edit — don't let them drift.
7. **Every experiment → same-day `docs/report-log.md` entry** (hypothesis, plain-
   English method, numbers, decision, report relevance); **every ladder ship →
   `training/ladder_history.csv` row.** The report is assembled from these —
   nothing is reconstructed in September.

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
- Author a backlog-ready spec/issue → invoke /spec

## Health Stack

- lint: ruff check .
- test: python training/regression/regression_states.py
- test2: python training/belief/test_determinize.py --games 2
- typecheck: none (mypy not installed)
- deadcode: none
