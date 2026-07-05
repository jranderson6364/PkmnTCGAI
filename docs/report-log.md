# Report Log — Experiment Journal & Method Glossary

*Every experiment gets a dated entry here THE DAY IT RUNS: hypothesis, method in
plain English, result with numbers, decision, report relevance. In September the
final report is assembled from this file — nothing gets retrofitted. Newest first.*

**Last updated:** 2026-07-05 (CORRECTION: the "v26/v27 ladder regression" conclusion was age-confounded, not real — see correction entry below; board-thinning fix gated positively both ways it's been tested; DMC rounds 1-3 show a real but far-too-slow climb (1.0%→1.7%→2.5%), paused to the 2026-07-19 checkpoint; Phase C real-replay behavioral check DONE — belief model is accurate and wall-anticipation is well-calibrated, but a real miscalibration found: only 39% of true-unknown decks correctly read as low-confidence)

---

## 2026-07-05 — Phase C real-replay behavioral check (read-only, no code changes)

**Why:** the re-spined report thesis rests on "the belief model measurably
improves piloting" — and per the corrected entry above, the noisy
`publicScore` reads can't currently confirm or deny that. Per advisor
guidance, checked the belief model's actual *behavior* against real ladder
replays instead of chasing more score deltas — signal independent of the
unstable score.

**Method:** downloaded all available ladder replay episodes via
`tools/download_replays.py` (725 replays with our team present, on disk
`replays/bulk/`). For each, computed a ground-truth archetype via
`tools/meta_survey.py`'s existing signature classifier (revealed cards →
name match), and separately ran `training/baselines/v27.py`'s
`_belief_posterior` (the exact shipped Phase C code) over every step of the
game, taking the last (most-informed) posterior. Purely a replay/behavior
read — no `main.py` changes, no ships.

**Results:**
- **Classifier-archetype agreement: 82.4% (360/437)** — real ladder replays
  where the true deck is one of the 5 classifier archetypes (lucario/
  dragapult/abomasnow/starmie/alakazam), the posterior's top class matches
  82.4% of the time. In the right ballpark of Phase A's 92.3% (that number
  was on the "easy" 5-fixed-bot setup, not real ladder decks, so some drop
  is expected) and comfortably above the honest 78.7% recognition ceiling
  reported for the archetype-library work — **the classifier itself is
  working reasonably well on real ladder data.**
- **Wall-anticipation false-positive rate on the 5 classifier archetypes:
  0.9% (4/437)** — very close to the tech survey's expected 0-3% true wall
  rate for those archetypes. Anticipation is NOT misfiring broadly against
  decks that don't run walls — this looks correctly calibrated.
- **Crustle wall-detection true-positive: 85.7% (48/56)** — correctly
  flags wall threat on crustle games most of the time.
- **Real finding — a genuine miscalibration: only 39.3% (64/163) of
  true `other/unknown` games show the posterior correctly below the 0.8
  confidence threshold.** The other ~61% of real ladder decks the
  classifier has no training signature for still produce a confident
  (≥0.8) — and therefore wrong — top-class guess. This is the deck-
  recognition long tail (the honest 78.7%/21.3% split documented in Phase
  B) behaving worse than assumed when the model is actually put in front
  of it: `opp_likely_ace`'s `b_conf<0.8` fallback path, meant to catch
  exactly this case and default to the conservative pre-Phase-C behavior,
  triggers far less often than the ~21-29% unknown-deck share would
  predict, because the softmax is often confidently wrong rather than
  appropriately uncertain on OOD (out-of-distribution) inputs — a known
  general failure mode of softmax classifiers, not something specific to
  this implementation, but not previously measured here.

**Report/decision relevance:** this is real, replay-verified evidence for
the hybrid-report thesis (the classifier + wall-anticipation genuinely
work on ladder-realistic data, independent of the noisy score), AND a
concrete, scoped follow-up target: tighten or recalibrate the confidence
threshold (or add an explicit OOD/novelty signal) so `other/unknown` decks
correctly fall back more often. Not fixed this session (per advisor's
explicit "don't touch main.py — isolation experiments in flight" —
this is investigation only); candidate next Phase C work once the current
isolation ships settle. Raw per-replay data: `training/phasec_replay_check.csv`.

**Follow-up (same session, still read-only/no ship): a data-driven
candidate fix for the miscalibration.** Swept the confidence threshold
against the same 725-replay data (`belief_conf` distribution by ground
truth): known-archetype confidence is extremely peaked near 1.0
(p25=0.992, median≈1.0), while unknown-deck confidence spreads out but
still often lands high. Raising the threshold from 0.8 to **0.97** nearly
doubles the unknown-catch-rate (39.3%→60.1%) while only modestly
increasing the false-low rate on known archetypes (19.2%→21.1% — a safe
direction, since "low confidence" just triggers the pre-Phase-C
conservative default, not a wrong action). About 10.3% of all 725 real
games have their fallback behavior flip under this change — meaningful,
not overwhelming. Drafted as `training/baselines/v27_recalibrated.py`
(one two-line diff: both `b_conf<0.8` occurrences → `b_conf<0.97`) —
**NOT applied to `main.py` and NOT shipped**, per the standing instruction
to leave the current isolation experiments (v25c-revert, v25c+boardthin)
uncontaminated until their ladder reads settle. Ready to A/B-gate and
ship once that's done.

---

## 2026-07-05 — CORRECTION: the "v26/v27 ladder regression" was age-confounded, not real

**What happened:** entries below this one describe pulling `publicScore` for
v26 and v27, seeing a decline (818.3 → 726.2 → 695.1 across reads), and
reverting main.py to v25c plus shipping an isolation candidate (v25c +
board-thinning fix only, no Phase C) — submissions `54354862` and
`54354935`. **Do not trust the "monotone decline → Phase C regressed"
narrative in those entries as established.** Immediately after those ships,
a fresh pull showed the pure-v25c revert (essentially identical code to the
original 818.3 submission) scoring **600.0**, while the just-shipped v27
simultaneously read **735.9** (up from 695.1 minutes earlier). Same-ish
code scoring 818 vs 600 rules out a real quality regression as the sole
explanation.

**Corrected read (per advisor, second consult):** this Kaggle ladder's
`publicScore` most likely climbs as a fresh submission accumulates games
over hours-to-days (or is simply high-variance early and settles later —
either way the fix is the same). Read through that lens, the original data
was never a monotone decline, it was a **maturity gradient**: v25c
(2 days settled) = 818; v26 (~6h old at read time) = 720-726; v27 (minutes
old, read 3 times) = 616→695→736, climbing in real time; the v25c-revert
(minutes old) = 600. This matches an earlier-documented data point in this
same file (v25b re-scored 861→748 over time) — `publicScore` is not a
stable snapshot for a fresh submission.

**What still stands:** the revert and the isolation ship are both harmless
and defensible regardless (both are v25c-based, both a clean, sensible
"latest 2" state) — keep them, don't ship a 4th variant today just to
chase more noisy reads. **What does NOT stand:** "v26 ladder-regressed,"
"v27 made it worse," "both offline gates were contradicted by the ladder"
— none of that is established; it rested on comparing differently-aged
submissions as if `publicScore` were a stable, comparable number.
**Real verdict on whether Phase C actually regressed the agent: still
unknown.** The honest way to find out is to let the isolation submission
(`54354935`) settle for the same amount of time v25c had (~2 days) before
comparing it against a similarly-settled v26/v27 reading, or — better,
and actionable without waiting — pull recent real ladder replays via
`tools/download_replays.py` and check directly whether `_belief_posterior`
/ wall-anticipation are firing sensibly, which is signal independent of
the noisy score entirely.

**Report-log methodology note (this one is real and worth keeping):**
single-point `publicScore` reads are age-confounded on this ladder and
must not be used to compare submissions of different ages. Future
regression/improvement claims from this API need either (a) reads taken
at matched submission age, or (b) corroborating replay/behavioral evidence,
not a single score delta.

**Update, ~2h after this correction:** re-pulled scores again — the
isolation candidate (`54354935`, v25c+boardthin) has moved 600.0 → 877.9
(now numerically ABOVE v25c's original 818.3), while the plain v25c-revert
(`54354862`) has moved 600.0 → 468.0 → 714.1. This is not a clean monotone
settling curve either — it's genuinely high-variance, swinging by
100-300+ points within a couple hours for unchanged code. **Do not read
877.9 as "the fix beats v25c"** — that would repeat the exact mistake this
correction entry describes, just in the opposite direction. Continuing to
wait for the swings to damp down before drawing any conclusion; no further
ships until there's a real basis to compare.

---

## 2026-07-05 — PRE-REGISTRATION: DMC (Deep Monte Carlo), the one remaining bounded learned-policy shot

**Per the Fable design-consult decision above:** at most one more learned-
policy operator gets attention, and it must be genuinely different from
the five already-closed negatives (BC, DAgger, AWR, PIMC-search,
oracle-critic) plus the flat exploiter-round-1 result. DMC (DouZero's
actual recipe) is that operator: train the network's per-action score as a
true Q-value via direct regression to the game's Monte Carlo return
(+1/-1/0), with NO BC/teacher-imitation data mixed in at all (unlike every
prior attempt, which all anchored back toward imitating `main.py` to some
degree) and epsilon-greedy exploration instead of temperature-sampling a
softmax policy.

**Method:** reused the existing action-conditioned architecture as-is — the
model already scores each legal action via `action_mlp`+`logit_mlp`
(originally trained as policy logits); DMC repoints the SAME output at a
regression target instead. New code: `training/nn/train_dmc.py` (Huber
loss between the taken action's logit and the episode outcome, warm-started
shape-filtered from `ptcg_dagger_r2.pth` for feature representations only —
the training objective itself has zero imitation signal) and
`training/nn/dmc_agent.py` (epsilon-greedy: prob ε uniform-random legal
action, else argmax over Q). Collection reuses the existing
`exploiter_collect.py` infra unchanged (self-play vs. frozen `main.py`,
seats alternated, outcome-labeled decisions) — round 1 bootstraps on
already-collected `exploiter_r1*.pkl.gz` data; round 2+ will collect fresh
games with `dmc_agent.py` against the round-1 checkpoint.

**Pre-registered gate (set before seeing results, per the discipline that
has held for every prior line in this file):** a **trajectory** gate, not
an endpoint gate — win rate vs. **frozen v25c specifically** (not v26/v27,
to keep the target fixed and comparable to the other 5 closed lines) must
show a **monotone climb crossing ~25-30%** within the time budget, checked
at each round. **Hard stops:** first checkpoint by ~2026-07-19, absolute
stop end of July 2026 regardless of outcome — per Fable's explicit warning
that DouZero's own published recipe needed substantially more compute
against much weaker baselines than v25c, so the honest expected value here
is low and this must never compete for attention with the ladder-scoring
work (Phase C consumers, the v26 regression investigation, more
replay-mining) that Fable ranked as primary. Runs in the background only.

**Result — round 1: 1.0% (2/200) vs frozen v25c.** Training itself ran
cleanly (`training/nn/train_dmc.py` on 222,341 raw / 131,610 usable
samples from the already-collected `exploiter_r1*.pkl.gz` — bootstrapped
from the OLD dagger_r2 policy's self-play, not yet a real DMC iteration):
`epoch 0 avg_loss=0.7593 val_sign_acc=0.8986`, `epoch 1 avg_loss=0.1541
val_sign_acc=0.9132`. That validation split is in-distribution (a random
split of the same self-play data, not a held-out game distribution) and
high sign-accuracy over states does not guarantee good argmax-over-action
decisions — confirmed exactly that gap: gated the resulting greedy
(`NET_EPS=0.0`) policy for 200 games vs. a reconstructed frozen-v25c
snapshot (`training/baselines/v25c.py`, pulled from git history at commit
`cb0a526`, verified pre-dates Phase C / the board-thinning fix) —
**1.0% win rate, 0 engine errors** (so not a crash-driven floor; a quick
synthetic-input sanity check of `dmc_agent.py` didn't turn up an obvious
bug either, though this wasn't exhaustively ruled out). This is far below
even the exploiter's already-weak 10.6%, consistent with Fable's warning
that a single round of regression from limited, policy-skewed data is a
classic DMC cold-start failure mode (state-level sign-accuracy ≠ a good
greedy action ranking within a state). Per the pre-registered trajectory
gate, one round this low doesn't itself close the line — the protocol
calls for watching the trajectory across rounds — but it is a very weak
start against a 25-30% target. Launched round 2 (fresh collection with
the round-1 checkpoint, `NET_EPS=0.1`, vs. frozen v25c, then retrain) in
the background; given the explicit low-expected-value framing, this stays
a background-only line and does not take attention from the ladder-facing
priorities (v26 regression, Phase C consumers) unless round 2 shows a
real climb.

**Round 2 collection (before retrain): 1.7% (600 games, `NET_EPS=0.15`
vs. frozen v25c)** — essentially flat vs. round 1's 1.0%, no sign of a
climb yet after 2 data points against a 25-30% target.

**Round 3 (retrained on round-1 + round-2 combined, 267,906 raw / 154,425
usable samples, `train_dmc.py` → `ptcg_dmc_r2.pth`, in-distribution
val_sign_acc 0.9377): 2.5% (5/200, greedy/`NET_EPS=0.0`) vs. frozen v25c.**
Note: the *monotone* shape (not flat) across 3 retrains also rules out the
"always picks a degenerate fixed action" bug hypothesis — a gross bug would
sit flat, not climb; the signal is genuinely flowing, just at cold-start
speed.

**Trajectory so far: 1.0% → 1.7% → 2.5%** — a real, monotone-so-far climb,
but at roughly +0.7-1pp per round. At this rate, reaching the pre-
registered 25-30% target would take on the order of 30+ further
collect/train/gate rounds — far outside both the compute budget and the
hard-stop schedule (first checkpoint ~2026-07-19, absolute stop end of
July). **Decision:** this is not grounds to close the line early — the
pre-registered protocol set a date checkpoint, not a "stop when the rate
looks slow" rule — but per Fable's explicit framing (must not compete for
attention with ladder-scoring work) and the now-evident rate, this line is
paused here rather than actively grinding rounds 4+ this session. It will
be revisited at the 2026-07-19 checkpoint with all 3 data points as
evidence; if the rate hasn't inflected upward by then, close it as the
6th converging negative.

**One bounded methodological check (not a new collection round — reused
existing round-1+round-2 data, no new games): does class imbalance explain
the slow climb?** Win-outcome samples are a small minority of the combined
corpus (13,306/154,425 = 8.6%), a classic sparse-positive-signal problem
for regression. Retrained with wins oversampled 6x (36.1% of the balanced
set) — same architecture, same warm start, no new data collection.
**Result: 2.0% (4/200) vs. frozen v25c — statistically indistinguishable
from round 3's 2.5%.** Class imbalance is not the bottleneck; whatever is
capping the climb is more fundamental (state representation, insufficient
state-diversity in the self-play data itself, or genuinely needing far
more total samples/rounds as DMC's own published recipe implies). Does
not change the decision above — line stays paused to the 2026-07-19
checkpoint — but rules out one specific, cheap fix so a future session
doesn't re-try it.

---

## 2026-07-05 — Exploiter-win replay mining: 18/18 losses are the SAME failure mode (deck-out/board-thinning race)

**Method:** dumped a fresh sample of games where the exploiter net
(`ptcg_exploiter_r1.pth`, temp 1.0) beat frozen `main.py`, alternating seats,
via a one-off script (`training/nn/_scratch_replay_dump.py`, `keep_steps=True`)
until 18 wins were collected across 5 batches of 40. Ran the existing
`tools/analyze_replay.py` (already built for this purpose from earlier
sessions) on all 18 to get human-readable decision logs — output kept at
`replays/exploiter_wins/win_*_summary.txt`.

**Finding:** every single one of the 18 examined games — not most, all —
ends the same way: `main.py`'s side is at deck=0-10 cards (8/18 hit the
harness's explicit `DECK_OUT` terminal cause; the other 10 are recorded as
`OTHER` but end in the identical state — deck count in single digits, or in
one case, `win_011`, the active slot goes fully empty, `you=NONE`, meaning
every Pokémon was already KO'd with nothing left on the bench to promote),
while the opponent sits comfortably on a fully-fueled `Alakazam(743) 140/140`
needing only 1 more prize. `main.py`'s side, in every case, is stuck on a
tech/non-attacker (`Kadabra`, `Genesect`, `Shaymin`, `Psyduck`, `Dunsparce`)
— it never completes its own Alakazam evolution+fueling line before running
out of resources. One clean example (`win_009`): by turn 11, hand size 11
(220 damage available via Powerful Hand if it could attack), 3 copies of
`Kadabra` sitting dead in hand the entire game because no `Abra` was ever
in play to evolve onto, while `main.py` played `Hilda` twice (a Stage-1/2 +
energy searcher that cannot fetch a Basic) instead of a Basic-capable
searcher — the game stalls out with zero prizes taken by either side before
hitting the step/time limit.

**This is not a set of newly-discovered tactical bugs** — it is a tight,
quantified confirmation of the already-flagged, still-open CLAUDE.md item:
*"board-thinning (ending up with 1-2 Pokémon in play and a bloated dead hand
after the attacker line gets repeatedly KO'd)."* The exploiter net's entire
10.6% win rate against `main.py` appears to be substantially explained by
triggering this one race condition (deck-out or full board wipe before the
attacker line comes online), not by out-playing `main.py` tactically —
consistent with the exploiter itself being a weak, still-mostly-imitating
policy (per the round-1 report entry above) that cannot win on skill, only
by exploiting a structural weakness that shows up often enough at a 10.6%
base rate.

**Report relevance:** this is strong, concrete, replay-verified material for
the report's model-approach section — it demonstrates the falsification
program surfaced a real, quantifiable weakness in the shipped heuristic
(not just negative ML results), and gives a precise, evidence-backed target
for the highest-priority next heuristic fix: harden `main.py`'s resource/
deck management to avoid completing the evolution line too late relative
to remaining deck depth, and/or fix Hilda-vs-Dawn-style search prioritization
when no Abra is in play (the v25c changelog already fixed one instance of
this exact Hilda/Dawn issue — `win_009` suggests it is not fully closed).

**Decision:** treat "harden the deck-out/board-thinning race" as the
top-priority next action item (ahead of scoping a new Phase C consumer,
per the reprioritization above) — it is evidence-backed, concrete, and
ship-able as a heuristic fix without needing further ML experiments.

---

## 2026-07-05 — v27 shipped: `hand_surplus` gate broadened (board-thinning fix), + v26 ladder-score regression flagged

**Fix (delegated investigation, reviewed):** root-caused the deck-out
sub-mode of the board-thinning pattern above to `main.py`'s `hand_surplus`
gate requiring `ready_attacker_exists`, which is false during a rebuild
turn (attacker just KO'd, no bench backup) — exactly when hand is already
2-3x `cards_needed` from earlier banking. Powerful Hand's damage is capped
by opponent HP regardless of attacker readiness, so a hand this far over
`cards_needed` is already-wasted value; confirmed directly in two replays
(`win_001`, `win_010`) burning the deck to 0 in a single turn via repeated
Poffin/Dawn/Hilda/Poké Pad plays at hand_n=16-23 vs `cards_needed`=7 while
stuck on a non-attacker. Fix: `hand_grossly_over = opp_hp<99999 and
hand_n>=cards_needed+6`, OR'd into the existing gate — one line, all 7
existing `hand_surplus` consumers benefit uniformly, no new call sites.

**Scope caveat (important, do not overclaim):** this addresses only the
"attacker line complete/near-complete but hand-surplus gate didn't engage"
sub-mode — confirmed directly in 2 of the 18 mined replays. It does NOT
address the "never builds the attacker at all" pattern (the majority
symptom in the other 16) or 2 anomalous short `OTHER`-cause games where the
terminal-cause classifier itself may have a gap (`tools/analyze_replay.py`,
not confirmed). Board-thinning is only partially closed by this fix.

**Gate:** 300-game mirror A/B (fixed vs. frozen pre-fix `main.py`, i.e. vs
v26) — **56.0% ± 5.6%** (95% CI [50.4%, 61.6%]) — modest but real, clears
50%. Smoke: 20 games vs `lucario_agent` (95%) + `abomasnow_agent` (95%), 0
errors across ~580 games during validation.

**Shipped:** submission `54354278` (v27), 2026-07-05.

**Important flag, found while shipping:** pulled `v26`'s actual ladder
`publicScore` for the first time — **720.4, DOWN from v25c's 818.3.** The
v26 gate (400-game A/B, 50.7%±4.9%) was explicitly a non-regression check
against anchor bots that don't exercise the wall-anticipation path — per
Design Principle #1 ("offline overrates"), the ladder is the only honest
evaluator, and this is exactly the failure mode that principle warns
about. **v26's ladder regression is NOT yet root-caused.** v27 ships on
top of v26 (same Phase C code, plus the fix above) — its own ladder score
needs to be watched closely to tell apart two possibilities: (a) the
board-thinning fix recovers some or all of the v26 regression, or (b) v26
has a separate, still-unfound problem (e.g. the belief posterior
misfiring against real ladder archetypes it wasn't validated against,
unlike the classifier's 5 training archetypes) that persists in v27 too.
**Next step once v27's score is in:** if still regressed vs 818.3, do a
targeted investigation of Phase C's real-ladder behavior specifically
(pull fresh ladder replays post-v26, check whether `_belief_posterior`
or the wall-anticipation triggers are firing sensibly against real
opponent decks) rather than assuming the board-thinning fix alone
explains any change.

---

## 2026-07-05 — DESIGN DECISION (Fable consult): re-spine the report, deprioritize pure learned-policy chase

**Context:** five independent operators have now converged on the same
outcome — BC (17% vs teacher), DAgger (fidelity 73%→82%, win-rate flat
~15-17%, paused not closed), AWR (15.8%/11.5% vs teacher, worse with more
aggressive weighting, closed negative), PIMC search-at-inference (0W-50L,
closed negative), oracle-critic value head (62.5% sign-acc vs 65%
threshold, closed negative), and now v25c exploiter round 1 (winner-only
self-imitation, flat ~12% vs ~11% baseline at both temp 1.0 and argmax,
closed negative for that specific method). This matches the 2026-07-04
literature review's prediction and an external data point (a Kaggle forum
thread: best public pure-RL agent for this competition sits ~30th
percentile, below v25c).

**Consulted Fable** (full transcript context: this file + CLAUDE.md) on
whether to (a) close the "pure learned policy exceeds heuristic" framing
and re-spine the report around a hybrid claim, (b) invest in a genuinely
different exploiter operator (real iterated best-response, low/no BC
anchoring), or (c) something else, given ~6 weeks of ladder time and ~10
weeks to the report remaining.

**Recommendation (adopted):** reject the binary framing — the team already
has a shipped, load-bearing learned component (the Stage 3 belief model,
driving v26's `opp_likely_ace_spec` and wall anticipation, gated 50.7% over
400 games). The report's 70%-model-approach axis should be won on: (1) a
hybrid agent whose learned belief module measurably improves piloting
decisions (published precedent: DouZero+ +5-7% from belief inputs), and (2)
a systematic, pre-registered, literature-corroborated falsification program
across five independent learned-piloting operators — itself strong
model-approach content, and something almost no competitor will have.

**Reprioritization (effective immediately):**
1. **Primary:** more Phase C belief-model consumers in `main.py` (each is
   independently ladder-A/B-gatable and directly evidences "learned model
   improves the agent"); the real-replay accuracy-by-turn figure (target
   figure #1 — Phase A's 92.3% is the easy 5-bot version, the honest number
   against the 78.7% recognition ceiling isn't built yet); mining the 106
   exploiter-round-1 win replays for v25c blind spots (nearly free, already
   collected, and replay-verified heuristic fixes have been the single most
   productive lever in this project's entire history — 861.8→~880-900
   ladder Elo came from that loop, not ML).
2. **Skip:** the determinization sampler (Phase C item 2) — its only
   consumer (determinized search) is closed with a named cause; building it
   now is speculative infrastructure against CLAUDE.md's simplicity rule.
3. **At most one bounded, background, pre-registered shot** at a genuinely
   different learned-policy operator: Deep Monte Carlo with action-as-input
   Q-network (DouZero's actual recipe — the one operator in the literature
   review never tried), continuous training against a FROZEN v25c, no BC
   mixing, no value-head mediation, epsilon-greedy exploration, Monte Carlo
   returns. Must be pre-registered with a trajectory gate (monotone climb
   crossing ~25-30% win-rate vs. frozen v25c) and a hard wall-clock stop
   (gate check ~2026-07-19, absolute stop end of July regardless of
   outcome) — expected value is low (DouZero's own recipe needed
   substantially more compute against weaker baselines) so it must not
   compete for attention with the ladder-scoring work above.
4. **Explicitly ruled out:** further oracle-critic variants, DAgger rounds
   3+, AWR beta sweeps, exploiter round 2 with tweaked hyperparameters —
   re-rolls inside an already-converged negative family.

---

## 2026-07-05 — PRE-REGISTRATION: v25c exploiter, round 1 (winner-only vs frozen main.py)

**Hypothesis:** DAgger (imitation, caps at teacher parity), AWR (reward routed
through a value head that saturates near ±1), and PIMC search (built on the
same saturated value/policy) all failed to exceed v25c. None of them optimize
directly against a fixed target with a real win/loss signal. Winner-only
self-play specifically vs. a FROZEN `main.py` (not a mirror, not the ladder)
gives a cheap, different operator: keep only the trajectories the net actually
won, retrain on those (uniform weight, no AWR advantage-shaping), and rerun —
an iterated best-response only against the exact deck/pilot we're trying to
beat.

**Method:** `training/nn/exploiter_collect.py` runs the current checkpoint
(temp 1.0, exploration on) vs. frozen `main.py`, seats alternated, and records
only the net's own actions/outcome (no teacher relabeling). Round 1: 1000
games from `ptcg_dagger_r2.pth` → **106/1000 wins (10.6%)**, 138,693 samples
(`training/exploiter_r1.pkl.gz` + `.part1`). Retrain via
`training/nn/train_sp.py --winner-only` (filters to `outcome>0`, uniform
weight, BC pool still mixed at `--bc-frac 0.4` for policy grounding).

**Pre-registered gate (set BEFORE retraining, per advisor guidance — don't
run multiple rounds and then look):** collect a fresh 200-400 game batch of
the round-1-retrained net vs. frozen `main.py` (same seat balance). Compare
win rate to the round-0 baseline (10.6%):
- **~10-13%: flat/asymptoting like the prior three lines — do not commit to
  further rounds** without a different lever (larger collection, softer
  filter, or close as a 5th negative).
- **≥20%: real single-round improvement — round 2 is justified**, and this
  becomes the leading live candidate.
**Caveat (both outcomes):** offline win-rate vs. a frozen main.py is a
necessary but not sufficient signal — an exploiter finds `main.py`'s
specific blind spots by construction, which may not transfer to the ladder's
actual deck diversity (Design Principle #1: offline overrates, v5 was 64%
offline / 0-5 live). A pass here queues a ladder-realistic gauntlet check,
not a direct ship.

**Result: GATE NOT MET — round 1 flat, matches the "asymptoting" bucket.**

Retrain hit the same checkpoint-widening `strict=True` load failure as the
oracle-critic work (`model.py`'s value_head is now permanently wider from
that change) — fixed identically in `train_sp.py` (shape-filtered
`strict=False` load). Also hit a **second live OOM** during this session:
the first retrain attempt (unbounded `--bc-data`, 579k v25c BC samples)
silently died, and — new failure mode — `Stop-Process` by the MSYS/cygwin
PID (1606) reported "not found" while the real Windows PID (32264) kept
running and holding 24.5GB, starving the retry. Root-caused via
`Get-Process | Select Id,WorkingSet64` (PowerShell) showing the true PID/
memory; killed correctly, then reran with `--bc-limit 100000`, which
completed cleanly (100,000 BC + 14,679 winner-only SP samples, 3 epochs,
`ptcg_exploiter_r1.pth`).

Gate batch (temp=1.0, matching how the 10.6% baseline was measured): 300
games, **12.3% win rate vs main.py** (vs 10.6% baseline) — inside the
pre-registered 10-13% "flat" bucket, not the ≥20% bar for round 2.
**Follow-up per advisor** (a sampled win-rate at temp=1.0 measures the
exploring policy, not the deployable one): re-measured both checkpoints at
temp=0.1 (argmax/deployment-mode), 300 games each — **exploiter_r1 12.7%
vs baseline dagger_r2 10.7%**. Same ~2pp gap at both temperatures — the
negative is not a measurement artifact.

**Scoped claim (per advisor, to avoid overclaiming in the report):** this
result licenses "cheap single-round winner-only self-imitation against a
frozen teacher does not meaningfully improve on it" — not "no learned
exploiter operator can beat the heuristic." At a 10.6% base rate, a large
share of "wins" in the training pool are plausibly main.py mulliganing or
dead-drawing rather than the net outplaying it (a variance trap flagged
before the run), and `--bc-frac 0.4` spends 40% of every training batch
pulling the policy back toward imitating `main.py`, which works against
the stated goal of exploiting it — this is close to the weakest possible
version of "train an exploiter," not a rigorous test of the idea.

**Decision:** do not commit to further winner-only rounds (lower `bc-frac`,
more games) without a different lever — per pre-registration this is
5th-negative territory for that narrow method, but the broader
"can a learned exploiter beat v25c" question is not yet closed; escalated
to Fable as an overarching design-change decision (2026-07-05) given 4-5
negatives now converging on what the 2026-07-04 literature review
predicted, and the choice between closing the learned-policy thesis vs.
investing in a genuinely different operator (e.g. real iterated
best-response with fresh collection per round and low/no BC anchoring, or
reward-weighted policy gradient) affects the report's spine.

---

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

## 2026-07-04 — PRE-REGISTRATION + BUILD: oracle-critic value head (PerfectDou-style)

**Hypothesis (literature-backed, upgrades the earlier diverse-data value-head
pre-registration below):** the value head's saturation/OOD failure is partly
an *information* problem, not just a data-diversity problem. PerfectDou
(arXiv:2203.16406) beat DouZero by letting the CRITIC see hidden state during
training while the policy stays legal; Suphx and OADMCDou independently
converged on the same trick. In our local self-play harness the opponent's
true hand is free (both seats' observations are recorded in every step
trace), so a privileged value head costs only plumbing.

**Method (built 2026-07-04, Sonnet implementation to a fixed design, reviewed):**
1. `training/nn/model.py`: `oracle_embed` EmbeddingBag over opponent-hand
   card ids + presence flag, concatenated into the VALUE head input only
   (`Linear(256+128+1,1)`); policy path verified bit-identical with/without
   oracle input. Old checkpoints load via shape-filtered `strict=False`
   (`net_common.py`); policy argmax on `ptcg_dagger_r2.pth` verified
   unchanged pre/post model change.
2. `training/nn/vd_collect.py`: diverse corpus (4 anchors + mirror, both
   seats) with a **validated backward-walk oracle join** — opponent hand
   taken from the most recent opponent observation whose own-hand length
   exactly equals the opp handCount we observe at the decision; no exact
   match → oracle flag 0 (only verifiably-current privileged info is used).
   Exact-match rate 87.6%; misses are concentrated (46/61) in turn-0/1
   setup states where no intermediate opponent observation exists —
   correctly degraded, near-complete mid-game coverage.
3. `training/nn/dataset.py`: `oracle_ids`/`oracle_flag` batching +
   `ORACLE_DROPOUT=0.25` (Suphx-style, trains the no-oracle path so the
   same head evaluates ladder states and determinized search leaves).
4. `training/nn/train_value.py`: warm-start `ptcg_dagger_r2.pth`
   (value_head reinitialized), loss = Huber value + 0.1×policy CE
   regularizer, saves on best val sign-accuracy.

**Pre-registered gates (unchanged from the earlier entry):** (a) evaluator
gate — `value_signal_probe` must show contested-state values off the ±0.95
rails with correct sign tendency, AND ≥65% held-out ladder outcome
sign-accuracy (`value_holdout_eval`, no-oracle path); (b) only if (a)
passes: plug in as leaf evaluator in `training/nn/mcts.py` (shallow
rollout→evaluate) and gate ≥55% vs v25c over 50+ games, else this line
closes and the story is the negative-results trilogy + belief model.

**Result: GATE NOT MET — closing this line as inconclusive-negative.**

Collection: `vd_collect.py`, 2000 games vs 4 anchors + mirror,
`our_winrate=0.889`, 252,823 samples, 3 shards, oracle exact-match rate
83.05% (backward-walk join; misses concentrated in unrecoverable turn-0/1
opening-hand states, as designed).

**OOM during first retrain attempt, diagnosed and fixed:** the full-corpus
run (`vd_diverse*.pkl.gz` + `vd_ladder_train.pkl.gz`) exited 0 with zero
training-progress output — a silent OOM kill, the same `dataset.py::
load_shards`-reads-full-shards-before-`--limit` pattern documented
elsewhere in this project's history. Confirmed via a working `--limit 4000`
run (val_sign_acc 0.955, proving the code path itself was correct) plus a
memory probe (`mem_probe.py`): 39.6GB total RAM, 15.1GB available, ~58KB/
sample measured directly, projecting ~16.1GB needed for the full 293k-sample
corpus — over budget. Fix: deterministic 1-in-3 seeded subsample
(`subsample_vd.py`) → `vd_diverse_sub.pkl.gz` (84,168 samples, oracle
fraction 0.809 preserved). Retrain on subsample + ladder-train (161,065
samples total) completed cleanly: `epoch 0: val_sign_acc=0.7567`,
`epoch 1: 0.7970`, `epoch 2: 0.8248` (best, saved).

**Pre-registered gate (a):** `value_holdout_eval.py` on
`vd_ladder_holdout.pkl.gz` (4000 real-ladder samples, never trained on) —
**ALL sign_acc=0.625** (EARLY turn≤4: 0.525 near-coinflip as expected;
MID turn 5-10: **0.565**, barely above coinflip; LATE turn≥11: 0.776).
**0.625 < the 0.65 threshold — gate fails.** `value_signal_probe.py` on the
4-anchor diverse set: frac_extreme(|v|>0.95)=57.1%, mean=0.709 (inflated by
the 88.9% win rate vs those anchors, not evidence of calibration).

**Root-cause check (advisor-directed, before accepting the negative):**
does the gate fail because the privileged info was useless, or because it
doesn't survive to the oracle-free inference path the gate (correctly)
tests? Discovered `net_common.py::encode_batch` / `value_estimate` never
pass oracle features at all — **the evaluator gate is structurally
oracle-blind by construction**, which is actually the right test (ladder
inference and search leaves never have the opponent's true hand either).
Built a direct ON/OFF diagnostic (`oracle_onoff_diag.py`) on 800
oracle-hand-populated samples from the diverse set: **oracle-ON sign_acc
=0.870 vs oracle-OFF=0.8425** — a real, positive, but modest gap
(+2.75pp), and this set was itself part of the training mix so the
absolute numbers are optimistic; the gap is the informative part, and
because this is TRAINING data, memorization should inflate the ON/OFF gap
to its maximum — a gap this small (+2.75pp) at best-case measurement
conditions means the value head extracts little decision-relevant signal
from the opponent's hand beyond what board state already gives it, not
that a real signal failed to survive dropout. (Corrected read, per
advisor, from an earlier draft of this entry that over-read the gap as
"real but dropout-blocked" — that framing would incorrectly invite
retrying with less dropout.) This is consistent with the value head still
not carrying much decision-relevant nuance beyond coarse board-state,
echoing the AWR finding (Stage 2, 2026-07-04: "value head saturates near
±1 on most states"), and it is weakest exactly where opponent-hand info
should matter most if it mattered at all (mid-game holdout sign-acc
0.565).

**Decision:** do not integrate into `mcts.py` (gate (b) is conditional on
(a) passing). Close this line — fourth data point (after DAgger, AWR,
PIMC-search) that value/policy signal built from this self-play data,
even with privileged information added, plateaus at-or-below teacher
parity rather than exceeding it. Given three prior negatives in adjacent
territory, further oracle-critic iteration (e.g. lower dropout, explicit
distillation loss) is deprioritized in favor of the v25c exploiter track
(orthogonal: trains directly against a frozen strong teacher rather than
trying to extract more signal from the same self-play distribution) and,
if that also plateaus, the report should treat "no operator in this
toolkit exceeds a strong heuristic teacher without external search/
verification" as a load-bearing finding rather than a gap to keep closing.

---

## 2026-07-04 — Stage 3 Phase C: belief posterior wired into main.py

**Hypothesis:** wiring the Phase A archetype posterior into the heuristic
(replacing the v14-era hardcoded `opp_likely_ace_spec=True` and adding
Mist/Rocky *anticipation*) is a cheap, published-precedent upgrade (DouZero+
got +5-7% from belief inputs) that must at minimum not regress vs the
pre-wiring pilot.

**Data first (new, this session):** a 679-replay per-archetype tech survey
(scratch script; supersedes guessing from bot decklists) — wall energy
(Mist #11 / Rock #20) is revealed by **crustle 35.8%** and the
**unrecognized long tail 29.0%**, vs **0-3% for all five classifier
archetypes** (lucario 2.7% rock, starmie 2.4% mist, rest ~0). Ace-spec
reveal rates: rockets-mewtwo/raging-bolt 71%, alakazam 63%, lucario 60%,
archaludon 48%, abomasnow 48%, dragapult 46%, starmie 29%, overall 44.6%.
Design consequence: Mist anticipation must key on crustle/unknown reads and
revealed-wall evidence, NOT the 5-class posterior alone (its classes don't
tech walls).

**Method (all in `main.py`, pure Python, no new deps):**
1. Phase A logistic weights embedded (~4KB, 71 features × 5 classes);
   `_belief_posterior(opp, turn)` mirrors `training/belief/collect.py`
   feature extraction (board+discard revealed ids incl. preEvolution+tools,
   fresh energy-type counts, scalar counts); sparse dot product + softmax;
   returns `(None, False, False)` on any failure → all consumers fall back
   to pre-Phase-C behavior. Pure — honors `_main_phase_features`' search
   purity contract.
2. `opp_likely_ace` = observed-in-logs OR posterior-confidence<0.8 (keeps
   the conservative True default) OR archetype ace rate ≥0.35. Net change:
   only a confident starmie read (0.29) demotes the Genesect+tool bench
   priority.
3. `mist_threat` = wall energy revealed anywhere on their side (discard or
   attached, incl. bench — old `opp_mist` only saw the active) OR
   Crustle/Dwebble line seen OR classifier can't place the deck at ≥0.8 by
   turn 2+. Two surgical consumers: (a) Boss *chip* savoring — when a wall
   is anticipated but not present, don't spend Boss on optional mega chip
   damage (hold it as the wall escape); (b) Enhanced Hammer at 36 (vs base
   3) when a wall energy is parked on their BENCH — strip the wall being
   prepared before it promotes (the existing select scorer already targets
   Mist/Rocky first).

**Verification:** synthetic-board sanity (lucario/starmie/alakazam boards →
0.996 correct posterior; crustle board → 0.325 max = correctly unknown-ish
+ both threat flags; empty board diffuse; None input → fallback path).
Smoke: 10 games each vs lucario (9-1) + starmie (10-0), 0 errors.

**Pre-registered gate (docs/belief-model.md §Phase C):** 400-game A/B vs
pre-wiring main.py (`training/main_pre_phasec.py` = git HEAD copy) — must
not regress; then ladder confirm. NOTE the A/B is neutral-by-construction
on the mist_threat path (the anchor bots tech no walls and classify at
0.996 confidence) — it gates the ace-spec change + any accidental
regression; the wall-anticipation payoff is only observable on ladder vs
crustle/unknown (~29% of the field combined).

**Result:** **GATE PASSED — 50.7% ± 4.9% (95% CI) over 400 games, 0 errors**
(203W-197L, per-seat 104-96 / 99-101 — no seat asymmetry). Non-regression
confirmed; as predicted, the A/B is statistically a mirror match since the
anchor pool never exercises the wall-anticipation path. Shipped for ladder
confirm same day (see ladder_history.csv).

**Report relevance:** the belief model's first deployed consumer (the
originality centerpiece goes live); the tech-survey table is report
material on its own (data-driven tech-rate priors per archetype).

---

## 2026-07-04 — Literature review: how learned agents actually exceed strong teachers

**Hypothesis (motivating question):** after three load-bearing negatives
(DAgger plateau, AWR, PIMC 0/50), is "learned can't beat this heuristic" a
fact about our methods or about the problem? Ran a three-thread literature
survey (self-play RL feasibility; privileged-info/offline-RL theory; Kaggle
competitive intelligence) via parallel research agents.

**Method:** web survey with citations, focused on imperfect-info card games
with strong heuristic baselines and solo-desktop compute. Full reports in
session transcript; key sources: DouZero (arXiv:2106.06135), DouZero+
(2204.02558), PerfectDou (2203.16406), Suphx (2003.13590), VLOG (ICLR'22),
OADMCDou (IJCAI'24), Kumar et al. offline-RL-vs-BC (2204.05618),
"Is Value Learning the Bottleneck" (2406.09329), Minimax Exploiter
(2311.17190), ISMCTS/strategy-fusion literature, Kaggle threads 711644 /
717697, github.com/wmh/ptcg-abc.

**Result — five findings:**
1. **Our negatives are predicted, not anomalous.** Kumar et al.: with
   near-expert-only data, no offline algorithm can provably beat BC; AWR is
   specifically identified as a weak policy extractor; value saturation at
   ±1 under sparse terminal reward from a deterministic expert is the
   textbook picture. The whole Kaggle field replicates it (thread 711644:
   BC 10%, PPO-league 25% vs teacher; best public pure-RL claim ~top 30th
   percentile, below v25c).
2. **Every published success at exceeding a strong baseline injected
   information the baseline lacked**: hidden state at training time
   (Suphx oracle guiding; PerfectDou's oracle *critic* — critic sees both
   hands, policy doesn't, beat DouZero in ~1.5M steps; VLOG; OADMCDou) or
   belief posteriors at inference (DouZero+: +5-7% win rate from feeding a
   predicted opponent-hand distribution into the decision net — published
   precedent for our Phase C wiring).
3. **From-scratch Deep Monte-Carlo is desktop-feasible**: DouZero used
   4×1080Ti/48 cores, beat a strong heuristic in half a day, prior SOTA in
   10 days; key transferable trick is action-as-input-features Q-network
   (fits our variable action space). No imitation ceiling by construction.
   Caveat: v25c is a far better-tuned baseline than DouDizhu's bots were.
4. **Confirmed dead ends at our scale**: DeepNash/R-NaD (cluster-scale,
   optimizes unexploitability not ladder strength), ReBeL/Student of Games
   (representation-infeasible for a TCG — infostate enumeration), NFSP
   (feasible but never exceeded expert systems). PIMC's 0/50 has a named
   cause (strategy fusion + non-locality) — our Stage 5 closure is evidence
   against *determinized* search, not search per se.
5. **Exploiter reframing** (Minimax Exploiter, AAMAS'24): train a net only
   to beat frozen v25c — a one-fixed-policy target, much easier than the
   ladder. Productive even on failure: every exploiter win is a
   machine-found v25c blind-spot replay feeding the v25b/v25c fix loop;
   a failure is quantitative unexploitability evidence for the report.
Competitive intel: ladder top-8 ≈ 1114+, v25c ~880 ≈ rank 545/4252; field
consensus is heuristics dominate the visible top; our local-engine unlock
is a real edge (96-vote thread shows most teams lack it); public
wmh/ptcg-abc repo ships an Alakazam Powerful Hand agent (Elo 836) → expect
mirrors + Mist tech to grow.

**Decision:** the pre-registered diverse-data value head plan (below) stands
and is upgraded by finding 2: add **privileged critic features** (opponent's
true hand/deck, free in self-play) to the value-head retrain, and treat
Phase C belief wiring as the published-precedent cheap win to do first.
Ranked options handed to user: (1) Phase C belief wiring into the heuristic,
(2) oracle-critic value head + pre-registered evaluator-in-search gate,
(3) v25c exploiter, (4) scaled-down DMC from scratch as the long-shot
background run. DeepNash/ReBeL/SoG/NFSP formally closed without trial.

**Report relevance:** finding 1 turns the negative-results trilogy into a
literature-corroborated narrative (cite 2204.05618 + thread 711644 as
external replication); findings 2-5 justify whichever next line is chosen
as research-backed rather than ad hoc.

---

## 2026-07-04 — PRE-REGISTRATION: diverse-data value head → learned-evaluator search

**Hypothesis (research-backed):** the common root cause behind all three of
today's negative results (AWR's saturated value head, PIMC's no-signal
rollouts) is that every value signal so far was built exclusively from
Alakazam-mirror self-play — a distribution where the net never saw a
non-Alakazam board or a genuinely contested cross-archetype state. The
literature on card-game MCTS says the winning recipe at our scale is
search + a **learned state evaluator trained on (state, outcome) pairs
from diverse games** (Świechowski & Tajmajer, "Improving Hearthstone AI by
Combining MCTS and Supervised Learning", arXiv:1808.04794; Zhang & Buro on
learned rollout policies; ISMCTS literature on why weak rollouts fail in
long card games), not search + rollouts, and not belief-state methods
(ReBeL/Student of Games — sound but far beyond a 6-week window).

**Supporting local evidence (value_signal_probe.py, 2026-07-04):** on 48
games vs the 4 diverse anchors (v25c won essentially all), the dagger_r2
value head scored a quarter of sampled states below −0.90 — confidently
predicting LOSSES in games that were won — with no difference between
contested and blowout games (73.8% vs 67.0% |v|>0.95). The head is not
"correctly decisive," it is **confidently wrong out-of-distribution** — a
training-data artifact, which retroactively explains AWR's null result and
is fixable with data that exists today.

**Method:** (1) collect ~400 games × 4 anchors via `bc_collect.py
--opponent` (diverse boards + real losses; dragapult is the known-hardest
matchup); (2) parse the 680 real ladder replays into (state, outcome)
samples via new `tools/replay_to_bc.py` — DONE: 85,657 samples, 692 of
our seats, **44.4% seat win-rate (nearly balanced labels)**; (3) retrain
warm-started from `ptcg_dagger_r2.pth` on the combined corpus; (4) **gate
the evaluator itself before any game gate**: re-run `value_signal_probe`
(want frac_extreme down, correct sign on won games) + held-out
outcome-prediction accuracy on ladder games; (5) only if (4) passes, plug
in as the leaf evaluator in the existing `training/nn/mcts.py` search
(shallow rollout → evaluate, which also removes the 600s/game rollout
cost) and run one pre-registered gate vs v25c.

**Pre-registered success criteria:** step 4: held-out ladder outcome
accuracy ≥65% AND probe shows the majority of contested-state values
off the ±0.95 rails with correct sign tendency on won/lost games. Step 5
gate: ≥55% vs v25c over 50+ games → continue to a larger confirm;
otherwise this line closes too and the session's story is the
negative-results trilogy + belief model.

---

## 2026-07-04 — Stage 3 Phase B: signature extension + archetype library built

**Hypothesis:** with Stage 5 search-at-inference closed negative (see below
entries), pivoted to the already-scoped Stage 3 Phase B work per a Claude
Fable consult's explicit "regardless of outcome" sequencing — building a
real ladder archetype library was never blocked on the search result.

**Method:** (1) Clustered the 25.9%-unknown replay slice's non-generic
revealed cards (filtering common trainers/energies) to find real archetype
gaps rather than guessing. (2) Extended `tools/meta_survey.py`'s
`SIGNATURES` with confirmed findings. (3) Built
`tools/build_archetype_decks.py` to reconstruct a representative 60-card
list per archetype from real replay card-reveal frequency, weighted by
game-presence fraction and capped at real copy limits.

**Result:**
- Signature extension: unknown dropped **25.9% → 21.3%** (176→145 of 680
  replays) via two pre-evolution aliases (`snover`→abomasnow,
  `dreepy`/`drakloak`→dragapult — games ending before the ace evolves were
  being missed) plus one genuine new archetype (`kyogre`, 13 replays).
  Re-clustered the remaining 145: **no further common clusters** — 144 have
  unique/scattered minor-card combos, a real long tail, not a few more
  signatures away from coverage. Honest ceiling for this method: **78.7%**
  (535/680) recognized.
- Archetype library: `training/archetype_decks.json` — reconstructed lists
  for crustle (53 replays), archaludon (27), bellibolt (11), kyogre (11),
  raging-bolt (7), rockets-mewtwo (7), gardevoir (4), grimmsnarl (3), all
  with the ace card at/near 100% game-presence and plausible tails.
  lucario/dragapult/abomasnow/starmie skipped — already have *exact*
  decklists via the official Kaggle sample bots in `opponents/*_agent.py`.
  5 signatures (charizard, gholdengo, pidgeot-control, snorlax-stall,
  terapagos) had 0 matching replays in the current 680 — left unbuilt
  rather than fabricated.

**Decision:** Phase B's two concrete deliverables (signature extension,
archetype library) are done; the honest-unknown-handling finding (21.3%
long tail, not a gap) is logged as the answer to Phase B item 2, not an
open task. Phase C (wiring the posterior into `main.py`'s
`opp_likely_ace_spec` + a determinization sampler) is next but not started
this session.

**Report relevance:** concrete, data-driven "originality" material for the
report's model-approach section — a real ladder meta-share table built
from actual replay evidence (not guessed), with an honestly-reported
recognition ceiling rather than an inflated one. Good complement to the
Stage 5 negative-results trio from earlier today: this is deliverable
progress on a different, non-search axis of the same competition problem.

---

## 2026-07-04 — Stage 5 search-at-inference: MCTS built, first gate negative (untuned)

**Hypothesis:** with Stage 2 direct-self-play AWR closed negative (both β values
flat-to-worse vs teacher — see below), advisor guidance was that inference-
time search over a heuristic-dominant prior is a shorter path to beating the
v25c teacher than another training campaign — search over a good prior
improves on it almost by construction, and needs no retraining loop. Two cheap
probes gated whether this was even worth building: (1) branching — does one
`search_begin` root support independent children via repeated `search_step`
calls (needed for real tree search, not just linear rollouts), (2) timing —
does search fit the 10-min/match clock. Both confirmed favorable (see
`docs/engine-api.md` "MCTS branching + timing probe": branching independent
and non-destructive; ~730 sims/decision affordable on engine cost alone).

**Method:** Built `training/nn/mcts.py` (`MCTSSearcher`) — PUCT selection at
our own decision nodes using the heuristic's own per-option score vector
(softmaxed) as the prior; heuristic argmax as a fixed opponent model at the
opponent's decision nodes (not searched — keeps branching factor down);
leaf evaluation via a full heuristic-vs-heuristic rollout to a real terminal
result, **not** the net value head (already confirmed saturated bimodally
near ±1 during the AWR diagnostic — using it as a leaf evaluator would carry
almost no per-decision signal). `training/nn/mcts_agent.py` wraps this in the
standard `agent(obs_dict)` contract so it plugs into `training/ab_test.py`
directly. Also built `training/setup_local_search.py`: pairs the native
engine binary already bundled in the locally-installed `kaggle_environments`
package with the fuller `cg-lib` dataset's Python source, giving a fully
working local `search_begin`/`search_step`/`search_end` — the whole MCTS
build/test loop now runs with **zero Kaggle round-trips** (see
`docs/engine-api.md` "Local search dev shim").

Smoke-tested first: 10 games at `sims=30`, 0 errors across all decision
types exercised. First real gate: 60 games (alternating seats) at
`sims=150`, `c_puct=1.4`, `prior_temp=2.0` (both untuned defaults) vs the
v25c heuristic.

**Result:** MCTS agent (B) **25W-35L over 60 games (41.7%)** — nominally
*worse* than the plain heuristic, though not statistically separable from
50% at this n (0 errors both directions, mechanically sound; `avg_game_s`
262-390s, confirming the timing-probe budget held in practice). Working
theory: `prior_temp=2.0` significantly flattens the heuristic's own score
ordering into a near-uniform prior for many select types, which fights the
"heuristic-dominant prior" design intent — PUCT ends up spending sims
exploring heuristic-disfavored branches instead of mostly refining the
heuristic's own top choices. Iterating: sharper `prior_temp` (trust the
heuristic ranking more) and higher `sims` (within the confirmed budget
headroom) before drawing any conclusion — this is a single untuned run, not
yet a load-bearing result either way.

**Decision:** Not shipping, not concluding failure. Logged honestly per
project discipline (every experiment same-day, negative results included).
Next: a controlled follow-up varying `prior_temp`/`c_puct`/`sims` before any
verdict on the search-at-inference approach.

**Follow-up same day — sharper prior + more sims made it WORSE, not better
(2026-07-04):** `prior_temp=0.6` (sharper, trust the heuristic ranking more),
`c_puct=1.0`, `sims=300` (2x). 40 games: MCTS **13W-27L (32.5%)** — worse than
the first gate's 41.7%, and with a stark seat split (MCTS as P1 vs heuristic
P0: 15% (3/20); MCTS as P0 vs heuristic P1: 50% tied (10/20)). Rules out "just
needed better hyperparameters" as the fix — the same direction change
(trust the heuristic prior harder, search deeper) made things worse, mirroring
the AWR finding that "more aggressive" wasn't the fix there either.

**Root-cause theory — echo chamber, not a tuning problem:** the prior,
the leaf-rollout policy, AND the opponent model are all literally the SAME
heuristic function. `_heuristic_action` is deterministic argmax, so any
rollout from a given state always replays the identical continuation (only
the engine's own internal randomness — draws, coin flips — varies between
visits). The search therefore can't inject genuinely new information beyond
one level of root branching; it mostly re-confirms whatever the heuristic
already believes, occasionally amplifying blind spots rather than correcting
them. Sharpening the prior and adding sims made this WORSE because it means
trusting the same self-referential signal more strongly, not because more
computation is inherently bad. This is a different, more specific failure
mode than the AWR value-head saturation, but shares the shape: an
evaluator built entirely from the artifact you're trying to improve on can't
easily produce information that artifact doesn't already have.

**Next planned test (not yet run):** make the *rollout* policy stochastic
(temperature-sampled from the heuristic's own score distribution instead of
deterministic argmax) to see whether diversifying rollouts alone recovers
some of the lost signal, isolating this one variable against the original
untuned baseline (`prior_temp=2.0`, `c_puct=1.4`, `sims=150`). If that also
doesn't clear parity, this closes the "pure heuristic-guided MCTS, no net"
line as a second real negative result for the report (after AWR) — the
methodological throughline being that self-referential evaluators
(rollout==prior==opponent-model, or a saturated value head) don't give
search room to improve on the thing it's built from.

**Correction, same day — the "echo chamber" framing was wrong; found two
real, separate bugs instead (advisor-guided):** the sub-50% sign itself
doesn't fit an echo-chamber theory (that predicts ~50%, a wash, not
systematically worse) — sub-50% means the search's value signal was
actively misaligned, not just uninformative. Two real bugs, found via
targeted diagnostics rather than more hyperparameter tuning:

1. **`score_options` is a materially weaker reconstruction of the teacher
   than assumed.** Isolated test: `argmax(score_options(obs, sel))` alone
   (zero search) vs `main.py` over 50 games — **15W (30%)**. `mcts.py`'s
   rollout policy AND opponent model both called this weak function, not
   the real `main.agent`/`_choose` (desperation mode, `_STALL_MEMO`, real
   `_pick_boss_target`) — so every prior gate measured "search over
   `score_options`" fighting a materially weaker phantom, and MORE search
   (gate 2's sharper prior + more sims) just committed harder to that
   phantom's blind spots, explaining the monotonic degradation. Fixed:
   rollout and opponent-model now call the real `main.agent`; `score_options`
   is kept only as the (weaker, less damaging per advisor) PUCT prior.
2. **Strategy fusion — the real cause of the sub-50% sign.** `filler()`
   (the hidden-zone determinization: opponent hand/deck identity, own deck
   order) was sampled ONCE per real decision and reused across all 150
   simulations, with `manual_coin=True` removing even coin-flip randomness.
   Every simulation therefore explored the exact identical fixed fictional
   world — textbook determinized-search-without-re-determinization: the
   search wasn't averaging over hidden-information uncertainty at all, just
   exploring one guessed world deeply and picking whatever won in that one
   fiction. Under a correct implementation a mirror match has a theoretical
   floor near 50% (one step of policy improvement over the same teacher
   can't do worse); landing at 32-42% is exactly what a single bad
   determinization produces. Fixed: `training/nn/mcts.py` rewritten as
   proper PIMC — fresh random `filler()` + fresh `search_begin` EVERY
   simulation, `manual_coin=False` so the engine injects real coin
   randomness. Structural consequence: since different simulations are now
   different fictional worlds past the root, tree statistics can only be
   shared at the root (single-ply PUCT + full rollout-to-terminal per sim,
   not a deep persistent tree) — this is standard for PIMC, not a
   regression.

**Diagnostic confirming the fix (per advisor's suggested check):** printed
per-simulation values for the same root action across a mid-game decision —
values now genuinely vary sim to sim (1.0, 0.0, 0.0, 1.0, 0.0, ...) instead
of being frozen, confirming re-determinization is actually diversifying the
search instead of just adding call overhead.

**Third bug found — the confirmatory gate collapsed to 1/60 (1.7%), WORSE
than every prior attempt.** Root cause: `main.agent`/`_choose` mutates a
module-global stall-avoidance cache (`_STALL_MEMO`) that detects "have I
seen this exact decision+state before with no progress" and, on a repeat,
rotates to a different (often much worse) answer instead of repeating —
correct behavior across one real game's linear history, but not across
hundreds of interleaved simulated rollouts that legitimately revisit similar
decision fingerprints. Every simulation's calls to the real teacher were
polluting the SAME global dict, so later rollouts (and even the real
subsequent game turns after `choose()` returned, since the global was never
restored) saw false-positive "stall" triggers and got rotated into
essentially garbage answers. Notably: **the codebase already knew about this
exact hazard** — `score_options`' docstring says it is "side-effect-free
(never touches `_STALL_MEMO`)... safe to call repeatedly inside a search
tree," and `_main_phase_features`' docstring says the same ("search may
call this thousands of times per game"). Routing the rollout/opponent-model
through the real `main.agent` (the fix for bug #1) reintroduced exactly the
hazard `score_options` was built to avoid — bug #1's fix and this bug are in
tension, and both matter (see below).

**Fixed:** `training/nn/mcts.py`'s `_rollout` now saves the real game's
`_STALL_MEMO`, resets it to an empty dict at the start of every simulation
(each rollout is its own independent fictitious playout — a fresh memo is
the correct scope, not a bug), and restores the real game's memo when the
rollout returns, so search never pollutes the actual game's stall-tracking
and different simulations never contaminate each other. Smoke-tested next.

**Report relevance:** three real, distinct bugs found via targeted
diagnostics in one session (weak-teacher rollout, strategy fusion, global
state leakage across simulations) rather than blind hyperparameter tuning —
strong methodology-section material either way this resolves. The
`_STALL_MEMO` finding is also a nice concrete illustration of a general
principle for search-over-heuristic designs: a heuristic function's
purity/side-effect contract that's safe for one real sequential game is not
automatically safe to reuse inside a search process that calls it many times
over many hypothetical, non-sequential states.

**CLOSED, negative — the actual limiting mechanism found (2026-07-04):**
post-`_STALL_MEMO`-fix smoke test was still poor (1/12), prompting one more
advisor consult rather than a 5th gate. Key reframe: **every result since
the very first, simplest version has been flat-or-worse** (41.7% → 32.5% →
1.7% → 8.3%) — the `main.agent`-rollout and re-determinization changes,
despite being individually well-motivated, never demonstrably helped. The
unifying explanation, confirmed by a targeted 2-minute diagnostic (not
another full gate) on 3 fixed mid-game positions, 30 sims each:

- **`N` piled entirely on one action every time** (e.g. `[0,0,0,0,0,0,0,30,0,0]`)
  — PUCT exploration collapsed to pure exploitation after the first result,
  meaning the search never actually explores alternatives to the prior's
  top pick. In practice: MCTS ≈ `argmax(score_options)`.
- **Every terminating rollout across all 3 positions was a win — 90/90,
  zero losses, zero draws.** The simulated "opponent" (a random hand drawn
  from our OWN deck, piloted by our OWN Alakazam-strategy heuristic) is not
  a meaningful adversary — it's hapless enough that the rollout wins
  regardless of which root action was chosen. With no losses to distinguish
  good root choices from bad ones, **the rollout carries zero discriminating
  signal** — there was nothing for 150 sims to average into a useful value.

This is precisely the leaf-evaluator-quality check that should have gated
the build from the start (skipped in favor of the timing probe alone) — and
it lands on the same underlying problem as the AWR value-head result from
earlier the same day, just via a different mechanism: **a leaf/value signal
built only from the artifact being improved on (in AWR's case, a saturated
value head; here, a rollout against an opponent modeled by the same
heuristic) cannot supply the information needed to beat that artifact.**
Heuristic-guided PIMC without a genuinely discriminating value function is
therefore closed as a load-bearing negative result, the same shape as AWR:
**exceeding the v25c teacher via search needs either (a) real Kaggle-gated
MCTS expert iteration with a trained (non-degenerate) value/policy pair, or
(b) a materially stronger/adversarial opponent model in the rollout** — not
achievable within the "no net, no Kaggle compute" version tested here.

**Decision:** Escalating to the user rather than unilaterally choosing (a),
(b), or a pivot to Stage 3 belief-model work — this is the critical
decision point the autonomous `/goal` loop's own guidance reserves for the
user.

**User directed a Claude Fable consult; Fable's recommendation (2026-07-04):**
run option (b) — swap the rollout's hapless mirror-opponent for a real
adversarial opponent module (one of `opponents/{lucario,dragapult,
abomasnow,starmie}_agent.py`) — as ONE strictly time-boxed test: re-run the
diagnostic first (not a full gate), and if it shows a real signal, run
exactly one pre-registered gate. Regardless of outcome, pivot to Stage 3
Phase B afterward (already scoped, no dependency on this result, and its
archetype-deck output would make any future search opponent model more
honest). Defer the big Kaggle MCTS-expert-iteration option — its premise
(that search adds value here) is exactly what's unproven, and it depends on
a non-degenerate value function the AWR result says doesn't exist yet.

**Implemented:** `training/nn/mcts.py` now plays the rollout's opponent
turns via `opponents/lucario_agent.py`'s real `agent()` (env-configurable
via `MCTS_OPPONENT_MODULE`) instead of a mirror of our own heuristic/deck,
with hidden-zone filler drawn from Lucario's real deck list. Found and
isolated a second instance of the exact `_STALL_MEMO`-style hazard:
`lucario_agent.py` also has module-global mutable state (`plan`,
`pre_turn`, `ability_used`) that needed the same per-simulation
save/reset/restore treatment.

**Re-run diagnostic (3 positions, 30 sims each) — real, if partial,
improvement:** losses now appear (1/30 in 2 of 3 positions, vs 0/90 with
the mirror opponent), and one position shows genuine PUCT exploration
spread (a clean 15/15 split between two actions, vs total pile-up on a
single action everywhere with the mirror opponent). Not fully resolved —
2 of 3 positions still show near-total pile-up on one action — but this is
a qualitatively different, healthier pattern than the mirror-opponent's
total collapse, and clears the bar Fable set for proceeding to one
pre-registered gate.

**PRE-REGISTRATION (per Fable's plan):** heuristic-guided PIMC with a
Lucario adversarial rollout opponent vs the v25c teacher, `sims=150`,
50 games. **Gate: ≥55% win rate → real signal, worth a larger confirmatory
run before any ladder consideration. Below that → close the search-at-
inference line for this deck/session as a third load-bearing negative
result (alongside DAgger and AWR) and move to Stage 3 Phase B**, per
Fable's explicit "regardless of outcome" sequencing.

**RESULT: 0W-50L (0%) — decisively below the pre-registered bar.** A
complete shutout, nominally even worse than the pre-fix mirror-opponent
runs (1.7%, 8.3%), though at these low true rates a 0/50 draw is not
statistically distinguishable from "still ~2-8%, same broken regime" (e.g.
a true 3% rate has ~22% chance of landing exactly 0/50) — this reads as
the same underlying failure, not a new regression from the opponent swap.
The diagnostic's encouraging signal (losses appearing, one position's
visits splitting 15/15) evidently wasn't enough to move real end-to-end
game outcomes: 2 of the 3 diagnostic positions still showed near-total
PUCT pile-up on one action even with the adversarial opponent, and even
where exploration improved, one flipped loss out of 30 sims is a weak
correction against a heavily-weighted prior. **Per the pre-registration,
CLOSING Stage 5 search-at-inference (heuristic-guided PIMC, no net) as a
third load-bearing negative result, alongside DAgger's imitation-ceiling
and AWR's saturated-value-head findings** — all three converge on the same
theme: a value/policy signal built without genuinely external information
(a real trained value function, or a strong-enough opponent model) cannot
supply what's needed to exceed the teacher it's built from. Per Fable's
explicit sequencing, pivoting to Stage 3 Phase B next regardless of this
outcome — not attempting a fourth architectural change without a fresh
check-in.

**Report relevance:** clean three-way negative result for the report's
methodology section — DAgger (imitation asymptotes toward, never above,
teacher parity), AWR (saturated value head carries no per-decision
advantage signal), and now heuristic-guided PIMC (self-referential or
too-weak opponent models give the rollout nothing to discriminate on) all
tested against the same v25c teacher, same day, with root causes
diagnosed via targeted tests rather than blind tuning throughout. Genuinely
strong "why search-without-a-trained-value-function doesn't work here"
narrative material.

**Report relevance:** a third independent load-bearing negative result for
the 70%-axis narrative (after DAgger's imitation-ceiling and AWR's
saturated-value-head), all converging on the same methodological point
via three different mechanisms — imitation-family and self-referential-
evaluator search methods plateau at or below teacher parity on this
problem; only genuinely external signal (real adversarial self-play with a
trained value function, or search against real opponents) has a chance to
exceed it. Strong, cohesive material for the report's methodology section.

**Report relevance:** If this pans out after tuning, it's the load-bearing
positive result for the whole 70%-axis narrative (imitation family plateaus
at teacher parity per DAgger/AWR; search is what finally exceeds it). If it
doesn't pan out even after tuning, it's still citable — a third method family
(after DAgger, AWR) tested against the same teacher, strengthening the
"why search-in-the-loop is the AlphaZero-style answer, not more imitation"
methodology narrative either way.

---

## 2026-07-04 — Ladder replay bulk download + real meta-share survey

**Hypothesis:** Stage 3 Phase B needs real ladder archetype data — we only had
28 replay JSONs on disk, not enough to build an honest archetype library or
measure real meta share (as opposed to Phase A's synthetic 5-bot set).

**Method:** discovered the Kaggle CLI exposes `kaggle competitions episodes
<submission_id>` (list episode ids for a submission) and `kaggle competitions
replay <episode_id> -p <dir>` (download one episode's replay JSON) — not
previously used. Built `tools/download_replays.py`: enumerates all COMPLETE
submissions, lists their episodes (1,282 unique episode ids across 26
submissions), and downloads any not already on disk, skipping ones that are
(resumable — safe to re-run). Ran it in the background; it died twice
mid-run with no fatal error in the log (once after a burst of connection
timeouts, once cleanly with zero failures) — same signature as the machine
sleep/wake kills seen earlier this session during belief-data collection.
Resumed twice, and on the third run started hitting real `429 Too Many
Requests` from the Kaggle API. Per the user's explicit call ("I only care
about the top 50% percentile or so of bots so that's fine"), stopped there
rather than fighting the rate limit — the goal is a representative meta
sample, not exhaustive completeness.

**Result:** 652 new replays downloaded (680 total on disk incl. the original
28). `tools/meta_survey.py --all --csv training/meta_survey.csv` over all 680:

| Archetype | Count | Share |
|---|---|---|
| other/unknown | 176 | 25.9% |
| lucario | 111 | 16.3% |
| starmie | 82 | 12.1% |
| dragapult | 74 | 10.9% |
| alakazam | 71 | 10.4% |
| abomasnow | 53 | 7.8% |
| crustle | 53 | 7.8% |
| archaludon | 27 | 4.0% |
| bellibolt | 11 | 1.6% |
| rockets-mewtwo | 7 | 1.0% |
| raging-bolt | 7 | 1.0% |
| gardevoir | 4 | 0.6% |
| grimmsnarl | 3 | 0.4% |

This is a real update from the original 28-replay sample (archaludon had
looked tied for top share at 17.9%; with 680 replays lucario/starmie/
dragapult/alakazam all clearly outrank it, and archaludon settles at 4.0%).
The 25.9% `other/unknown` slice is itself information: a quarter of ladder
opponents don't match `tools/meta_survey.py`'s current `SIGNATURES` list at
all — Phase B's archetype library needs to either extend that signature set
or budget for a large honest `unknown` posterior mass.

**Decision:** stop the download here (rate-limited, and the user only needs
top-percentile-bot coverage, not exhaustive completeness); use this 680-replay
snapshot as Phase B's starting corpus. `tools/download_replays.py` remains
resumable if more replays are wanted later (e.g. after a cooldown on the
Kaggle API rate limit).

**Report relevance:** real ladder meta-share is direct material for the
report's meta-share figure/table (item 4, "meta-weighted expected win rate").
The `other/unknown` share is also evidence for Phase B's honest-coverage
framing ("coverage % of ladder opponents recognized").

---

## 2026-07-04 — Stage 3 Phase A: archetype classifier, 92.3% held-out, gate cleared

**Hypothesis:** the opponent's archetype leaks fast from public board state
(active/bench Pokemon, discard, energy types) — early enough and reliably
enough that a simple multinomial logistic regression beats a hand-written
key-card lookup, especially at turns 1-2 where partial evidence is all
either method has to go on.

**Method (plain English):** `training/belief/collect.py` plays `main.py`
(us) vs each of 4 opponent bots (lucario, dragapult, abomasnow, starmie) plus
a `main.py`-vs-`main.py` mirror (label=alakazam), 2000 games each. For every
game it walks the FULL step trace and extracts, from OUR OWN observation,
the opponent's public board state at each turn boundary: revealed card ids
(active/bench Pokemon + their pre-evolution chain + tools + discard pile —
a monotonically-growing public snapshot, not a windowed log parse) plus
simple counts (bench/discard/hand size, prizes taken, energy-type counts).
`training/belief/train.py` fits a multinomial `LogisticRegression` over a
`DictVectorizer` multi-hot of these features, split by game (not row) to
avoid same-game leakage across turns.

**Data:** 93,006 total rows across the 5 labels (lucario 18,685; dragapult
22,137; abomasnow 17,912; starmie 8,235 — shorter games, not a bug, verified
via turn-count distribution; alakazam 26,037).

**Result — gate cleared decisively:**
- Overall held-out accuracy: **92.3%** (pulled down almost entirely by
  turn-0 rows, before any cards are played, where there's genuinely no
  evidence yet — 28.5% there, matching chance-ish expectations).
- **Accuracy by turn: turn 1 = 99.1%, turn 2+ = ~100%** — clears the design
  doc's ">90% by turn 3" target by two turns early.
- **Beats the key-card baseline at both gate turns** (baseline derived from
  training data, not hand-guessed): turn 1 classifier 99.1% vs baseline
  80.2%; turn 2 classifier 100% vs baseline 85.7%.
- Confusion matrices at turns 1/2/3/5: diagonal from turn 1 on, one small
  leak (14 alakazam→dragapult at turn 1) fully resolved by turn 3.
- Posterior entropy collapses from ~1.55 nats (turn 0, near-maximum
  uncertainty over 5 classes) to ~0 by turn 2, mirroring the accuracy curve.

**Caveat, stated plainly for the report:** this is the "easy" 5-bot
classification with fixed, maximally-distinctive decklists — not the real
ladder, which has partial/noisy evidence and off-meta decks. Phase B (real
ladder archetype library + explicit `unknown` mass) is the honest test and
is unstarted. 92% here validates the pipeline and feature design; it is not
a ladder-readiness claim.

**Decision:** Phase A done, gate cleared. Weights exported
(`training/belief/belief_weights.json`, plain dict — pure-python dot product
at inference, no sklearn needed) but NOT yet wired into `main.py`
(`opp_likely_ace_spec` still hardcoded `True`) — that's Phase C, gated on
Phase B's real-archetype library existing first per the phase plan.

**Side-finding (not part of this experiment, logged since it corrects prior
project state):** `opponents/dragapult_agent.py` does not actually crash
locally — confirmed 0/20 errors in a direct spot-check and 22,137 clean rows
collected here. The existing "crashes 100% of local games" claim
(`CLAUDE.md` item 6) is stale; it has a working try/except fallback around
its Kaggle-only `cg.api` import. The gauntlet's dragapult column may be
trustworthy again — not re-verified via a full gauntlet re-run, flagged for
the user to decide.

**Report relevance:** target figure #1 (archetype-identification accuracy by
turn) is now real data, not a placeholder — this is the report's stated
"originality centerpiece" (`docs/belief-model.md`) delivering its first
concrete result.

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

**Update, same day — β sweep done, line closed:** ran the one pre-registered
follow-up, `--awr-beta 0.5` (more aggressive reweighting, `awr_norm=6.95` vs
β=1.0's 1.83) on the same self-play corpus → `ptcg_awr_beta0.5.pth`. Gated:
**11.5% ± 3.1% vs teacher** (worse than β=1.0's 15.8% — more aggressive
weighting hurt, not helped) and **53.7% ± 4.9% vs seed** (CI still spans
50%, not significant, though nominally above β=1.0's 47.7%). This rules out
"just needed a stronger β" as the fix.

| β | vs teacher | vs seed (`dagger_r2`) |
|---|---|---|
| 1.0 | 15.8% ± 3.6% | 47.7% ± 4.9% (tied) |
| 0.5 | 11.5% ± 3.1% | 53.7% ± 4.9% (not significant) |

**Final decision:** Stage 2 direct-self-play AWR line is **closed, negative
result**, per the pre-registered "1-2 follow-ups then stop" rule. Neither β
exceeds teacher parity; the more aggressive setting made vs-teacher
performance worse while not producing a significant vs-seed gain either —
consistent with the value-head-saturation working theory (advantage signal
carries little per-decision nuance beyond terminal outcome on this deck, so
AWR reweighting mostly just overfits to a winner-biased slice rather than
learning genuinely better actions). Advancing past teacher parity via
self-play needs Kaggle-gated MCTS/search-in-the-loop; alternatives in the
meantime are Stage 3 belief-model work (parallel, doesn't depend on this
result) or treating `ptcg_dagger_r2.pth`'s 81.9% fidelity as the practical
ceiling of the imitation-family approach while `main.py` (v25c heuristic)
remains the ladder submission. Both β results are real, citable material for
the report's ablation table (target figure #4) and plateau-without-search
narrative (target figure #3) — a load-bearing negative result, not a wasted
run. Full detail: `docs/nn-training.md` Stage 2 section.

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
**Update, ~2.5h after the correction:** another pull — isolation candidate
878→841.5, plain revert 714.1→751.7. Both readings are narrowing toward
each other and toward v25c's 818.3, more consistent with genuine settling
now than pure noise, though one more data point still isn't proof. Holding
the same stance: no ship, no conclusion, until the swings visibly damp out
further (scheduled wakeup will re-check).

**Update, ~3h after the correction:** isolation candidate 841.5→766.9,
plain revert 751.7→735.9 — swing amplitude visibly narrowing (last two
deltas -74 and -16, vs. the earlier +278/-37/-16). Isolation has now read
above the revert in 3 of 4 pulls. Still holding: not calling this settled
or drawing a conclusion yet, but the trend is consistent with genuine
convergence rather than pure noise. Will re-check per the scheduled wakeup.

**Bug-check (code review, no training/collection): confirmed dmc_agent.py's
action selection is not the cause of the low numbers.** Diffed against
`selfplay_agent.py` (the code path underlying the exploiter's already-
verified 10.6% baseline) — both share the identical `encode_batch`/`clamp`
pipeline; the only differences are argmax-over-Q vs softmax-sampling-over-
policy-logits and which checkpoint is loaded, exactly as intended. No
indexing/decode bug. The DMC trajectory (1.0%/1.7%/2.5%/2.0%) is a genuine
signal, not an artifact — consistent with the literature's warning that
cold-start Q-regression is markedly less sample-efficient early on than
cross-entropy imitation. This closes the cheap/available diagnostic space
for this line without new data collection; nothing further to check before
the 2026-07-19 checkpoint.

**Update: isolation candidate held exactly at 766.9 across two consecutive
pulls (first zero-movement reading) while the plain revert continues to
drift (735.9->720.7). This first stable read is a good sign of genuine
settling. Still one stable read is not two -- want to see the isolation
number hold again before treating it as final.

## 2026-07-05 — v28 shipped: board-thinning + Phase C + confidence recalibration (matched-age decision)

**Decision basis:** the isolation candidate (`54354935`) and plain v25c
revert (`54354862`) were shipped simultaneously that morning, giving a
genuinely matched-age comparison (unlike the earlier same-day mistake of
comparing a 2-day-old v25c read against minutes-old v26/v27 reads). Both
showed large early swings that visibly narrowed over ~3 hours (revert:
600→468→714→751.7→735.9→720.7→720.7, holding steady twice; isolation:
600→878.9→841.5→766.9→766.9→746.9). Final matched-age comparison: isolation
746.9 vs. revert 720.7, a modest +26-point (~3.6%) edge — consistent in
direction with the board-thinning fix's positive offline gates (54.3%,
56.0%), though the swings hadn't fully flatlined to zero movement.

**Combined with the separately-collected, ladder-score-independent
behavioral evidence** (Phase C's classifier + wall-anticipation confirmed
correct against 725 real replays; the confidence-recalibration candidate
data-driven from the same replays), shipped `main.py` = board-thinning fix
+ Phase C + recalibration as v28 (submission `54356683`).

**Explicitly not repeating the earlier mistake:** this is a matched-age
comparison for the base fix, plus independent replay-behavioral evidence
(not ladder score) for the additions — a different and more defensible
evidentiary basis than the earlier single-read panic-revert. **Still
flagged honestly:** v28 as a whole (fix+PhaseC+recalibration combined)
has no ladder read of its own yet at ship time; watch its score over the
next few hours the same way, and don't over-read early volatile numbers.

---

**v28 first read: 600.0** -- the same fresh-submission floor seen on every
prior first read (v25c-revert, v25c+boardthin isolation, all started at
600.0). Not a signal yet, expected pattern. Isolation candidate (54354935)
confirmed settled: 746.9->748.0, essentially unchanged. Holding for v28's
own score to climb/settle before drawing any conclusion about it.

**v28 second read: 709.1 (up from 600.0)** -- following the expected
climb pattern seen in every prior fresh submission. Isolation candidate
holds stable: 748.0->750.3. Still too early to compare v28 against
anything; continuing to wait for it to finish climbing/settling.

**v28 third read: 910.7 (up from 709.1)** -- notably ABOVE v25c's original
settled 818.3. Per the established pattern this session (every submission
has overshot before narrowing -- the isolation candidate hit 878.9 before
settling ~750-756), this is NOT being treated as a confirmed "beats v25c"
result yet. Isolation candidate holds essentially stable: 750.3->756.5.
Need at least one more read, ideally showing v28 stabilize near or above
this level rather than falling back, before drawing any conclusion.

**v28 fourth read: 829.8 (down from 910.7)** -- still moving, not settled
despite the prior read looking stable. Now very close to v25c's original
818.3 (within ~12 points), suggesting possible convergence toward
near-parity rather than a decisive win -- but with an 81-point swing just
observed, one more read is needed before concluding even "roughly even."
Isolation candidate held steady a second consecutive time (743.1,
unchanged) -- that one looks genuinely settled now. Continuing to hold on
v28 specifically.

**v28 confirmed stable: 829.8 held across two consecutive pulls.**
Isolation candidate confirmed stable: 743.1 held across three consecutive
pulls. **This is now a real, matched-methodology conclusion (not a single
noisy read):** v28 (board-thinning + Phase C + confidence recalibration)
settles at 829.8, modestly above v25c's original settled 818.3 (+11.5,
~1.4%) and clearly above the board-thinning-only isolation candidate's
743.1 (+86.7, ~11.7%). This suggests Phase C + the recalibration fix add
real further improvement on top of the board-thinning fix alone, and the
combined v28 agent is, on the best current evidence, at least on par with
and modestly ahead of v25c. **Caveat:** this is one settled comparison, not
a large-n statistical test — the margin over v25c (1.4%) is small relative
to the natural game-to-game/day-to-day variance this ladder has shown
throughout this session, so treat "v28 is a real, if modest, improvement
over v25c" as the working conclusion, not a certainty. v28 remains the
active submission; no further ship needed today.

**Further reads confirm a stable RANGE rather than one exact point (expected
given real match-to-match variance, not the earlier unbounded settling
noise): v28 829.8->840.4 (~830-840 band), isolation 743.1->718.1 (~720-745
band).** Both bands are now narrow relative to the earlier 100-300+ point
swings. Conclusion stands: v28 sits modestly above v25c's 818.3, and
clearly above the board-thinning-only isolation candidate.

## 2026-07-05 — CORRECTION #2: v28 was NOT actually settled; DMC more-epochs also negative

**v28 ladder score correction:** the prior "CONFIRMED SETTLED: 829.8, held
across 2 consecutive pulls" conclusion was premature. A subsequent pull
showed v28 at **724.4** — a swing of -116 points, back below both v25c's
818.3 and even below the isolation candidate's own concurrent read
(753.3). Two consecutive identical reads is not the same as true
convergence; this ladder's `publicScore` may not converge to a fixed
value at all within the timeframe available (possibly a genuinely live
metric reflecting an ever-changing pool of opponents, not settling toward
one true number the way a fixed offline benchmark would). **Retracting
the "v28 modestly beats v25c, confirmed" claim** — the honest state is:
v28, the isolation candidate, and v25c's original 818.3 are all within a
noisy band roughly 720-910 that has not stopped moving after ~4 hours of
observation. No confident ranking between them is currently possible from
`publicScore` alone. This is now the second time this session a "looks
settled" read turned out not to be (the earlier age-confound correction
being the first) — **the corrected methodology note stands even more
strongly: single or even double-confirmed `publicScore` reads cannot be
trusted for ranking submissions on this ladder within a same-day
timeframe.** A real ranking needs either much longer observation (days,
matching how long the original v25c reading had to mature) or a
direct large-n offline A/B (which IS trustworthy per this project's
Design Principle #1 for relative comparisons, even though single-model
`publicScore` reads are not).

**DMC more-epochs (8 vs. round 3's 2, same existing data, no new
collection): 2.0% (4/200) vs. frozen v25c** — statistically identical to
round 3 (2.5%) and the oversampling variant (2.0%), despite in-distribution
val_sign_acc jumping substantially (0.9377→0.9684). **This is another
instance of the "wrong measured quantity" pattern recurring throughout
this session** (the oracle-critic gate script that never passed oracle
features; temp-1.0 vs argmax exploiter measurement) — in-distribution
validation accuracy on the training data's own outcome labels does not
predict actual greedy-policy game performance, almost certainly because it
reflects memorizing/overfitting the specific self-play distribution rather
than learning transferable Q-value structure. **Confirms DMC's bottleneck
is data-limited (needs more diverse self-play rounds), not undertrained**
— training longer on the same narrow data just overfits harder without
improving actual play. No further training-recipe tweaks on existing data
are likely to help; the only lever left is more collection rounds, which
stays paused to the 2026-07-19 checkpoint per the standing pre-registration.

---

**Pivoted to a trustworthy measurement: large-n offline A/B (v28's exact
main.py vs pure v25c, 400-game mirror, seats alternated) — 50.2% ± 4.9%
(95% CI). Statistically indistinguishable from parity.** This is the
first low-noise, controlled comparison of the actual combined v28 changes
against v25c (the live ladder reads are not low-noise, per the correction
above). Honest current state: **there is no confirmed evidence v28 beats
v25c.** The board-thinning fix alone showed a real edge in its own
isolated tests (54.3%, 56.0%), but combined with Phase C + recalibration
in a mirror match against pure v25c, that edge is not observable at n=400
— plausibly because (a) Phase C's payoff specifically depends on facing
non-mirror decks with walls or misclassified archetypes, which a v25c
mirror match doesn't reliably surface (the same caveat noted for the v26
gate back on 2026-07-04), diluting the combined test's power to detect
the board-thinning fix's real but narrow-state effect, or (b) genuine
noise at this sample size. **Decision:** stop treating any individual
live `publicScore` read (or even a same-value pair of reads) as
conclusive for this project going forward — it has now been shown wrong
twice. The offline 50.2%±4.9% is the most trustworthy number available
right now, and it says: no confirmed win, not a confirmed loss either.
v28 remains shipped (harmless, reasoned basis at the time); no further
ships today without either much longer ladder observation or a properly
powered offline test that specifically includes non-mirror wall/unknown-
archetype opponents (which none of today's offline gates did).
