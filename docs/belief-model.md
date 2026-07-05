# Belief Model — Design & Plan (Stage 3)

*Predicting the opponent's hidden information: `P(archetype | their observed
plays, turn)`, then sampling their unseen zones from the inferred archetype's
decklist. The report's originality centerpiece. This file is the canonical home
for the design, the phase plan, and (as they land) the results.*

**Last updated:** 2026-07-05 (Phase C consumer 1 gate passed + shipped as v26; real-replay behavioral check DONE — classifier + wall-anticipation confirmed correct on 725 real replays, but found and fixed a confidence-miscalibration issue on unrecognized decks, shipped as part of v28; determinization sampler deprioritized, no live consumer)

---

## What It Is (and Is Not)

A **belief model** infers *hidden information* — which archetype the opponent is
piloting and what's plausibly in their hand/deck/prizes — from what they've
visibly done. It is **not** a win predictor (that's the value head; see
`docs/report-log.md` glossary). Three consumers, in deployment order:

1. **Heuristic wiring (Stage 0b channel):** replace `opp_likely_ace_spec`
   (hardcoded `True` since v14) and sharpen Mist/Rocky-wall anticipation —
   e.g. expect Rocky Energy the moment the opponent reveals a Lucario line,
   before it's attached. Teacher upgrade → compounds through DAgger.
2. **Determinization sampler (Stage 5):** `search_begin` currently fills the
   opponent's hidden zones with `[1072]*n` placeholders. Replace with: inferred
   archetype's 60-card list minus everything already observed, sampled into
   hand/deck/prizes. Ablation: placeholder vs belief determinization.
3. **Net input (optional, Stage 2+):** archetype posterior as a small feature
   vector appended to the encoder, so policy/value condition on the matchup.

---

## Why It's Tractable

Card games leak archetype fast: decklists are ~60% archetype-defining. The
engine gives us, from OUR seat's observation, everything needed:

- **Log type 10 (PLAY):** every card the opponent plays, with cardId
- **Log types 8/9 (ATTACH/EVOLVE):** their energy types and evolution lines
- **Board state:** their active/bench pokemon ids, attached energies, tools
- **Attack usage:** attackIds they've used

By turn 2–3 a Lucario deck has benched Riolu and attached Fighting energy;
a Dragapult deck has shown Dreepy. The question isn't *whether* this works —
it's how early, how calibrated, and how well it degrades on off-meta decks.

---

## Model

**Start dumb, stay dumb until forced otherwise:**

1. **Baseline (must-beat):** hand-written key-card lookup (Staryu=1030 →
   Starmie, etc.). One dict. This is also the report's ablation floor.
2. **Main model: multinomial logistic regression** (or naive Bayes) over a
   multi-hot vector of observed opponent card ids + turn number + simple counts
   (energy types attached, bench size, prizes taken). Trained offline with
   sklearn; **exported as a plain dict of weights** and evaluated in pure
   Python — the submission `main.py` must stay dependency-free and fast
   (timeout = instant loss).
3. Upgrade to anything fancier only if the calibration curve says the simple
   model is the bottleneck. (It won't be, for the 4-bot label set.)

**Output:** posterior over archetype labels + an explicit `unknown` mass
(see Phase B). Never a hard argmax — consumers use the posterior:
the heuristic thresholds it, the sampler samples from it.

---

## Phase A — 4-Bot Classifier + THE Figure (local, ~1 day)

Labels: `lucario, dragapult, abomasnow, starmie` (+ `alakazam` from mirror
games — 5 classes). Data: `harness.run_matches` vs each bot, ~2,000 games per
label, both extractable seats; every TURN (not decision) emits one training row
of cumulative observed-opponent features. Wall time: a few hours, machine only.

Deliverables:
- `training/belief/collect.py` — labeled feature rows → `belief_data.pkl.gz`
- `training/belief/train.py` — fit + calibration; exports `belief_weights.json`
- **Figure: archetype-identification accuracy by turn number** (with per-class
  curves + posterior entropy). Target shape: >90% by turn 3 on the 4-bot set.
- Confusion matrix at turns 1/2/3/5.

Gate: beats the key-card baseline at turns 1–2 (where partial evidence — an
energy type, a trainer card — is all you have; that's where ML earns its keep
over the lookup).

**DONE 2026-07-04.** `training/belief/collect.py` collects labeled rows by
running `main.py` (us) vs each labeled bot via `harness.run_matches`, walking
the FULL step trace of each game, and extracting from OUR OWN observation the
opponent's public board state (active/bench Pokemon ids + preEvolution chain
+ tools, discard pile, energy-type counts on their Pokemon) at every turn
boundary. This sidesteps log-window ambiguity entirely — board state is a
complete, monotonically-growing public snapshot regardless of how the
engine's `logs` field windows/resets, and it's exactly what `main.py` sees
at ladder inference time (no extra plumbing needed for Phase C). 2000 games
per label (lucario, dragapult, abomasnow, starmie, alakazam-mirror) →
93,006 total rows (`training/belief/belief_data_*.pkl.gz` per label +
merged `belief_data.pkl.gz`). `training/belief/train.py` fits a multinomial
`LogisticRegression` over a `DictVectorizer` multi-hot of observed card ids +
turn + simple counts (bench/discard/hand size, prizes taken), split by
group (game) to avoid same-game leakage across turns.

**Results:**
- **Overall held-out accuracy: 92.3%** (dragged down almost entirely by
  turn-0 rows, before any cards are played — expected, no evidence yet).
- **Accuracy by turn: turn 0 = 28.5%, turn 1 = 99.1%, turn 2+ = ~100%** — the
  design doc's ">90% by turn 3" target is cleared by turn 1.
- **Beats the key-card baseline at both gate turns:** turn 1: classifier
  99.1% vs baseline 80.2%; turn 2: classifier 100% vs baseline 85.7%.
- **Confusion matrices (turns 1/2/3/5):** essentially diagonal from turn 1
  on — the only leak is 14 `alakazam`→`dragapult` misclassifications at
  turn 1, resolved by turn 3 (0/1 leaked).
- Figure: `training/belief/belief_accuracy_by_turn.png` (accuracy-by-turn +
  posterior-entropy-by-turn panels — entropy collapses from ~1.55 nats at
  turn 0 to ~0 by turn 2, mirroring the accuracy curve).
- Weights exported (not yet wired into `main.py` — that's Phase C):
  `training/belief/belief_weights.json` (plain dict: classes, feature names,
  coef, intercept — pure dot-product at inference, no sklearn needed).

**Caveat — this is the "easy" 5-bot classification, not the ladder.** These
5 opponents each play one fixed, maximally-distinctive decklist; Phase B
(real ladder archetypes, partial/noisy evidence, an explicit `unknown`
class) is the honest test and hasn't been attempted yet. Don't over-read
92% as "solved" — it's the correct, expected result for a clean synthetic
label set and validates the pipeline/feature design before spending effort
on the harder Phase B problem.

**Side-finding, not directly part of this task:** `opponents/dragapult_agent.py`
does NOT crash locally — it has a try/except fallback around its `cg.api`
import and ran cleanly (0/20 errored in a direct spot-check, 22,137 real
rows collected here). This contradicts the "crashes 100% of local games"
claim in `CLAUDE.md` item 6 / `training/gauntlet_results.csv`'s dragapult
column notes — that claim needs correcting, and the gauntlet's dragapult
column may be trustworthy again (not re-verified via a full gauntlet re-run
here; flagged for the user to decide whether to re-run it).

## Phase B — Ladder Reality (the honest version)

The ladder is not 4 bots. Two additions:

1. **Archetype library from ladder replays.** ~~Download our submissions'
   episodes via the Kaggle API (only 8 replay JSONs on disk today — need
   bulk download into `replays/`)~~ **Blocker resolved 2026-07-04:**
   `tools/download_replays.py` uses `kaggle competitions episodes
   <submission_id>` + `kaggle competitions replay <episode_id>` (previously
   undiscovered CLI subcommands) to bulk-download episode replays,
   resumable (skips ids already on disk). 680 replays now on disk (up from
   28), across 26 complete submissions / 1,282 total known episode ids —
   stopped short of the full set after hitting Kaggle API rate limiting
   (`429`) and per the user's call that top-percentile-bot coverage is
   sufficient, not exhaustive completeness. Real meta share via
   `tools/meta_survey.py --all --csv training/meta_survey.csv`: lucario
   16.3%, starmie 12.1%, dragapult 10.9%, alakazam 10.4%, abomasnow 7.8%,
   crustle 7.8%, archaludon 4.0%, bellibolt 1.6%, rockets-mewtwo 1.0%,
   raging-bolt 1.0%, gardevoir 0.6%, grimmsnarl 0.4% — and **25.9%
   other/unknown** (opponents `meta_survey.py`'s `SIGNATURES` list doesn't
   recognize at all). Full writeup: `docs/report-log.md` 2026-07-04 "Ladder
   replay bulk download + real meta-share survey" entry.

   **Signature extension DONE 2026-07-04:** clustered the unknown slice's
   non-generic revealed cards and found two real gaps — pre-evolution
   aliases for archetypes whose ace hadn't evolved yet when a short game
   ended (`snover`→abomasnow, `dreepy`/`drakloak`→dragapult), and one
   genuine new archetype standalone in the data (`kyogre`, 13 replays).
   Added to `tools/meta_survey.py`'s `SIGNATURES`; unknown dropped
   **25.9% → 21.3%** (176 → 145 of 680 replays). Re-clustered the remaining
   145 and found **no further common clusters** — 144 of them each have a
   unique/scattered set of minor cards (only 2-3 tiny repeat groups of size
   ≤2), i.e. a genuine long tail of rare decks, not a few more signatures
   away from being covered. This is the honest Phase B item 2 finding
   below, not a to-do.

   **Archetype library DONE 2026-07-04:** `tools/build_archetype_decks.py`
   reconstructs a representative 60-card list per archetype from real
   replay evidence — counts every card id revealed across an archetype's
   tagged games, weights assumed copy-count by how consistently each card
   appears (≥60% of games → 4 copies, ≥35% → 3, ≥15% → 2, else 1), capped
   at 60 total. Four of the biggest archetypes (lucario, dragapult,
   abomasnow, starmie) already have **exact** decklists via the official
   Kaggle sample bots in `opponents/*_agent.py` — skipped, no need to
   reconstruct. Built reconstructed lists for the rest:
   `training/archetype_decks.json` (crustle 53 replays, archaludon 27,
   bellibolt 11, kyogre 11, raging-bolt 7, rockets-mewtwo 7, grimmsnarl 3,
   gardevoir 4 — all with the ace card at or near 100% game-presence,
   plausible support/energy tails; kyogre's list is honestly short,
   30/60, reflecting genuinely thin evidence rather than fabricated
   filler). charizard/gholdengo/pidgeot-control/snorlax-stall/terapagos:
   0 matching replays in the current 680 — either rare on-ladder or the
   signature needs a revealed-card example we haven't hit yet; left
   unbuilt rather than guessed. `main.DECK`-mirror ("alakazam") also got
   reconstructed as an incidental cross-check against the known real v23
   `deck.csv` — not the deliverable, just a free sanity data point.
2. **Honest `unknown` handling.** Posterior mass on `unknown` when evidence
   matches nothing known; consumers fall back to current behavior (ace-spec
   assumed, placeholder determinization). Report this honestly: coverage % of
   ladder opponents recognized, by month. **2026-07-04 finding:** current
   honest coverage is 78.7% (535/680) after the signature extension above;
   the remaining 21.3% is a real long tail of rare/varied decks each
   revealing only a handful of non-distinguishing cards before the game
   ended, not a gap that more signature engineering closes cheaply. Report
   this figure as-is — it is the honest ceiling for revealed-card-based
   classification on short/early-ending games, not a bug.

## Phase C — Consumers

1. Wire posterior → `opp_likely_ace_spec` + Mist/Rocky anticipation in
   `main.py` (weights dict embedded; pure Python). Gate: 400-game A/B vs
   pre-wiring main.py — must not regress; ladder confirm. Kills outstanding
   item on the hardcoded flag.

   **WIRED 2026-07-04 (gate pending):** `_belief_posterior(opp, turn)` in
   `main.py` — embedded Phase A weights (~4KB), feature extraction mirrors
   `collect.py`, pure + exception-safe (failure → pre-Phase-C behavior).
   Consumers: (a) `opp_likely_ace` = observed OR conf<0.8 OR archetype ace
   rate ≥0.35 (rates from a new 679-replay tech survey — see report-log
   Phase C entry); (b) `mist_threat` = wall revealed anywhere / Crustle line
   seen / unconfident read by turn 2+ — drives Boss-chip savoring and
   bench-wall Enhanced Hammer (36 vs base 3). Key survey fact shaping the
   design: walls are teched by crustle (35.8%) and the unknown tail (29.0%),
   NOT the 5 classifier archetypes (0-3%) — so anticipation keys on
   crustle/unknown evidence, not the posterior argmax. **Gate PASSED
   2026-07-04: 50.7% ± 4.9% over 400 games, 0 errors (non-regression;
   mirror-by-construction on the anticipation path). Shipped for ladder
   confirm — v26 (see ladder_history.csv).**

   **Real-replay behavioral check DONE 2026-07-05** (`docs/report-log.md`
   2026-07-05 "Phase C real-replay behavioral check" entry): ran the
   shipped `_belief_posterior` against 725 real ladder replays (ground
   truth via `tools/meta_survey.py`'s signature classifier). Classifier-
   archetype agreement 82.4% (360/437) — reasonably close to Phase A's
   92.3% (that number was the easier 5-fixed-bot setup) and above the
   78.7% honest recognition ceiling. Wall-anticipation false-positive rate
   on the 5 non-wall archetypes: 0.9% (4/437) — correctly quiet. Crustle
   true-positive: 85.7% (48/56). **Real miscalibration found:** only 39.3%
   (64/163) of true `other/unknown` games correctly read below the 0.8
   confidence threshold — the `b_conf<0.8` fallback under-triggers on
   ~61% of the ladder's unrecognized long tail, a known general failure
   mode (softmax classifiers are often confidently wrong on OOD inputs,
   not appropriately uncertain). **Fix drafted + validated same session:**
   swept the threshold against the same 725-replay data — known-archetype
   confidence is extremely peaked near 1.0 (p25=0.992), so raising the
   threshold 0.8→0.97 nearly doubles the unknown-catch-rate (39.3%→60.1%)
   for only a modest false-low increase on known archetypes (19.2%→21.1%,
   a safe direction since it just triggers the conservative default).
   ~10.3% of all 725 games have their fallback behavior flip under this
   change. Shipped as part of **v28** (submission 54356683, 2026-07-05,
   ladder score pending) alongside the board-thinning fix.
2. Determinization sampler for the Stage 5 MCTS spike (Kaggle-gated with the
   rest of search). **Deprioritized 2026-07-05** per Fable design consult
   (`docs/report-log.md` 2026-07-05 "DESIGN DECISION" entry) — its only
   consumer (determinized search) is closed with a named cause; no live
   consumer justifies building it now.
3. Ablation runs for the report: key-card baseline vs classifier; placeholder
   vs belief determinization (once search exists).

---

## Decisions Taken (defaults — flag disagreement early)

| Decision | Choice | Why |
|----------|--------|-----|
| Label set (Phase A) | 4 bots + alakazam-mirror | What we can generate labeled data for today |
| Model | Logistic regression, exported weights | Submission must be dependency-free; interpretable for the report |
| Row granularity | Per TURN, cumulative features | The consumer asks "what do I believe NOW"; turns, not decisions, are the natural clock |
| Inference in main.py | Pure-python dot product | Timeout = loss; no sklearn at inference |
| Unknown class | Explicit posterior mass | Ladder coverage is partial; honest fallback beats confident nonsense |
| Ladder replays | Bulk-download via Kaggle API (Phase B blocker) | 8 on disk; need hundreds for the library |
