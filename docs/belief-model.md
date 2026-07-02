# Belief Model — Design & Plan (Stage 3)

*Predicting the opponent's hidden information: `P(archetype | their observed
plays, turn)`, then sampling their unseen zones from the inferred archetype's
decklist. The report's originality centerpiece. This file is the canonical home
for the design, the phase plan, and (as they land) the results.*

**Last updated:** 2026-07-02

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

## Phase B — Ladder Reality (the honest version)

The ladder is not 4 bots. Two additions:

1. **Archetype library from ladder replays.** Download our submissions'
   episodes via the Kaggle API (only 8 replay JSONs on disk today — need
   bulk download into `replays/`). Tag each opponent by key cards into the
   known meta list (Crustle, Gholdengo, Raging Bolt, Charizard, Bellibolt,
   Alakazam mirror, …) — `docs/matchups.md` has the meta list. Each tagged
   archetype needs a representative 60-card list (from `EN_Card_Data.csv` +
   observed cards) before the determinization sampler can use it.
2. **Honest `unknown` handling.** Posterior mass on `unknown` when evidence
   matches nothing known; consumers fall back to current behavior (ace-spec
   assumed, placeholder determinization). Report this honestly: coverage % of
   ladder opponents recognized, by month.

## Phase C — Consumers

1. Wire posterior → `opp_likely_ace_spec` + Mist/Rocky anticipation in
   `main.py` (weights dict embedded; pure Python). Gate: 400-game A/B vs
   pre-wiring main.py — must not regress; ladder confirm. Kills outstanding
   item on the hardcoded flag.
2. Determinization sampler for the Stage 5 MCTS spike (Kaggle-gated with the
   rest of search).
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
