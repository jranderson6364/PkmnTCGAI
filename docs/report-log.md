# Report Log — Pre-registered Trials & Results

**Last updated:** 2026-07-03

> Merge note: this file was created on a remote branch that predates the local
> `training/` work. If a local `docs/report-log.md` already exists, merge these
> sections into it — everything below is additive (new target figures/tables +
> two pre-registration entries), nothing supersedes earlier entries or figures
> #1–#5.

Every trial that feeds the report is **pre-registered here before any games run**:
hypothesis, protocol, decision rule, dated. Results entries land in the same
session as the run. Evidence rule (see Mission in `docs/competition-strategy.md`):
no claim in the report without an entry here behind it.

---

## Target figures & tables (additions)

- **Figure #6 — robustness panel:** per-anchor matchup table, per-seat win split,
  cross-run (seed-to-seed) variance, opening-hand-quality conditional win rate.
- **Figure #7 — pro-metrics panel:** consistency table (`tools/deck_math.py`:
  mulligan %, key-combo-by-turn odds per deck), prize-trade efficiency (prizes
  taken per prize conceded), setup speed (mean first-attack turn), meta-weighted
  expected win rate.
- **Table A — deck bake-off:** 5 decks × (tier-1 BT rating, tier-2 BT rating,
  head-to-head vs Alakazam ±CI, meta-weighted win rate, mulligan %, prize-trade
  efficiency).
- **Table B — method comparison:** one row per method (random, generic-greedy,
  heuristic, bc, dagger-r2, + future), same gauntlet protocol, with a one-line
  keep/reject verdict each.

---

## Pre-registration — Deck bake-off (Stage 0c)

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

## Pre-registration — Method bake-off

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

*(Results entries append below, dated, in the same session as each run.)*
