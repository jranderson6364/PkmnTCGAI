# Endgame Plan — 2026-07-23 to competition close

*The plan from the public-notebook survey to the Strategy report deadline.
Supersedes `docs/next-session-plan.md` (that file is a closed experiment record,
kept as-is). Evidence base: `docs/competitor-notebook-survey.md`.*

**Last updated:** 2026-07-23

---

## The calendar we are running against

**VERIFIED 2026-07-23 against the official competition timeline page.**

| Date | Days out | Event |
|---|---|---|
| 2026-07-23 | 0 | today |
| **2026-08-09** | 17 | Entry deadline **and** team-merger deadline (rules must be accepted — we already have) |
| **2026-08-16** | 24 | **Final Submission Deadline** — submissions lock |
| 2026-08-17 → ~08-31 | 25–39 | games keep running on the locked pair; leaderboard finalises at the end |
| **2026-09-06** | 45 | Hackathon/Strategy report due (separate competition page) |

Two mechanics confirmed from the rules page that change how we ship:

- **Only the latest 2 submissions are active**, and **the leaderboard shows the
  best of them**. Our current active pair is v30-exp (637.8) and the *weaker*
  v29d copy (620.5); the 708.9 copy is third-latest and therefore inactive. That
  is exactly why the team reads 637.8.
- **Games continue for ~2 weeks after the Aug 16 lock.** The final pair keeps
  playing, so σ shrinks after we stop being able to intervene. Whatever is in
  those two slots on Aug 16 is what the final ranking is built from.

---

## Framing: what we are actually optimizing

**CORRECTION (2026-07-23), read this before planning anything.** An earlier draft
of this plan said the ladder was "instrumental, not the goal" because the money
sits in the Strategy/Hackathon track. **The official rules say otherwise:**

> "The Competition track itself does not include monetary prizes. However,
> participants who submit a report to the Hackathon track will be eligible for
> prize awards. **Final rankings for Hackathon prizes will be determined based on
> both the Competition leaderboard performance and the Hackathon evaluation.**"

So ladder performance is a **direct input to prize ranking**, not merely a
credibility signal. The two objectives are coupled, and the earlier framing
under-weighted the ladder.

**What stays true:** top-8 on the ladder is not reachable from 637.8 in 24 days
against a 1114.1 cutoff and an ~88-point noise floor. A plan that targets ladder
rank alone is still a plan to fail.

**What changes:** ladder work is no longer the cheap side-quest. Revised
objectives, both first-class:

> **Objective A — leaderboard performance.** Close as much of the piloting gap as
> the calendar allows. Realistic target 800–900, i.e. at or above the public
> Alakazam agents rather than 140 below them.
> **Objective B — the report.** A Hackathon submission whose model-approach axis
> is unusually well-evidenced.

The good news is that the work overlaps heavily: the calibrated panel (P1) and
the disagreement mining (P3) serve both at once.

---

## The unlock this plan is built on

The survey turned up something the project has wanted since Design Principle #1
was written: **public agents whose live ladder scores are known, whose source is
downloadable, and which run offline.**

| Agent | publicScore |
|---|---|
| `aristophanivan/probablity-v2` | 933.8 |
| `lucifer19/battlecore-compact-agent` | 846.8 |
| `raunakdey07/pok-mon-tcg-advanced-heuristic-agent` | 796.8 |
| `prvsiyan/…search-audited-alakazam-v9` | 778.2 |
| `prvsiyan/…field-audited-alakazam-v8` | 739.7 |
| our v29d (54481189) | 673.5 |
| our v30-exp | 637.8 |

This fixes the project's two chronic measurement problems at once:

1. **No discriminating offline opponent.** Every reference anchor reads ≤6%
   against our champion; `grimmsnarl_agent` at 57.5% was celebrated as "the first
   discriminating offline opponent this project has ever had." Now we have five
   more, all *stronger* than us.
2. **Offline systematically overrates, and the ladder is too noisy to arbitrate.**
   With opponents at known ratings we can **fit offline-strength → publicScore
   directly** and stop guessing.

**Licence boundary:** these are Apache-2.0 public notebooks. Using them as
offline sparring partners and as diff targets for our own development is fine.
Submitting their code, or a derivative of it, as our own agent is not, and we
will not. Any technique we adopt gets reimplemented from the mechanism and
credited in the report.

---

## Phase 0 — Triage (target: 1 day, by 2026-07-24)

| # | Task | Verify |
|---|---|---|
| 0.1 | Read the Simulation and Strategy rules pages; reconcile Aug 9 / Aug 16 / Sep 6 against CLAUDE.md | Three dates confirmed from the official page, CLAUDE.md updated |
| 0.2 | Put the known-good agent in **both** counting slots (v30-exp is currently live at 637.8 with v29d's better copy displaced) | `kaggle competitions submissions` shows the latest 2 are the intended agent |
| 0.3 | Formally retire the 2026-07-16 v30-exp revert rule as **unresolvable** | Report-log entry: the ≥30pp threshold is a third of the measured 88.4pp identical-code spread; v30-exp sits between the two v29d copies |

**Note on 0.2:** with an 88-point noise floor, v30-exp vs v29d cannot be decided
on ladder score, now or ever. Decide it on the offline evidence (v30-exp's gate
was mirror 51.5%, adopt bar not met, all four instruments point-positive) and
move on. Recommendation: ship v29d in both slots for safety, and revisit only if
Phase 1's calibrated panel separates them.

---

## Phase 1 — The calibrated offline panel (target: 2026-07-24 → 07-28)

**This is the highest-ROI item in the plan.** It is infrastructure that makes
every later gate trustworthy, and it is itself report material.

| # | Task | Verify |
|---|---|---|
| 1.1 | Extract the 5 public agents' `main.py` + `deck.csv` from their notebooks into `opponents/public/` | Each imports and returns legal indices on the regression states |
| 1.2 | Adapt each to our harness (they expect Kaggle paths / bundled `cg`) | `python training/harness.py` runs 10 games vs each, 0 errors |
| 1.3 | Round-robin gauntlet: our {v29d, v30-exp} × the 5 public agents, n=200/pairing, seats alternated | `training/gauntlet_results.csv` rows; offline gElo per agent |
| 1.4 | **Fit gElo → publicScore** across the 7 agents with known scores | R² reported honestly; a stated prediction interval, given n=7 and an 88pt noise floor on the y-axis |
| 1.5 | Pre-register the resulting offline gate threshold for all later phases | Report-log entry with the threshold and its predicted ladder delta |

**Success criterion:** a documented mapping from offline gElo to expected
publicScore, with honest error bars. **Failure is informative too** — if offline
strength does not predict ladder score across 7 agents spanning 300 points, that
is a headline finding about this competition's evaluator and goes straight into
the report.

**Risk:** these agents may not run outside Kaggle (bundled `cg`, `/kaggle_simulations`
paths, `search_begin_input`). Mitigation: `training/setup_local_search.py` already
solves local `cg.api`. Budget half a day; if an agent resists, drop it and
proceed with the rest — even 4 points is a usable fit.

---

## Phase 2 — Sham-search placebo (target: 2026-07-24 → 07-25, parallel, cheap)

Control for the RNG-contamination risk found in `battlecore`'s notebook.

| # | Task | Verify |
|---|---|---|
| 2.1 | Build a placebo variant of `endgame_agent.py`: runs the full search, discards the result, returns the heuristic's choice | Behaviorally identical to `main.py` by construction |
| 2.2 | A/B placebo vs plain `main.py`, n=400, seats alternated — the same protocol that produced v29's +59.0% | Win rate with CI |
| 2.3 | Report either way | Report-log entry |

**Pre-registered reading:** a clean harness gives 50%. A reading whose CI excludes
50% confirms in-process search perturbs the shared engine RNG, and v29's +59.0%
needs re-measurement under process isolation. **Scope discipline:** a ~3pp
artifact cannot flip the 0W-50L ISMCTS/PIMC closures or the Φv4 20–34pp gaps —
those stand regardless. Do not reopen them on this result.

---

## Phase 3 — Close the Alakazam piloting gap (target: 2026-07-28 → 08-05)

Two public agents pilot our own deck 100–140 points better than we do. After
Phase 1 they are playable opponents, which converts our one reliably-positive
workflow (loss mining) into something much sharper: **mine per-decision
disagreements against a known-better pilot of the same deck**, instead of against
our own losses.

| # | Task | Verify |
|---|---|---|
| 3.1 | Log per-decision disagreements between v29d and `alakazam-v9` over ~200 shared states | Disagreement corpus on disk, categorized by decision type |
| 3.2 | Rank disagreement classes by frequency × outcome delta | A ranked list, top 5 classes named |
| 3.3 | Implement the top 2–3 as surgical `main.py` changes, one at a time | Each passes `python training/regression/regression_states.py` |
| 3.4 | Gate each on the Phase 1 calibrated panel, pre-registered | Offline gElo delta + its predicted publicScore delta |
| 3.5 | Ship the best-gated combination | `training/ladder_history.csv` row |

**Kill rule:** if 3.2 finds no disagreement class with a plausible mechanism by
2026-08-01, stop and put the time into Phase 4. Do not grind heuristic tweaks.

**Deliberately excluded:** deck switching. Mega Lucario ex reads stronger in the
meta (76.4% vs Alakazam's −238 aggregate delta), but our deck freeze was closed
on a pre-registered rule, two public agents reach 778 with Alakazam, and the
Strategy track scores *deck concept and its defense*, not raw tier. Switching
decks 24 days out would invalidate the piloting work and the deck chapter of the
report simultaneously. **Flagged as a live decision for the user, not taken
unilaterally.**

Cheap deck lever that is *not* a switch: **Night Stretcher / Sacred Ash**. Top-2
winner-correlated cards in the format, we run neither, and Night Stretcher
plausibly fixes board-thinning (our #1 live failure mode) while raising hand size
(our damage stat). The correlation is archetype-confounded, so this needs a real
gate, not adoption on the strength of the delta. Fits in Phase 3 as one more
gated candidate.

---

## Phase 4 — The architecture bet (target: 2026-07-28 → 08-12, parallel)

**The model-training answer, and the primary contribution to the 70% axis.**

Every net this project has trained scores a fixed-size action-slot vector. The
official pinned sample and both `fishcat37` nets are **action-conditioned**: each
candidate action is embedded and cross-attends the board, emitting one scalar per
action. This is the one axis never varied across nine algorithm variations.

| # | Gate | Threshold | Verify |
|---|---|---|---|
| 4.1 | Build the action-conditioned net | — | Trains; inference under the timing budget |
| 4.2 | **Gate 1 — isolation.** BC on the new architecture vs BC on the existing MLP, **identical corpus, seed, epochs** | fidelity > BC-MLP's 74.9% and DAgger-r2's 81.9%, CI-separable | paired bootstrap |
| 4.3 | **Gate 2 — win rate** vs v29d, n=400, seats alternated | > 25% (the best any public team reports; our historical band is 12–17%) | CI reported |
| 4.4 | **Gate 3 — external teacher.** Only if Gate 2 clears: retrain on top-20-team replays rather than our own heuristic | beats Gate 2's number | CI reported |

**Critical design constraint:** test the architecture **in isolation, not wrapped
in search.** Our own Φv4 closure showed a verified-better eval transfers zero
improvement through the search wrapper. Wrapping this in search first would
re-derive a search-shaped negative and tell us nothing about the architecture.

**Pre-registered kill rule:** if Gate 1 fails by 2026-08-05, the architecture
hypothesis is closed and the remaining time goes to Phase 3 and Phase 6. If Gate
1 passes and Gate 2 fails, that is *itself* the strongest result in the report —
a third independent architecture showing the fidelity/win-rate decoupling, this
time with the organizers' own reference design.

**Honest prior:** neither `fishcat37` net publishes a ladder score, and every
learned arm in this project and in `nursrijan`'s has failed. Expected value here
is report evidence, not a ladder jump. Budget it accordingly and do not let it
eat Phase 3.

---

## Phase 5 — Ladder endgame (target: 2026-08-12 → 08-16)

| # | Task | Verify |
|---|---|---|
| 5.1 | Freeze the agent; final clean-room validation from the extracted tarball | `get_last_callable` + 5 full `env.run` games, 0 errors |
| 5.2 | Ship it into **both** counting slots | `kaggle competitions submissions` |
| 5.3 | **Re-verify on 2026-08-15** that the latest 2 are as intended | screenshot / CLI output in report-log |

**Hard rule:** do not let an experimental read be final. This has nearly gone
wrong twice (v-dmc1b at 374.7; v30-exp at 637.8 right now).

---

## Phase 6 — Report assembly (target: 2026-08-16 → 09-06)

Three weeks, no ladder distractions. The report is assembled from
`docs/report-log.md`, per the standing rule that nothing gets reconstructed in
September.

**The spine.** We ran ~16 pre-registered learned-agent experiments and closed
nearly all of them negative. That is not a weak submission — `battlecore`'s
846.8 notebook leads with *"why negative results are the headline"* and it is one
of the best-received in the competition. The differentiator is that we can also
show **the measurement itself is trustworthy**, which almost nobody does.

Chapters, mapped to assets we already hold:

| § | Content | Asset |
|---|---|---|
| 1 | Problem framing: POMDP, variable action space, why rule-based caps out | `docs/game-nature.md` |
| 2 | Deck concept and its defense (20% axis) | Stage 0c bake-off, pre-registered freeze rule |
| 3 | The learned-agent graveyard: 16 arms, each with hypothesis, gate, number, verdict | `docs/report-log.md` |
| 4 | **The structural finding:** every method that distils a signal from the teacher can approach it, never exceed it — by construction (imitation) or empirically (search, AWR, IQL, DMC, AlphaZero) | advisor consult + 16 gates |
| 5 | **Fidelity/win-rate decoupling** across independent architectures | BC-MLP, DAgger, sequence-transformer, + Phase 4 |
| 6 | **Measurement integrity** — the chapter almost nobody else can write | see below |
| 7 | Belief model (92.3%) and Φ v4 eval — the components that did work | `docs/belief-model.md`, `docs/eval-function-research.md` |
| 8 | External replication: `nursrijan` (LB 1091) hit the same plateau independently | survey §1 |

**§6 is the standout chapter** and it is already three-quarters written:

- the **accidental A/A test**: one 24,857-byte tarball, uploaded twice, scored
  708.9 and 620.5 — an 88.4-point noise floor on identical code, verified by
  provenance
- the **sham-search placebo** (Phase 2), whether it confirms contamination or clears us
- the **offline→ladder calibration curve** (Phase 1)
- the project's own retraction history — every CORRECTION entry in the report-log,
  presented as the process working rather than hidden

---

## What could go wrong

| Risk | Mitigation |
|---|---|
| Phase 4 eats the calendar | Hard kill date 2026-08-05 on Gate 1 |
| Public agents will not run offline | Half-day budget; proceed with whatever subset works |
| Phase 1's calibration has no signal | That result is itself a report headline |
| Ladder close catches an experimental submission live | Phase 5.3, re-verify 08-15 |
| Deadlines are not what we think | Phase 0.1, first task in the plan |

---

## Sequencing summary

```
Jul 23-24  P0 triage ────────────────────────────────┐
Jul 24-25  P2 sham placebo (cheap, parallel) ────────┤
Jul 24-28  P1 calibrated offline panel  ← the unlock ┤
Jul 28-Aug 05  P3 piloting gap (kill 08-01) ─────────┤
Jul 28-Aug 12  P4 architecture bet (kill 08-05) ─────┤
Aug 12-16  P5 ladder endgame + 08-15 verify ─────────┤
Aug 16-Sep 06  P6 report assembly ───────────────────┘
```

---
