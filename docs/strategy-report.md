# Strategy Report — PTCG AI Battle Challenge (working draft)

*The Hackathon/Strategy-track writeup, assembled from `docs/report-log.md`. This
is the delivery channel for the 70% "model approach" score. Draft in progress;
prose being tightened, figures marked `[FIG-n]`.*

**Last updated:** 2026-07-23
**Rubric it targets** (verbatim, `docs/competition-strategy.md`): 70% Model Score
(articulation + rationale; originality + technical soundness; *consistency under
repeated matches*; *avoids over-reliance on specific matchups/states*; competition
performance) · 20% Deck Score · 10% Report Score (structure + figures).

---

## Abstract

We set out to build a competitive agent for a partially-observable, variable-
action-space card game where the organizers themselves note that "rule-based
programming alone may not ensure a high ranking." Our central finding is a
methodological one: **on this problem, the binding constraint is not the learning
algorithm but the measurement.** We document (1) a graveyard of eight learned-
agent methods that each fail for one structural reason — a signal distilled from a
teacher cannot exceed that teacher; (2) three independent, quantified measurement
pathologies that explain why offline evaluation systematically misled us and,
we argue, the field; and (3) the one method that broke the ceiling — a
*conservative-override* forward search — identified by mining public agents,
explained mechanistically against our own five failed search attempts, and shipped
to a live-ladder result of **776**, the top of our deck's public range. The
through-line is that every claim is backed by a pre-registered trial, and several
of our most valuable results are the honest reversals of our own earlier
conclusions.

---

## 1. Problem framing

The competition is a two-player, imperfect-information game: ~40% of the state
(opponent hand, both decks, prizes) is hidden, the action space is variable
(2–50 legal options per decision, with context-dependent semantics), turns are
sequences of sub-decisions, and reward is sparse (win/loss only). Formally a
POMDP. Two properties dominate method choice:

- **Sparse terminal reward + long horizon** (~30–50 turns, ~150 decisions/game)
  makes credit assignment hard for model-free RL.
- **A strong hand-authored heuristic is available and cheap.** This turns the
  problem from "learn to play" into "learn to *exceed* a strong teacher" — a
  materially harder and, as we show, differently-shaped problem.

The two scoring axes pull apart: the ladder (Simulation track) rewards raw win
rate; the Strategy track rewards the *method and its justification*. We treat the
report as the delivery channel for the 70% model-approach axis, per
report-driven development — every experiment logged the day it ran, with a
pre-registered hypothesis and decision rule.

---

## 2. Deck concept [20% axis]

*(To expand from the Stage 0c bake-off, `docs/report-log.md` 2026-07-03.)*

Alakazam single-prize control. Win condition: **Powerful Hand** (`damage = 20 ×
hand size`) — hand size is simultaneously the resource and the damage stat, which
makes card economy the central strategic tension and rules out the usual
draw-hard staples. Single-prize, non-ex: gives up one prize per KO but denies the
opponent the 2–3-prize swings that ex attackers concede. The deck was frozen via a
pre-registered bake-off (tier-1 as-piloted ≥93% vs four challengers; tier-2 fixed-
pilot ~50/50 — establishing that the deck's value is pilot-dependent, itself a
finding). `[FIG-deck]` decklist as text + the bake-off Bradley-Terry table.

---

## 3. The method: learning to exceed a strong teacher

### 3.1 The graveyard, and why it is one finding, not eight

We tested eight learned-agent methods, each pre-registered and gated:

| method | result vs the heuristic | why it closed |
|---|---|---|
| Behavioral cloning (MLP) | ~12–17% | imitation asymptotes *to* the teacher |
| DAgger (2 rounds) | fidelity 73→82%, win-rate flat | on-policy correction doesn't lift win-rate |
| Advantage-weighted self-play (AWR) | 11–16% | value head saturates ±1; no per-decision advantage |
| PIMC search | 0W-50L | leaf-value signal too weak |
| Belief-ISMCTS | 0W-50L | same |
| Endgame-gated rollout search | reverted | (see §4 — measurement) |
| Sequence-transformer BC | fidelity 83% / win-rate 12% | fidelity and win-rate *decouple* |
| DMC (round 3→6) | 2.5%→8.25%, then negative slope | more data made it worse |
| AlphaZero-style self-play + MCTS+value-net | ~20–24% vs heuristic | value net can't beat the teacher it's trained from |

**These are one result.** Every method builds its signal — a cloned policy, a
fitted value, a self-play target — from the teacher or from self-play against the
same checkpoint, and therefore **cannot exceed the teacher by construction or by
empirical closure.** A second architecture (the sequence transformer) independently
reproduced the fidelity/win-rate *decoupling*: past the mid-70s%, higher
per-decision accuracy stops converting to wins — evidence the ~12–17% ceiling is
not primarily a per-decision-accuracy problem. `[FIG-graveyard]` the ablation
table; `[FIG-decouple]` fidelity vs win-rate across BC/DAgger/sequence arms.

### 3.2 The breakthrough: conservative-override search

The escape came not from a new learning algorithm but from reading the public
leaderboard. The strongest public agent on our own deck (778.2) is a heuristic
*plus a forward search*; an isolation test (its `USE_SEARCH` switch off) showed its
bare heuristic *ties ours* — **its entire edge is the search.** Yet our own five
search attempts had all closed negative. The reconciling insight is structural:

> Every failed attempt **replaced** the policy with the search (the tree picks the
> move). The winning recipe is a **conservative override**: the heuristic drives
> every turn, and the search only *vetoes* its choice when a bounded 2-ply
> look-ahead shows a clearly better line (≥ half-a-prize margin). The override is
> ≥ the heuristic by construction — it preserves plan coherence, which the
> policy-replacement wrapper destroys.

We built it on our own heuristic and belief model (not the public code):
belief-determinized opponent decks, greedy-completed turns, a minimax ply-2, and a
leaf value that includes hand size (our win condition — without it the search
traded away tempo and collapsed our best matchups). Shipped to the live ladder:
**750.7**, then **776.2** after upgrading placeholder determinization to belief
determinization — the top of our deck's public range and the first agent in the
project to exceed the plain heuristic on the ladder. `[FIG-override]` override vs
replacement, decision-flow; `[FIG-ladder]` the ladder trajectory 673→750→776.

**A negative that sharpens the story:** we then tested whether a *learned* value
function (Φ v4, our best fitted state-value) sharpens the search as its leaf eval.
It did not (51.2%, null) — the leaf eval is not the bottleneck for a shallow
search. So the win is the *structure* (override) and the *opponent model* (belief),
not a learned evaluator.

---

## 4. Measurement integrity — why offline evaluation misled us

This is the report's technical core, and the axis the rubric rewards directly
("consistency under repeated matches; avoids over-reliance on specific matchups").
We quantified three distinct pathologies, each of which had already caused a wrong
conclusion earlier in the project.

### 4.1 The live-evaluator noise floor — an accidental A/A test
Two **byte-identical** submissions (verified: one 24,857-byte tarball, SHA-256
`acef3750…`, uploaded twice 13 s apart) scored **708.9 and 620.5** — an
**88-point spread on identical code.** This is a clean A/A control we ran without
meaning to. It retroactively voids every version-to-version `publicScore`
comparison the project made inside that band, and it sets the bar any real ladder
claim must clear. `[FIG-aa]` the two identical submissions and their scores.

### 4.2 Local search evaluation is contaminated — a placebo control
Following a public notebook's lead, we ran a **sham-search placebo**: an agent that
runs the search API every turn and *discards the result*, returning the pure
heuristic's move (behaviorally identical by construction). In the mirror it read a
benign 53.2%. But extended to **asymmetric** opponents it exposed a large hidden
confound: the search calls alone — changing *no decision* — swing matchups by up to
**−39pp** (dragapult 99%→60%), because the search API perturbs the shared engine
RNG in our local harness. **This voids all local evaluation of search agents
against the real field**, corrects our own over-confident "harness is clean"
conclusion from six entries earlier, and — crucially — is a *local-engine*
artifact: the winning agent scores 776 on Kaggle, which isolates the search RNG.
`[FIG-placebo]` per-matchup swing from search-calls-alone.

### 4.3 Offline strength does not predict ladder strength — a calibration collapse
We assembled the project's first discriminating offline panel: four public agents
of *known* ladder score (739.7–933.8), every one of which beats our champion (all
prior anchors read ≤6%). A single-reference fit looked usable (R²=0.537), but a
full round-robin **collapsed it to R²=0.004** — offline win rate is nearly
*orthogonal* to ladder score, because the panel is archetype-skewed and ranks
agents by matchup composition rather than skill (the highest-ladder agent, a
Lucario deck, ranks 4th of 5 offline). This is Design Principle #1 ("offline
overrates") converted from a slogan into a measured transfer function with an error
bar. `[FIG-calibration]` gElo vs publicScore, R²=0.004.

**The unifying lesson:** on this competition, the evaluator is the hard part.
Three independent pathologies — a large noise floor, a matchup-specific
serialization confound, and a composition-dependent calibration — each produced a
wrong conclusion until it was isolated. Our most valuable methodological
contribution is not any single agent; it is the discipline that caught these, and
the honesty that reversed our own conclusions in public.

---

## 5. Components that worked (even where the end-to-end arm didn't)

- **Belief model:** archetype classifier, 92.3% held-out, 99.1% by turn 1 — used
  live for wall anticipation and now for search determinization (the +26 ladder
  points from placeholder→belief). `[FIG-belief]` accuracy-by-turn.
- **Φ v4 state-value:** literature-driven 11-feature antisymmetric fitted eval,
  the project's best state-value signal (0.675) — valuable as a gate baseline even
  though it did not transfer through search.
- **Disagreement mining** against a byte-identical stronger pilot of our own deck:
  a clean method that localized where our piloting differed from a +105-point pilot.

---

## 6. Results, and honest limitations

**Result:** 776 on the live ladder (top of our deck's public range), from a
conservative-override belief-determinized search, with v29d (the pure heuristic) as
a validated backstop. Consistency (a rubric criterion) is argued from the shipped
result plus the measurement work that separates real edges from matchup artifacts.

**Limitations, stated plainly:** we did not produce a genuinely *learned* ladder
agent — every learning arm hit the can't-exceed-teacher wall, and we argue (from
the AlphaZero-push precedent and the null leaf-eval probe) that the remaining
learned path (deep search + trained value net) is a low-probability gamble rather
than an expected win. The search win is algorithmic, not learned. We consider the
rigorous documentation of *why* the learned methods fail — with pre-registered
gates and quantified measurement pathologies — to be the honest and defensible
model-approach contribution.

---

## Figure list (to produce)

| tag | content | source |
|---|---|---|
| FIG-deck | decklist (text) + bake-off BT table | report-log 2026-07-03 |
| FIG-graveyard | 8-method ablation table | §3.1 |
| FIG-decouple | fidelity vs win-rate, BC/DAgger/seq | report-log 2026-07-09 |
| FIG-override | override-vs-replacement decision flow | §3.2 |
| FIG-ladder | ladder trajectory 673→750→776 | ladder_history.csv |
| FIG-aa | the 88-point identical-code A/A | report-log 2026-07-23 |
| FIG-placebo | per-matchup swing from search calls alone | report-log 2026-07-23 |
| FIG-calibration | gElo vs publicScore, R²=0.004 | round_robin_matrix.csv |
| FIG-belief | archetype accuracy-by-turn | report-log 2026-07-04 |

---
