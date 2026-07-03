# Competition Strategy & Report Guide

*Summary of the strategic analysis for the PTCG AI Battle Challenge.*

**Last updated:** 2026-07-03

---

## Mission

The deck and the learning problem are designed **together**: a simple, single-prize,
flat-damage deck exists to make piloting learnable, and the learned pilot exists to
extract the deck's ceiling — every major choice (deck, method, architecture) is
quantitatively defended against named alternatives, not asserted. This maps directly
onto the rubric: 70% model approach (the method bake-off), 20% deck concept (the deck
bake-off), 10% report (the trial log that makes both citable). Evidence rule: **no
claim goes in the report without a pre-registered trial behind it** — hypothesis,
protocol, and decision rule are written into `docs/report-log.md` *before* the games
run.

---

## The Thesis in One Paragraph

Top 8 is **not** won by the best forward-search engine or the flashiest RL run. It is won by a **learned-piloting agent on a deliberately simplified deck**, trained by **BC → DAgger → advantage-weighted self-play** from a strong scripted teacher, optionally deployed with **fast, shallow belief-determinized search** under the 10-minute clock — then written up as a rigorous "model approach." This is the one path that scores on *both* axes: real ladder performance *and* the 70%-weighted "model approach" rubric. And the report is not the 10% axis — it is the delivery channel for the whole 70%: **report-driven development** (`docs/report-log.md`).

The heuristic (v23) is the live ladder placeholder and the DAgger teacher. Ladder
replays are still downloaded and audited (`tools/analyze_replay.py`); the heuristic
changes only through three sanctioned channels (replay-confirmed bugs, weight
search, belief-classifier wiring — see Stage 0b). All new effort goes to Stage 0
(deck + Gauntlet) and then the NN track.

---

## What "Top 8" Actually Means

**Scoring:** 70% model approach / 20% deck concept / 10% report.

1. **Your method is the whole ballgame.** 70% is the algorithm. A clever, well-justified learning method is worth more than three points of ladder Elo. Pure rule-based programming caps out low on the dominant axis.
2. **You still need real ladder results.** Simulation performance feeds the score. A beautiful method that goes 0-5 live is a hard sell.
3. **Entry is gated on a submitted ladder agent.** Keep something on the ladder at all times.

**Mechanics:**
- 10 minutes total per match; running out the clock is an instant loss
- Only W/L/T affects rating; margin is irrelevant
- 5 submissions/day; only your latest 2 count for final standing
- "Reasonableness standard" against excessive paid compute → favors lean, imitation-heavy approach

---

## What the Field Says Wins

### This Competition's Meta (wmh/ptcg-abc reference)
- **The real ladder is the only reliable judge.** Both custom harness and official `cabt` mispredicted ladder rank. Offline win-rates systematically overrate (documented by both us and competitors).
- **Deck choice dominates — but only up to a point.** Their *simplest* deck (Iono's Bellibolt ex, ~Elo 836) beat their "stronger" combo decks because rule-based pilots play simple decks cleanly.
- **The meta shifts fast.** Crustle (immune to ex/megaEx attacks) ballooned toward ~half the field — instantly devaluing big-ex bomb decks.
- **Single-prize Alakazam is flagged as current top meta.** The field agrees with the deck choice.
- All deployed agents in wmh/ptcg-abc are rule-based, with MCTS/RL only in `research/`. **That's the gap we exploit.**

### Comparable Kaggle 1v1 Ladders (Lux AI, Hungry Geese)
- **BC from self-play → RL → light MCTS at inference** is the proven recipe.
- Imitation is the cheap, high-leverage half; self-play RL is the expensive half.
- BC alone needs only a modest GPU and the self-play data — closes most of the piloting gap.

---

## The Master Plan (2026-07-01 roadmap, second revision)

**Game-changer:** the cabt engine runs fully locally (`training/README.md`) at
~0.5s/game. Self-play, A/B evaluation, weight tuning, and BC data collection no
longer need Kaggle sessions. The Vivobook (16 cores) multiplies throughput but is
no longer a hard blocker.

**Governing principle — report-driven development.** The report is not the 10%
axis; it is the *only channel through which the 70% "model approach" score is
delivered*. Every stage below must produce a figure, an ablation row, or a
finding, logged in `docs/report-log.md` the day it happens. The seven target
figures and Tables A/B are listed at the top of that file.

**Method narrative the stages build** (each step motivated by a diagnosed
failure of the previous one): BC from a strong scripted teacher → compounding
error diagnosed (85.9% action match yet 22% head-to-head) → **DAgger** on the
net's own state distribution → **advantage-weighted self-play** to exceed the
teacher → **belief-modeled hidden information** (originality centerpiece) →
optionally shallow search with blended priors, justified by a measured latency
curve.

### Stage 0 — Deck simplification + Gauntlet (NOW, before any rigorous training)

Deck changes invalidate collected teacher data, so the 60 gets settled first:

1. **Deck audit at scale:** `python tools/deck_audit.py --games 1000` — per-card
   utilization (plays per game drawn, rot rate, end-hand rate, win-rate deltas).
   4-game smoke run already fingered Genesect and Psyduck (0 plays, 100% rot).
2. **Variant A/B:** copy `main.py` → `variants/<name>.py`, edit `DECK`, then
   `python training/ab_test.py variants/<name>.py main.py 600`. Test at most
   1–3 swaps; likely outs are low-utilization passengers, likely ins are
   consistency/redundancy (e.g. 4th Alakazam). One decision point, then the 60
   is **frozen permanently** and `deck.csv` regenerated from `DECK`.
   *(2026-07-03 update: the freeze is re-opened pending the Stage 0c deck
   bake-off below — that section supersedes this one on deck-choice finality.)*
3. **Gauntlet baseline:** `python training/gauntlet.py --candidate main.py
   --name v24-<deck> --games 200` — establishes the new agent's gElo against
   the fixed 8-anchor panel. All future candidates get gauntleted under a
   distinct `--name`; results accumulate in `training/gauntlet_results.csv`,
   ladder outcomes go to `training/ladder_history.csv`. This pair is the
   offline/online calibration dataset.
4. **The pro list is the control**, not just the starting point: "started from
   the Indianapolis 1st-place list, instrumented utilization over N thousand
   games, swapped X for Y, +Z% over 600 games, confirmed on ladder" is the 20%
   deck-concept story.

### Stage 0b — Heuristic tuning on the frozen deck (parallel, ongoing)

The heuristic is the live submission AND the DAgger teacher — every Elo added
compounds through the student. Three sanctioned channels only:

- **Replay-forensics bug fixes** (bar unchanged: confirmed game-losing bug in
  2+ ladder replays).
- **Weight search** (`training/weight_search.py`, unattended overnight; winner
  needs a 600+ game A/B and a ladder confirm). Ship any teacher upgrade
  *before* the big DAgger collection runs.
- **Belief-classifier output wired into the heuristic** (fixes the hardcoded
  `opp_likely_ace_spec` — shared work with Stage 3).

No open-ended manual feature engineering — it competes with the 70% axis.

### Stage 1 — DAgger (target: week of Jul 6)

The net pilots self-play games; at every decision `main.score_options()` is
also queried and the teacher's choice recorded as the label. Retrain on the
aggregate (original BC data + DAgger rounds), iterate 2–3 rounds. This attacks
the state-distribution problem directly — the net gets teacher supervision
exactly on the states *it* reaches. **Gate: ~50%+ vs the teacher on the
Gauntlet → ship the learned agent to the ladder** (single forward pass, no
timeout risk) and start logging its real-bracket results.

### Stage 2 — Advantage-weighted self-play (mid-Jul → early Aug)

Past the teacher: weight each self-play sample's policy loss by
`exp(advantage/β)` using the n-step value targets already computed — actions
that outperformed expectation get imitated harder. Keep the 40% BC / 60% SP
batch mix (non-negotiable — SP-only collapsed 46%→20%). Winner-only filtering
runs as the dumb-baseline ablation row. **Gate: 55–60% vs the teacher over 400
local games.** Retired checkpoints join the Gauntlet panel and the sparring pool.

### Stage 3 — Belief model (parallel B-track; design: `docs/belief-model.md`)

Supervised classifier `P(archetype | opponent's observed plays, turn)` trained
from local games vs the four opponent bots, extended to ladder archetypes via
bulk replay download. Deliverables: the accuracy-by-turn figure (report
centerpiece), the `opp_likely_ace_spec` fix (teacher upgrade), and the
determinization sampler for Stage 5. Full phase plan, model choice, and the
decisions table live in `docs/belief-model.md`.

### Stage 4 — Hardening (Aug)

PFSP league vs the opponent pool + checkpoint hall-of-fame; frozen
hard-position and bad-hand eval suites (`training/curriculum.py`) as regression
gates on every checkpoint; curriculum training starts from mined tight positions.

### Stage 5 — Search at inference (Kaggle-gated, freeze-week go/no-go)

MCTS with the already-built λ-blended heuristic/net priors
(`training/nn/prior_blend.py`) + belief-based determinization. Gated on the
`search_begin` Kaggle spike and a measured latency curve (sims/decision vs p99
move time vs win rate). The go/no-go is a lookup in that curve, and the curve
is a report figure either way.

---

## Concrete Timeline

| Window | Goal |
|--------|------|
| **Jul 1–6** | Stage 0: deck audit at scale → variant A/B → **freeze the 60**; Gauntlet baseline for the new deck; weight search overnight; re-collect BC data on the frozen deck |
| **Jul 6–15** | Stage 1: DAgger rounds → learned agent to ~teacher parity → ship to ladder |
| **Jul 15 – Aug 5** | Stage 2: advantage-weighted self-play iterations; Stage 3 belief model in parallel; gauntlet + ladder A/B each checkpoint |
| **~Jul 20** | Merger decision point: solo if AWR shows a gradient; else recruit (GPU/RL partner) before Aug 9 |
| **Aug 5–16** | Stage 4 hardening; freeze best 2 submissions; Stage 5 MCTS go/no-go via latency curve; no risky changes |
| **Aug 16 – Sep 13** | Report assembly from `docs/report-log.md` (method section drafted during Stage 2, not after) |

---

## Official Strategy-Track Rubric (verbatim, retrieved 2026-07-03)

From the Kaggle Strategy competition Overview → Evaluation. This is the exact
text the judges score against.

| Category | Criteria (verbatim) | Weight |
|----------|--------------------|--------|
| **Model Score** | • How clearly is the chosen approach articulated, and how well is the rationale for the model and methods explained? • How original and technically sound is the proposed approach? • How consistently does the model perform under repeated matches and stable conditions? • How well does the strategy avoid over-reliance on specific initial states, matchups, or situational advantages? • Performance within the competition track. | 70% |
| **Deck Score** | • How clearly is the deck concept articulated, and how well does it align with the intended strategy? • How effectively are the key cards selected and utilized to support the deck's overall game plan? | 20% |
| **Report Score** | • How logically and clearly is the report structured and written? • How effectively are figures, charts, tables, or other visual elements used to support the explanation? | 10% |

**Submission format facts (from the same page):**
- Deliverable is a **Kaggle Writeup** (title + subtitle + body), not a PDF; a
  Track must be selected to submit; drafts left unsubmitted at the deadline
  are not judged. Final deadline **Sep 13, 2026, 7:59 PM EDT**.
- **"Your Writeup should not exceed 2000 words. Submissions over this limit
  may be subject to penalty."**
- Optional **Media Gallery** for images/video; other assets (code repos,
  Kaggle notebooks, external links) may be attached. Private attached
  resources are auto-made public after the deadline.
- **"Submissions containing images that violate the license granted for
  Pokémon Elements will not be evaluated and are subject to
  disqualification."** → no card art / scans in figures; deck lists as text
  tables, damage math as abstract charts.
- Judges: shige (Data Scientist, Matsuo Institute), choya (Data Scientist,
  MI), plus three The Pokémon Company judges — a **mixed ML + game-domain
  panel**; the report must read for both.
- Host framing (Description + shige's welcome post): "Why was a particular
  strategy chosen? What hypotheses were tested?"; "creative approaches,
  interesting findings, and lessons learned"; "decision-making when dealing
  with unknowns"; explicitly: middle/lower ladder tiers "can still achieve
  high overall scores through deep analysis, originality, and
  well-structured reporting."

**Rubric→plan gaps flagged 2026-07-03:** the two Model-Score robustness
bullets (consistency under repeated matches; no over-reliance on initial
states/matchups/situational advantages) had no dedicated report figure —
added as target figure #6: a robustness panel (per-anchor matchup win-rate
table, seat/going-first split, variance across repeated gauntlet runs,
win-rate conditional on opening-hand quality). Data mostly already collected
by the gauntlet; keep logging per-matchup splits every run.

---

## Stage 0c — Deck Bake-off (RESOLVED 2026-07-03: freeze re-closed on Alakazam)

**Outcome:** both tiers ran 2026-07-03 (2,000 games each, 0 errors; full
numbers + decision in `docs/report-log.md`). No challenger came near the
pre-registered ≥10pp bar — Alakazam won ≥93% of every tier-1 pairing (BT-Elo
1010 vs 577 for the best challenger), and tier 2 (fixed generic pilot)
discriminated nothing: all pairings ~50/50, 80% of games ending in DECK_OUT —
the "deck value is pilot-dependent" finding in its most extreme form. **The
freeze is re-closed, now quantitatively justified.** Original protocol below
for the record; re-run only if the ladder meta shifts against Alakazam
(watch `tools/meta_survey.py` shares — Archaludon at ~18% is not covered by
the current challenger pool).

The Alakazam freeze was **re-opened** pending a controlled 5-deck comparison, so the
final freeze is a measured decision, not a prior. Decks: Alakazam + the 4 opponent
anchor decks in `opponents/`. Two tiers, both required:

- **Tier 1 (as-piloted):** each deck with its own specialist agent, full seat-alternating
  round-robin (200 games/pair), Bradley-Terry ranking + matchup matrix with CIs.
- **Tier 2 (controlled):** all decks piloted by one deck-agnostic greedy heuristic
  (`training/generic_pilot.py`), same protocol — isolates deck strength from pilot
  quality. Tier-1/tier-2 rank disagreement is itself a finding (deck value is
  pilot-dependent), not a trigger.

**Pre-registered decision rule:** Alakazam is replaced only if a challenger
(a) beats Alakazam head-to-head by ≥10pp with the 95% CI excluding 0 in **both**
tiers, and (b) has a meta-weighted expected win rate (weights = observed ladder
archetype spread from the replay meta survey) ≥ Alakazam's. Otherwise the freeze
stands, now quantitatively justified. Switch cost, stated up front: invalidates the
BC/DAgger corpora, requires new heuristic piloting work, ~6 weeks of ladder left.

### Measurement standard (pro-aligned)

Match how the field measures decks (Trainer Hill / Limitless matchup tables; the
IEEE DataPort formal analysis of Trainer Hill data uses Wilson CIs + bootstrap —
our conventions match; JustInBasil / SixPrizes hypergeometric deck math;
competitive prize-mapping practice):

1. **Matchup matrix + meta-weighted win rate** — pairwise win rates with raw W-L-T,
   then expected field win rate weighted by observed meta share.
2. **Consistency panel** (analytic, `tools/deck_math.py`) — mulligan probability,
   P(key combo by turn N) per deck.
3. **In-play efficiency** — per game: turns, prizes taken per side, first-attack
   turn, end reason; derived: **prize-trade efficiency** (prizes taken per prize
   conceded — the Alakazam thesis, measured), setup speed, game length.
4. **Statistical conventions** — ties = 0.5 wins; crashes excluded and reported
   (>2% errors → fix and re-run, don't interpret); Wilson 95% CIs; recorded seeds
   with ≥2 independent-seed runs per headline pairing; seat-alternated with per-seat
   splits persisted.
5. **External anchor** — real-TCG Alakazam (Powerful Hand): 48% win rate at 4.07%
   usage (Campinas 2026 Regional), 4 regional top-8s incl. 1 win (Cerys Jones,
   Indianapolis) — the archetype is competitively real; card pool/meta differ here.

---

## Method Bake-off Protocol (first full run 2026-07-03 — Table B in report-log)

**Outcome:** gElo ladder random 5 → generic-greedy 57 → bc-v2 239 →
dagger-r2 246 → heuristic-v25c 568 (8,000 games, one protocol). DAgger not
CI-separable from BC on win-rate (third confirmation of the fidelity plateau);
kept as Stage 2 initialization. Future rows (AWR, search+belief) append to the
same table under new seeds — no re-runs of old rows without a protocol bump.

Every method row is produced by the same `training/gauntlet.py` version on the same
anchor panel, 200 games/anchor, recorded seed, statistical conventions above. Rows:
`random`, `generic-greedy` (tier-2 pilot on the Alakazam deck), `heuristic`
(`main.py`), `bc`, `dagger-r2`; future rows (AWR self-play, search+belief) join the
same table. Deliverable: one comparable ranking table plus a written "why kept /
why rejected" line per method, backed by these numbers and the existing negative
results (SP-only collapse, BC compounding error, DAgger plateau). No from-scratch RL
baseline: its cost competes directly with Stage 2 compute; imitation-first is
justified by the SP-only collapse data point and literature precedent, and the report
says so explicitly.

---

## Writeup Strategy

2,000 words ≈ 4 pages: a tight research note where **figures and tables carry
the evidence and prose carries the argument**. Assembled from
`docs/report-log.md` (which also holds the seven target figures and the plain-
English method glossary). Skeleton with word budget:

| § | ~Words | Content |
|---|--------|---------|
| 1. Problem framing | 200 | Imperfect information (hidden hand/deck/prizes), stochasticity, 100+ decision episodes, 10-min hard clock, positional option schema. State explicitly why vanilla AlphaZero doesn't apply and what does (determinization / belief-state methods). Correct problem taxonomy before any method. |
| 2. Deck as thesis | 200 | Not a list — an argument: single-prize trade math, Crustle-immunity vs the ex meta, and the ML-native point that **hand size = damage yields a dense per-turn learning signal**, freeing model capacity for micro-decisions. Plus the measured adaptation story: pro list as control, utilization audit, validated swaps. |
| 3. Method | 700 | The failure→fix chain: BC (85.9% action match) → why that coexists with 22% head-to-head (compounding error, explained) → DAgger → advantage-weighted self-play → belief-model determinization → (if shipped) shallow search with λ-blended priors. Every arrow is a diagnosed failure mode and its named cure. |
| 4. Evaluation | 400 | The methodology itself (seat-alternating A/Bs with CIs, frozen baselines, the Gauntlet's Bradley-Terry scale), the offline↔ladder calibration scatter, the ablation table, ladder trajectory of shipped versions. |
| 5. Findings & honesty | 300 | Engine discoveries (positional options, non-blind searches, the setup-active bug), the SP-only collapse (46%→20%) as a negative result, limitations. Honest negative results are rare in competition reports and score disproportionately. |
| 6. Compute statement | 100 | The "reasonableness standard" favors us: one consumer machine + free Kaggle T4 quota. State total GPU-hours and game counts; frame lean-ness as a design principle. |

**The seven figures** (logged continuously — see `docs/report-log.md` header):
archetype-inference accuracy by turn; gElo-vs-ladder-Elo calibration scatter;
win-rate-vs-teacher across training stages with CIs; the ablation table;
the latency budget curve; the robustness panel (matchup table, seat split,
seed-to-seed variance, opening-hand conditional); the pro-metrics panel
(consistency, prize-trade efficiency, setup speed, meta-weighted win rate).
Plus **Table A** (deck bake-off) and **Table B** (method comparison).

**Ablations to schedule (one table row each):** binary-terminal vs n-step value
targets; BC/SP mix ratios (the 46%→20% collapse is already a data point);
placeholder vs belief determinization; DAgger on/off; prior-blend λ sweep.

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Latency cliff (timeout = loss) | Hard wall-clock guard; guaranteed legal fallback in `_safe_return` |
| Chasing offline numbers | Ladder-only evaluation discipline |
| Sep 13 deadline vs other commitments | Front-load method by mid-Aug; writeup is not for the final week |
| NN track not ready in time | Heuristic (v25c) is a credible standalone submission; NN is upside |
| Team merger window closes Aug 9 | Actively seek partner with GPU/RL experience in July |

---

## Deck Ratings Summary

| Deck | Prizes | Complexity | Crustle? | Project fit |
|------|--------|------------|----------|-------------|
| **Alakazam (Powerful Hand)** | 1 ✓ | Stage 2 (med) | Beats it ✓ | **9.0** |
| Bellibolt ex (fallback) | 2 | Stage 1 (low) | Walled | 8.5 |
| Crustle (wall) | 1 ✓ | Control (subtle) | n/a | 6.5 |
| Raging Bolt ex | 2 | Linear | Walled | 6.0 |
| Gholdengo ex | 2 | Low-med | Walled | 5.5 |
| Charizard ex | 2 | Stage 2 high | Walled | 5.0 |
| Dragapult ex | 2 | Spread (very high) | Walled | 4.0 |
| Gardevoir ex | 2 | Toolbox (very high) | Walled | 4.0 |
| Mega ex boxes (incl. Lucario) | 3 ✗ | High + variance | Walled | 3.0 |
| Stall / mill | varies | Subtle + clock risk | n/a | 2.0 |

**Deck status: freeze re-opened 2026-07-03 pending the Stage 0c bake-off** (see
that section for the pre-registered decision rule). Original doctrine — one
sanctioned simplification pass, then frozen forever — resumes once Stage 0c
resolves. The 60
started as the meta backbone (Cerys Jones, Indianapolis Regional 1st) — a list
tuned for *human* pilots. Stage 0 adapts it for a machine pilot: instrumented
per-card utilization (`tools/deck_audit.py`), at most 1–3 swaps validated by a
600+ game A/B plus ladder confirm, at a single decision point. Early audit
suspects: Genesect and Psyduck (fine-grained human meta calls, ~0 plays/game in
bot games); likely ins: consistency/redundancy (4th Alakazam). After that pass
the 60 is frozen — no churn on uncertain data, and every deck change before the
freeze invalidates collected teacher data (recollect after).
