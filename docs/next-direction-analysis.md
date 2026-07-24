# Next-Direction Analysis — after the search line landed (776.2)

*In-depth reasoning on the three candidate directions, grounded in the full
project history. Written 2026-07-23 at the user's request ("explore each option
thoroughly and report what you think is next best").*

**Last updated:** 2026-07-23

---

## Where we actually are

| Fact | Value |
|---|---|
| Best live agent (belief-twoply, counting) | **776.2** |
| Placeholder-twoply (non-counting, settled) | 750.7 |
| v29d backstop (counting) | ~555–570, still settling |
| Public ceiling on our deck (alakazam_v9) | 778.2 |
| Field top-8 cutoff | ~1114 |
| Days to ladder close (Aug 16) | 24 |
| Days to Hackathon/report (Sep 6) | 45 |
| Report draft | none yet (136 report-log entries, no assembled doc) |

**The headline:** the conservative-override search is the **first agent in the
project's history to exceed the plain heuristic on the ladder**, and belief
determinization pushed it to 776.2 — at the public best for this deck. That is a
banked, real win. Everything below is about what to do with the remaining 24
ladder-days and 45 report-days.

---

## Option 1 — Deep search + trained value net (AlphaZero-style)

### What it is
Deepen the 2-ply override toward a real search tree (MCTS/PUCT), and put a
*trained value net* at the leaf where, at depth, the leaf evaluation dominates the
outcome. This is the genuine "learned model guiding search" the user has wanted.

### What the history says — this is not untried
The project **already ran an "AlphaZero-style push"** (2026-07-05 → 07-07). The
composed system was `mcts_leafeval_agent.py`: limited-sim PUCT, leaf evaluation
via the value net's `max_a Q(s,a)`. Its result, vs the plain heuristic:

| checkpoint | win rate vs heuristic |
|---|---|
| pre-Phase-0 net | 6.7% |
| round-1 (300-game corpus) | ~20% |
| round-2 (1500-game corpus, n=100 re-gate) | ~19–24% |

**It LOST to the heuristic ~76–80% of the time.** And the self-play *training* loop
(Phase 2, rounds 2–3) closed negative: "retrain does NOT clear the pre-training
baseline (0.566 vs 0.584)." The documented reason is the one that has killed
**every** learned arm this project has tried — BC, DAgger, AWR, DMC,
sequence-policy, oracle-critic, IQL, AlphaZero self-play: *a value/policy signal
distilled from the teacher (or from self-play against the same checkpoint) cannot
exceed the teacher it is built from.*

### Why the failure is structural, and why it collides with *this* session's win
The AlphaZero push was a **policy replacement** — the MCTS chooses the move,
overriding the heuristic — and it lost because the value net was never good enough
to make the tree beat the heuristic. The current twoply works for the *opposite*
reason: it is a **conservative override** (heuristic drives, search vetoes only on
a ≥half-prize margin), so it is ≥ the heuristic *by construction*. Deepening toward
a real MCTS+value-net tree moves the agent **back toward the policy-replacement
regime that already scored 20–24%.** The two are in tension: the thing that made
search work (staying subordinate to the heuristic) is exactly what a deep tree
gives up.

### And this session already probed the premise — null
The **leaf-eval null** (2026-07-23): swapping the crude formula for Φ v4 (the
project's *best* fitted value function, 0.675 sign-acc) as the leaf eval made **no
difference** (51.2%). If a strong fitted value doesn't help the search, a trained
net is very unlikely to — at least at shallow depth. The counter-argument ("value
matters more at *deep* depth") is true in principle but requires the deep tree,
which is the regime that failed.

### The one genuinely new angle
The history explicitly names the only escape: **"search-derived targets that reach
further than the value net's own priors."** We now have that — the working
override search beats the heuristic (776), so twoply-search-derived value targets
*do* exceed the old teacher. Training a value net on those and using it in deeper
search is the one combination never tried. **But** the leaf-eval null already
discourages the "better value → better search" link, and validating any of it is
crippled: local search measurement is contaminated (−39pp asymmetric, proven this
session), so **every iteration would need a live-ladder read (days each)** — the
slowest possible loop for the highest-variance build.

### Cost / risk
- **Effort:** multi-day. Train a net, deepen the search, manage the 0.8s budget
  and the variance that grows with depth, re-solve packaging.
- **Compute-safety:** deep MCTS was measured at ~3.1s/decision, ~216s/game — under
  the 600s clock but 4× heavier than twoply; deeper is tighter.
- **Measurement:** offline-invalid for search agents → ladder-only iteration.
- **Prior:** the closest precedent closed at 20–24%; the value-net-can't-exceed-
  teacher wall has held across 8+ arms; the leaf-eval probe is null.

### Verdict
**High effort, high risk, strong negative prior.** This is the user's RL dream and
the only path to a genuinely *learned* ladder agent — but the evidence says it is
much more likely to reproduce the AlphaZero-push negative than to beat a 776
override. If pursued, it must be **time-boxed and gated behind a cheap probe**
(train a value net on twoply-search value targets; test it as a leaf eval vs Φ v4
in the mirror — if it can't even beat Φ v4 there, stop), NOT an open-ended
commitment, and NOT at the expense of the report.

---

## Option 2 — Bank the search win, assemble the report

### What it is
Treat 776.2 as the ladder result and invest the remaining runway in the
Hackathon/Strategy writeup.

### Why this is where the value is
- **The $240k prize pool is entirely in the Strategy/Hackathon track**, judged on
  "reasoning, methodologies, and design decisions" — and prize ranking uses *both*
  leaderboard performance AND the report (verified 2026-07-23). The ladder result
  (776, a real improvement) now *backs* the report instead of undercutting it.
- **The material is exceptionally strong and mostly already written** (136
  report-log entries): recipe identification from public agents; the mechanistic
  "why conservative-override search succeeds where 5 policy-replacement attempts
  failed" story; the **measurement-integrity chapter** almost nobody else can
  write (the 88-point A/A noise floor, the mirror→asymmetric placebo that quantifies
  and localizes a −39pp hidden confound, the R²=0.004 calibration collapse); the
  learned-agent graveyard with pre-registered gates; and now a *shipped ladder
  improvement* as the capstone.
- **It needs runway.** No assembled draft exists; 45 days is comfortable but a
  strong writeup (figures, narrative, honest negative results) is real work.

### Cost / risk
- **Low risk, high value on the axis that pays.** The only downside is opportunity
  cost — 24 ladder-days left where cheap gains might exist (addressed by running
  Option 3 in parallel).

### Verdict
**The highest-EV single use of time**, and the one directly tied to the prize.
Should be the *primary* thread starting now, precisely because it needs the runway
and the ladder win has made its story complete.

---

## Option 3 — Squeeze the current search (+ let belief read)

### What it is
Cheap, low-risk tuning of the working agent while the ladder is open: override
margin, N_DET (more determinizations → lower value-estimate variance), a possible
3-ply, more candidates, and the one untried *deck* lever from the notebook survey
(Night Stretcher / Sacred Ash — top winner-correlated cards that fix board-thinning
and raise hand size).

### What the history/this session says
- Belief determinization already delivered (+10pp mirror → +26 on the ladder,
  750→776). So this class of change *can* pay.
- But the leaf-eval avenue is now **null** (exhausted). Margin tuning last session
  moved everything toward 50% under contamination (uninformative). Offline
  validation is limited to the ±11pp mirror; real gains need ladder reads (days).
- We are at 776, ~2 points under the public ceiling (778) — the *piloting* headroom
  on this deck is nearly gone. Further ladder gains likely need the deck lever
  (Night Stretcher/Sacred Ash) or depth, not more override tuning.

### Cost / risk
- **Low effort, low-moderate, uncertain gains, slow validation.** A couple of
  targeted experiments are cheap; open-ended tuning has poor signal.

### Verdict
**Worth a small, bounded amount in parallel** — specifically the deck lever (real
mechanistic prior, cheap to gate on the mirror + a ladder read) and possibly a
3-ply probe — because 24 ladder-days remain and we're at a medal-relevant score.
Not a primary thread; diminishing returns beyond a couple of experiments.

---

## Recommendation — what I think is next best

**A sequenced combination, not a single option:**

1. **PRIMARY — start assembling the report now (Option 2).** It is the prize axis,
   the material is now complete and capped by a real ladder win, and it needs the
   45-day runway. Begin with the structure and the three standout chapters
   (measurement integrity; why-override-beats-replacement; the search-recipe arc).

2. **PARALLEL, BOUNDED — one or two cheap ladder squeezes (Option 3)** while the
   ladder is open and we sit ~2 points under the public ceiling: the Night
   Stretcher / Sacred Ash **deck** lever (the real remaining headroom on this deck,
   with a mechanistic prior), gated on the mirror + a ladder read; optionally a
   3-ply probe. Cap it — do not open-ended-tune.

3. **DEFER / gate — deep-search AlphaZero (Option 1).** Do **not** commit multi-days
   to it on the current evidence: the closest precedent scored 20–24%, the
   value-net-can't-exceed-teacher wall has held across 8+ arms, and this session's
   leaf-eval probe is null. IF the user wants the RL swing anyway (a legitimate want
   given the "learned model" goal), run it **only** as a time-boxed, probe-gated
   experiment: train a value net on twoply-*search*-derived targets (the one new
   information source), and kill it immediately if it can't beat Φ v4 as a leaf in
   the mirror. Frame it honestly as a gamble, not a plan.

**One-line version:** the search line already won (776, first ever) — convert that
win into the report that actually pays, take the cheap deck-lever shot at the last
few ladder points in parallel, and treat the AlphaZero dream as a gated gamble, not
the main road, because the whole history says that road ends at ~20–24%.

**The honest tension to name for the user:** Option 1 is the only path to a
*genuinely learned* ladder agent, which has been a stated goal. My analysis is that
it is very likely to fail on the evidence — but if the goal is "I want to have
seriously tried a learned model," a *gated, time-boxed* Option-1 probe is
defensible and its negative would itself be strong report material. What is not
defensible is an open-ended AlphaZero build that eats the report runway.

---
