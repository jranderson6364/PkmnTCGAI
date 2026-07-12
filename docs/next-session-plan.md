# Next-Session Plan — Scaled Sequence-Policy Experiment (capacity vs. information)

*The detailed plan for the next work session, written 2026-07-09 at the end of
the Φ v4 overnight run. Headline: resolve whether the imitation-fidelity
plateau (~82%) is a MODEL-CAPACITY ceiling or an INFORMATION ceiling, using
the first real-capacity, GPU-trained sequence model this project has ever
attempted. Everything else tried from the literature and the 120-method
survey is closed with receipts — see §Why This, and `docs/report-log.md`
2026-07-09 entries.*

**STATUS: FULLY EXECUTED AND CLOSED, 2026-07-09 (corrected same day, then
carried through to a real conclusion).** The first "CLOSED, information
ceiling confirmed" verdict (76.6% fidelity at a 5-epoch checkpoint) was
PREMATURE — the training curve was still descending steeply at epoch 4,
not a plateau — and was retracted before any GPU compute was spent
(caught via an `advisor` consult). Resumed training with fresh-game
fidelity (not just train_acc, since there's no held-out val split) as the
real stopping signal:

| checkpoint | fidelity |
|---|---|
| epoch 1 | 62.8% |
| epoch 4 (premature stop) | 76.6% — retracted |
| epoch 5 | 82.17% |
| epoch 6 | 82.90% (+0.73pp) |
| epoch 7 | **83.03%** (+0.13pp — plateau confirmed) |

**Real Gate 1 result: fidelity plateaus at ~83.0%, genuinely above BOTH
reference points (BC-MLP 74.9%, DAgger-r2 81.9%)** — a real, replicated,
if modest capacity signal once trained to actual convergence. Because
83.0% cleared the pre-registered 82% threshold, Gate 2 (win-rate, n=400
vs. the v29d teacher, seats alternated) was run next (built
`training/nn/seq_agent.py` for live inference; confirmed CPU-safe timing
first).

**Real Gate 2 result: 12.2% ± 3.2% win rate (49W-351L/400)** — BELOW the
35% threshold, and at the LOW end of the historical 12-17% BC/DAgger
plateau, not above it, despite the fidelity gain being real. **This is the
actual, non-premature conclusion: fidelity and win-rate decouple once
fidelity clears roughly the mid-70s% — a second independent architecture
(after DAgger's own MLP) shows the same disconnect.** More capacity,
history-context, and Φ v4 features, trained correctly, buy real fidelity
but not games. Full numbers, both gates, in `docs/report-log.md`'s
2026-07-09 entry.

**CONCLUSION FOR THE KAGGLE GPU QUESTION:** a bigger version of this same
architecture (the plan's original "10-50x capacity" ask) is very unlikely
to convert to win-rate either — the bottleneck this experiment isolated is
not model capacity or missing context, it is something structural to pure
imitation against a strong teacher (most plausibly a small number of
game-deciding low-probability branches that an aggregate fidelity number
can't see). **Do not spend Kaggle GPU quota scaling this architecture
further on the same objective.** If a GPU push is still wanted, per
`advisor`'s framing it should target the one lever with real precedent —
DAgger-style on-policy correction, which is a data-collection/labeling
change, not a capacity change, and doesn't obviously need GPU either.
There's also an unresolved goal question that determines whether ANY
bigger imitation model is the right target: imitation (BC/DAgger) can
only asymptote toward teacher (v29d) parity by construction, never exceed
it — every non-imitation path tried (search, AWR, IQL, DMC, AlphaZero-
style self-play) is already closed negative. If the goal is to actually
beat v29d rather than produce a well-documented learned clone of it, no
tested method — including this one — gets there.

**Last updated:** 2026-07-09 (fully executed: real plateau found, win-rate gated, final conclusion reached)

---

## Why This (context for a fresh session)

- Every learned-policy arm (BC, DAgger, AWR, DMC, IQL, AlphaZero-style,
  winner-BC) and every eval/search consumer (PIMC, ISMCTS, endgame search,
  Φ v4 leaf, 1-ply advisor ×2, Gumbel root, RV leaf) is CLOSED with
  pre-registered gates. The survey's full A-shortlist is executed.
- The one shared bottleneck: **no learned policy has come near the
  heuristic** (best: 17% win rate vs teacher; fresh-state fidelity plateau
  74.9% → 81.9% across DAgger rounds).
- The one untried lever with a live causal story: **capacity + compute.**
  All prior policy nets were small CPU MLPs (BC v2: ~106 min CPU, 10
  epochs). The literature's working card-game agents (DouZero, Suphx,
  VGC-Bench baselines) are sequence models trained on GPUs for days. The
  fidelity plateau was measured at tiny capacity only — we never learned
  which ceiling it is.
- Honest prior: MODEST. DAgger showed fidelity gains don't convert 1:1 to
  win rate, and imitation asymptotes at parity by construction. Frame as a
  decisive measurement, not a likely win. Either outcome is strong report
  material (the 70% axis rewards the experimental program).

---

## Phase 0 — Pre-registration (do FIRST, before any training)

Log in `docs/report-log.md` with these pre-committed gates:

1. **Fidelity gate (the capacity verdict):** fresh-state argmax
   teacher-agreement on 3,000 deployment-realistic states (same protocol
   as the 2026-07-03 DAgger measurements — comparability matters).
   Reference points: BC-MLP 74.9%, DAgger-r2 81.9%.
   - ≥90% → capacity was (part of) the ceiling; proceed to win-rate gate.
   - ~82% or below at 10-50x capacity → **information ceiling confirmed;
     the line closes** and the plateau becomes a headline report result.
2. **Win-rate gate:** n=400 vs the current teacher, seats alternated.
   ≥35% = meaningfully above the 17% plateau → consider DAgger rounds on
   the big model. Below → close.
3. **Clock-safety gate:** per-decision CPU inference (Kaggle has no GPU at
   MATCH time) must project <2s/decision at ~69 decisions/game against the
   600s clock, measured with the existing `MCTS_TIMING_LOG`-style hook.
   If the transformer is too slow, distill to a smaller student and gate
   the student.

---

## Phase 1 — Data

1. **Re-collect BC data from the CURRENT teacher (v29d main.py)** — the
   existing corpora (`bc_data_v25c*`, 579k samples; `dagger_data_r1/r2`)
   imitate v25c-era logic. 2,000 self-play games ≈ hours locally at
   ~0.5s/game with `workers=15` (`training/bc_collect.py`). Keep the old
   corpora for a data-scaling ablation arm.
2. **Sequence format:** per game, the ordered list of (obs, chosen option
   index) for our seat — the model consumes the game HISTORY, not isolated
   states (the current `encode.py` is per-state; a new
   `encode_seq.py` is needed).
3. **Auxiliary inputs (the calculated-values thesis, proven twice):**
   append the 12 Φ v4 features (`training/nn/eval_v4.features_v4`) to
   every state's encoding. Cheap, and the eval ladder proves they carry
   signal raw encodings don't surface easily.
4. Optional aux targets (survey #113): opponent-archetype label (from
   `tools/meta_survey.py` signatures) and outcome — multi-task heads
   regularize sequence models on small data.

## Phase 2 — Model + training (Kaggle GPU)

- Architecture: 2-4 layer transformer (or GRU control), d_model 256-512,
  ~5-20M params — 10-50x current capacity. Context = full game history
  (~150 decisions max), per-decision causal mask.
- **Controls trained on the SAME data:** (a) the existing small MLP
  (isolates capacity from data-freshness), (b) transformer WITHOUT the
  Φ v4 feature inputs (isolates the calculated-values contribution).
- Train on Kaggle GPU (quota ~30h/week; the local engine means only
  TRAINING needs Kaggle, not collection). **Checkpoint every epoch to
  /kaggle/working and download incrementally** — a Kaggle stall already
  ate 9-11h of this project once (2026-07-07 entry).
- Local smoke first: 1 epoch on 5% data on CPU end-to-end (collect →
  encode → train → fidelity eval) before any GPU run.

## Phase 3 — Gates, in order (kill early)

1. Fidelity gate (Phase 0 #1) — cheapest, decides the headline question.
2. Win-rate gate vs teacher (Phase 0 #2).
3. Clock-safety gate (Phase 0 #3).
4. Only if all pass: diverse-anchor gauntlet (lucario/abomasnow/starmie +
   mirror — NEVER mirror-only, per the v29 lesson) before any ship talk.
   No ladder ship without explicit user approval.

---

## Parallel / fallback tracks (lower priority, no new decisions needed)

- **Blunder-mining v29d losses** with `training/nn/blunder_scan.py` (the
  one reliably positive workflow; 60 fresh replays already on disk at
  `replays/v29d_ladder/`, more downloadable via `tools/download_replays.py`).
- **Board-thinning trade-gate fix:** designed, documented (report-log
  2026-07-09 audit entry), NOT implementable with a measurable gate
  offline (no reproducing opponent — main.py beats the archaludon pilot
  96.7%). Needs a user decision to ship replay-reasoned.
- **Report assembly** (due ~Sep 13): the eval ladder (Φ v1 0.563 → v2
  0.610 → v4 0.650 → MLP 0.675) and the four-override-mechanisms/
  one-ceiling table are the two strongest figures; both fully sourced in
  report-log 2026-07-09.

---

## Key assets from 2026-07-09 (where things are)

| Asset | Path |
|---|---|
| Champion state eval (MLP, 0.675/0.724/0.752) | `training/eval_v4_mlp.pth` + `training/nn/eval_v4_mlp.py` |
| Φ v4 features + linear weights | `training/nn/eval_v4.py`, `training/eval_v4_weights.npy` |
| Eval-guided loss miner | `training/nn/blunder_scan.py` |
| Literature research | `docs/eval-function-research.md` |
| All 2026-07-09 gates + closures | `docs/report-log.md` 2026-07-09 entries (7) |
| Closed action-ranker family agents (reference) | `training/nn/{phi4_agent,advisor_agent,cem_tune,audit_agent}.py` |
