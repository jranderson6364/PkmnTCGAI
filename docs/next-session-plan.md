# Next-Session Plan — Scaled Sequence-Policy Experiment (capacity vs. information)

*The detailed plan for the next work session, written 2026-07-09 at the end of
the Φ v4 overnight run. Headline: resolve whether the imitation-fidelity
plateau (~82%) is a MODEL-CAPACITY ceiling or an INFORMATION ceiling, using
the first real-capacity, GPU-trained sequence model this project has ever
attempted. Everything else tried from the literature and the 120-method
survey is closed with receipts — see §Why This, and `docs/report-log.md`
2026-07-09 entries.*

**Last updated:** 2026-07-09

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
