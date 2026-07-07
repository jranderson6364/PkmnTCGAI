# PkmnTCGAI

*Pokemon TCG AI Battle Challenge (Kaggle) — Alakazam single-prize heuristic agent
(ladder submission + NN teacher) and the learned-piloting neural-net track.*

**Last updated:** 2026-07-06

---

## Current State

- **Active ladder submission:** v28 Alakazam heuristic (`main.py` + `deck.csv`),
  shipped 2026-07-05 — board-thinning fix + belief-driven wall anticipation
  (Phase C) + confidence recalibration.
- **NN track:** AlphaZero-style push in progress — self-play-with-search
  collection validated; retrain on the corrected corpus is the next step
  (`docs/nn-training.md` §Resume Here).
- **Deadlines:** Ladder ~Aug 16–17 · Report ~Sep 13 · Team merger ~Aug 9, 2026.
- **Engine runs locally:** `pip install kaggle_environments --no-deps` → ~0.5s/game
  (see `training/README.md`). No Kaggle session needed for self-play or A/B testing.

---

## Orientation (start here)

1. **`CLAUDE.md`** — slim always-loaded orientation: current state, working rules,
   pointers. Read first.
2. **`docs/project-reference.md`** — the full reference layer: repo tree, engine
   API summary, complete deck tables, `main.py` architecture, packaging/shipping.

---

## Layout

| Path | What it is |
|------|------------|
| `main.py` + `deck.csv` | The submission: v28 heuristic agent + 60-card deck |
| `docs/` | All project documentation (canonical homes per topic — see table below) |
| `training/` | Local training & evaluation rig: harness, A/B, gauntlet, NN track (`training/README.md`) |
| `opponents/` | Meta opponent pool agents (gauntlet anchors + collection sparring) |
| `tools/` | Replay forensics, deck audit/math, meta survey, replay download |
| `replays/` | `bulk/` (downloaded ladder replays) + `exploiter_wins/` (board-thinning evidence) |

---

## Docs Map

| Doc | Canonical home for |
|-----|--------------------|
| `docs/project-reference.md` | Full reference breakdown (tree, deck, architecture, shipping) |
| `docs/competition-strategy.md` | Stage 0–5 roadmap (§Master Plan), writeup strategy |
| `docs/report-log.md` | Experiment journal + glossary — the report is assembled from this |
| `docs/engine-api.md` | cabt engine API (enums, option schema, verified behaviors) |
| `docs/version-history.md` | Agent version change log |
| `docs/nn-training.md` | NN architecture, training log, phased roadmap |
| `docs/belief-model.md` | Belief model design + phase plan (Stage 3) |
| `docs/game-nature.md` | Game mechanics / decision-structure rundown |
| `docs/piloting-guide.md` | Expert Alakazam piloting logic (NN training target spec) |
| `docs/matchups.md` | Matchup reference + tech cheat-sheet |
