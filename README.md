# PkmnTCGAI

*Pokemon TCG AI Battle Challenge (Kaggle) — Alakazam single-prize heuristic agent
(ladder placeholder + DAgger teacher) and the learned-piloting neural-net track.*

**Last updated:** 2026-07-02

---

## Current State

- **Active ladder submission:** v24 Alakazam heuristic (`main.py` + `deck.csv`) —
  shipped 2026-07-02, rating pending; v23 sits at ladder public score 796.3.
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
| `main.py` + `deck.csv` | The submission: v24 heuristic agent + 60-card deck |
| `docs/` | All project documentation (canonical homes per topic — see table below) |
| `training/` | Local training & evaluation rig: harness, A/B, gauntlet, SPSA tune, NN track (`training/README.md`) |
| `opponents/` | Meta opponent pool agents (+ `samples/`: official Kaggle sample notebooks) |
| `tools/` | `analyze_replay.py` (replay forensics), `deck_audit.py` (per-card utilization) |
| `variants/` | Scratch dir for deck-variant A/Bs (`variants/README.md`) |
| `replays/` | Ladder replay forensics by version (summaries tracked; raw JSONs local-only) |

---

## Docs Map

| Doc | Canonical home for |
|-----|--------------------|
| `docs/project-reference.md` | Full reference breakdown (tree, deck, architecture, shipping) |
| `docs/competition-strategy.md` | Stage 0–5 roadmap (§Master Plan), writeup strategy |
| `docs/report-log.md` | Experiment journal + glossary — the report is assembled from this |
| `docs/engine-api.md` | cabt engine API (enums, option schema, verified behaviors) |
| `docs/version-history.md` | v1–v24 agent change log |
| `docs/nn-training.md` | NN architecture, training log, phased roadmap |
| `docs/belief-model.md` | Belief model design + phase plan (Stage 3) |
| `docs/piloting-guide.md` | Expert Alakazam piloting logic (NN training target spec) |
| `docs/matchups.md` | Matchup reference + tech cheat-sheet |
