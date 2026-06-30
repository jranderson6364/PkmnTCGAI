# PkmnTCGAI

Pokemon TCG AI Battle Challenge (Kaggle) — Alakazam heuristic agent + neural net track.

**Active ladder submission:** v14 Alakazam heuristic (`main.py` + `deck.csv`)  
**Competition deadline:** Ladder ~Aug 16–17 · Report ~Sep 13 · Merger ~Aug 9, 2026

## Quick Start (AI sessions)

Read `CLAUDE.md` — it is the single source of truth for this project.

## Docs

- `CLAUDE.md` — master context: competition facts, engine API, deck card IDs, v14 architecture, outstanding items
- `docs/nn-training.md` — full NN training log, architecture, self-play roadmap
- `docs/piloting-guide.md` — expert Alakazam piloting logic (BC target spec)
- `docs/matchups.md` — matchup reference and tech cheat-sheet
- `docs/version-history.md` — v1–v14 change log
- `docs/competition-strategy.md` — strategy analysis and report guide

## Files

- `main.py` — v14 heuristic agent source (564 lines)
- `deck.csv` — 60-card Alakazam deck, one card ID per line
