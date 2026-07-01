# PkmnTCGAI

Pokemon TCG AI Battle Challenge (Kaggle) — Alakazam heuristic agent + neural net track.

**Active ladder submission:** v21 Alakazam heuristic (`main.py` + `deck.csv`) — currently ~750 Elo (1600/4000)  
**Competition deadline:** Ladder ~Aug 16–17 · Report ~Sep 13 · Merger ~Aug 9, 2026

## Quick Start (AI sessions)

Read `CLAUDE.md` — it is the single source of truth for this project.

## Docs

- `CLAUDE.md` — master context: competition facts, engine API, deck card IDs, v21 architecture, outstanding items
- `docs/nn-training.md` — full NN training log, architecture, self-play roadmap, "Resume Here" checklist
- `docs/training-setup.md` — self-play + curriculum training plan for the opponent pool
- `docs/piloting-guide.md` — expert Alakazam piloting logic (NN training target spec)
- `docs/matchups.md` — matchup reference and tech cheat-sheet
- `docs/version-history.md` — v1–v21 change log
- `docs/competition-strategy.md` — strategy analysis and report guide

## Files

- `main.py` — v21 heuristic agent source
- `deck.csv` — 60-card Alakazam deck, one card ID per line
- `opponents/` — training-opponent stubs (Mega Starmie ex, Mega Lucario ex, Dragapult ex) for the self-play pool; deck IDs still placeholder, see `docs/nn-training.md`
- `tools/analyze_replay.py` — kaggle-env replay decoder/auditor (missed lethals, bad retreats, bad Boss targets, wasted energy attaches, timeouts); `python3 tools/analyze_replay.py <replay.json> [more...]`

## Local Development

`cg-lib` (the game engine, `kiyotah/cg-lib`) is currently only known to be available
as a Kaggle "Add Input" notebook dataset, not a pip package — unverified whether it
can be downloaded and used standalone outside a Kaggle session
(`kaggle datasets download -d kiyotah/cg-lib` is the thing to try). Until that's
confirmed:

- **Self-play / NN training** requires a Kaggle notebook session (GPU for training,
  CPU for data collection).
- **Heuristic-only work** — reading/editing `main.py`, running
  `tools/analyze_replay.py` against downloaded replay JSONs, editing docs — needs
  nothing beyond Python 3, no engine dependency.
