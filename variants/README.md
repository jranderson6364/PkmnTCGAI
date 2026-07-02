# variants/ — Deck Variant Agents (Stage 0)

*Scratch directory for deck-variant A/B candidates. Losers get deleted; git has
the history.*

**Last updated:** 2026-07-02

---

## Workflow

Each deck variant = a full copy of `main.py` with only the `DECK` list (and its
count constants) edited, so `training/ab_test.py` and `training/gauntlet.py` can
load it directly (they need `agent` + `DECK` per file). Same piloting logic, so
any win-rate difference is the deck.

Workflow per variant:

```bash
cp main.py variants/no_psyduck_4th_zam.py   # then edit DECK in the copy (keep 60 cards)
python training/ab_test.py variants/no_psyduck_4th_zam.py main.py 600
```

Winner gets ladder-confirmed, promoted into `main.py`, `deck.csv` regenerated
from `DECK`, and the 60 is then frozen permanently (see
`docs/competition-strategy.md` §Stage 0). Delete losing variants — this
directory is scratch, not history (git has the history).
