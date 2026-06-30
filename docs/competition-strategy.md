# Competition Strategy & Report Guide

*Summary of the strategic analysis for the PTCG AI Battle Challenge.*

---

## The Thesis in One Paragraph

Top 8 is **not** won by the best forward-search engine or the flashiest RL run. It is won by a **learned-piloting agent on a deliberately simple deck**, trained primarily by **imitation from ladder replays** (which include strong players), optionally fine-tuned with light self-play, and deployed with **fast, shallow determinized search** under the 10-minute clock — then written up as a rigorous "model approach." This is the one path that scores on *both* axes: real ladder performance *and* the 70%-weighted "model approach" rubric.

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
- **BC from high-rated replays → RL → light MCTS at inference** is the proven recipe.
- Imitation is the cheap, high-leverage half; self-play RL is the expensive half.
- BC alone needs only a modest GPU and the replay data — closes most of the piloting gap.

---

## Recommended Method

**(1) Behavior cloning from ladder replays — do this first, 60% of the win.**
Download daily episode datasets, filter to high-rated agents' games, train a network to predict the action taken from the observation. Plain supervised learning: cheap, stable, no self-play infrastructure. Directly transplants *good piloting* into the agent.

**(2) Use the cloned net as policy/value inside light search.**
Wrap in **shallow determinized / IS-MCTS**: sample plausible opponent hands/decks consistent with public info, run a *small* number of simulations guided by the net's priors, average. Budget search by wall-clock with a hard latency guard.

**(3) Optionally fine-tune with self-play RL — only if compute allows.**
Once cloning works, self-play RL (policy-gradient or AlphaZero-style) squeezes out the last increment and lets you exceed the players you imitated. This is the expensive part. Cloning alone is plausibly enough to clear rule-based bots.

**(4) Deck: simple, single-prize, meta-aware. Lock early, stop touching it.**
Alakazam — single-prize, low-branch, consistent, flat deterministic damage. Single-prize sidesteps prize-trade math, dodges Crustle ex-immunity, gives clean learning targets.

---

## Concrete Plan

**Phase 0 — now → July 1:** Keep v14 (or better, a simple rule-based) on the ladder. Prep the replay-download + dataset-builder script and observation→tensor encoder.

**Phase 1 — July 1 → mid-July:** Download and filter replays. Train the policy/value net. Ship inside the existing search scaffold with a strict latency guard. **Success metric: beats the untrained baseline and a clean rule-based bot in real-ladder A/B.**

**Phase 2 — mid-July → mid-Aug:** Improve encoder, add strongest opponents' games, tune search depth vs clock. If compute permits: self-play RL fine-tune. Lock the deck. Freeze best two submissions before the Aug 16 ladder deadline.

**Phase 3 — mid-Aug → Sep 13:** The writeup. Budget real time here — it's 30% of the score directly (deck + writing) and the *presentation* of the 70% method axis.

---

## Writeup Strategy

The 2,000-word report should read like a tight research note:

- **Lead with the method and its justification.** Frame it as: imperfect-information game → determinized/IS-MCTS with a learned value/policy → trained by imitation from real ladder data → optional self-play refinement. Cite the imperfect-information rationale explicitly (why not vanilla AlphaZero). This is your 70%.
- **Show ablations / evidence.** "Untrained search went 0-5; cloned policy went X-Y on the same ladder bracket." Quantified, real-ladder, honest.
- **Make the deck a thesis, not a list.** Single-prize, consistency-over-burst, immune to Crustle ex-counter meta, clean learning signal. That's your 20%, and it's a *story*.
- **Use hard-won findings as credibility.** The draw-scoring bug, real prize values, text-driven damage formulas, the lone-Pokémon ability forfeit trap — these show you understood the engine at a level most won't.
- **Be honest about the offline/ladder gap.** Stating "local sims overrate; we optimized on the real ladder" is exactly the kind of mature insight that scores.

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Latency cliff (timeout = loss) | Hard wall-clock guard; guaranteed legal fallback in `_safe_return` |
| Chasing offline numbers | Ladder-only evaluation discipline |
| Sep 13 deadline collides with Stanford move-in | Front-load method by mid-Aug; writeup is not for the final week |
| Team merger: highest single odds move | Seek RL-experienced partner with GPU compute; float post in July before ~Aug 9 deadline |

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

**Two-deck plan:** Primary = Alakazam (learned approach unlocks the Stage-2 setup). Fallback = Bellibolt ex (maximum simplicity, proven top-Elo, use to hold ladder now).
