Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.
2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.




# PTCG AI Battle Challenge — Orientation

This file is the slim always-loaded layer: current state, working rules, and
pointers. **The full breakdown — repo tree, engine API summary, complete deck
tables, `main.py` architecture, NN details, packaging/shipping — lives in
`docs/project-reference.md`. READ IT before any task that touches the engine,
the deck, `main.py` internals, training, or shipping.** It is the reference
layer; this file only orients.

---

## Quick Orientation

**Competition:** Pokemon TCG AI Battle Challenge (The Pokemon Company × HEROZ × Matsuo Institute × Kaggle).
**Goal:** Top-8 in the Strategy track → $30k + Tokyo finals.
**Scoring:** 70% model approach / 20% deck concept / 10% report — but the report
is the *delivery channel* for the 70%: report-driven development, every experiment
logged same-day in `docs/report-log.md`.
**Key insight:** Rule-based bots cap out at ~0% on the 70% axis. A learned piloting
agent is the only path; the heuristic is the ladder placeholder and DAgger teacher.

**Current agent:** v25c (`main.py` + `deck.csv`, submission 54282648, shipped
2026-07-03, user-reported ladder Elo peaked ~900, settled ~880; gauntlet gElo
589, top of the whole table) — see the v25c paragraph below. Previously
v25b — v23 deck (reverted from v24 on
2026-07-03; v24's ladder reading (680) prompted a user call to revert ahead of
the 48h decision checkpoint) + 5 heuristic logic fixes from a full retreat/
energy/evolution audit (stype==9 deck-out, Kadabra retreat, Wondrous Patch
targeting, Enriching deck-safety gate, manual-evolve-vs-Candy racing gate,
shipped as submission 54277762) + 2 more replay-verified fixes: retreat
scoring no longer force-retreats a fully-fueled Alakazam whenever the
opponent has a Mist/Rock wall regardless of whether we can exploit it, and
Rare Candy racing gained an offensive trigger (races to Alakazam when it sets
up a near-term KO, not just when already in danger). Shipped 2026-07-03,
submission 54279766, **publicScore 861.8** — up from v25's 732.0 and above
v24's 698.1, and gauntlet gElo 551 (top of the whole table, first version to
clear v24 offline too). Also fixed a `training/harness.py` scoring bug that
miscounted opponent-crash games as ties in one seat direction (root cause of
the "Dragapult ties" item below — not a game anomaly). See
`docs/version-history.md` and `docs/report-log.md` 2026-07-03 entries for the
full data and caveats.
**v25c (shipped 2026-07-03, submission 54282648):** replay-verified fixes from
5 fresh ladder losses — a `desperation` mode (opponent ≤1-2 prizes from
winning) that overrides deck-out caution and forces racing to Alakazam
instead of trying to out-survive a lethal hit; a `lone_active_opportunity`
heuristic (opponent's bench is empty + a rough max-hand estimate clears the
KO threshold → stop banking draw and go for the kill); fixed `_score_deck_search`
preferring a dead-weight Alakazam/Kadabra fetch over Abra when no line piece
exists anywhere; fixed Hilda's search (Stage-1/2 + energy only, can't fetch
Basics) beating Dawn (Basic+Stage1+Stage2) by raw weight even with zero Abra
in play; and fixed Boss's Orders' `PHASE_CLOSING` branch firing an
unconditional flat score regardless of target quality, confirmed wasting Boss
plays in 2 separate replays. User-reported ladder Elo peaked ~900, settled
~880 as of 2026-07-03 evening. Gauntlet re-run 2026-07-03 overnight (200
games/anchor, dragapult excluded — crashes locally): gElo 589, top of the
whole table (above v25b's 559, v24's 516, v23's 499). See
`training/ladder_history.csv`, `docs/report-log.md` 2026-07-03 entry, and
`docs/version-history.md` v25c entry for full replay evidence. Open,
unconfirmed lead: `main.py` has no handling for the MULLIGAN select context.
Open, unfixed pattern: "board-thinning" (ending up with 1-2 Pokémon in play
and a bloated dead hand after the attacker line gets repeatedly KO'd).
**Roadmap (canonical: `docs/competition-strategy.md` §Master Plan):**
Stage 0 deck freeze + Gauntlet baseline → Stage 0b heuristic tuning → Stage 1
DAgger → Stage 2 advantage-weighted self-play → Stage 3 belief model (parallel) →
Stage 4 hardening → Stage 5 search-at-inference (Kaggle-gated).
**Engine runs locally:** `pip install kaggle_environments --no-deps` → ~0.5s/game.
Rig + workflows: `training/README.md`.
**NN track:** BC done on the OLD deck (22% vs teacher = compounding error →
DAgger next). Re-collect after deck freeze. `docs/nn-training.md` §Resume Here.

---

## Competition Facts

| Item | Detail |
|------|--------|
| Ladder ends | ~Aug 16–17, 2026 (slug: `pokemon-tcg-ai-battle`) |
| Strategy report due | ~Sep 13, 2026 (slug: `pokemon-tcg-ai-battle-challenge-strategy`) |
| Team merger deadline | ~Aug 9, 2026 |
| Submissions | 5/day; only the latest 2 count for final standing |
| Submission format | `main.py` + `deck.csv` → `submission.tar.gz` (ship via Kaggle CLI — see project-reference §Packaging) |
| Agent contract | `def agent(obs_dict: dict) -> list[int]` returning legal option indices |
| Clock | 10 minutes total per match; timeout = instant loss |
| Rating | Win/loss/tie only; margin irrelevant |

---

## Deck Essentials (full tables: `docs/project-reference.md` §Deck)

**Alakazam single-prize control (v23 deck, 60 cards).** Win condition: **Powerful Hand**
(Alakazam 743, cost 1 Psychic) — `damage = 20 × hand_size`. Hand size IS the damage
stat; never discard unnecessarily. KO threshold: `ceil(opp_active_hp / 20)` cards.
**Blocked by** Mist Energy (#11) and Rock Fighting Energy (#20) on the opponent's
Active. Energy routing: exactly one Psychic on Alakazam, then Kadabra → Abra;
Enriching (13) → Dudunsparce only, never Alakazam.

---

## Where Things Live

| Need | Go to |
|------|-------|
| **Full reference breakdown** | `docs/project-reference.md` |
| Roadmap / writeup strategy | `docs/competition-strategy.md` |
| Experiment journal + glossary + target figures | `docs/report-log.md` |
| Engine API (canonical) | `docs/engine-api.md` |
| Version change log (v1–v24) | `docs/version-history.md` |
| NN training log + pipeline | `docs/nn-training.md` |
| Belief model design + phases (Stage 3) | `docs/belief-model.md` |
| Training rig workflows (A/B, gauntlet, tuning) | `training/README.md` |
| Piloting logic spec / matchups | `docs/piloting-guide.md`, `docs/matchups.md` |

---

## Outstanding Items (Priority Order)

Stage numbers refer to `docs/competition-strategy.md` §Master Plan.

1. ~~Ship the v23-deck revert + v25 logic fixes~~ DONE 2026-07-03: submission
   54277762 (v23-deck revert + 5 heuristic fixes — stype==9 deck-out, Kadabra
   retreat, Wondrous Patch targeting, Enriching deck-safety gate, manual-evolve-
   vs-Candy racing gate); ~~ship v25b~~ DONE 2026-07-03: submission 54279766
   (Mist-wall retreat fix + Candy-racing offensive trigger, publicScore 861.8,
   gElo 551); ~~ship v25c~~ DONE 2026-07-03: submission 54282648 (desperation
   mode + lone_active_opportunity + 3 replay-verified fixes, user-reported
   ladder ~880 settled, gElo 589 — top of the table). v24 deck swap had been
   reverted 2026-07-03 on early ladder trend, ahead of the documented 48h/50-point
   decision rule — see `docs/report-log.md` 2026-07-03 entries (the local A/B
   favoring v24 stands unexplained; revisit the deck-simplification question
   with more ladder data before retrying it). ~~Stage 0b weight tune~~ DONE
   2026-07-02: gate not cleared (52.0% ± 4.0% over 600 games) → default `W`
   kept; see `docs/report-log.md`.
2. ~~Stage 0: full Gauntlet baseline~~ DONE 2026-07-03 (overnight run): v25c
   gElo 589 vs the 7 working anchors (dragapult excluded, see item 6), top of
   the table. See `training/ladder_history.csv` and `docs/report-log.md`
   2026-07-03 entry.
3. **Stage 1: BC re-collect + retrain on the frozen deck, then DAgger.**
   ~~BC re-collect~~ DONE 2026-07-03: `training/bc_data_v25c*.pkl`, 2000 v25c
   self-play games, 579,169 samples. ~~Build `dagger_collect.py`~~ DONE
   2026-07-03: `training/nn/dagger_collect.py`, smoke-tested end-to-end (0
   relabel errors). ~~Retrain on frozen deck~~ DONE 2026-07-03:
   `training/ptcg_bc_v2.pth`, 10 epochs (~106 min CPU, confirmed runs locally,
   no Kaggle GPU needed), val_top1_acc 0.875 peak. Gated: 86% vs random
   (matches v1), 17% vs v25c heuristic (expected BC-plateau signature — v25c
   is a stronger teacher than v22 was). ~~DAgger round 1 collection~~ DONE:
   1000 games, `training/dagger_data_r1*.pkl`, 326,240 samples, 0 relabel
   errors. ~~Retrain + gate round 1~~ DONE: both a 3-epoch and a heavier
   10-epoch retrain (`ptcg_dagger_r1.pth`/`ptcg_dagger_r1b.pth`) gated flat
   at 12%/15% vs teacher (statistically tied with BC's 17% at n=100). **The
   decisive check (per advisor): 100-game win-rate can't resolve DAgger's
   actual target.** Measured teacher-agreement on 3000 FRESH deployment-
   realistic (argmax) states instead: BC 74.9% → DAgger round 1 79.7% — a
   real +4.8pp gain. **DAgger is confirmed working**; the flat win-rate is a
   measurement-resolution artifact (single-decision gains compound over
   ~150 decisions/game but need more rounds or larger n to show up in
   head-to-head win-rate), not a broken pipeline. Also fixed the REAL root
   cause of two silent OOM kills (not just papered over with `--bc-limit`):
   `dataset.py::load_shards` always read every glob-matched shard fully
   before slicing, so even a small final cap needed the whole ~37GB corpus
   in RAM transiently — fixed to cap during read (shuffles shard order, not
   post-load samples). See `docs/report-log.md` 2026-07-03 entries for the
   full story. ~~DAgger round 2 (user-directed, testing lower collection
   temperature)~~ DONE: collected 1000 games at temp 0.2 (vs round 1's 1.0),
   retrained on BC+round-1+round-2 combined → `ptcg_dagger_r2.pth`, gated 81%
   vs random, **16% vs teacher — still flat**. Fresh-state fidelity re-check:
   BC 73.1% → r1b 81.1% (+8pp, confirmed) → r2 **81.9%** (only +0.8pp) —
   **diminishing returns; the temperature lever helped a little but fidelity
   is flattening near 80-82%, not accelerating.** Two rounds of evidence now
   agree further DAgger rounds are unlikely to move the needle much more.
   **DAgger paused here** — `ptcg_dagger_r2.pth` is the best checkpoint
   (strongest report evidence yet: fidelity gains don't linearly convert to
   win-rate on this deck) but not ship-ready. Gate 50%+ vs teacher → ship
   (per advisor: imitation asymptotes to parity, never above — exceeding it
   needs Stage 2 AWR/search, not more imitation rounds).
4. **Stage 2: advantage-weighted self-play** — gate 55–60% vs teacher over 400 games.
5. **Stage 3 (parallel): belief model** — archetype classifier + accuracy-by-turn
   figure; fixes `opp_likely_ace_spec` hardcoded to True.
6. ~~Dragapult step-limit ties~~ DIAGNOSED 2026-07-03: not step-limit draws at
   all — `opponents/dragapult_agent.py` crashes 100% of local games (Kaggle-
   only `cg.api` import unavailable locally), and `training/harness.py`'s
   `summarize()` miscounted opponent-crash-as-P0 as a tie instead of a win.
   Both are fixed except the anchor's underlying crash (optional follow-up:
   vendor `cg/api.py` from the downloadable `kiyotah/cg-lib` Kaggle dataset —
   confirmed it's plain ~26KB Python, not urgent). The `dragapult` column in
   `training/gauntlet_results.csv` has never reflected real play for any prior
   version; disregard it until the anchor itself is fixed.

---

## Design Principles

1. **Real-ladder A/B is the only honest evaluator.** Offline win rates systematically overrate (v5: 64% offline, 0-5 live). Gauntlet gElo ranks candidates; the ladder decides.
2. **Timeout = instant loss.** Every inference path needs a guaranteed fast fallback (`_safe_return`).
3. **Single-prize + non-ex beats the meta.** Crustle immunity walls ex attackers; Alakazam hits it normally. Prize trade: give 1, take 2-3.
4. **Hand size IS damage.** No Professor's Research, no Iono, no Ultra Ball.
5. **Rule-based caps at ~0% on the 70% axis.** The NN track is the goal; the heuristic (v23) is placeholder + teacher.
6. **5 submissions/day, latest 2 count.** The ladder is an A/B rig; ship via the Kaggle CLI without asking.

---

## Documentation Standards

**MANDATORY — after ANY change (code, deck, plans, results), update the docs in
the same session:** the canonical `docs/*.md` home for that fact, the matching
section of `docs/project-reference.md`, and the summary lines here (Quick
Orientation / Outstanding Items) if the current state changed. A change without
its doc update is an unfinished change — this is how the v24 deck swap briefly
went missing from this file.

Every doc file (`CLAUDE.md`, every `docs/*.md`) follows the same shape:

```
# Title

*One- to three-line italic description of what this file is for.*

**Last updated:** YYYY-MM-DD

---

## Section
content
---
## Next Section
content
```

Rules:
1. **`---` separates every top-level (`##`) section**; never stack two `---`.
2. **`**Last updated:**` is mandatory** — bump on material changes, real date.
3. **One canonical home per fact** — link, don't copy-paste. `CLAUDE.md` stays a
   slim orientation layer; durable reference lives in `docs/project-reference.md`;
   narrative history, API dumps, and matchup detail live in their dedicated docs.
4. **Version/change-log entries newest-first**, each under its own `##` heading.
5. **Tables over prose** for anything enumerable.
6. **When a file's status changes**, update the home doc AND the pointer/summary
   here in the same edit — don't let them drift.
7. **Every experiment → same-day `docs/report-log.md` entry** (hypothesis, plain-
   English method, numbers, decision, report relevance); **every ladder ship →
   `training/ladder_history.csv` row.** The report is assembled from these —
   nothing is reconstructed in September.
