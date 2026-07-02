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

**Current agent:** v24 (`main.py` + `deck.csv`) — v23 logic on the simplified deck
(Psyduck/Genesect → 4th Alakazam + 4th Dunsparce; 60% ± 6.8% vs frozen v23 over
200 games). Shipped 2026-07-02, rating pending; v23 sits at ladder public score
796.3. See `docs/version-history.md`.
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

**Alakazam single-prize control (v24, 60 cards).** Win condition: **Powerful Hand**
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

1. **Ladder-confirm v24** (submission 54265639) → fill the
   `training/ladder_history.csv` row → declare the 60 frozen. Day-one reading
   (690 vs v23's 773) is provisional noise; decision rule: if v24 is still 50+
   below v23 after ~48h of episodes, audit v24 loss replays (specifically: losses
   to Rocky/Mist walls — Genesect's Nullifier is the one cut that could matter).
   ~~Stage 0b weight tune~~ DONE 2026-07-02: gate not cleared (52.0% ± 4.0% over
   600 games) → default `W` kept; see `docs/report-log.md`.
2. **Stage 0: full Gauntlet baseline** — `python training/gauntlet.py
   --candidate main.py --name v24 --games 200` (all 8 anchors).
3. **Stage 1: BC re-collect + retrain on the frozen deck, then DAgger** — build
   `training/nn/dagger_collect.py`; 2–3 rounds; gate 50%+ vs teacher → ship.
4. **Stage 2: advantage-weighted self-play** — gate 55–60% vs teacher over 400 games.
5. **Stage 3 (parallel): belief model** — archetype classifier + accuracy-by-turn
   figure; fixes `opp_likely_ace_spec` hardcoded to True.
6. **Dragapult step-limit ties** — 50/100 local games tie in one seat direction;
   diagnose if ladder Dragapults match (ties = half-losses).

---

## Design Principles

1. **Real-ladder A/B is the only honest evaluator.** Offline win rates systematically overrate (v5: 64% offline, 0-5 live). Gauntlet gElo ranks candidates; the ladder decides.
2. **Timeout = instant loss.** Every inference path needs a guaranteed fast fallback (`_safe_return`).
3. **Single-prize + non-ex beats the meta.** Crustle immunity walls ex attackers; Alakazam hits it normally. Prize trade: give 1, take 2-3.
4. **Hand size IS damage.** No Professor's Research, no Iono, no Ultra Ball.
5. **Rule-based caps at ~0% on the 70% axis.** The NN track is the goal; the heuristic (v24) is placeholder + teacher.
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
