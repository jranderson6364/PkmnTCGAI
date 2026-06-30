# Agent Version History

*From Mega Lucario to Alakazam. Each version listed with what changed and why.*

---

## v1–v6: Progressive Heuristic Bug Fixes (Mega Lucario era)

- v1-v3: Initial agent, basic scoring
- v4-v5: Added determinized forward search (MCTS scaffold), energy type detection, Mist Energy detection
- v6: ~median ladder position (~1500/3000). Complete scorer with board census.

**Key lesson:** Mega Lucario ex went 0-5 on live ladder despite ~64% offline win rate. Root causes: (1) 3-prize target = terrible prize math; (2) megaEx = Crustle-walled; (3) options are positional with no cardId — v7's card-specific rules were silently dead code.

---

## v7: Full Logic Audit + Option Resolution Fix

**The audit finding:** Engine options are *positional*, carry no `cardId`. All of v6's card-specific rules keyed on `o.get('cardId')` which is always absent → silently fell to generic fallbacks. The agent was far simpler than designed.

**Fix:** Resolve options through visible hand/board: `PLAY/ATTACH/EVOLVE → hand[o['index']].id`; `ABILITY → (area,index) → active/bench pokemon`.

**83% vs random** as BC teacher. Served as teacher for NN Phase 0 BC warm-start (113k samples).

---

## v8: P0-P3 Core Fixes

- P0: Option resolution via hand/board (unlocked everything)
- P1: Energy routing by type (Enriching → Dudunsparce, Psychic → Alakazam)
- P2: Psychic-aware "ready" check (`5 in pokemon.energies`, not just any energy)
- P3: Attack discipline (draw to lethal before swinging)

---

## v9: Bench Priority

Never sit on a lone Active. Bench a Basic scaled by board emptiness. Poffin first.

---

## v10: Speed Setup + Boss Combo

- Reach Alakazam fast (Rare Candy score 13 / evolve-to-Alakazam score 14)
- Boss + Powerful Hand combo (P4)
- De-prioritize busywork (Sacred Ash/Lana's Aid 3.0, Handheld Fan 1.5)

---

## v11: Supporter Preservation

Hold Supporter for a Boss snipe (partial implementation). Hit ~700 Elo (~1600/3200, ≈median).

**Ceiling identified:** greedy can't SEQUENCE a full turn. The Boss/Supporter conflict is the canonical example — greedy plays the draw Supporter first, then can't Boss the target.

v11 became the **BC teacher for NN track** (113k samples).

---

## v12: Turn Planner — FAILED (regression)

Added a bounded turn planner at pivotal nodes (engaging only when Alakazam is Active + attack offered + KO-able target exists).

**Result:** 39.8% vs v11 — a regression. Lines C/D burned Supporters incorrectly. Never ladder-tested; superseded.

---

## v12b: Partial Revert

Fixed the regression but `boss_ex_snipe` never fired due to name-based ex detection bug (checked pokemon name for " ex" suffix, unreliable).

---

## v12c: 8 Audited Fixes

- Enriching/Dudunsparce loop
- Rare Candy timing
- Direct `pokemon.ex`/`pokemon.megaEx` boolean detection (replaces name-based)
- Evolution scoring fix
- Hybrid boss-target picker

Built but never ladder-tested; superseded by v13/v14 detour.

---

## v13: Phase Rewrite (PDF-Driven) — Never Validated

Introduced 4-phase state machine (ESTABLISH/CONVERT/PRESSURE/CLOSING) and hand-conservation-as-damage principle, based on user-provided "Alakazam Manual" PDF.

**A/B harness was run but results were never reported back.** v13's actual win rate vs v11/v12c is UNKNOWN. Superseded by v14.

---

## v15: Heuristic Fixes + Training Infrastructure — CURRENT ACTIVE SUBMISSION

Three targeted fixes to the greedy scorer, plus opponent pool and training plan staged for Vivobook.

**Heuristic changes:**
1. **Bench Alakazam evolution scoring** — `inPlayArea==5` (bench) Kadabra→Alakazam now scores 50/40/25/12 by phase (was 16/10). Getting a second Alakazam on bench is critical for continuity when the active gets KO'd.
2. **Enhanced Hammer escalation** — scores 45 when opponent has Mist/Rocky Energy (was 28). Blocking energy makes Powerful Hand deal 0; removing it is near-mandatory, so the old score was dangerously low.
3. **Battle Cage reactive** — scores 22 when bench damage detected in logs (was flat 6). Now reacts to Dragapult/Starmie spread with same urgency as Shaymin.

**Training infrastructure (code complete, Vivobook needed to run):**
- `opponents/starmie_agent.py` — Mega Starmie ex spread (330 HP megaEx)
- `opponents/lucario_agent.py` — Mega Lucario ex + Rocky Energy lock (340 HP megaEx)
- `opponents/dragapult_agent.py` — Dragapult ex Stage 2 spread (Phantom Dive)
- `docs/training-setup.md` — full training plan: opponent pool, curriculum, reward shaping, checkpoint pool

**Status:** Committed. A/B harness validation pending.

---

## v14: Replay-Driven Fixes — SUPERSEDED BY v15

Built directly from 4 real replay JSONs (vs Shachify, 3fk, Nicholas Low, Evan Liu — 2 losses, both to deck-out).

**What changed:**
1. **Fez suppressed by default** (-1.0 score) — 2-prize target; only activates reactively
2. **Sacred Ash deck-out prevention** — scores 35 at deck<5, 25 at deck<10 (was 2-5)
3. **Dudunsparce overdraw guard** — suppressed when deck<10 or hand≥14
4. **Evolution scoring fix** — Kadabra→Alakazam scores 250-270 (was 13), near-mandatory
5. **Boss prize-value guard** — requires `prize_value(target) >= prize_value(opp_active)`
6. **Rock Energy detection** — Enhanced Hammer scores 28 for Mist (#11) OR Rock (#20)
7. **Genesect role** — Bench+Fan scores 11 (ACE Nullifier blocks Rocky Energy plays)
8. **Shaymin reactive** — scores 16 when bench damage detected in logs
9. **Psyduck reactive** — scores 18 when opponent self-damage cards detected (placeholder IDs — see CLAUDE.md Outstanding Items)

**Status:** Submitted to ladder. A/B harness validation pending.

---

## NN Track (parallel to heuristic v12+)

See `docs/nn-training.md` for full details.

- Phase 0 (BC): complete. 52% net vs v11 teacher.
- Phase 1 (self-play): attempt 2 running. Best: sp2_iter2.pth at ~55% vs teacher.
- Paused while v14 heuristic detour runs.

---

## Deck Evolution

| Era | Deck | Why abandoned |
|-----|------|---------------|
| v1-v11 | Mega Lucario ex | 3-prize giveaway; Crustle-walled; 0-5 live |
| v12+ | Alakazam (Powerful Hand) | Single-prize; non-ex; flat deterministic damage; meta-proven |
