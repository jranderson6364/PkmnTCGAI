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

**83% vs random.** Served as teacher for NN Phase 0 imitation warmup (113k samples of v7 self-play).

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

v11 became the **imitation warmup teacher for NN track** (113k self-play samples). Note: opponent BC (imitating other decks' action sequences) is not viable — decks differ too much.

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

## v17: Competitive-Research Alignment + Threshold Discipline — CURRENT ACTIVE SUBMISSION

Did a full research pass on how the Alakazam/Dudunsparce deck is actually piloted at
the top level (Cerys Jones' 1st-place Indianapolis Regional list, CL Osaka 2026, the
Limitless meta lists, and the TCGplayer/Cardsrealm/Deltia guides) and rewrote
`docs/piloting-guide.md` into a comprehensive v3 strategy doc — every card's role, the
card-economy table, turn-by-turn sequencing, Boss target priority, energy routing,
Iono play-around, deck-out avoidance, all matchups, an explicit our-list-vs-meta-list
comparison, and a strategy→heuristic map with the remaining gaps.

**Key finding:** our 60-card backbone is *identical* to the meta list (4 Abra / 4
Kadabra / 3 Alakazam / 3 Dunsparce / 3 Dudunsparce / 4 Poffin / 4 Poké Pad / 4 Dawn /
3 Hilda / 3 Rare Candy / 2 Enhanced Hammer / 2 Handheld Fan / 1 Sacred Ash / 1 Lana /
4 Telepath / 1 Enriching), so the win has to come from piloting, not the decklist. The
single most-cited skill in every guide — "draw to the KO threshold, then **stop** and
bank surplus on the bench" — was exactly the principle our deck-out losses violated.

**Heuristic changes:**
1. **`hand_surplus` threshold discipline.** When a ready attacker exists (Active or
   bench Alakazam) and `hand_n >= cards_needed`, with no Boss-snipe plan and not an
   emergency, all non-essential draw is suppressed: Dudunsparce ability → 0.5, Fez
   ability → -3, Dawn/Hilda/Poké Pad → 2.0. Re-running the deck-out replay through the
   patched agent shows it now ENDs/attacks instead of firing Dawn/Poké Pad on five
   separate overdraw turns, conserving the deck cards it previously milled itself out on.
2. **Dudunsparce Run Away Draw** now hits a hard -8 floor at `deck_danger` (<5), not
   just the -2 at `deck_critical` (<10).
3. **`active_immobile` rescue attach prefers Psychic** (65 vs 55 for colorless) so a
   stranded Alakazam gets energy that enables both retreat and attack, and the lone
   Enriching isn't wasted on the rescue.

**Verification:** agent runs clean on all 1,672 real selections across the 6 replays
(0 exceptions, 0 illegal empty returns); guaranteed-lethal lines still taken. Deck
unchanged (deck is 20% of scoring; the validated 60 is meta-identical).

**Status:** Committed. Not yet ladder-validated.

---

## v16: Replay-Driven Bug Hunt

Six real ladder losses (5 freshly uploaded + 1 from the v15 session) were decoded turn-by-turn with a one-off replay analyzer that reverse-engineered the kaggle-env log format: `steps[i]['action']` resolves `steps[i-1]`'s `select.option` list, not its own. This let every PLAY/ATTACH/EVOLVE/RETREAT/BOSS decision be reconstructed with card names (cross-referenced against `docs/EN_Card_Data.csv`), instead of guessing from raw IDs.

**Two systemic, game-losing bugs found, both confirmed across multiple independent games:**

1. **ATTACK/Boss score tie in PHASE_CLOSING.** `ATTACK` when `can_ko` and `BOSS` when `phase==PHASE_CLOSING` both scored exactly `200.0`. On ties, Boss sometimes won (order-dependent), so the agent played Boss's Orders *instead of* taking a guaranteed lethal attack — caught directly in 2 of 4 full games (e.g. `dmg=420 vs opp_hp=320`, chose PLAY Boss anyway), both times also taking a smaller-prize KO afterward instead of the lethal one. **Fix:** `can_ko` attack now scores 500; `boss_ex_snipe` (Boss repositions into a *bigger*-prize KO than the current target, still strictly better than a plain attack) scores 600 to preserve that one legitimate case where Boss-then-attack beats attacking now; generic closing-phase Boss dropped to 199 and gated behind `not can_ko`.
2. **Energy-starved stuck Active → deck-out.** A non-attacker (Fezandipiti ex in two separate games, Dudunsparce in a third) ended up Active with 0 energy attached, unable to attack *or retreat* — both options were simply absent from that turn's `select.option` list. With no way to fix its own position, the agent spent 30-40 consecutive turns spamming Dawn/Poké Pad searches and Fez's "Flip the Script" draw ability, ballooning its hand to 17-19 cards while burning the deck from 7-12 cards down to 0, then lost to deck-out at 1-2 prizes remaining — once with a tied 1-1 prize count, the closest possible loss. **Fix:** new `active_immobile` flag (no attack, no retreat, 0 energy on active) makes attaching any energy card to Active score 60 — overriding the normal "route Psychic to Alakazam" rule — since freeing the stuck Active is more urgent than optimal routing. Dawn/Poké Pad (can't fetch energy) drop to -3 while immobile; Hilda (can fetch energy) jumps to 18. A hard `deck_danger` (<5 cards) floor of -8 was also added to all three search cards so none of them fire that close to decking out, and Fez's ability — previously a flat, deck-size-blind 5.0 — now respects `deck_critical`/`deck_danger`.

A 5th game (the short one) was an unfixable bad-luck loss: opening hand had only one Basic Pokémon (the starting Abra), the blind Poké Pad search whiffed onto an unplayable Stage-1 Dudunsparce (no Dunsparce in play to evolve from), and a turn-2 Mega Lucario ex one-shot the lone Active before any recovery was possible. No heuristic bug — pure variance, the kind curriculum training on bad-hand starts is meant to target.

**Status:** Committed. Not yet ladder-validated.

---

## v15: Heuristic Fixes + Training Infrastructure

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

- Phase 0 (imitation warmup on v11 self-play): complete. 52% net vs v11.
- Phase 1 (self-play vs diverse pool): attempt 2 running. Best: sp2_iter2.pth at ~55% vs v11.
- Paused while v15 heuristic runs. Opponent BC (imitating other decks) is not viable — action sequences don't transfer.

---

## Deck Evolution

| Era | Deck | Why abandoned |
|-----|------|---------------|
| v1-v11 | Mega Lucario ex | 3-prize giveaway; Crustle-walled; 0-5 live |
| v12+ | Alakazam (Powerful Hand) | Single-prize; non-ex; flat deterministic damage; meta-proven |
