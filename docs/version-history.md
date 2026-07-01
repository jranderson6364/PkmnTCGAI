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

## v21: Phase-Stuck-at-ESTABLISH + Mist-Wall Escape Fixes — CURRENT ACTIVE SUBMISSION

Analyzed 4 more v19 replays. All 4 lost; `check_bad_energy_attach()` (v20's tool
improvement) still flagged wasted Psychic attaches in 3 of the 4 — expected, since
these predate the v20 fix. Digging into the two that *weren't* explained by that
already-known issue surfaced two new, more consequential bugs.

**1. Phase permanently stuck at `PHASE_ESTABLISH`, even mid/late-game with a fully
fueled, attacking Alakazam.** Traced a game where deck hit 0 with an absurd 20-25
card hand while `you_prizes` sat at 2 (needing 2 more) against an opponent needing
5 — we were dominating the prize race and still lost to our own deck-out. Root
cause: `_detect_phase`'s `not_established` check required BOTH `backup_abra` (2+
Abra simultaneously in play) AND `draw_count>0` (a Dunsparce/Dudunsparce currently
in play) — but Dudunsparce's own "Run Away Draw" ability shuffles ITSELF back into
the deck on use, so `draw_count` routinely drops to 0 the instant the draw engine
fires, and a spare Abra is rarely available once it's been used climbing the
evolution line. Whenever either dropped to zero — which happens constantly,
independent of actual board development — phase reverted to ESTABLISH and
re-enabled its overdraw-permissive scoring (Poffin 25/19, Dawn/Hilda 22/24) for the
rest of the game. **Fix:** removed both conditions from `not_established`, leaving
only the two that actually gate "is the engine running": `has_alakazam` and
`has_energy_plan`. Also closed a related gap: POFFIN's scoring had no `hand_surplus`
gate at all (unlike Dawn/Hilda/Poké Pad), so it kept firing during the deck-out
stretch even once the fixed phase logic would otherwise have suppressed it.

**2. Mist/Rocky-walled Active with no removal left → agent kept overdrawing chasing
an unreachable goal, AND under-prioritized Boss's Orders as the actual escape.**
Two of the four games showed the opponent's Active permanently blocking Powerful
Hand (Mist Energy) for the rest of the game, with both Enhanced Hammer copies
already in the discard — confirmed via the discard pile, not just absent from hand.
More cards can't fix a card-*type* block, so continuing to search only accelerates
deck-out for zero payoff. **Fix:** new `hopelessly_walled` flag (`opp_mist and not
hammer_in_hand and not boss_in_hand`) added as a hard suppressor to Poffin, Dawn,
Hilda, Poké Pad, and Dudunsparce's ability. Separately — and this was the sharper
find — `boss_target_exists` explicitly required `not opp_mist`, which is backwards:
Mist only blocks the *current* Active, and gusting a *different*, unwalled bench
target with Boss is exactly the escape valve, most valuable precisely when Mist is
up. This bug meant Boss fell through to a generic, un-informed `12.0` fallback that
routine search plays (Poké Pad `13.0`) could beat, even with a fully killable
non-Mist target sitting on the opponent's bench. Fixed the condition to `(opp_mist
or opp_hp>my_dmg)`. Also hardened `_pick_boss_target` itself: it scored candidate
bench targets by prize/HP/energy alone, with no check for whether the target
*also* carried Mist/Rocky Energy — gusting another walled Pokémon would just
recreate the same dead end. Non-walled candidates are now strictly preferred, with
walled-only targets used strictly as a last resort when literally every option is
blocked.

**Verification:** full regression clean across all 15 saved replays (4,776
selections, 0 errors, 0 illegal empties). Re-verified both phase-stuck games with
the fixed agent: phase now correctly reads CONVERT/PRESSURE instead of ESTABLISH,
and the agent chooses END/RETREAT/basic plays instead of continuing to search.
Confirmed the Boss-under-Mist fix with an isolated synthetic decision (Mist-walled
Active, Boss + a killable non-Mist bench target + a routine search card all legal)
— agent now correctly plays Boss instead of the search card.

**Status:** Committed. Not yet ladder-validated.

---

## v20: Preemptive Energy on Support Mons Fix

User-requested audit: "supporter mons should not get energy at all unless they are
attacking or retreating — no preemptive giving." Per instructions, first validated
the replay-analyzer tool by manually tracing one game (`4eeb92ef`, raw JSON, no
script) before trusting it on the rest — found the exact bug independently within
the first 6 steps: Shaymin (a support mon with **free retreat** — it never needs
energy to retreat at all) got a scarce Telepath Psychic energy attached while active,
then sat on it uselessly for the game's remaining 16 turns since the bench never
developed. Cross-checked the current agent against this exact historical
observation and confirmed it still made the same choice — a live bug, not stale data.

**Root cause:** the `active_immobile` flag (added v16, refined v18) was meant to free
a genuinely stuck Active by prioritizing an energy attach, but its gate was too loose
in two ways: (1) it didn't exclude free-retreaters (`PIVOT_FREE_RETREAT_IDS` = Shaymin
— energy was never what was blocking its retreat), and (2) it only checked
`bench_count > 0`, not whether that bench actually contained a **ready** attacker —
so retreating a stuck Dunsparce/Psyduck/Fezandipiti into an un-fueled Kadabra/Abra/
other support mon still leaves you unable to attack, meaning the energy fixed
nothing. Since this flag also drives Dawn/Hilda/Poké Pad's search suppression, the
bug was silently distorting five separate scoring decisions, not just the ATTACH one.

**Tool improvement:** added `check_bad_energy_attach()` to
`scratchpad/analyze_replay.py` — explicitly flags any Psychic energy attach landing
on a non-attacker (i.e. not Alakazam/Kadabra/Abra) with no legitimate retreat-cost
reason, rather than requiring a manual eyeball of the decision log. Re-running the
improved analyzer on all 15 saved replays found this pattern in **9 of 15 games**
(including the one win) — Shaymin, Dunsparce, Psyduck, and Fezandipiti all repeatedly
received energy they could never use.

**Fixes (`main.py`):**
1. `active_immobile` now excludes `active_free_retreat` (Shaymin) and requires
   `bench_has_alak_ready` instead of the looser `bench_count>0` for any non-Alakazam
   Active — i.e. energy only counts as "freeing" the Active if it enables an attack
   (Alakazam) or a retreat into an attacker that's actually ready to swing.
2. The generic `PSYCHIC_ENERGY_IDS` ATTACH fallback (any target that isn't Alakazam/
   Kadabra/Abra) dropped from a flat `3.0` to `-2.0` — no more "preemptive giving" to
   bench support mons by default; the one legitimate case (paying a real retreat cost
   into a ready attacker) is already covered by the corrected `active_immobile` block.

**Verification:** full regression (3,425 selections across all 15 saved replays) 0
errors, 0 illegal empties. Re-ran every historical main-phase decision point through
the fixed agent (using the actual board state as recorded, not a full re-simulation)
and confirmed the wasted-energy count drops to **0 across all 15 games**. Also
confirmed with two isolated synthetic ATTACH-decision tests.

**Status:** Committed. Not yet ladder-validated.

---

## v19: Psychic Energy Over-Attach Fix

User-requested fix: Powerful Hand costs exactly 1 Psychic energy — a 2nd Psychic on
the same Alakazam does nothing (damage scales with hand size, not energy count), so
attaching a spare Psychic to an already-fueled Alakazam is pure waste of a scarce
resource (only 6 Psychic sources in the 60-card deck: 2 Basic Psychic + 4 Telepath
Psychic).

**Fix:** reordered the `PSYCHIC_ENERGY_IDS` ATTACH scoring priority. Was: Alakazam
without Psychic (16) > Alakazam *with* Psychic already (8) > Kadabra (7) > everything
else (3) — meaning a redundant re-attach to a fueled Alakazam could beat pre-loading
Kadabra. Now: Alakazam without Psychic (16) > **Kadabra (9) > Abra (6)** > other bench
support (3) > **Alakazam that already has one (1, lowest)**. Kadabra/Abra are
pre-loaded ahead of other bench so the energy is already there the instant they evolve
into the next stage, without needing another attach that turn.

**Verification:** full regression (2,654 selections across 11 replays) still 0 errors,
0 illegal empties. Built two synthetic isolated-ATTACH-decision tests (Alakazam active
already fueled; bench has Kadabra+Abra in one, Abra+other-support in the other) and
confirmed the agent now routes to Kadabra first, then Abra, in preference to both the
redundant Alakazam re-attach and generic bench support.

**Status:** Committed. Not yet ladder-validated.

---

## v18: Ladder Result + Prize-Selection Stall Diagnosis

v17 climbed to ~900 Elo on the ladder, then dropped to ~660 after a losing streak.
Analyzed 5 fresh replays from that streak with the same replay-analyzer approach as
v16. Result: **0 missed-lethal, 0 bad-retreat, 0 bad-Boss-target across all 5 games**
— the v16/v17 fixes for those are holding clean. The losses are dominated by two
different things, one already-known and one newly (and more correctly) diagnosed:

1. **One pure-variance loss.** Opening 7-card hand had zero Basics and zero search
   cards (Poffin/Poké Pad/Dawn) — no possible path to a bench that turn. Same failure
   class as a game from the v16 batch. No heuristic fix changes this without a
   mulligan mechanic, which the engine doesn't appear to expose.

2. **Prize-card-selection stall — corrected diagnosis of a bug misidentified last
   session.** Three of the five new losses (plus, in hindsight, the v16 deck-out game)
   show an identical signature immediately after a scoring Powerful Hand attack:
   `stype=1, context=7, option[].area=6, option[].playerIndex==self, option[].type=3`,
   with `minCount == maxCount == N`. Cross-checking `N` against the KO'd Pokémon's
   prize value across multiple games confirms an exact match every time (1-prize KO →
   N=1, 2-prize/ex KO → N=2) — **this is prize-card selection**, not "Sacred Ash /
   Lana's Aid discard recovery" as guessed in the v16 write-up (both those cards say
   "up to X"; only prize-taking requires an exact count). In 2 of the 3 games this
   selection never resolved — the identical option list repeated for 6-12 ticks with
   no state change until the match ended in a recorded loss, despite the opponent's
   Pokémon being confirmed dead. Root cause is **not conclusively identified**: our
   own resolution logic for this select shape was audited and is already a valid,
   generically-correct blind pick (`_clamp(list(range(n)), sel)` for a face-down
   selection where any index is equally good) — the fact that the identical pattern
   sometimes resolves in 4 ticks and sometimes never resolves points at engine/timing
   behavior we can't fully pin down from static replay logs.

**Fixes:**
1. **Handheld Fan no longer counts as the `active_immobile` rescue energy.** Traced a
   stuck-Alakazam game and found the ATTACH scoring's "free the stranded Active" bonus
   (score 55-65, added in v16/v17) fired for *any* card attached to Active, including
   Tools. Handheld Fan provides zero Energy and can't pay a retreat or attack cost, so
   attaching it did nothing to fix the immobility it was supposedly solving. Gated the
   bonus behind `cid in PSYCHIC_ENERGY_IDS or cid==ENRICHING`. Verified: replaying the
   exact stuck state now has the agent EVOLVE instead of wasting the attach on the Fan.
2. **Stall-detection hedge for the prize-selection freeze (defensive, not a proven
   fix).** Added `_select_fingerprint`/`_resolve_stalled_or`: if the identical
   select+game-state signature is seen again with no progress, rotate to a different
   (still valid) combination of indices instead of resubmitting the same answer
   forever. Scoped narrowly to the exact generic blind-pick fallback path this bug
   lands in — every other select path is untouched. Costs nothing on the first call of
   any select (same behavior as before); only diverges on a detected repeat. This is a
   hedge against an unconfirmed root cause, not a guaranteed fix — flagged honestly as
   such.

**Verification:** agent runs clean on all 2,654 real selections across all 11 saved
replays (6 from v16 + 5 new), 0 exceptions, 0 illegal empties. Replayed the exact
stuck-selection sequence from `f094c5ad` through the patched agent and confirmed the
stall hedge now rotates `[0]→[1]→[2]→[3]→[4]→[5]→[0]...` instead of resubmitting `[0]`
every time.

**Status:** Committed. Not yet ladder-validated.

---

## v17: Competitive-Research Alignment + Threshold Discipline

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
