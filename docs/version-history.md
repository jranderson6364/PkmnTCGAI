# Agent Version History

*Newest first. From Mega Lucario to Alakazam.*

**Last updated:** 2026-07-02

---

## v24: Deck Simplification — Psyduck/Genesect Out, 4th Alakazam + 4th Dunsparce In — CURRENT

First (and only sanctioned) deck change since the list was adopted; logic untouched.
The Stage 0 deck audit (`tools/deck_audit.py`) showed Psyduck (858) and Genesect
(142) at ~0 plays per game drawn with ~100% rot — pure Powerful Hand fuel, a job
any card does equally well, so their unique effects (Damp, ACE-blocking) were
contributing nothing in the bot meta. Replacements target consistency: **4th
Alakazam** (win-condition redundancy; more live Dawn/Hilda/Rare Candy targets)
and **4th Dunsparce** (draw-engine redundancy; Poffin is now always fully live —
its search pool is just Abra/Dunsparce).

**Code change: `DECK` list only.** All PSYDUCK/GENESECT logic paths remain but are
dead (they fire only when the card is seen in play). `deck.csv` regenerated; 60
cards verified.

**Verified:** 200-game seat-alternating A/B vs frozen v23 (old deck, same logic):
**120W–80L, 60.0% ± 6.8%, 0 errors** — CI excludes parity; the swap is a real
improvement on its own. Overnight SPSA weight tuning (`training/overnight_tune.py`)
launched on top of this deck the same night; winner (if any) validates via 600-game
A/B + ladder confirm before shipping.

---

## v23: Replay Forensics — Promotion Ties, Enriching Misroute, Lone-Attacker Risk

Analyzed 5 fresh ladder replays (all losses) turn-by-turn by reconstructing board
state (`current.players[i]`) directly from the raw kaggle-env JSON at each decision
point, not just the existing decision-log summaries — `analyze_replay.py`'s flags
(missed lethal, bad retreat, wasted energy, Boss targeting) all read 0 across all 5
games, so the bugs here are new categories the tool doesn't check yet.

**Finding 1 — bench-promotion scoring ties Kadabra/Abra with pure-support mons.**
`_score_bench_target` only special-cased ALAKAZAM/DUDUNSPARCE/free-retreaters; every
other bench occupant — including an already-energized Kadabra one evolution from
attacking — fell into the same `-10` catch-all as Psyduck/Fez/Genesect. `max()`
breaks ties by array order, so a forced post-KO promotion between Kadabra and Psyduck
was a coinflip on bench slot order, not board value. **Confirmed root cause of a real
loss**: game `83173539` reached 1-1 prizes (sudden death), our Active got KO'd with
bench = [Psyduck, Abra, Dunsparce] (no Alakazam/Kadabra left), and the agent promoted
Psyduck — tied at -10 with Abra and picked first by index. Psyduck died to the
opponent's returning attacker next turn; game over. Replaying the exact position
through the fixed scorer (Kadabra/Abra now get dedicated tiers between Dudunsparce
and the -10 floor) flips the pick to Abra. Fix: `_score_bench_target` in `main.py`.

**Finding 2 — Enriching Energy could score positively on Alakazam.** CLAUDE.md's
deck notes are explicit ("Never attach Enriching to Alakazam" — it's Colorless and
can't pay Powerful Hand's Psychic cost), but the ATTACH scorer's fallback for
`tid==ALAKAZAM and not _has_psychic(tgt)` only covered the *unfueled* case (score
1.0, low but positive); an **already-fueled** Alakazam fell through to the generic
`return 6.0` — a positive score for a strictly wasted attach when Dudunsparce (which
wants Enriching for draw) or simply holding the card was always better. Found via a
replay decision (`83175768`, step 87) that turned out on inspection to be the
legitimate `active_immobile` rescue path (zero Psychic anywhere in hand, Active
otherwise fully stuck) rather than this bug — but the scoring gap itself was real and
independently reachable any time a fueled Alakazam is offered a redundant Enriching
attach with nothing better competing. Fixed: `tid==ALAKAZAM` now always scores -8.0
in the Enriching branch, regardless of fuel state.

**Finding 3 — lone-attacker risk: zero bench is an instant-loss single point of
failure that phase/hand-surplus gating didn't treat as urgent.** Game `83166796`
spent a long stretch with exactly one Pokemon in play (Active Alakazam, empty bench)
while comfortably ahead on prizes (6 vs opp's 3) — then that Alakazam got KO'd with
nothing to promote, an automatic loss regardless of the prize lead. Poffin/Poké Pad
(the direct bench-refill cards) were gated behind `hand_surplus`/phase checks that
don't independently recognize "you have exactly one Pokemon total" as its own risk
tier — `hand_surplus` in particular would suppress them once the hand hit the KO
threshold, even with an empty bench. Checked whether this specific game had Poffin/
Poké Pad in hand during the bench-empty window and it didn't (genuine bad draw, not
this bug in this instance) — but the gap is real and will bite in a future game.
Fixed: new `bench_empty` flag short-circuits Poffin/Poké Pad to their highest search
tier whenever bench_count==0, ahead of the surplus/phase gates.

**Not fixed — logged as variance, not a bug:** game `83168738` ended with 3 Alakazam
+ 1 Kadabra + energy dead in an 8-card hand because all 4 deck copies of Abra were
dead/discarded with zero replacement drawn (Poffin never appeared in the 8-card
hand either) — a legitimate thin-resource bad-beat (Abra is a 4-of) against early
aggression from a 340 HP Mega Lucario ex, not a selection-logic error; the existing
search-priority code already escalates Abra/Poffin/Poké Pad correctly once one is
actually drawn. Also declined to chase the exact Battle-Cage-vs-bench-damage
interaction in `83175768` (opponent's Boss forced our Fezandipiti into Active, which
then died to spread damage over their turn) — that's the opponent choosing our
forced promotion target, not a play we control.

**Verified:** replayed all 691 real (obs, select) decision points across the 5
saved v23 replays through the fixed `main.py` — 0 exceptions, 0 illegal picks.
120-game local A/B vs frozen v21 held at parity (50.8% ± 8.9%, small sample —
not a regression signal either way; re-confirm on a larger run before shipping).

---

## v22: Full API Audit — Selection Intelligence + Local Engine Unlock

Read the official cabt docs end-to-end (https://matsuoinstitute.github.io/cabt/,
all four reference sub-pages) and audited v21 against them. Full findings in
`docs/engine-api.md`. Three discoveries changed the game:

**Discovery 1 — the engine runs locally.** `pip install kaggle_environments
--no-deps` ships the full cabt engine with native binaries (cg.dll on Windows).
Full games run at ~0.5s each on any machine. This unblocks the entire training
plan without Kaggle sessions or the Vivobook being strictly required — see
`training/`.

**Discovery 2 — deck searches are NOT blind.** `select.deck` lists the deck and
options index into it. v21 was picking arbitrary first-N cards on every Poffin/
Poké Pad/Dawn/Hilda search. v22 scores search candidates by board need
(`_deck_search_pick`): missing Alakazam line pieces, backup Abra, draw engine,
energy plan.

**Discovery 3 — `_pick_setup_active` was a live bug.** It read `o['cardId']`,
which is never populated (verified: 0 of 1,287 options in a full game), so every
option scored the default and it picked opts[0] — an effectively random game
opener since v7's option-resolution era. v22 resolves through the hand
(options are `{area:HAND, index}`), restoring the Dunsparce>Abra>… preference.

**Also fixed:**
- Enhanced Hammer's energy pick (stype=4) now targets Mist/Rocky explicitly via
  `energyCards[energyIndex]` instead of picking the first special energy.
- Rare Candy's target pick (stype=7, ctx=EVOLVE) now prefers the Psychic-fueled
  Abra (evolving it yields an attack-ready Alakazam immediately), then the active.
- ~20 hand-guessed scoring constants moved into a module-level `W` dict —
  defaults identical to v21 — enabling empirical weight search
  (`training/weight_search.py`, SPSA vs frozen v21).

**Verification (real local games, not just replay regression):**
- 400-game alternating-seat A/B vs frozen v21: **225W–175L (56.3% ± 4.9%)**, 0 errors.
- Matchup baselines (100 games each): 94% vs Lucario, 94% vs Abomasnow,
  79W–3L–18T vs Starmie, 50W–0L–50T vs Dragapult (ties = step-limit draws —
  the sample Dragapult bot stalls in one seat direction; not our loss vector).
- Corrected area-code semantics: DISCARD=3, PRIZE=6 (project notes previously
  said 6=discard). The v18 "prize-selection stall" is confirmed prize-taking
  (ctx=TO_HAND, area=PRIZE).

---

## v21: Phase-Stuck-at-ESTABLISH + Mist-Wall Escape Fixes

Analyzed 4 more v19 replays. All 4 lost; wasted Psychic attaches in 3 of the 4 were
expected (pre-v20 fix). Two new, more consequential bugs surfaced from the other two.

**1. Phase permanently stuck at `PHASE_ESTABLISH`, even mid/late-game with a fully
fueled, attacking Alakazam.** Root cause: `_detect_phase`'s `not_established` check
required BOTH `backup_abra` (2+ Abra simultaneously in play) AND `draw_count>0`
(a Dunsparce/Dudunsparce currently in play) — but Dudunsparce's own "Run Away Draw"
shuffles itself back into the deck on use, so `draw_count` routinely drops to 0 the
instant the draw engine fires, and a spare Abra is rarely available once used climbing
the evolution line. This reverted phase to ESTABLISH and re-enabled overdraw-permissive
scoring for the rest of the game. One game decked out at 0 with a 20-25 card hand while
winning the prize race 2v5 — we were dominating and still lost. **Fix:** removed both
conditions from `not_established`, keeping only `has_alakazam` and `has_energy_plan`.
Also added a `hand_surplus` gate to POFFIN's scoring (it had none, unlike Dawn/Hilda/Poké Pad).

**2. Mist-walled Active with no removal left → agent kept overdrawing, AND Boss was
under-prioritized as the escape.** New `hopelessly_walled` flag (`opp_mist and not
hammer_in_hand and not boss_in_hand`) suppresses Poffin/Dawn/Hilda/Poké Pad/Dudunsparce-
ability (more cards can't fix a card-type block). Separately, `boss_target_exists`
required `not opp_mist` — backwards, since Mist only blocks the current Active and
gusting a different target is the escape valve, most valuable exactly when Mist is up.
Fixed to `(opp_mist or opp_hp>my_dmg)`. Also hardened `_pick_boss_target` to avoid
gusting another Mist/Rocky-walled target.

**Verification:** full regression clean — 4,776 selections across 15 replays, 0 errors.

---

## v20: Preemptive Energy on Support Mons Fix

User-requested audit: "supporter mons should not get energy unless attacking or retreating."
Validated replay analyzer first by manually tracing one game — found Shaymin (free
retreat, never needs energy) got a scarce Telepath Psychic while active, then sat on
it uselessly for 16 turns. Confirmed current agent still made the same choice — a live bug.

**Root cause:** `active_immobile` flag didn't exclude free-retreaters and only checked
`bench_count>0` instead of `bench_has_alak_ready` — retreating into an un-fueled
Kadabra/Abra/support still leaves you unable to attack, so the energy fixed nothing.
This flag also drove Dawn/Hilda/Poké Pad suppression, silently distorting 5 scoring paths.

**Tool improvement:** added `check_bad_energy_attach()` to replay analyzer — found the
pattern in 9 of 15 saved replays (including the one win).

**Fixes:** `active_immobile` now excludes `active_free_retreat` and requires
`bench_has_alak_ready`. Generic `PSYCHIC_ENERGY_IDS` ATTACH fallback dropped from
`3.0` to `-2.0`.

**Verification:** 3,425 selections across 15 replays, 0 errors. Wasted-energy count
drops to 0 across all 15 games.

---

## v19: Psychic Energy Over-Attach Fix

Powerful Hand costs exactly 1 Psychic — a 2nd on the same Alakazam does nothing.
Only 6 Psychic sources in the 60-card deck; wasting one is real cost.

**Fix:** reordered `PSYCHIC_ENERGY_IDS` ATTACH priority. Was: Alakazam-without(16) >
Alakazam-with(8) > Kadabra(7) > else(3). Now: Alakazam-without(16) > Kadabra(9) >
Abra(6) > other bench support(3) > Alakazam-that-already-has-one(1, lowest).

**Verification:** 2,654 selections across 11 replays, 0 errors.

---

## v18: Ladder Result (900→660 Elo) + Prize-Selection Stall Diagnosis

Analyzed 5 fresh replays from a post-v17 losing streak. 0 missed-lethal, 0 bad-retreat,
0 bad-Boss-target — v16/v17 fixes held. Two issues found:

1. **Handheld Fan no longer counts as rescue energy.** ATTACH scoring's "free the stranded
   Active" bonus fired for any attached card, including Tools. Fan provides zero energy —
   gated behind `cid in PSYCHIC_ENERGY_IDS or cid==ENRICHING`.
2. **Prize-card selection stall (defensive hedge).** 3 of 5 losses show: `stype=1, context=7,
   area=6, minCount==maxCount==N` immediately after a scoring attack, where N exactly matches
   the KO'd Pokémon's prize value. In 2 of 3 games the selection never resolved, freezing
   until match end. Root cause not conclusively identified (engine/timing behavior). Added
   `_select_fingerprint`/`_resolve_stalled_or` to rotate indices instead of resubmitting
   the same answer forever.

**Verification:** 2,654 selections across 11 replays, 0 errors.

---

## v17: Competitive-Research Alignment + Threshold Discipline

Full research pass on how the deck is piloted at the top level (Cerys Jones' Indianapolis
Regional 1st, CL Osaka 2026, Limitless meta lists). Rewrote `docs/piloting-guide.md`
into v3. Key finding: our 60-card backbone matches the meta list exactly.

1. **`hand_surplus` threshold discipline.** Once a ready attacker exists and
   `hand_n >= cards_needed`, all non-essential draw suppressed: Dudunsparce ability → 0.5,
   Fez ability → -3, Dawn/Hilda/Poké Pad → 2.0.
2. **Dudunsparce Run Away Draw** now hits hard floor at `deck_danger` (<5).
3. **`active_immobile` rescue attach prefers Psychic** (65 vs 55).

**Verification:** 1,672 selections across 6 replays, 0 errors.

---

## v16: Replay-Driven Bug Hunt

Six real ladder losses decoded turn-by-turn with a custom replay analyzer. Two systemic,
game-losing bugs confirmed across multiple independent games:

1. **ATTACK/Boss score tie in PHASE_CLOSING.** Both scored 200.0 — agent sometimes chose
   Boss over a guaranteed lethal attack. Fixed: `can_ko` attack → 500; `boss_ex_snipe`
   (Boss into bigger-prize KO) → 600; generic closing Boss → 199, gated behind `not can_ko`.
2. **Energy-starved stuck Active → deck-out.** Non-attacker Active with 0 energy couldn't
   attack or retreat — agent burned deck to 0 over 30-40 turns. Fixed: `active_immobile`
   flag makes energy-to-Active score 60; Dawn/Poké Pad → -3 while immobile; Hilda → 18.
   Added `deck_danger` (<5 cards) floor of -8 on all search cards.

---

## v15: Heuristic Fixes + Training Infrastructure Staged

1. Bench Alakazam evolution scoring: 50/40/25/12 by phase (was 16/10).
2. Enhanced Hammer escalation: scores 45 when opponent has Mist/Rocky (was 28).
3. Battle Cage reactive: scores 22 when bench damage detected (was flat 6).

Training infrastructure (opponents/ agents + docs/training-setup.md) staged for
when Vivobook compute is available.

---

## v14: Replay-Driven Fixes (4 replays)

1. Fez suppressed by default (-1.0) — 2-prize liability; only activates reactively
2. Sacred Ash deck-out prevention: 35 at deck<5, 25 at deck<10
3. Dudunsparce overdraw guard: suppressed at deck<10 or hand≥14
4. Evolution scoring fix: Kadabra→Alakazam scores 250-270 (was 13)
5. Boss prize-value guard: target must have prize_value ≥ prize_value(opp_active)
6. Rock Energy detection: Enhanced Hammer scores 28 for Mist (#11) OR Rock (#20)
7. Genesect role: Bench+Fan scores 11 (ACE Nullifier)
8. Shaymin reactive: scores 16 when bench damage detected
9. Psyduck reactive: scores 18 when opponent self-damage cards detected

---

## v13: Phase Rewrite — Never Validated

Introduced 4-phase state machine (ESTABLISH/CONVERT/PRESSURE/CLOSING). A/B harness
run but results never reported back. Win rate vs v11/v12c is unknown. Superseded by v14.

---

## v12, v12b, v12c: Turn Planner Attempt — Regression, Then Partial Recovery

v12 added a bounded turn planner — resulted in 39.8% vs v11 (regression). v12b partially
reverted but `boss_ex_snipe` never fired (name-based ex detection bug). v12c audited 8
fixes but was never ladder-tested; superseded by v13/v14.

---

## v11: Supporter Preservation

Held Supporter for Boss snipe (partial). Hit ~700 Elo (~1600/3200 ≈ median).

**Ceiling identified:** greedy can't sequence a full turn. Boss/Supporter conflict is
the canonical example — greedy plays the draw Supporter first, then can't Boss the target.

---

## v8–v10: Core Fixes

- **v8:** Option resolution via hand/board (P0 — unlocked everything). Energy routing
  by type. Attack discipline (draw to lethal before swinging).
- **v9:** Bench priority — never sit on a lone Active. Poffin first.
- **v10:** Speed setup. Boss + Powerful Hand combo. Reach Alakazam fast via Rare Candy.

---

## v7: Full Logic Audit + Option Resolution Fix (83% vs random)

**The audit finding:** Engine options are *positional*, carry no `cardId`. All of v6's
card-specific rules keyed on `o.get('cardId')` which is always absent → silently fell
to generic fallbacks.

**Fix:** resolve options through visible hand/board: `PLAY/ATTACH/EVOLVE → hand[o['index']].id`;
`ABILITY → (area,index) → active/bench pokemon`.

**83% vs random.**

---

## v1–v6: Mega Lucario Era

- v1-v3: Initial agent, basic scoring.
- v4-v5: Determinized forward search scaffold, energy type detection, Mist Energy detection.
- v6: ~median ladder position (~1500/3000). Complete scorer with board census.

**Key lesson:** Mega Lucario ex went 0-5 live despite ~64% offline win rate. Root causes:
(1) 3-prize target = terrible prize math; (2) megaEx = Crustle-walled; (3) options are
positional with no cardId — all card-specific rules were silently dead code.

---

## Deck Evolution

| Era | Deck | Why abandoned |
|-----|------|---------------|
| v1–v11 | Mega Lucario ex | 3-prize giveaway; Crustle-walled; 0-5 live |
| v12+ | Alakazam (Powerful Hand) | Single-prize; non-ex; flat deterministic damage; meta-proven |

---

## NN Track

See `docs/nn-training.md`. All prior training data lost; track reset.
Architecture and self-play design are preserved. Restart requires Vivobook access.
