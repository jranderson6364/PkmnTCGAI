# Agent Version History

*Newest first. From Mega Lucario to Alakazam.*

**Last updated:** 2026-07-05 (v27)

---

## v27: Broaden `hand_surplus` Draw-Suppression Gate (Board-Thinning Fix)

Ships the top-priority fix identified by mining 18 exploiter-vs-frozen-v25c
replays (`docs/report-log.md` 2026-07-05 "Exploiter-win replay mining"
entry): `main.py`'s `hand_surplus` gate (suppresses further draw/search once
hand already clears the KO threshold) required `ready_attacker_exists`,
which is false during a rebuild turn (attacker just KO'd, no bench backup)
— exactly when hand is already 2-3x `cards_needed` from earlier banking.
Powerful Hand's damage is capped by opponent HP regardless of attacker
readiness, so a hand this far over `cards_needed` is already-wasted value;
confirmed in two replays (`win_001`, `win_010`) burning the deck to 0 in a
single turn via repeated Poffin/Dawn/Hilda/Poké Pad plays at hand_n=16-23
vs `cards_needed`=7 while stuck on a non-attacker. Fix: added
`hand_grossly_over = opp_hp<99999 and hand_n>=cards_needed+6`, OR'd into the
existing gate condition — one line, no new call sites (all 7 existing
`hand_surplus` consumers benefit uniformly).

**Scope caveat:** only addresses the deck-out sub-mode where the attacker
line is complete/near-complete (~4-5 of the 18 mined replays); does not fix
the "never builds the attacker at all" pattern or 2 anomalous short
`OTHER`-cause games — those need separate investigation.

**Gate:** 300-game mirror A/B (fixed vs. frozen pre-fix `main.py`) —
**56.0% ± 5.6%** (95% CI [50.4%, 61.6%]) — modest but real, CI clears 50%.
Smoke tests: 20 games vs `lucario_agent` (95%) and `abomasnow_agent` (95%),
0 errors/crashes across ~580 total games run during validation.

---

## v25c: Desperation Mode, Lone-Active-Opportunity, Deck-Search/Boss's Orders Target-Quality Fixes

Submission `54282648`, shipped 2026-07-03. 5 replay-verified fixes from user-flagged
ladder losses (replays `83429870`, `83458785`, `83461698`, `83462350`), plus one
new scoped heuristic feature:

1. **`desperation` mode** — when the opponent is ≤1 prize from winning (or ≤2
   with an ex in play), overrides deck-out caution and forces racing to
   Alakazam/maximum hand size instead of trying to out-survive a lethal hit.
2. **`lone_active_opportunity`** — opponent's bench is empty and a rough
   max-hand estimate (current hand + untapped Dudunsparce/Hilda/Dawn/Enriching
   draw sources) already clears the KO threshold on their active → stop
   banking draw for later and go for the kill now.
3. **`_score_deck_search` Alakazam-priority bug** — Poke Pad/Dawn/Hilda/Lana's
   Aid/Sacred Ash preferred fetching a dead-weight Alakazam or Kadabra over an
   Abra when no line piece existed anywhere (replay `83458785`). Fixed via a
   `have_line_piece` gate (in-play line piece or Abra-in-hand only — a
   Kadabra-in-hand doesn't count, since it's equally unplayable without an
   Abra already in play).
4. **Hilda-vs-Dawn context bug** (replay `83461698`) — Hilda's search pool
   structurally cannot fetch Basic Pokémon (Stage 1/2 + energy only), but its
   flat scoring weight beat Dawn's (which can fetch Basics) even with zero
   Abra anywhere in play or hand. Fixed via `need_basic_abra`, which drops
   Hilda's score when no line piece exists and there's no Abra in hand.
5. **Boss's Orders `PHASE_CLOSING` target-quality gate** (replays `83458785`
   and `83462350`) — the `PHASE_CLOSING` branch returned an unconditional flat
   199.0 regardless of target quality, confirmed wasting Boss's Orders twice:
   once swapping a KOable 3-prize Starmie for a 1-prize Staryu, once gusting
   a fueled Mega Lucario ex for a 1-prize Solrock (though replaying that exact
   state showed the "440 damage" alternative wasn't actually legal that turn —
   active was a same-turn Abra, Powerful Hand needs Alakazam, Rare Candy was
   blocked by `appearThisTurn`). Fixed: `PHASE_CLOSING`'s 199.0 now requires
   `boss_target_exists` (a real, KOable target) rather than firing unconditionally.

400-game A/B and full gauntlet not yet re-run against this exact patch (see
outstanding items in `CLAUDE.md`). Known open gap, not fixed here: `main.py`
has no handling for the engine's `MULLIGAN` select context — zero confirmed
firings found across 6 replays checked, so left unfixed pending a live example.
Known unfixed pattern: "board-thinning" — ending up with 1-2 Pokémon in play
and a bloated, mostly-dead hand after the attacker line gets repeatedly KO'd;
not a card-sequencing problem, flagged for a dedicated look later.

---

## v25b: Mist-Wall Retreat Fix + Candy-Racing Offensive Trigger + Harness Tie Bug

User watched the first v25 replay live and flagged 3 suspected misplays plus the
long-standing Dragapult gauntlet ties. Traced against replay `83429870.json`.

**1. Confirmed: retreat scoring force-retreated a fully-fueled Alakazam whenever
the opponent walled with Mist/Rocky Energy, with no check for whether we could
exploit it.** `main.py`'s RETREAT branch had `if opp_mist and active_is_alak:
return 9.0` unconditionally — turn 20, a 140/140 HP Alakazam with a Psychic
energy attached retreated into a Kadabra (no Powerful Hand, can never attack)
for zero benefit, confirmed by running the exact game state through
`score_options_main`. In this deck the only escapes from a Mist/Rocky wall are
Enhanced Hammer and Boss's Orders, both of which target the *opponent's* side
and work regardless of who's active — retreating our own Alakazam never helps,
and re-promoting it later costs a second wasted retreat. **Fix:** removed the
branch; falls through to the existing `active_can_attack: return -2.0`.

**2. Not reproduced: Boss's Orders allegedly skipping a lethal fueled Trevenant
for a weak unevolved target with "no wall present."** Traced all 3 Boss's
Orders plays in the same replay by energy card ID (Mist Energy displays as
colorless type but carries card id 11) — all 3 targeted a genuinely
Mist-walled active, and the bench target chosen was the best legal one each
time. No fix applied; need the specific replay if this recurs.

**3. Confirmed gap (not reproduced in this replay, but real): Rare Candy racing
had no offensive trigger.** `racing_for_alakazam` only fired on defense
(`active_below_half`) or in late phase — no notion of "Candying now sets up a
near-term KO." **Fix:** added `candy_lethal_soon` (current hand size × 20 ≥
opponent's active HP, no Mist/Rock wall, Abra active) as a third OR-branch.

**4. Confirmed: "Dragapult ties" were never step-limit draws — a harness bug.**
`opponents/dragapult_agent.py` imports `cg.api` for rich dataclasses, only
available in the Kaggle-hosted `kiyotah/cg-lib` dataset; the local
`try/except Exception` fallback never defines `Pokemon`, so every local game
vs this anchor crashes with `NameError` in both seats — reproduced 20/20.
Separately, `training/harness.py::summarize()` read only `rewards[0]`; when
the crash landed in slot 0 it fell through to the tie branch instead of
reading slot 1's reward of `1` as a win. **Fix:** `summarize()` now checks
both reward slots. Post-fix: 20/20 vs `dragapult_agent.py` in both seat
orders now correctly show 100% win, 0 ties. `lucario`/`abomasnow`/`starmie`
anchors don't have this crash. The anchor's own crash is left unfixed
(optional: vendor `cg/api.py` from the dataset, confirmed downloadable and
plain ~26KB Python); the `dragapult` column in `gauntlet_results.csv` has
never reflected real play for any prior version.

**Verification:** 400-game A/B vs frozen v23: 53.0% ± 4.9% (up from 52.0%
pre-fix, CIs overlap). Gauntlet gElo 767 (200 games/anchor, post-harness-fix):
below v24 (791), above v23 (753)/v22/v21. 0 errors throughout. Shipped as
submission 54279766.

---

## v25: Replay Analyzer Rebuild + Ability-Prompt Deck-Out Fix + Kadabra Retreat Fix

Rebuilt `tools/analyze_replay.py` from scratch — the prior version never gated on
each step's `status` field, so the `select` object's echo into the opponent's
INACTIVE steps was read as a fresh decision, fabricating 28–102 phantom "timeouts"
per game and swamping any real signal (its analytical predicates read 0 across all
5 games in the v23 forensics for the same reason: wrong decision extraction, not
wrong predicates). Rebuilt as a faithful transcript: real decisions require
`steps[i-1][you].status=='ACTIVE'` with a `select`, action read from `steps[i]`
regardless of its own status. Added a terminal-cause triage classifier
(`PRIZED_OUT`/`DECK_OUT`/`NO_POKEMON_IN_PLAY`/`EMPTY_OR_ILLEGAL_RETURN`/`OTHER`)
and enriched each decision line with hand contents, active HP/energy, deck count,
and `cards_needed` — replacing the old noise predicates that never fired on a real
bug.

**Confirmed root cause (replay `83348630`, `DECK_OUT` loss):** evolving a bench
Kadabra into Alakazam triggers a separate "may use this Ability?" Yes/No prompt
(select `stype==9`) for Psychic Draw — `_choose`'s stype==9 handler answered YES
unconditionally, with no deck-count check, unlike every other draw source in the
file (Dawn/Hilda/Poffin/Poké Pad/Dudunsparce-ability all gate on `deck_danger`).
In this game the prompt fired at deck=3 with hand already at 17 (`cards_needed`=7)
and a 5-2 prize lead — drew 3 more cards and emptied the deck the same turn.
**Fix:** stype==9 now declines (answers NO) when `deck_count<5` and
`hand_n >= cards_needed+3` — mirrors the existing deck_danger convention.
Verified end-to-end by replaying the exact `83348630` step-160 obs through the
patched `main._choose` directly: confirmed `deck_count=3`/`hand_n=16`/
`cards_needed=7` (opp HP correctly populated) and the patched code returns `[1]`
(NO) where the old code always returned `[0]` (YES). Note this game had a
*separate* leak too (Psychic energy misrouted onto the non-attacking Dudunsparce
instead of the developing Alakazam/Kadabra line, so `hand_surplus`'s
`ready_attacker_exists` gate never engaged) — the fix closes one deck-out path,
it doesn't convert this specific game. 400-game A/B vs frozen v23: 52.2% ± 4.9%
(a null result, expected and uninformative here — it only shows the fix didn't
break the common case, not that it fires; the direct-replay check above is what
verifies it). Ships regardless of the A/B per the contract-bug policy, since it
prevents a category of instant self-losses.

**Also surfaced (documented, not yet fixed) — triage is incomplete:**
- **Recurring engine stall, 2 of 3 spot-checked `OTHER` losses:** replays
  `83166796` and `83168738` both show a healthy full-HP board, the opponent
  having taken only 2-4 of their 6 prizes, the game stalling in `INACTIVE` for
  7-14 steps, then an abrupt `DONE` loss with an empty action. Fresh evidence for
  the previously-unconfirmed "prize-selection engine stall" gap
  (`docs/piloting-guide.md` §13) — but with only 2 data points and ~11 `OTHER`
  losses still unread, this isn't root-caused yet, and the triage classifier
  itself likely under-detects it (only catches `NO_POKEMON_IN_PLAY`/`DECK_OUT`/
  `PRIZED_OUT==6`, not "stalled with a healthy board"). **Follow-up needed:** add
  a stall detector to `tools/analyze_replay.py`'s classifier and read the
  remaining `OTHER` games.
- `NO_POKEMON_IN_PLAY` loss (replay `83344386`): opening hand had no Poffin/Poké
  Pad/Dawn/Abra to build a bench; single Dunsparce active got KO'd on turn ~3-4
  with nothing to promote — an instant loss under standard TCG rules. Consistent
  with the already-documented mulligan/dead-hand gap (piloting-guide.md §6); no
  new fix, logged as supporting evidence.

**Second confirmed fix, same v25 (full logic audit of retreat/energy/evolution,
requested separately, before shipping):** `KADABRA` was missing from
`NON_ATTACKER_IDS`. Kadabra has a real attack (Super Psy Bolt, {P}→30 flat dmg),
but this deck's `ATTACK` scoring only ever rewards Alakazam's Powerful Hand
(`active_can_attack` requires `is_alak`; any Kadabra attack scores -5
unconditionally) — so Kadabra is functionally a non-attacker here, just not
tagged as one. Consequence: a stuck Kadabra active (no energy, no attack) with a
fully-fueled, ready Alakazam waiting on the bench fell through every
retreat-priority tier in the `RETREAT` scoring and landed on a flat 0.5 — *lower
than simply ending the turn (1.0)*. Verified empirically both before and after
the fix by constructing that exact state and calling `score_options_main`
directly: pre-fix, RETREAT=0.5 < END=1.0; post-fix, RETREAT=22.0 > END=1.0.
400-game A/B vs frozen v23: 53.2% ± 4.9% (CI includes parity — expected and
uninformative for a narrow-state fix; this run and the stype==9 fix's 52.2% run
are two independent noisy samples of the same ~50% true rate, not a trend — 0
errors, no regression is the actual signal). Ships bundled with the stype==9
fix above.

**Two more fixes from the same audit, after checking each was real and cleanly
fixable (not just plausible-looking):**
- **Wondrous Patch's target-select was reusing `_score_bench_target`.** That
  function's tiebreak (prefer an *already-fueled* Alakazam over an unfueled
  Kadabra/Abra) is correct for retreat/promotion targeting (swap into whoever
  can attack right now) but backwards for Wondrous Patch, which *attaches* the
  recovered Psychic energy to the selected bench Pokémon — there you want
  whoever needs the energy, not whoever already has it. Checked the actual
  select object across replays before fixing: Wondrous Patch's follow-up select
  carries `effect.id==1146` (its own card id), cleanly distinguishable from
  plain retreat/promotion selects (`effect=None`) and Boss's Orders
  (`effect.id==1182`) — so this didn't need new context-tracking machinery, just
  a dispatch branch on that existing field. Added `_score_wondrous_patch_target`
  (prefers unfueled Alakazam > Kadabra > Abra; -10 if already fueled) routed via
  `sel['effect']['id']==WONDROUS_PATCH`. Verified with a constructed
  already-fueled-Alakazam + unfueled-Kadabra bench: old shared scorer would pick
  the fueled Alakazam (wasting the attach), new scorer correctly picks the
  Kadabra. 400-game A/B: 53.2% ± 4.9%, 0 errors (expected null result — Wondrous
  Patch is a 1-of, this is a rare-state fix).
- **Enriching Energy's "draw 4" had no deck-safety gate at all.** Unlike
  Kadabra/Alakazam's Psychic Draw (a may-use prompt, now gated via the stype==9
  fix above), Enriching's draw fires unconditionally on attach — and it drew
  literally zero scrutiny for deck safety anywhere in the ATTACH branch. Traced
  to a real contributing factor in the *other* `DECK_OUT` loss, replay
  `83156504`: attached at deck=5 with hand already at 18, dropping the deck to 1
  in one action (the game continued a few more turns before finally hitting 0,
  so this wasn't the sole cause, but it burned nearly all the remaining margin
  for zero reason). Fixed: `if deck_critical and not emergency_draw: return -6.0`
  before the existing Dudunsparce/Alakazam routing — `deck_critical` (<10) rather
  than `deck_danger` (<5) since Enriching's single draw (4) is roughly double
  Dawn/Hilda's, so it needs more margin. Verified by replaying the exact
  `83156504` step-174 obs: the option the agent originally chose now scores -6.0,
  below the new best available option (9.0). 400-game A/B: 54.0% ± 4.9%, 0
  errors.
- (Originally considered discouraging Enriching-on-support-mon as an inconsistency
  with the Alakazam case — **rejected on closer reading**: the draw-4 fires
  regardless of target, so a support mon attach is a real +4 cards, not a wasted
  attach like the Alakazam case which blocks the {P} attack-cost slot. The actual
  bug was the missing deck-safety gate above, not the target preference.)

**Fifth fix, closing the evolution gap the user asked about directly:**
`docs/piloting-guide.md` §13 had long flagged "manual evolve > Rare Candy" as
"⚠️ approximate." Nailed down precisely via a synthetic zero-time-pressure state
(full-HP Abra, full-HP opponent, 6-6 prizes, normal hand): Rare Candy (45.0)
unconditionally beat manual Abra→Kadabra evolve (2.0) whenever both were legal —
there was no "do we have time" signal in this matchup at all, exactly matching
the guide's own gap. Added `racing_for_alakazam` (no Alakazam in play yet, AND
either our active is below-half HP while the opponent already leads the prize
race, or the opponent is down to ≤2 prizes) and gated both the Candy-on-active-
Abra score and the manual-Kadabra-evolve score on it — Candy wins when racing
(unchanged 45 vs 2), manual wins when there's time (new: 15 vs 10), per
piloting-guide §3's "bank the extra card when you have time" principle.

**Two bugs caught in the fix's OWN construction, by testing rather than trusting
the code:** (1) first draft reused `active_vulnerable` for the danger check, but
that has an `active_hp<60` absolute-HP clause that's *always true for Abra*
(50 max HP) regardless of actual health — made `racing_for_alakazam`
unconditionally true and the fix a no-op until caught by a synthetic full-HP
test. Switched to the relative `active_below_half`. (2) first draft also
included `emergency_draw` (hand≤4), which is spuriously true turn 1-2 before the
draw engine has run — exactly when there's the *most* time to climb manually,
not an emergency. Dropped it. Final version verified against three synthetic
states (neutral, hurt-and-behind, opponent-closing) all producing the correct
choice. 400-game A/B vs frozen v23: 52.7% ± 4.9%, 0 errors, no regression.

**Audit scope, stated plainly:** retreat, energy/ATTACH, and now evolution's
manual-vs-Candy tradeoff got full branch-by-branch scrutiny with
replay/synthetic verification. Boss's Orders and bench-promotion targeting were
not re-audited here — they were already reviewed and fixed in the v23
replay-forensics pass and showed no new issues in this session's spot checks.

**Shipped to Kaggle 2026-07-03** bundled with the other agent's v23-deck revert
— one submission carrying both the deck revert (`deck.csv`) and these five
logic fixes (`main.py`). See `training/ladder_history.csv` for the submission
row.

---

## v24 REVERTED (2026-07-03) — back to v23 deck

Live ladder trend (680 Elo at ~7h, 780 at ~24h) prompted a user call to revert
before the documented 48h/50-point decision checkpoint. `main.py`'s `DECK` list
and `deck.csv` reverted to the exact v23 composition (3× Alakazam, 3× Dunsparce,
Genesect + Psyduck back in) — no scoring-logic changes, since v24's change was
deck-list-only. Caveat for the record: v23 itself had decayed to 773 by its own
day-2 reading, so 780-at-24h for v24 isn't an unambiguous regression against
that baseline — see `docs/report-log.md` 2026-07-03 entry for the full data.
The 200-game local A/B that favored v24 (60.0% ± 6.8%) stands unexplained.

---

## v24: Deck Simplification — Psyduck/Genesect Out, 4th Alakazam + 4th Dunsparce In — SUPERSEDED, see revert above

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

Training infrastructure (opponents/ agents + a training-setup plan doc, since
folded into `training/README.md`) staged for when Vivobook compute is available.

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
