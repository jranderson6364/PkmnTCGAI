# Alakazam / Dudunsparce — Comprehensive Strategy Guide (v3)

*The single source of truth for how this deck is piloted. Written from competitive
play research (Cerys Jones' Indianapolis Regional win, CL Osaka 2026, Limitless
meta lists, TCGplayer/Cardsrealm/Deltia guides) and cross-checked against our own
ladder replay losses. Section 13 maps every principle to the `main.py` heuristic
that encodes it, and flags the gaps.*

---

## 1. Game Plan (one sentence)

Draw a huge hand, then **Powerful Hand** the opponent's Active for `20 × hand size`
as placed damage counters — every turn, as the aggressor, in every matchup. It is a
**glass cannon**: powerful but fragile, and it can lose to *itself* (deck-out) as
easily as to the opponent. Almost every non-core card exists to **pre-empt a
specific answer** to the plan, not to add nuance to it.

The deck is **all single-prize Pokémon.** You win the prize race by trading 1 for 2
or 3 — they take one prize for KO'ing your Alakazam; you take two or three for
KO'ing their ex/mega.

---

## 2. Core Math

**Powerful Hand damage = 20 × (cards in hand)**, placed as **damage counters** →
**ignores Weakness, Resistance, and all damage reduction.** A KO needs
`ceil(target_HP / 20)` cards.

| HP  | Cards | | HP  | Cards | | HP  | Cards |
|-----|-------|-|-----|-------|-|-----|-------|
| 60  | 3     | | 170 | 9     | | 280 | 14    |
| 70  | 4     | | 200 | 10    | | 300 | 15    |
| 110 | 6     | | 210 | 11    | | 320 | 16    |
| 120 | 6     | | 230 | 12    | | 330 | 17    |
| 140 | 7     | | 250 | 13    | | 340 | 17    |
| 150 | 8     | | 260 | 13    | | 350 | 18    |

**Boss cost:** playing Boss's Orders is a card out of hand, so on a Boss-then-attack
turn your damage is `20 × (hand − 1)`. Always cost the Boss in before deciding it
sets up a KO.

**Hard wall:** **Mist Energy (#11)** and **Rock Fighting Energy (#20)** both read
"prevent all effects of attacks" → Powerful Hand places **0** counters on a Pokémon
holding one. Against these you must **Enhanced Hammer** the energy off, **Boss
around** to a different target, or simply not attack into it.

---

## 3. The Draw Engine & Card Economy

Every action has a **net card value**. This deck has no Professor's Research / Iono /
Ultra Ball — all card flow is "free" (no hand discard), because **hand size is your
damage stat.**

| Play | Net cards | Note |
|------|-----------|------|
| Kadabra (manual evolve) | **+1** | −1 to play, +2 from its evolve-draw |
| Alakazam from Kadabra (manual) | **+2** | −1 to play, +3 from Psychic Draw |
| Alakazam via Rare Candy from Abra | **+1** | −1 Candy −1 Alakazam, +3 draw — skips Kadabra's +2 |
| Dudunsparce — Run Away Draw | **+3** | draw 3, shuffle *itself* back into deck |
| Enriching Energy (attach) | **+3** | draw 4, −1 to attach |
| Dawn | **+2** | grab 3 Pokémon to hand, −1 supporter |
| Hilda | **+1 … +6** | grab evolution + energy; chains into evolve-draws + Enriching |
| Telepath Psychic (attach to {P}) | **+1 board** | benches 2 Basic {P} (Abra) — tempo, not hand |

**Manual evolution beats Rare Candy by +1 card** (you keep Kadabra's +2 draw). Use
**Rare Candy only for speed** — to land a turn-2 Alakazam, or to rebuild after a KO
when you can't afford the two-turn manual climb. When you have the time, climb the
line by hand to bank the extra cards.

**Enriching Energy → Dudunsparce, always.** Dudunsparce shuffles itself (and the
Enriching) back with Run Away Draw, so you can redraw and re-attach the same
Enriching repeatedly. Never burn Enriching on Alakazam — it provides only {C} and
**cannot pay Powerful Hand's {P} cost.**

---

## 4. Threshold Management — *the* skill (and our #1 leak)

You do **not** blindly maximise hand size. You compute how big your hand needs to be
to KO the relevant target, draw to exactly that **threshold**, then **stop**:

> `cards_needed = ceil(opponent_active_HP / 20)` → draw/evolve until
> `hand ≥ cards_needed` (+ whatever you must still play this turn), then **suppress
> all further draw and swing.**

**Why stopping matters — deck-out is a real loss condition.** The deck draws
10–15 cards a game; if you keep firing the engine after you've already hit lethal,
you mill yourself to zero and lose with a fat, useless hand. *Every one of our
long-game ladder losses was a self-inflicted deck-out at 1–2 prizes left, hand
bloated to 17–19 cards.* The fix is discipline, not more draw.

**Bank surplus draw on the bench.** Instead of popping a benched Dudunsparce's
Run Away Draw the moment it's available, **leave it unused** as draw *in reserve* —
it's there for the turn *after* an Iono, or when you actually need to climb to a new
threshold. A benched, un-popped Dudunsparce is stored value; a popped one is spent.

**Deck-out insurance:** **Sacred Ash** shuffles up to 5 Pokémon from discard back
into your **deck** — this is not just recovery, it is the explicit answer to running
low. Fire it *before* the deck reaches ~5 cards, not after.

---

## 5. Card-by-Card Roles (every card in the 60)

### Attackers / evolution line
- **Abra (741, Basic, 50HP) ×4** — the line's foundation. You want **two Abra down
  early** so a KO'd front Abra doesn't end the plan. Never promote an Abra to Active
  voluntarily (it can't attack and walls your own turn).
- **Kadabra (742, Stage 1, 80HP) ×4** — `+2` evolve-draw. A bench Kadabra is a
  one-step-from-Alakazam reserve; keep at least one when possible.
- **Alakazam (743, Stage 2, 140HP) ×3** — the engine and the wincon. `+3` Psychic
  Draw on evolve, then **Powerful Hand ({P})** every turn. Retreat cost {C}1, so a
  single energy frees it. A **second Alakazam on the bench is your continuity plan** —
  when the Active is KO'd, you promote and keep swinging without missing a beat.

### Draw engine
- **Dunsparce (305, Basic, 70HP) ×3** — Basic that evolves into Dudunsparce. Fine as
  an early Active (70HP soaks a hit while you build).
- **Dudunsparce (66, Stage 1, 140HP) ×3** — **Run Away Draw: draw 3, shuffle self
  back.** Primary engine + the **Enriching battery.** Don't over-pop (see §4).

### Tech Pokémon (one-of toolbox)
- **Fezandipiti ex (140, Basic, 210HP, 2-prize) ×1** — **"Flip the Script": draw 3
  after one of your Pokémon was KO'd last turn.** Comeback draw after a trade.
  Caveat: it's a **2-prize liability** and a poor Active (no useful attack for us) —
  bench it, use the ability reactively, and **never let it get stuck Active with no
  energy** (see deck-out failure mode). Respect deck count: its draw can mill you.
- **Genesect (142, Basic, 110HP, ACE-relevant) ×1** — Nullifier role: with a Handheld
  Fan attached it blanks opposing ACE-SPEC plays (notably an opponent's Rocky/Mist
  shenanigans and Unfair-Stamp-style ACE disruption). Bench + tool it vs ex/ACE decks.
- **Shaymin (343, Basic, 70HP) ×1** — bench-damage insurance vs spread (its presence
  + Battle Cage zero out chip damage to your fragile bench line).
- **Psyduck (858, Basic, 70HP) ×1** — "Damp"-style tech that strips opposing
  self-KO / ability shenanigans. Reactive, bench-only.

### Search / draw trainers
- **Buddy-Buddy Poffin (1086, Item) ×4** — search **2 Basics ≤70HP** to bench (Abra,
  Dunsparce, Psyduck). Your fastest board-builder — lead with it.
- **Poké Pad (1152, Item) ×4** — put **any non-Rule-Box Pokémon** into hand (the whole
  Abra/Kadabra/Alakazam line, Genesect, Shaymin, Psyduck — not Fez ex). Consistency glue.
- **Dawn (1231, Supporter) ×4** — grab **Basic + Stage 1 + Stage 2** to hand: the full
  Alakazam line in one card. Premier setup supporter; net +2 cards.
- **Hilda (1225, Supporter) ×3** — grab **Evolution + Energy** to hand; chains into
  evolve-draws and the Enriching attach. The energy fetch makes it the right card when
  the Active is energy-starved.
- **Rare Candy (1079, Item) ×3** — Abra → Alakazam, skipping Kadabra. **Speed only**
  (costs the +1 card vs manual). Hold when you have time; fire to hit turn-2 Alakazam
  or to rebuild fast after a KO.

### Disruption / reach
- **Boss's Orders (1182, Supporter) ×3** — gust a benched opposing Pokémon to the
  Active spot. Two uses: (a) drag up something you can **KO this turn** for a better
  prize than the current Active, (b) drag up a fragile support piece they were hiding.
  See §7 for target priority.
- **Enhanced Hammer (1081, Item) ×2** — discard a **Special Energy** from the
  opponent. *The* answer to Mist / Rocky Fighting walling Powerful Hand. Near-mandatory
  vs Crustle/Team Rocket; treat as a closing resource, not a throwaway.

### Recovery / sustain
- **Lana's Aid (1184, Supporter) ×1** — return up to 3 non-Rule-Box Pokémon + Basic
  Energy from discard **to hand**. Rebuild a wiped Alakazam line.
- **Sacred Ash (1129, Item) ×1** — shuffle up to 5 Pokémon from discard **into deck**.
  Both line recovery **and** the deck-out brake (§4).
- **Wondrous Patch (1146, Item) ×1** — attach a Basic {P} from discard to a benched
  {P} Pokémon. Energy acceleration for a bench Alakazam / recovery of discarded energy.

### Field / tools / energy
- **Battle Cage (1264, Stadium) ×4** — prevents bench damage from opponents' attacks.
  The answer to spread (Dragapult Phantom Dive, Starmie snipe, Munkidori). Keep one in
  play vs spread decks; it protects the fragile Abra/Kadabra bench.
- **Handheld Fan (1161, Tool) ×2** — anti-deck-out tool **and** the enabler for
  Genesect's nullifier role.
- **Telepath Psychic Energy (19, Special, {P}) ×4** — pays Powerful Hand **and**, on
  attach to a {P} Pokémon, searches **2 Basic {P}** (Abra) to bench. Your best energy:
  it's a board-builder and an attack-enabler in one. Route to Alakazam.
- **Basic Psychic Energy (5, {P}) ×2** — plain {P} to pay Powerful Hand. Route to
  Alakazam.
- **Enriching Energy (13, ACE SPEC, {C}) ×1** — draw 4 on attach; **{C} only, cannot
  pay Powerful Hand.** Route to **Dudunsparce** exclusively (§3).

---

## 6. Turn-by-Turn Sequencing

**Setup (mulligan logic):** keep any hand with a Basic + a path to draw. The dead
hands are "lone Abra, no Poffin/Poké Pad/Dawn" — nothing to develop. Lead the
Active with a **Dunsparce** (70HP wall + future engine) over a lone Abra when you
have the choice.

**Turn 1–2 (ESTABLISH):** Poffin out 2 Abra (+ a Dunsparce). Telepath onto a {P}
Pokémon to bench more Abra. Get the **first Alakazam online ASAP** (Rare Candy if it
lands turn 2; otherwise climb manually). Goal state: Alakazam Active with a Psychic
energy, a **backup Abra**, and a draw engine (Dunsparce/Dudunsparce) on bench.

**Mid game (CONVERT / PRESSURE):** every turn — evolve to bank draws, route one
energy, draw **to threshold**, then **Powerful Hand.** Keep a second Alakazam line
maturing on the bench. Hold Battle Cage up vs spread. Enhanced Hammer the instant a
Mist/Rocky wall appears.

**Closing (≤2 opp prizes):** take the most efficient lethal line. If a same-turn
Boss drags up a **bigger-prize** KO than the current Active, do that; otherwise just
attack. **Do not draw past lethal** — that's how you deck out on the last turn.

---

## 7. Boss's Orders — Target Priority

Only play Boss when it **changes the outcome this turn.** Costing the Boss in
(`20 × (hand−1)`):

1. **Bigger-prize KO** — if you can KO a benched ex/mega (2–3 prizes) that's worth
   strictly more than the current Active, Boss it up and KO it. This is the highest-
   value Boss and outranks a same-turn plain KO of the Active.
2. **Meaningful chip on a 3-prize mega** — even without the KO, dragging a megaEx
   into range to start the prize you'll finish next turn can be correct.
3. **Strand a support piece** — gust a fragile bench attacker/engine they were
   protecting, denying their next turn.

Target selection among KO-able options: **most prizes → highest HP → most energy**
(deny the biggest investment). **Never** Boss when you already have lethal on the
Active for the same prize count — just attack.

---

## 8. Energy Routing Rules

- **Psychic (Telepath 19, Basic 5) → Alakazam** — it needs exactly one {P} to fire.
  Prefer Telepath early (it benches 2 Abra on attach).
- **Enriching (13) → Dudunsparce** — never Alakazam (can't pay {P}; wastes the ACE).
- **One attach per turn matters** — when the Active is **energy-starved and can
  neither attack nor retreat**, fixing it is the top priority; route the attach to the
  Active even if it breaks the "Psychic-to-Alakazam" default (a single energy frees a
  retreat-cost-1 Pokémon).

---

## 9. Playing Around Hand Disruption (Iono / Unfair Stamp)

When you search and **don't need anything specific**, deliberately pull the cards you
*don't* want and **leave the good cards in the deck** — Iono puts your hand on the
*bottom*, so pieces left in the deck stay reachable while a hand full of key cards
gets buried. Corollary: **avoid shuffle effects** (don't pop Dudunsparce just to draw —
use Dawn/Hilda) when a disruptor is live and you want to preserve deck order.
Recognise the "opponent must Iono and pray" spots and bank a benched Dudunsparce so
you redraw into gas afterward. *(Requires deck-state modelling — currently a documented
gap, not yet in the heuristic.)*

---

## 10. Deck-Out Avoidance (do not lose to yourself)

This deck mills itself. Hard rules, in priority order:

1. **At/over threshold with a ready attacker → stop drawing.** No Dudunsparce pop,
   no Dawn/Hilda/Poké Pad "just because." Attack (or retreat into the ready Alakazam
   and attack).
2. **Deck < 10 → no non-essential search/draw.** **Deck < 5 → hard stop**; the only
   draw allowed is genuine emergency (you'll lose this turn otherwise).
3. **Fire Sacred Ash to refill the deck** before you reach the danger zone.
4. **Never strand a 0-energy non-attacker Active** — it can't attack or retreat, and
   the only "legal" plays left become more draw, which decks you. Attach energy to
   free it *immediately*.

---

## 11. Matchups

- **Dragapult ex (hardest):** 16 cards to KO and Phantom Dive shrinks your bench
  line. **Race prizes, don't try to KO the attacker** — Battle Cage the spread, trade
  single-prize for their 2-prize ex. Bot Dragapults often misplay → easier on ladder
  than in human events.
- **Mega Starmie ex (spread):** 50 bench snipe + a big ignore-effects attack.
  **Battle Cage** shuts the snipe; win on prize math (you take 3 for a Starmie, they
  take 1 for an Alakazam).
- **Crustle / ex walls:** you're non-ex and hit normally; Crustle's ex-immunity does
  nothing to you. Watch for **4 Mist Energy** → **Enhanced Hammer mandatory.**
- **Mega ex decks generally:** ideal prize math (1-for-3). Genesect blanks their ACE
  disruption. Boss around damage-reduction walls.
- **Team Rocket (Articuno) — worst:** Articuno walls Powerful Hand and protects
  Basics; TR Mewtwo ≈ unwinnable. Out: rush before the wall sets, Enhanced Hammer,
  direct-damage tech.
- **No-disruption aggro (Ceruledge etc.):** get two Abra down, set up, and you "can't
  lose" — pure speed race you win.
- **Mirror / single-prize:** prize-race and threshold discipline; deck-out avoidance
  decides it.

---

## 12. Our List vs the Meta List (Cerys Jones, Indianapolis 1st)

Shared core (identical): **4 Abra, 4 Kadabra, 3 Alakazam, 3 Dunsparce, 3 Dudunsparce,
4 Poffin, 4 Poké Pad, 4 Dawn, 3 Hilda, 3 Rare Candy, 2 Enhanced Hammer, 2 Handheld
Fan, 1 Sacred Ash, 1 Lana's Aid, 4 Telepath Psychic, 1 Enriching.** This is the
backbone and we match it exactly — strong validation of the build.

Where we differ (and why we keep ours): the human meta trims to **2 Boss** and runs
**Dedenne/Elgyem + Night Stretcher + Lucky Helmet + a stadium** package for
consistency and recursion. We instead run **3 Boss, 4 Battle Cage, 1 Wondrous Patch,
1 Genesect, 1 Shaymin, 1 Psyduck, 1 Fezandipiti ex.** Rationale: the **bot ladder is
spread-heavy** (Dragapult/Starmie agents), so the 4 Battle Cage + Shaymin
anti-spread package earns its slots here even though humans cut it. The extra Boss
suits a meta where reach-for-KO closes more games than raw consistency. **Deck is 20%
of scoring vs 70% for model approach — we don't churn the validated 60 on uncertain
data; the win comes from piloting it better.**

---

## 13. Strategy → Heuristic Map (and remaining gaps)

| Principle (§) | Encoded in `main.py` | Status |
|---|---|---|
| Lethal first (§2,6) | `can_ko` ATTACK = 500 | ✅ dominant |
| Boss bigger-prize snipe (§7) | `boss_ex_snipe` = 600 > plain KO | ✅ |
| Boss never over plain lethal (§7) | `can_ko → BOSS` = 1.0 | ✅ |
| Boss target = prize→HP→energy (§7) | `_pick_boss_target` sort | ✅ |
| Threshold draw then **stop** (§4,10) | `hand_surplus` suppresses draw/search | ✅ (v17) |
| Bank Dudunsparce on bench (§4) | over-pop guards on Run Away Draw | ✅ partial |
| Deck-out brakes (§10) | `deck_critical`/`deck_danger` floors; Sacred Ash 35/25 | ✅ |
| Free a stranded Active (§8,10) | `active_immobile` → energy-to-Active = 55/65, energy cards only | ✅ (v16-v18) |
| Energy routing (§8) | Psychic→Alakazam, Enriching→Dudunsparce | ✅ |
| Manual evolve > Rare Candy (§3) | evolve scores ≥ candy when not racing | ⚠️ approximate |
| Battle Cage / Shaymin vs spread (§5,11) | reactive scoring on `bench_dmg_received` | ✅ |
| Enhanced Hammer vs Mist/Rocky (§2,11) | `opp_mist` → 45 | ✅ |
| Never promote Abra/Kadabra (§5) | `_pick_bench_target` Abra/Kadabra ≤ 3 | ✅ |
| **Iono play-around (§9)** | — | ❌ gap (needs deck modelling) |
| **Threshold vs overfill nuance (§4)** | coarse (`hand_surplus` is binary) | ⚠️ partial |
| **Prize-selection engine stall (v18)** | `_resolve_stalled_or` rotation hedge | ⚠️ unconfirmed root cause |

The remaining true gaps — Iono play-around and fine-grained threshold/overfill — are
exactly the spots a learned policy should later beat the rules. The prize-selection
stall (v18) is a different kind of gap: a reproducible engine-adjacent freeze,
identified by prize-value cross-checking across multiple replays, that our own
selection logic looks correct for — hedged defensively rather than root-caused.
