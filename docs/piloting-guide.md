# Alakazam Expert Piloting Guide (v2)

*The behavior-cloning target spec. This is what the heuristic tries to encode and what the net should learn.*

---

## 1. Game Plan

Draw a ton, blow up the Active with Powerful Hand, every turn, as the aggressor in every matchup. Glass cannon: powerful but fragile. Most tech slots exist to **pre-empt the specific answers** to your plan, not to add nuance to it.

---

## 2. Core Math

**Powerful Hand = 20 × (cards in your hand)**, placed as damage counters → **ignores Weakness, Resistance, and damage reduction.** KO = `ceil(HP / 20)` cards.

| HP | Cards needed | | HP | Cards needed |
|----|-------------|---|----|----|
| 60 | 3 | | 200 | 10 |
| 120 | 6 | | 250 | 13 |
| 140 | 7 | | 300 | 15 |
| 150 | 8 | | 320 | 16 |
| 170 | 9 | | 330 | 17 |

**Hard vulnerability:** Mist Energy (#11) and Rocky Fighting Energy (#20) make Powerful Hand deal 0. This drives Enhanced Hammer and the worst matchup (Team Rocket).

---

## 3. Card-Counting: The Real Skill

You don't blindly maximize hand size — you **count how big your hand can get next turn and hold exactly the threshold you need.** Every action has a *net card* value:

| Play | Net cards | Why |
|------|-----------|-----|
| Kadabra (manual) | **+1** | -1 to play, +2 from Psychic Draw |
| Alakazam from Kadabra | **+2** | -1 to play, +3 from Psychic Draw |
| Alakazam via Rare Candy from Abra | **+1** | -1 Candy -1 Alakazam +3 draw |
| Dawn | **+2** | grab 3 Pokémon, -1 to play supporter |
| Hilda | **+1** (up to **+6** with Enriching + evolve + attach) | |
| Enriching Energy | **+3** | draw 4, -1 to attach |
| Dudunsparce (Run Away Draw) | **+3** | and shuffles itself back |

Manual evolve beats Rare Candy by +1 card — use Candy only for speed.

**The discipline: hit the threshold, then stop.** Don't overfill. If you need 16 cards to KO a Dragapult, don't bloat to 20 — extra cards are wasted and expose you to disruption. **Bank surplus draw on the bench** instead: leave spare Dudunsparce unplayed so you have draw *in reserve* for after an Iono.

*Encoding:* target `cards_needed = ceil(opp_active_HP / 20)`. Draw/evolve until `hand >= cards_needed` (+ buffer for what you must play this turn), then **suppress further draw** and swing. Track "banked draw" count (benched Dudunsparce) separately.

---

## 4. Playing Around Hand Disruption (Iono / Unfair Stamp)

When you search and **don't need anything specific**, deliberately grab the cards you *don't* want and **leave the cards you *do* want in the deck.** Reason: Iono puts your hand on the *bottom* of the deck — so cards left in the deck become reachable again, while a hand full of key pieces gets buried.

Corollary: **avoid shuffle effects** (don't pop a Dudunsparce just to draw — play Dawn/Hilda instead) when you want to preserve deck order so a post-Iono draw finds what you need.

Recognize the "**opponent must Iono and pray**" spots — when they're forced to disrupt and hope you miss the 2–3 cards you need.

*Encoding (later pass):* when a search has no required target, prefer pulling low-value cards; prefer non-shuffling draw (supporters) over Run Away Draw when a known disruptor is live.

---

## 5. Tech Package — What Each Slot Answers

- **2 Enhanced Hammer** — *the* answer to Special Energy that walls Powerful Hand (Mist, Rocky Fighting). Treat as a closing resource, not a throwaway. Running out vs Mist-heavy Crustle decks (up to 4 Mist) often means you must rush instead.
- **4 Battle Cage** — zeroes bench spread (Munkidori, Dragapult, Froslass). Full 4-of; keep one in play vs spread.
- **2 Handheld Fan** — anti-deck-out. You draw so much that loops like repeated Genesect Clutch can deck you.
- **4 Poké Pad** — supporter search engine (this deck runs no traditional draw supporters; Poké Pad + Dawn + Hilda do the work).
- **Lana's Aid / Sacred Ash** — recovery: rebuild Alakazam lines and recur energy/Pokémon.
- **Wondrous Patch** — setup consistency + energy acceleration for bench Alakazam.

---

## 6. Decision Priority (Heuristic Ordering)

1. **Lethal:** if `hand * 20 >= active_HP` (or a gust target is KO-able), take the KO.
2. **Guards:** never empty the board (suppress Abra's Teleporter; lone-Pokémon guard); keep Battle Cage up vs spread; **never swing into an Active under Mist/Rocky Energy** (damage = 0) — instead Enhanced Hammer or Boss a different target.
3. **Threshold draw, then stop:** draw/evolve toward `ceil(opp_HP/20)`; once met, **bank** (bench Dudunsparce) rather than overdraw.
4. **Advance setup:** manual evolve > Rare Candy (the +1 card); bench only what you need.
5. **Energy:** 1 attach to the attacker; Psychic → Alakazam, Enriching → Dudunsparce.
6. **Hold + attack** with the threshold-sized hand.

---

## 7. Matchups

- **Dragapult ex (hardest):** 16 cards to KO and they shrink your hand — **race prizes, don't try to KO the attacker.** Battle Cage; Lillie's Clefairy ex shines (spawn through the line-wipe). Bot players pilot Dragapult badly → fewer strong Dragapult agents on ladder than in human events.
- **Mega Starmie ex (spread):** 50 bench snipe; Nebula Beam 210 ignores active effects. **Battle Cage** shuts the snipe. Win on prize math (3 prizes when you KO a Starmie vs 1 when they KO Alakazam).
- **Crustle / ex walls:** you're non-ex, hit normally. Watch for Crustle running 4 Mist Energy → **Enhanced Hammer mandatory**. Crustle immune to ex attacks — Alakazam bypasses this entirely.
- **ex decks generally:** trade 1-for-2 prize math; Genesect blanks Unfair Stamp.
- **Team Rocket (Articuno) — worst:** Articuno walls Powerful Hand, protects Basics. TR Mewtwo ≈ unwinnable; TR Honchkrow is close. Real outs: Lillie's Clefairy ex (direct damage, bypasses effect blocks) + rush before the wall sets.
- **No-disruption aggro (Ceruledge):** get 2 Abra down, set up, you "can't lose."
- **Mirror / single-prize:** skill + prize-race; even.

---

## 8. Honest Ceiling

The win condition is a formula and the role never changes, so heuristics go far. The leak points:
- **Threshold management vs overfill** — biggest current gap in v14
- **Iono play-around** — requires deck-state modeling, beyond current heuristic
- **Enhanced Hammer resource timing vs special-energy walls** — reactive not proactive

These three are exactly where a cloned policy should later beat the rules.
