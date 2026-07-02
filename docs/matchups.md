# Alakazam Matchup Reference

*Tech cheat-sheet and matchup verdicts for the heuristic and learned agents.*

**Last updated:** 2026-07-01

---

## Core Principle: Win the Prize Trade

Alakazam is a single-prize attacker. Every key matchup is decided by: **can you answer their damage while landing knockouts?** Against the 2-prize (ex) and 3-prize (Mega ex) field, the math is lopsided — you give up 1, they give up 2–3.

---

## Tech Cheat-Sheet

| Threat | What it does | Your answer |
|--------|--------------|-------------|
| **Mega Starmie ex (spread)** | 50 bench snipe; Nebula Beam 210 ignores active effects | Battle Cage; win on prize math |
| **Dragapult ex (spread)** | 200 active + 60 bench spread | Battle Cage + Psyduck |
| **Crustle (ex wall)** | Immune to ex attacks | You're non-ex — hit it normally |
| **Mist Energy (#11)** | Prevents all effects of attacks on holder | Enhanced Hammer to remove it |
| **Rock Fighting Energy (#20)** | Prevents all effects of attacks — Rocky Energy ACE SPEC | Enhanced Hammer + Genesect+Fan blocks it from being played |
| **Unfair Stamp (disruption)** | Shrinks your hand, gutting Powerful Hand | Genesect (ACE Nullifier with Tool) blocks it |
| **Self-KO abilities (Dusknoir)** | Snipe your ability Pokemon | Psyduck (Damp) blocks self-damage abilities both sides |
| **Wrong/awkward active** | Best target is benched or stalling | Boss's Orders to gust it up |
| **Team Rocket Articuno** | Protects Basics, zeros effect-based attacks | No clean answer; rush or Lillie's Clefairy ex (direct damage) |

---

## Matchup Verdicts

| Matchup | Verdict | Notes |
|---------|---------|-------|
| **ex / Mega field (general)** | Favorable | Prize math dominates |
| **Mega Starmie ex** | Favorable with Battle Cage | Disciplined Cage play; win on prize race |
| **Crustle** | Favorable | You're non-ex; immunity doesn't apply |
| **Dragapult ex** | Hardest; mitigated | Battle Cage + bot pilots Dragapult badly |
| **Mega Abomasnow ex** | Favorable | Prize math; race before Kyogre loads |
| **Mirror / single-prize** | Even | Decided by piloting |
| **Team Rocket (Articuno)** | Worst matchup | Mewtwo ≈ unwinnable; Honchkrow hard; rush |

---

## vs Mega Starmie ex (Spread) — Favorable

**Their card:** Stage 1, 330 HP, 3 prizes. Jetting Blow {W}: 120 to Active + 50 to one Benched Pokemon. Nebula Beam {C}{C}{C}: 210, ignores effects on Active.

- **Their plan:** snipe Abra/Kadabra off the bench before you set up.
- **Your answer:** Battle Cage stops damage counters on benched Pokemon — zeroes out the 50 bench snipe. Starmie reduced to 120-to-your-active only.
- **Prize math:** KO a Starmie = 3 prizes (half the game); they KO an Alakazam = 1. You can lose the attrition and still win the race.
- **Risks:** OHKO on 330 HP needs ~17 cards, so grind via prize math and gust. Their deck runs Boss's Orders — guard against them dragging up a low-HP bench target.
- **Important:** Nebula Beam *ignores effects on the Active* — this means it bypasses Mysterious Rock Inn (Crustle immunity). Not directly relevant to you but worth knowing.

---

## vs Dragapult ex (Spread) — The Hardest Matchup

**Their card:** Phantom Dive does 200 to Active + 60 spread across bench — wipes your evolution lines.

- **Your answer:** Battle Cage shuts the spread. Psyduck blocks self-KO ability tricks.
- **Arena softener:** Dragapult is brutal for a bot to pilot (target selection every attack). Fewer strong Dragapult agents on the ladder than in human events. Your worst matchup is softer here.

---

## vs Crustle (The Ex Wall) — Favorable

**Their card:** Stage 1, 150 HP, retreat 3. Mysterious Rock Inn prevents all damage to Crustle from opponent's ex attacks. Superb Scissors: 120.

- **Why you beat it:** Alakazam is **not** an ex, so Mysterious Rock Inn does nothing against you — you damage Crustle normally.
- **The bypass rule (general):** any attack that "isn't affected by effects on the opponent's Active" ignores Rock Inn (the immunity counts as an effect on the active). Mega Starmie's Nebula Beam, Mega Lopunny's Spiky Hopper, Cornerstone Ogerpon's Demolish all punch through.
- **Watch for:** Crustle decks can run up to 4 Mist Energy → Enhanced Hammer is mandatory to keep Powerful Hand active.

---

## Why Not Alakazam + Crustle Hybrid

- Crustle immunity covers only itself; spread Megas still snipe your Alakazam line.
- Starmie's Nebula Beam (210) bypasses immunity and one-shots the 150 HP Crustle.
- Wrecks consistency — second evolution line + Grass energy + retreat-3 mode-switch dilute the single-energy engine.
- Wrong shape for behavior cloning — wall+combo control decks are high-branching.

---

## vs Mega Lucario ex (Important Matchup — 3-Prize Opponent)

**Their card:** 340 HP, Fighting type, 3 prizes (megaEx=True). Weak to Psychic → Alakazam gets +20 effective damage.

- KO threshold: ceil((340-20) / 20) = 16 cards in hand
- Rocky Energy (card #20) is their ACE SPEC counter — blocks Powerful Hand entirely
- Korrina + Arena of Antiquity buffs their damage to 190 against ex — **do not bench Fez against Lucario**
- **Our answers:** (a) Enhanced Hammer removes Rocky Energy once attached; (b) Genesect+Handheld Fan (ACE Nullifier) prevents Rocky Energy from being played at all
- **Boss note:** don't gust Lucario off the field — you *want* to KO it for 3 prizes (the whole game)

---

## vs Mega Abomasnow ex (Energy Mill)

**Their card:** Stage 1 megaEx, 3 prizes. Hammer-lanche: discard top 6 cards of their
own deck, deal 100 damage per Basic Water Energy discarded. Kyogre (backup): Riptide —
20 × Basic Water Energy in discard. Their whole game is self-milling Water Energy for
damage.

- **Their plan:** pile Water Energy into discard fast, then hit for massive damage with
  Kyogre once Abomasnow is gone or set up.
- **Your answer:** they're self-milling, not attacking your bench — Battle Cage is less
  critical here. Race the prize trade: KO Abomasnow for 3 prizes, then KO Kyogre for 2.
  You give up 1 per Alakazam. Pure prize math — heavily favored.
- **Watch for:** if they get enough Water in discard quickly, Kyogre can spike for
  200+ damage. Don't let them stall while building Kyogre — Boss it up before it's ready.
- **Enhanced Hammer note:** their deck runs Basic Water Energy, not Special, so Hammer
  doesn't help here. Win through speed.

---

## One Discipline Note

Don't optimize for any single matchup on paper. Decide every matchup call by **real-ladder A/B** once the cloned net exists, never by offline win-rate. If spread genuinely floods the ladder, that's the signal to revisit Battle Cage counts or build variants.
