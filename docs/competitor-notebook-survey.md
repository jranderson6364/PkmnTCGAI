# Competitor & Official Notebook Survey

*What the public Kaggle notebooks for this competition actually do, what the
leaderboard says about where we stand, and which of their techniques are
genuinely untested by us. Read this before choosing the next training arm.*

**Last updated:** 2026-07-23

---

## Why this file exists

Session of 2026-07-23: the user reported the project felt stagnant and asked for
a survey of the public competition notebooks, weighted toward **model training**
rather than more heuristic tuning. 12 notebooks were pulled via the Kaggle CLI
(`kaggle kernels pull`) and read in full, plus the live leaderboard.

The survey produced one uncomfortable diagnosis, one methodological bombshell
that may invalidate a chunk of our own experiment history, and three concrete
untested levers. All three are recorded here; the same-day narrative entry is in
`docs/report-log.md` 2026-07-23.

---

## 1. Where we actually stand (the diagnosis)

Pulled live 2026-07-23 from `kaggle competitions leaderboard pokemon-tcg-ai-battle`.

| Cut | Score |
|---|---|
| #1 (Luca) | 1169.3 |
| **#8 (top-8 cutoff)** | **1114.1** |
| #50 | 1037.2 |
| #100 | 998.0 |
| #200 | 943.9 |
| **Us (Jason Anderson, team 16394479)** | **637.8 — rank 2902 / 5578** |

**We are at the 52nd percentile and ~476 points below the top-8 cutoff.** This is
the honest answer to "are we stagnating" *on the ladder*: yes, and by more than
the project's internal narrative suggested. Our internal docs track offline win
rates and `publicScore` reads in the 650–750 band; the field has moved to 1100+.

**Read this against the right yardstick.** This leaderboard is the **Simulation**
category — points and medals, plus a participation prerequisite for Strategy. The
**$240k is entirely in the Strategy track**, scored 70% model approach / 20% deck
concept / 10% report and judged from the writeup, not from ladder rank. So
"476 short" is 476 short of a *Simulation* placing, **not of the prize bar.** The
two-part honest diagnosis is:

- *On the ladder:* stagnant and far behind, with a piloting gap we can name.
- *On the axis that is actually judged:* this survey alone produced three
  methodology-grade findings (§2, §6, and the external replication in §1) and
  identified the one architecture we have never tried (§3a). That is real
  progress on the 70% axis.

Both are true. Neither cancels the other, and the ladder cannot be abandoned
because Strategy entry requires Simulation entry.

Multiple **public** notebooks — free, forkable, no team required — outscore our
champion by 100–450 points:

| Public notebook | Score | Nature |
|---|---|---|
| `nursrijan` Lucario heuristic (via writeup) | **1091** | pure heuristic |
| `aristophanivan/probablity-v2` | **933.8** | pure heuristic (Mega Lucario ex) |
| `lucifer19/battlecore-compact-agent` | **846.8** | pure heuristic (Archaludon) |
| `raunakdey07/pok-mon-tcg-advanced-heuristic-agent` | 796.8 | heuristic + MCTS |
| `prvsiyan/…search-audited-alakazam-v9` | 778.2 | **our deck**, heuristic + bounded search |
| `prvsiyan/…field-audited-alakazam-v8` | 739.7 | our deck |
| `fishcat37` v8/v9 attention BC (learned) | **no score published** | learned |
| our v29d / v30-exp | 673.5 / 637.8 | heuristic |

Two independent teams pilot **our own Alakazam deck 100–140 points better than we
do**. That is a piloting gap, not a deck problem.

### The discriminating fact: the ladder ceiling is currently non-learned

Every agent above ~730 that publishes a score is **rule-based or
rule-based-plus-bounded-search**. The two purely-learned public entries
(`fishcat37` v8/v9 attention BC) publish **no ladder score at all**. Independent
corroboration from `nursrijan`, who reached LB 1091:

> BC pre-training + PPO self-play produced a policy with **25% win rate** vs the
> heuristic […] MCTS […] suffered from time-budget exhaustion.

That is the *same plateau we hit* (our BC/DAgger family: 12–17%; our DMC arc:
2.5%→8.25%). A different team, different deck, different algorithm (PPO not
DMC/AWR), same wall. **This is strong external validation of our negative
results** — and it means the learned track is currently a *report* asset, not a
ladder asset. Do not conflate the two goals again.

---

## 2. The methodological bombshell: our search A/Bs may be contaminated

From `lucifer19/battlecore-compact-agent` (846.8), §3.3, "the sham-search
placebo control":

They added a Search-API attack oracle, then ran a **placebo arm** — a variant
that performs every `search_begin`/`search_step` call and then **discards the
result**, so it is behaviorally identical to baseline by construction. It should
have measured the baseline's 0.48. It measured **0.450**.

> *In a shared process, agent-side searches perturb the live engine's RNG
> stream. The measurement gate itself was contaminated.*

They rebuilt their arena with **one OS process per agent** (Kaggle's actual
isolation model) before trusting any verdict.

### Why this matters to us, specifically

`training/harness.py:119` parallelizes across **games** (`mp.Pool`, one process
per game) — but inside a game, `_worker` runs **both agents and the engine in the
same process**. So whenever one side is a search agent (`mcts.py`,
`ismcts_agent.py`, `endgame_agent.py`, `rv_endgame_agent.py`,
`gumbel_endgame_agent.py`, `phi4_agent.py`, `advisor_agent.py`) and the other is
plain `main.py`, the search side is drawing from the shared engine RNG and the
baseline side is not.

**This is the configuration battlecore showed is invalid.** But scope the claim
carefully — the observed artifact was **~3pp** (0.480 → 0.450), and a 3pp
measurement bias cannot manufacture a 0-of-50 sweep or a 15pp anchor gap. Sorting
our search results by whether an effect that size could plausibly flip them:

| Result | Margin | Could ~3pp move the verdict? |
|---|---|---|
| v29 endgame search, **59.0% ±4.8%** vs plain heuristic (2026-07-07) | ~9pp over 50 | **Yes — this is the real target.** Near enough the decision boundary to matter |
| v29d gauntlet revert: lucario 73.0%, abomasnow 75.0% vs an 88% bar | 13–15pp | Unlikely to flip, but the *sign* is informative (see below) |
| ISMCTS 0W-50L (2026-07-07); PIMC 0W-50L (2026-07-04) | 50pp | **No — structural.** The agent played badly; these closures stand |
| Φ v4 leaf-eval Gate 2: 74.0%/62.0% vs 94.0%/96.0% same-day | 20–34pp | **No.** Closure stands |

So: **do not treat the 0W-50L closures as reopened.** The honest statement is
that we have *identified and bounded a measurement risk*, and that exactly one
headline result — v29's +59.0% — sits close enough to the boundary for a few-pp
artifact to matter.

One nuance worth keeping: battlecore's placebo scored *lower* than baseline, i.e.
the searching side paid the penalty. If that sign holds for us, contamination
would have **understated** v29's endgame edge and **overstated** the severity of
the v29d gauntlet failure — the opposite of the convenient direction.

It does **not** affect the non-search arms (BC/DAgger/AWR/DMC/sequence-policy
gates), which never call the Search API at inference.

**This is cheap to test and we should test it: run our own sham-search placebo.**
Take `endgame_agent.py`, make a variant that runs the full search and then throws
the answer away and returns the heuristic's choice, and A/B it against plain
`main.py` at n=400 — the same protocol that produced the +59.0% number, so the
placebo is a direct control for *that* measurement. Expected reading under a
clean harness: 50%. **Either outcome is report-grade**: a confirmed contamination
is a first-class methodology finding (and re-dates v29's edge), a clean placebo
hardens every existing closure.

---

## 3. The official RL/MCTS sample (`kiyotah`, 862 votes, pinned)

The single highest-value artifact in the competition. Full source pulled to
scratchpad; key structure:

### 3a. Action-conditioned architecture — the lever we have never pulled

```
encoder: SparseVector -> EmbeddingBag(22000, d_model, mode="sum")
         -> 24 "words" -> TransformerEncoder -> tanh -> VALUE (scalar)
decoder: per-ACTION SparseVector -> EmbeddingBag(decoder_size, d_model)
         -> cross-attention over encoder_out -> Linear(d_model,1)
         -> one scalar PER CANDIDATE ACTION
```

The decoder **embeds each candidate action** (its option type, the card it
plays, the target it hits, the attack id) and scores it against the board
representation. `fishcat37`'s v8/v9 use the same shape independently — a
`BCPolicy` with board self-attention over 51 card channels plus
**option cross-attention**, a shared `score_head` emitting one logit per option.

**We have never built this.** `grep -rl "MultiheadAttention\|option_encoder"
training/` returns nothing. Every net this project has trained — BC-MLP,
DAgger, AWR, IQL, DMC (`model.py`/`model_big.py`), the oracle critic, the
sequence-policy — scores a **fixed-size action slot vector** from a
hand-engineered state encoding.

This matters because it is the one axis our own prior `advisor` consult
identified as never varied: *"nine algorithm variations, four other axes never
varied."* Architecture was one of those axes. An action-conditioned net
generalizes across actions it has never seen in that slot; a fixed-slot MLP
cannot.

### 3b. Native determinization + the direct battle driver

```python
from cg.game import battle_start, battle_finish, battle_select   # direct engine driver
search_begin(obs, your_deck=…, your_prize=…, opponent_deck=…,
             opponent_prize=…, opponent_hand=…, opponent_active=…)
```

- `cg.game.battle_start/battle_select/battle_finish` **exists in our vendored
  `training/local_cg/cg/game.py` but is used by none of our training code** — we
  drive every game through `kaggle_environments` (~0.5 s/game). The official
  sample drives the engine directly. Worth benchmarking: collection is our
  CPU bottleneck, and this removes the whole env wrapper + JSON round-trip.
- `obs_dict["search_begin_input"]` is a **real field on the live observation**
  (`training/local_cg/cg/api.py:443`) — the engine hands the agent its own
  determinization seed at inference time. `probablity-v2` reads it via
  `obs_dict.get("search_begin_input")`.
- The sample fills the opponent's hidden zones with **Snorlax ×N and basic
  energy** and says so explicitly: *"There is no deep meaning."* `nursrijan`
  independently names this as their MCTS's fatal flaw ("massive simulation
  bias") and proposes archetype inference as the fix — **which is precisely what
  our Stage 3 belief model already does at 92.3% held-out accuracy.** We have
  the asset the strongest public teams say they lack.

### 3c. A training-target recipe we have not tried

Our DMC arc swept full-MC (weak) and n-step=5 (encouraging, deferred). The
official sample uses neither:

```python
LAMBDA = 0.9
value = 1.0 if i == result else -1.0
for sample in reversed(samples[i]):
    label = (value + sample.value) * 0.5              # blend outcome with MCTS root value
    value = value * LAMBDA + sample.value * (1.0 - LAMBDA)   # λ-return backup
```

Value label = **mean of the λ-return and the MCTS root value**. And the policy
target is **not** an AlphaZero visit-count softmax — it is the per-child
*advantage* `Q(child) − V(root)` clipped to [−1,1], regressed with HuberLoss, and
converted back to a prior at inference via `exp(10·p)`. Both are third options
we never swept.

---

## 4. What the top public agents actually do

### `aristophanivan/probablity-v2` — 933.8, and its search is dead code

Mega Lucario ex heuristic with a bolted-on beam search. **The beam search never
runs.** `def evaluate_state(obs, original_yourIndex)` takes two arguments; all
three call sites are `evaluate_state(ar.state.observation)` — one argument.
Every call raises `TypeError`, caught by the bare `except Exception: return None`
wrapping `SEARCH_ALGO`, which falls through to the plain heuristic. Verified
against the notebook source.

**So the highest-scoring public agent in this competition is a pure heuristic,
and its author probably does not know the search is inert.** That is the
cleanest possible statement of "search is not what separates the top of this
ladder."

Techniques worth stealing from its heuristic regardless:
- `prize_count()` correctly decrements for **Legacy Energy** and **Lillie's
  Pearl** — real rules interactions we should check we handle.
- `target_score = prize_count×2000 + energies×300 + tools×200 + stage bonus + hp`
  — denying invested energy/tools is scored explicitly.
- Anti-stall branch keyed on Crustle/Snorlax that **inverts** the hand-size term.

### `lucifer19/battlecore-compact-agent` — 846.8

Beyond the placebo control (§2), two reusable items:

- **Two real Search-API bugs**, both of which we should check for in our own
  search code: (1) *KO mis-measurement* — after a KO the opponent promotes a new
  active, so naïve "HP before − HP after" scores a kill shot as ~0; detect **your
  own prize count dropping** instead. (2) *Turn-boundary contamination* —
  stepping past your own decisions lets a phantom opponent act; halt at the turn
  boundary.
- **A ladder-noise model**: version-to-version `publicScore` deltas smaller than
  ~2 stationary rating SDs carry no quality signal. They observed a 600→1054
  range and rejected the pure-luck null. This is the principled version of our
  own repeatedly-rediscovered "do not trust a single publicScore read."

### `prvsiyan/…search-audited-alakazam-v9` — 778.2, our deck

Four Abra / four Kadabra / four Alakazam / four Rare Candy, transcribed from
public episode 87347575. Also publishes an audit of a Great Tusk control deck
over 50 official episodes: 36.8% vs Alakazam, 44.4% vs Archaludon, 25.0% vs
Grimmsnarl, 0/5 vs Crustle.

### `fishcat37` v8/v9 — the only serious learned entries

**Behavioral cloning on the top-20 leaderboard teams' replays**, not on their own
heuristic. Dataset `fishcat37/ptcg-v8-daily-top20`, aligned per-day by replay
`agent_index` to exact team names. Config: `top_k=20`, 3 days of data
(2026-07-13→15), 8 epochs, lr 6e-4, batch 4096, T4×2, `bc-static-v3` feature
contract, Parquet sharding.

This is the *external teacher* our own advisor said we lacked — cloning agents
**stronger than ours** rather than cloning `v29d`. Note our own winner-BC
replay-imitation family closed negative on 2026-07-08 (RP-1 77%/5%, RP-2 89%/8%),
but that ran without top-20 filtering and without an action-conditioned net.
Also note: **neither v8 nor v9 publishes a ladder score**, so there is no
evidence this works either.

---

## 5. Meta intelligence (deck axis, 20% of the score)

From `nursrijan`'s analysis of 25,000+ top replays across 5 days:

| Finding | Number |
|---|---|
| Night Stretcher (#1097) — winner-card correlation | **+1556** (highest) |
| Boss's Orders (#1182) | **+1526** |
| Sacred Ash (#1129) | +924 |
| **Alakazam — overall delta** | **−238** |
| Crustle wall | −176 |
| Mega Abomasnow ex | −393 (worst) |
| Going first vs second | 56% / 44% |

Archetype win rates: Mega Lucario ex 76.4%, Alakazam 74.5%, Hop's Trevenant
73.5%, Hop's Snorlax 72.7%, Dipplin Grass 72.1%.

Two things to sit with:

1. **Alakazam is −238 overall but 74.5% in the hands of one team.** Their read:
   *"high-skill, low-floor — weak implementations drag down its aggregate."*
   Combined with the two public Alakazam agents outscoring us, the evidence says
   our deck is fine and our pilot is the problem.
2. **The two highest winner-correlated cards are recovery cards we do not run.**
   Night Stretcher returns a Pokémon from discard **to hand** — which in our deck
   simultaneously (a) fixes board-thinning, our #1 documented live failure mode
   (10/27 fresh v29d losses end with zero Alakazam-line pieces in play), and
   (b) *increases hand size*, which is literally our damage stat. Sacred Ash
   returns Pokémon from discard to deck, which addresses our deck-out losses.
   This is the rare change that is synergistic on all three axes at once.

**Caveat on the card deltas:** these are winner/loser *correlations*, and they are
archetype-confounded — strong decks tend to run recovery packages *and* win, so
the delta attributes the deck's strength to the card. Treat Night Stretcher /
Sacred Ash as a cheap offline deck lever to gate properly, not as an established
causal effect, and do not assert causation from these numbers in the report.

Also available as maintained public datasets, refreshed daily — free meta
telemetry we currently recompute ourselves:
- `busyaprime/pokemon-tcg-ai-battle-live-meta` (tier list, matchup grid, deck recommender)
- `fishcat37/ptcg-v8-daily-top20` (top-20 team replay index)

---

## 6. Our own accidental A/A test (self-generated, not from a notebook)

Surfaced while pulling our submission history for this survey:

| Submission | Content | publicScore |
|---|---|---|
| 54760870 | v29d re-ship, copy 1 | **708.9** |
| 54760877 | v29d re-ship, copy 2 — **byte-identical tarball** | **620.5** |
| 54766181 | v30-exp | 637.8 |

**Two byte-identical submissions scored 88.4 points apart.** We ran a perfect
A/A control on the live ladder without meaning to.

**Provenance (verified 2026-07-23, not taken from the submission description):**
there is exactly one local artifact, `training/v29d_reship.tar.gz` —
24,857 bytes, SHA-256 `acef37506436e6a641ea05ce4f166bac570c0b7ff6cb1c7a19f8463ccc14d5be`,
mtime 2026-07-16 08:51 — and it was uploaded twice, at 12:52:58 and 12:53:11
(13 s apart). `training/ladder_history.csv:22` records copy 2 as "second copy of
54760870." Same file, two submissions, two scores.

Consequences:

- The 2026-07-16 pre-registered v30-exp revert rule ("≥30 below the lower v29d
  copy on 2 consecutive reads") is **statistically meaningless** — its threshold
  is a third of the demonstrated identical-code spread. v30-exp's 637.8 sits
  *between* the two identical copies. It is not distinguishable from v29d.
- Every historical `publicScore` comparison in this project — the
  818.3→726.2→695.1 "monotone decline," the 861.8 v25b reading, the 116-point
  v28 swing — is inside or near this noise band. Our repeated
  "CORRECTION" entries about ladder reads were all circling this one fact.
- It independently confirms battlecore's noise model (§4) with our own data.

**This is one of the best pieces of methodology evidence the project owns** and
should be a figure in the report: an unintentional identical-agent A/A test
quantifying a live evaluator's noise floor at ~88 points.

**Operationally urgent:** our leaderboard score tracks our *latest* submission,
and it is currently v30-exp at 637.8, below v29d's 708.9 copy. The scheduled
2026-07-18 revert check never ran (this session is 2026-07-23).

---

## 7. Untested levers, ranked

| # | Lever | Why it is new | Cost |
|---|---|---|---|
| 1 | **Sham-search placebo on our own harness** | Directly tests whether ~6 closed search results are measurement artifacts | Low — one agent variant, n=400 |
| 2 | **Action-conditioned net** (option cross-attention, one scalar per candidate action) | The one architecture axis never varied across 9 algorithm variations | Medium |
| 3 | **Night Stretcher / Sacred Ash deck slots** | Top-2 winner-correlated cards; fixes board-thinning *and* raises hand size (= damage) | Low |
| 4 | Belief-model determinization into the official Search API | Both strong public teams name crude determinization as their blocker; we have a 92.3% classifier | Medium |
| 5 | `cg.game.battle_start` direct driver instead of `kaggle_environments` | Collection is CPU-bound; removes the env wrapper entirely | Low, infra-only |
| 6 | λ-return ⊕ MCTS-root value targets; advantage-regression policy head | Third target recipe, never swept | Low |
| 7 | Top-20-filtered replay BC | External teacher stronger than v29d | High; unproven publicly |

Levers 1 and 3 are cheap and independent of everything else. Lever 2 is the real
research bet.

---

## 8. Reconciling the competition facts

The organizers' LinkedIn post (2026-07-23) vs `CLAUDE.md`:

| Item | LinkedIn post | Our CLAUDE.md | Status |
|---|---|---|---|
| Simulation entry deadline | **Aug 9, 2026** | "team merger deadline Aug 9" | **conflict — verify** |
| Ladder close | (not stated) | ~Aug 16–17 | Kaggle page reads "24 days to go" on 07-23 → **~Aug 16**, consistent |
| Strategy deadline | **Sep 6, 2026** | ~Sep 13 | **conflict — verify** |
| Simulation prize | Points and medals only | — | note |
| Strategy prize | **$240,000 pool** | "top-8 → $30k" | consistent with a pooled $240k |
| Dependency | Strategy **requires** Simulation participation | implied | consistent |

The post's Strategy goal wording is worth quoting into the report plan verbatim:
*"strategic data analysis and agentic gameplay, while analyzing and sharing the
reasoning, methodologies, and design decisions behind each approach."*

**Values check: the report-driven direction is correct and the money is entirely
in the Strategy track.** Our graveyard of pre-registered negative results is
legitimate, high-value content there — battlecore's 846.8 notebook leads with
*"why negative results are the headline"* and won 17 votes doing it. But the
ladder and the report are **two different goals**, and §1 shows we have been
letting offline learned-model work stand in for ladder progress. Strategy
participation *requires* Simulation participation, so the ladder cannot be
abandoned — but it should be pursued with the piloting/deck levers (§7 items 1,
3), not with another learned arm.

---

## 9. Source material

All 12 notebooks pulled to scratchpad via `kaggle kernels pull -m`:

| Ref | Votes / score |
|---|---|
| `kiyotah/reinforcement-learning-and-mcts-sample-code` | 862, pinned |
| `aristophanivan/probablity-v2` | 933.8 |
| `lucifer19/battlecore-compact-agent` | 846.8 |
| `raunakdey07/pok-mon-tcg-advanced-heuristic-agent` | 796.8 |
| `prvsiyan/ptcg-ai-battle-search-audited-alakazam-v9` | 778.2 |
| `nursrijan/pokemon-tcg-ai-my-efforts-and-tries-on-rl` | LB 1091 writeup |
| `fishcat37/ptcg-v8-attention-end-to-end-submission` | learned, no score |
| `fishcat37/ptcg-v9-attention-all3d-end-to-end` | learned, no score |
| `myso1987/ptcg-ai-battle-leaderboard-deck-meta-by-score-band` | 43 votes |
| `busyaprime/what-actually-wins-on-the-ladder` | 21 votes |
| `beicicc/ptcg-public-experiment-snapshot-jul20` | public experiment log + GitHub |
| `llccqq624/ptcg-replay-data-miner` | 11 votes |

Not yet mined: `myso1987`'s `score_band_top10.csv` (deck archetype share per
100-point leaderboard band, 500→1100+) — the CLI `kernels output` call returned
empty; the file is browsable on the notebook's Output tab. It would answer
"which decks actually live at 1100+" directly.

---
