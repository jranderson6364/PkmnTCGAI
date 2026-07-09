# Game-State Evaluation Function — Literature Research

*External research (2026-07-09) on how successful card-game AIs build hand-crafted
state evaluation functions, gathered to design Φ v3 — the richer evaluation
function intended to fix the leaf-value bottleneck that closed every prior
search line. Companion to `docs/method-survey.md` (broad ML survey) and
`docs/nn-training.md` (Φ v1/v2 history).*

**Last updated:** 2026-07-09

---

## Why This Research

Every search arm this project closed (PIMC, ISMCTS, endgame-search-vs-aggro,
AlphaZero-style self-play) converged on the same diagnosis: **the leaf/state
value signal is the binding constraint, not the search machinery.** The best
real-replay value signal the project has ever measured is the hand-designed
Φ v2 potential (0.604 ALL / 0.696 LATE sign-accuracy) — beating every trained
network. This research asks: how do successful CCG AIs build such functions,
what features do they use, and how do they tune and deploy them?

---

## Source 1 — Miernik & Kowalski, "Evolving Evaluation Functions for Collectible Card Game AI" (arXiv:2105.01115)

Testbed: Legends of Code and Magic (LoCM), the Strategy Card Game AI
Competition game. Full paper read (all 8 pages).

**Architecture.** Eval = `evalState(global features) + Σ evalCard(own cards
in play) − Σ evalCard(opponent cards in play)`. This global+per-card
decomposition is described as the standard simplification in the field.

**Feature set (their Linear genome, 20 weights):**
- Global, 6 per player (as own-vs-opponent pairs): current mana, deck size,
  health, max mana, cards to draw next turn, next rune (draw indicator).
- Per-card, 8: attack, defense, and one 0/1 flag per keyword.

**Representation result.** Linear combination beat both tree/GP
representations under limited compute (tournament avg 54.8% vs ~45.6%);
trees only paid off when *bootstrapped from a converged linear solution*
(Tree-from-Linear 60.2%, the top of the whole tournament). **Takeaway:
start linear; consider nonlinear only as a second stage seeded by the
linear solution.**

**Tuning result.** Weights evolved by a plain GA (pop 50, 50 generations,
elitism 5, mutation 5%; fitness = win-rate over ~100 games/pairing).
Choice of fitness opponent matters a lot: self-play ("progressive",
vs best of previous generation) > fixed strong opponent > fixed weak
opponent (evolving vs a weak opponent produced the weakest agents in the
tournament — "no point in learning against a weak opponent").

---

## Source 2 — Santos, Santos & Melo, "Monte Carlo Tree Search Experiments in Hearthstone" (IEEE CIG 2017)

Full paper read (all 10 pages). The most concrete recipe for using a
hand-crafted eval **inside** MCTS.

**Heuristic 1 (5 features, all difference-form):** minion advantage,
tough-minion advantage, hand advantage, trade advantage, board mana
advantage. GA-evolved weights: `[1.05, 0.21, 9.66, 0.94, 5.22]` — note
**hand advantage weighted ~10x minion advantage**; card/resource advantage
dominated board presence. Heuristic 2 (the simulator's richer built-in one,
adds hero health etc.) consistently beat Heuristic 1, with the gap *growing
with search depth* — richer eval compounds with more search.

**Integration recipe (all components individually ablated, 250 games each):**
1. **Progressive bias** in UCT: `argmax Q/N + c·sqrt(2lnN(v)/N(w)) + H(w)/(1+N(w))`.
2. **Heuristic-guided rollouts via tournament selection:** at each rollout
   step sample k of the legal actions, take the best by eval. Best values:
   k=75% of actions on own nodes, k=50% on opponent nodes. Both extremes
   hurt — k=0 (pure random rollout) and k=100% (fully greedy rollout) were
   each worse than the middle.
3. **Max-child** action return (most wins) beat robust/secure-child.
4. **Tree reuse** across moves helped clearly.
5. Opponent hidden hand handled by matching played cards against a **deck
   database** (= exactly our Stage 3 belief model + archetype library).

**Headline result:** identical MCTS budget, vanilla ≈21% vs the SOTA
alpha-beta baseline; with the eval integrated ≈42% — **the eval function
doubled search strength with a 5-feature linear formula.**

---

## Source 3 — AAIA'17 Data Mining Challenge, "Helping AI to Play Hearthstone" (arXiv:1708.00730)

Task: predict the game winner from 2,000,000 mid-game Hearthstone states
(AUC metric), 188 teams.

- **Winner AUC 0.802; a plain 2-layer NN baseline scored 0.785** — the top
  solutions beat a simple model by <2pp. State-value prediction in a CCG has
  a hard ceiling well below certainty, and simple models get most of the way.
- Models trained on random-play states transferred to MCTS-player states
  with only ≈0.79→≈0.75 AUC drop — features generalize across play styles.
- Feature engineering (domain knowledge) was still what separated the top
  solutions; winners used NNs/XGBoost + ensembles.

**Calibration for us:** Φ v2's 0.604 ALL sign-accuracy vs a 0.80-AUC-class
ceiling suggests real headroom, but "perfect assessment" does not exist —
the goal is a *better* leaf signal, not an oracle. Santos shows even a
modest eval transforms search performance.

---

## Source 4 — LoCM contest post-mortems & related

- Top LoCM contest players used depth-limited minimax/MCTS **with a
  heuristic cutoff** — evaluate the eval function at a shallow depth instead
  of rolling out to terminal states. Different top players disagreed on
  which features mattered — feature usefulness is game/deck-specific and
  must be measured, not assumed.
- COG'20 LoCM winner: static card weights via harmony search + MCTS with
  opponent prediction (same shape as our belief model consumer).
- Broader Pokemon-AI literature (VGC/showdown agents) leans on TD learning
  where handcrafted eval was judged too hard — but those are 6v6 stat games;
  the CCG literature above is the closer match to PTCG.

---

## Synthesis → Φ v3 Design Directions

1. **Form:** linear, difference-form features (`mine − theirs`), global
   part + per-Pokemon-in-play part, exactly the standard decomposition.
   Nonlinearity only later, seeded from the tuned linear solution.
2. **Feature classes to encode (mapping the literature to PTCG):**
   - *Win-progress:* prize differential; **turns-to-lethal for each side**
     (opponent expected damage next turn vs our active HP, and our KO
     threshold `ceil(opp_hp/20)` vs our hand size) — the user-proposed
     lethality features; the literature's "trade advantage" analog.
   - *Resource advantage:* hand size diff (for us, hand IS damage), deck
     size / deck-out clock (stype==9 losses are a known failure mode),
     cards-drawn-next-turn effects.
   - *Board development:* energy on attackers vs their retreat/attack costs,
     evolution-line progress (Abra→Kadabra→Alakazam stage counts), bench
     size / fodder count, "armed attacker" counts both sides (the
     `_opp_threatening` proxy, generalized and graded).
   - *Blockers:* Mist/Rock wall on their active (hard gate on our damage),
     our access to Boss's Orders-style outs.
3. **Tuning, two cheap complementary gates already in the rig:**
   - *Supervised:* fit/tune weights against real ladder replay outcomes,
     gated by `dmc_replay_gate.py` sign-accuracy vs Φ v2's 0.604/0.696
     (AAIA'17 says simple models get close to the ceiling — a linear fit
     is a legitimate contender, not a toy).
   - *Simulation-based:* GA/CEM over weights with fitness = win-rate vs the
     **strong diverse anchor pool** (lucario/abomasnow/etc.), never vs weak
     or mirror-only opponents — this directly encodes both the paper's
     fitness-opponent finding and our own v29b/c mirror-blindness lesson.
4. **Deployment order (each step separately gated):** (a) Φ v3 standalone on
   the replay gate; (b) as the leaf/cutoff eval + rollout action-scorer in
   the existing endgame search skeleton (progressive bias + tournament-k
   rollouts per Santos), gated vs aggro anchors at n=400 before any mirror
   read is trusted; (c) only then consider a net trained on top of the
   features.

---

## Sources

- [Miernik & Kowalski, Evolving Evaluation Functions for Collectible Card Game AI](https://arxiv.org/abs/2105.01115)
- [Santos, Santos & Melo, Monte Carlo Tree Search Experiments in Hearthstone](https://ieeexplore.ieee.org/document/8080446/) ([PDF](https://fenix.tecnico.ulisboa.pt/downloadFile/1970719973966524/paper.pdf))
- [Janusz, Tajmajer & Świechowski, Helping AI to Play Hearthstone: AAIA'17 Data Mining Challenge](https://arxiv.org/abs/1708.00730)
- [Świechowski, Tajmajer & Janusz, Improving Hearthstone AI by Combining MCTS and Supervised Learning](https://arxiv.org/abs/1808.04794)
- [LoCM CC05 Feedback & Strategies thread](https://www.codingame.com/forum/t/legends-of-code-magic-cc05-feedback-strategies/50996)
- [VGC-Bench: Competitive Pokemon benchmark](https://arxiv.org/html/2506.10326v3)
