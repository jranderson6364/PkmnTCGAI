# ML/RL Method Survey — 120 Methods, Pros/Cons, Project Fit

*Systematic catalog of machine-learning and reinforcement-learning methods
evaluated against this project's constraints, with a shortlist and proposed
alterations. Commissioned 2026-07-07 alongside the Φ-shaping failure autopsy.*

**Last updated:** 2026-07-07

---

## Project Constraints (the lens every method is judged through)

| Constraint | Consequence |
|---|---|
| 10-min match clock, timeout = loss | Inference must be bounded; heavy per-move compute only where gated |
| CPU-mostly compute, free Kaggle T4 quota, "reasonableness standard" | No large-scale distributed RL; sample-efficient methods favored |
| Imperfect information (hidden hand/deck/prizes), stochastic draws | Perfect-info methods need determinization or belief handling |
| ~150 decisions/game, sparse ±1 terminal reward | Credit assignment is the core learning difficulty |
| Strong scripted teacher (v29c heuristic, ~880 ladder Elo) | Imitation asymptotes to parity — CONFIRMED 3x (BC, DAgger, AWR) |
| Self-play data from own checkpoint carries no new signal | CONFIRMED (Phase 2 rounds 2–3, IQL r1) — needs EXTERNAL information |
| External sources available: 1,400+ real ladder replays, real archetype bots, engine clone, endgame terminals | Methods that consume these are structurally favored |
| Confirmed positives: endgame-gated rollout search (59.0%±4.8% n=400); n-step targets; belief classifier | Methods that compose with these are favored |
| Confirmed negatives: full-game search (5 configs), Φ-in-target, oracle-critic w/ fixed dropout, mirror self-play | Methods repeating these mechanisms need a named fix for the named failure |

Fit key: **A** = shortlisted; **B** = viable with modification / conditional;
**C** = wrong regime or dominated by an A/B method; **D** = already tried and
closed here (kept for completeness), or infeasible under constraints.

---

## Family 1 — Imitation & Learning from Demonstration

| # | Method | What it is | Pros | Cons | Fit |
|---|--------|-----------|------|------|-----|
| 1 | Behavioral Cloning (BC) | Supervised s→a on teacher data | Cheap, stable, no env interaction | Compounding error off-distribution; ceiling = teacher | D (done; 17% vs teacher) |
| 2 | DAgger | Iteratively relabel learner's own states with teacher | Fixes distribution shift directly | Still capped at teacher parity; needs cheap teacher queries | D (done; fidelity 82%, win-rate flat) |
| 3 | Ensemble/HG-DAgger | DAgger w/ uncertainty-gated teacher queries | Fewer queries; safer rollouts | Same parity ceiling; ensemble cost | C (teacher queries are free here — no gain) |
| 4 | GAIL | Adversarial imitation via discriminator reward | Learns from states w/o action labels; can generalize past demos | Notoriously unstable; discriminator hacking; needs online RL loop | C (instability + we HAVE action labels) |
| 5 | AIRL | GAIL variant recovering a transferable reward | Reward transferable to new dynamics | Same instability; transfer irrelevant (fixed game) | C |
| 6 | IQ-Learn | Imitation as inverse soft-Q learning | Single Q function, no adversarial loop; strong sample efficiency | Still bounded by demonstrator; soft-Q assumptions | C (parity ceiling again) |
| 7 | Implicit BC (energy-based) | s,a energy model, argmin at inference | Handles multimodal teacher policies | Slow inference (sampling/optimization per move) — clock risk | C |
| 8 | SQIL | Imitation via RL with r=+1 on demo transitions, 0 else | Dead simple; off-policy | Crude reward; parity ceiling | C |
| 9 | MARWIL / exponentially-weighted imitation | BC weighted by advantage estimates from logged data | Can exceed demonstrator if advantages are real | Advantage quality is the whole game — our value heads are weak | B (only with the replay-value net's advantages) |
| 10 | Filtered BC (%BC / best-trajectory cloning) | Clone only top-k% return trajectories | Trivial; robust; beats naive BC on mixed data | Discards most data; return-conditioning coarse at game level | B (cheap ablation row on real replays: clone only WINNERS' lines vs all) |
| 11 | DART (noise-injected teacher) | Perturb teacher during collection to widen support | Cheap robustness to covariate shift | Teacher parity ceiling unchanged | C |
| 12 | Policy distillation | Compress/ensemble teachers into one net | Merges multiple experts; smaller/faster student | Needs multiple teachers worth merging | B (distill SEARCH choices, not the heuristic — see shortlist) |

---

## Family 2 — Offline RL (learning from fixed datasets)

| # | Method | What it is | Pros | Cons | Fit |
|---|--------|-----------|------|------|-----|
| 13 | CQL | Q-learning with conservative penalty on OOD actions | Strong offline benchmark record; prevents value overestimation | Penalty tuning brittle; pessimism can collapse to BC | B |
| 14 | IQL | Expectile-based value learning, no OOD action queries | Stable; no bootstrapping from unseen actions; simple | Expectile τ sensitive; still needs decent data coverage | D→B (r1 anti-predictive = label-imbalance artifact; r2 pre-registered on balanced/replay data) |
| 15 | AWAC | Advantage-weighted actor-critic warm-started from offline data | Designed exactly for offline→online finetune | Online phase = self-play here, which is our dead end | C |
| 16 | TD3+BC | TD3 + BC regularizer, minimal offline RL | Simplest competitive offline method | Continuous-action design; needs porting to discrete | C (discrete port ≈ CQL/IQL anyway) |
| 17 | BCQ | Restrict Q backup to actions a VAE says are in-data | Early, well-understood | Generative model overhead; superseded by IQL/CQL | C |
| 18 | EDAC / SAC-N | Ensemble Q pessimism via min over N critics | Strong results; uncertainty for free | N× training cost; continuous-action origins | C |
| 19 | BEAR | MMD-constrained policy near data support | Principled support constraint | Fiddly; superseded | C |
| 20 | Onestep RL | Single policy-improvement step from behavior Q, no iteration | Avoids iterative error amplification — matches our "one step past teacher" need | Only one step of improvement | **A** (see shortlist: one-step improvement over the heuristic using replay-fit Q) |
| 21 | Decision Transformer | Return-conditioned sequence model a~p(a\|R,s,...) | No bootstrapping/TD at all; exploits transformer context; Metamon precedent in Pokémon | Needs return diversity; struggles to exceed best-in-data; GPU training | B (T4-feasible; real replays have both sides = wins AND losses per game) |
| 22 | Trajectory Transformer | Model full (s,a,r) sequences + beam search plan | Planning at inference | Beam search per move = clock risk; heavy | C |
| 23 | RvS (reward-conditioned supervised) | MLP conditioned on outcome/goal, pure supervised | Simplest offline method that can exceed BC | Same exceed-best-in-data doubts as DT | B (cheap ablation vs DT) |
| 24 | COMBO | Model-based offline RL w/ conservative values on model rollouts | Data augmentation via model | We have the REAL engine — a learned model is strictly worse | C |
| 25 | MOPO / MOReL | Offline RL with uncertainty-penalized learned model | Same as above | Same — real clonable engine dominates | C |
| 26 | Fitted Q Evaluation (FQE) | Off-policy evaluation of a fixed policy from logged data | Cheap policy ranking without games; could pre-screen candidates | Evaluation only, not improvement; bias under distribution shift | B (useful RIG addition: rank candidates on replays before 400-game A/Bs) |

---

## Family 3 — Value-Based RL

| # | Method | What it is | Pros | Cons | Fit |
|---|--------|-----------|------|------|-----|
| 27 | DQN | Q-learning + replay + target net | Foundational, off-policy | Overestimation; sparse-reward struggles | C (subsumed) |
| 28 | Double DQN | Decoupled argmax/eval to cut overestimation bias | One-line fix, always worth having | Marginal alone | B (hygiene for any Q training here) |
| 29 | Dueling networks | Separate V(s) and A(s,a) streams | Better V learning when actions don't matter — most of our 150 decisions! | Architecture change only | B (cheap; matches "most decisions low-impact" diagnosis) |
| 30 | Prioritized Experience Replay | Sample high-TD-error transitions more | Focuses on informative states (endgames!) | Bias corrections fiddly | B |
| 31 | n-step returns | Bootstrap after n steps instead of terminal | CONFIRMED here: +0.02 ALL/+0.04 LATE over full-MC | Bootstrap quality bounds gains | **A** (already adopted; keep in every value-training recipe) |
| 32 | Rainbow | All DQN tricks combined | Strong baseline | Kitchen-sink; hard to attribute | C (violates our one-lever-at-a-time discipline) |
| 33 | C51 / categorical value | Predict distribution over discrete return atoms | Directly fixes "value head saturates near ±1" (our named AWR failure); richer gradient | More output dims; needs calibration | **A** (as the two-hot head below) |
| 34 | QR-DQN | Quantile regression over returns | Distribution w/o fixed atoms | Overkill for ±1 outcomes | C (C51/two-hot suffices for W/L) |
| 35 | IQN | Implicit quantile networks | Risk-sensitive policies possible | Complexity unneeded | C |
| 36 | R2D2 | Recurrent replay distributed DQN | Handles partial observability via LSTM memory | Distributed infra; burn-in complexity | C (belief features already summarize history more cheaply) |
| 37 | NGU / Agent57 | Episodic + lifelong curiosity exploration | Solves hard-exploration games | Massive compute; exploration is NOT our bottleneck | C |
| 38 | Retrace(λ) | Safe off-policy multi-step correction | Principled off-policy n-step from replay data | Importance weights need behavior probs (replays lack them) | C (heuristic teacher is deterministic — no probs) |
| 39 | Tree-backup / Q(λ) | Off-policy multi-step w/o importance sampling | No behavior probs needed | Cuts traces at off-policy actions — short traces on replay data | B (the correct n-step form for REPLAY-based value training) |
| 40 | Expected SARSA | Backup expectation over policy instead of max | Lower variance than Q-learning | On-policy flavor | C |
| 41 | Munchausen DQN | Add scaled log-policy bonus to reward | Implicit KL regularization, strong gains cheaply | Interacts with our ±1 reward scale | B (one-line trial in any DQN-style run) |
| 42 | Bootstrapped DQN | Ensemble heads for deep exploration | Uncertainty estimates | Exploration not the bottleneck | C |
| 43 | Deep Monte Carlo (DouZero-style) | Regress Q on full-episode returns, ε-greedy self-play | Proven in large-action card games (DouDizhu); simple; our current base | High variance targets; no bootstrapping; self-play signal ceiling (confirmed here) | D (current pipeline; superseded by n-step + external data) |
| 44 | TD(λ) / eligibility traces | Exponentially-weighted multi-step blend | Classic credit assignment; TD-Gammon pedigree | Online form awkward with replay-buffer training | B (offline λ-return on replay games ≈ generalizes our n=5 win) |

---

## Family 4 — Policy Gradient / Actor-Critic

| # | Method | What it is | Pros | Cons | Fit |
|---|--------|-----------|------|------|-----|
| 45 | REINFORCE | Monte-Carlo policy gradient | Trivial | Extreme variance on 150-step ±1 games | C |
| 46 | A2C/A3C | Parallel advantage actor-critic | Simple online AC | Needs many env workers; on-policy sample hunger | C |
| 47 | TRPO | Trust-region policy steps | Monotonic improvement theory | Second-order machinery; heavy | C |
| 48 | PPO | Clipped surrogate objective | Industry default, robust | On-policy sample hunger; self-play signal ceiling applies | C (dominated: PPO-from-scratch already rejected in gap analysis) |
| 49 | PPG | Phasic policy/value decoupling | Better value learning within PPO | Inherits PPO's cons | C |
| 50 | IMPALA / V-trace | Distributed off-policy AC with correction | Scales; V-trace is principled | Distributed infra we don't have | C |
| 51 | SAC | Max-entropy off-policy AC | Sample-efficient, stable | Continuous-action native; discrete-SAC exists but niche | C |
| 52 | TD3 | Twin critics + delayed policy | Strong continuous control | Wrong action space | C |
| 53 | DDPG | Deterministic PG | Historical | Brittle | C |
| 54 | MPO | EM-style KL-constrained policy improvement | Robust, off-policy, discrete-friendly | Complexity vs. AWR-family gains unclear | C |
| 55 | V-MPO | On-policy MPO variant | Strong in multi-task | On-policy hunger | C |
| 56 | AWR | Regress policy toward exp(A/β)-weighted actions | Simple, offline-friendly | Advantage quality gates everything | D (done; both β negative — root cause: saturated value head) |
| 57 | GAE | λ-blended advantage estimator | Variance/bias dial for any AC method | Component, not a method per se | B (use inside any future policy-improvement step) |
| 58 | ACER | Off-policy AC w/ Retrace + trust region | Sample-efficient AC | Complexity; superseded | C |

---

## Family 5 — Model-Based RL & Search

| # | Method | What it is | Pros | Cons | Fit |
|---|--------|-----------|------|------|-----|
| 59 | MCTS / UCT | Tree search, UCB at nodes, rollout leaf eval | Anytime; no training needed | Rollout quality gates everything — CONFIRMED (weak leaf signal killed 5 full-game configs) | D full-game / **A** endgame-gated (shipped, +9pp) |
| 60 | PUCT | MCTS with policy prior in selection | Priors focus narrow search | Visit collapse at low sims — CONFIRMED here | B (only with Gumbel fix, #64) |
| 61 | AlphaZero | Self-play MCTS + policy/value net, visit-count targets | The gold standard in perfect-info games | Needs huge compute; imperfect info unsound (strategy fusion); self-play ceiling confirmed here | D (Phase 2 rounds closed) |
| 62 | Expert Iteration (ExIt) | Search = expert, net imitates search, search uses net | The general improvement operator behind AZ | Only improves if search genuinely beats net — needs good leaf values | **A** (our endgame search IS a confirmed expert — distill it) |
| 63 | MuZero | AZ + learned dynamics model | No env needed at plan time | We have a fast clonable REAL engine — learned model strictly worse | C |
| 64 | Gumbel MuZero/AlphaZero | Sequential-halving root selection + Q-based policy targets | Guaranteed policy improvement at 2–16 sims — exactly our budget; fixes PUCT collapse | Still needs a trustworthy Q at root | **A** (drop into endgame searcher root) |
| 65 | Stochastic MuZero | MuZero + chance nodes | Handles stochasticity properly | Learned-model objection stands | C |
| 66 | EfficientZero | Sample-efficient MuZero (SPR + others) | Atari-100k SOTA | Same objection | C |
| 67 | Dreamer v3 | Latent world model + imagination training | General, robust defaults | Latent model of a symbolic card game = wasted effort vs real engine | C |
| 68 | MBPO | Short model rollouts augment real data | Sample efficiency | Real engine already free/fast | C |
| 69 | PETS / CEM-MPC | Ensemble model + cross-entropy-method planning | Strong low-dim control | Wrong domain shape | C |
| 70 | Minimax / alpha-beta | Exact adversarial tree search | Optimal in small perfect-info endgames | Needs determinization here; branching too big mid-game | B (exact solver for FINAL 2-3 turns inside determinizations?) |
| 71 | Expectimax / *-minimax | Chance-node search | Correct for stochastic draws | Explodes fast; needs sampling anyway | C |
| 72 | Rollout policy iteration (Tesauro) | One-step lookahead + base-policy rollouts = improved policy | PROVEN here (endgame agent is exactly this); theory guarantees improvement over base policy | Improvement modest; rollout cost | **A** (this IS the working method — extend its region) |
| 73 | TD-Gammon-style TD(λ) self-play | Incremental value learning through self-play games | Historic proof for stochastic games | Self-play-only signal — our confirmed dead end | C |

---

## Family 6 — Imperfect-Information Methods

| # | Method | What it is | Pros | Cons | Fit |
|---|--------|-----------|------|------|-----|
| 74 | PIMC (determinize + perfect-info search) | Sample hidden worlds, search each, vote | Simple; parallel | Strategy fusion + non-locality errors; CONFIRMED negative full-game here | D full-game / A inside endgame gate (shipped form) |
| 75 | ISMCTS | Single tree over info-sets, fresh determinization per sim | Reduces strategy fusion vs PIMC | Leaf signal still gates it — CONFIRMED (0W-50L, pinned on leaf values) | B (reopen ONLY with a real leaf value net) |
| 76 | POMCP | Particle-filter belief + UCT in POMDP | Principled belief tracking | Particle collapse on long games; our belief model already approximates this cheaper | C |
| 77 | DESPOT | Determinized sparse POMDP tree w/ bounds | Stronger guarantees than POMCP | Same regime; complexity | C |
| 78 | CFR | Regret matching over info-sets → Nash | Game-theoretic soundness | State space astronomically beyond tabular reach | C |
| 79 | CFR+ / Discounted CFR | Faster CFR variants | Better constants | Same tabular objection | C |
| 80 | MCCFR | Sampled CFR traversals | Scales further | Still needs abstraction engineering for this game | C |
| 81 | Deep CFR | Neural regret/strategy approximation | No hand abstraction | Compute-heavy; rejected in gap analysis (stands) | C |
| 82 | DREAM | Model-free deep regret w/ baselines | Cheaper than Deep CFR | Still a research-grade lift | C |
| 83 | NFSP | Mix best-response net + average-policy net | First deep self-play→Nash method | Slow convergence; exploitability focus wrong for a ladder of non-Nash bots | C |
| 84 | PSRO / Double Oracle | Iteratively add best responses to a meta-game pool | Directly targets a POPULATION (the ladder is one!) | Each BR is a full training run | B (lite version: rule-based BRs to observed archetypes — partially built already) |
| 85 | R-NaD (DeepNash) | Regularized Nash dynamics | Stratego-scale success | DeepMind-scale compute; rejected (stands) | C |
| 86 | ReBeL | Belief-state value nets + depth-limited subgame solving | Sound imperfect-info search | Public-belief-state machinery heavy; poker-shaped | B (endgame-only lite variant conceivable) |
| 87 | Player of Games | Unified perfect+imperfect search (GT-CFR) | Generality | Same scale objection | C |
| 88 | Depth-limited subgame solving (Pluribus) | Re-solve reached subgame with value abstractions at leaves | Superhuman poker on ONE workstation — the compute story fits | Needs blueprint strategy + leaf value sets; engineering-heavy | B (the principled upgrade path for the endgame searcher if it plateaus) |
| 89 | Suphx-style oracle distillation w/ ANNEALED dropout | Train w/ hidden info, anneal it away 0→1 | Documented fix for exactly our oracle-critic failure (fixed dropout fails, anneal works); infra exists | Anneal schedule = new hyperparameter; Mahjong≠TCG transfer risk | B (pre-approved in five-family plan; run only if slack) |
| 90 | Supervised belief model (ours) | Classify archetype / predict hidden cards from observations | CONFIRMED (92.3% held-out; drives determinizer + wall anticipation) | Recognition ceiling 78.7% on long-tail decks | A (keep feeding consumers) |

---

## Family 7 — Self-Play, Population & League Methods

| # | Method | What it is | Pros | Cons | Fit |
|---|--------|-----------|------|------|-----|
| 91 | Fictitious self-play (XFP) | Best-respond to opponents' AVERAGE strategy | Convergence properties in zero-sum | Averaging machinery; slow | C |
| 92 | Prioritized fictitious self-play (PFSP) | Sample sparring partners ∝ difficulty | Focuses compute on beatable-but-hard opponents | Needs a real pool first | B (already planned Stage 4; pool now exists via opponent_pool.py) |
| 93 | AlphaStar league | Main agents + main-exploiters + league-exploiters | Robustness to strategy cycles | Team-of-agents compute scale | C (full form) |
| 94 | Exploiter agents | Train agents specifically to beat the current champion | CONFIRMED value here (exploiter mining → v27 board-thinning fix) | Each exploiter = compute; weak exploiters only probe shallow flaws | **A** (as a permanent flaw-mining rig, not a training loop) |
| 95 | MAP-Elites / Quality-Diversity | Archive of diverse-behavior elites | Finds varied strategies/decks | Behavior descriptors unclear here; compute | C |
| 96 | POET | Co-evolve agents + environments | Open-ended discovery | Environment is fixed by the competition | C |
| 97 | PBT (population-based training) | Evolve hyperparams during training across a population | Free hyperparam tuning | Needs population-scale compute | C |

---

## Family 8 — Credit Assignment & Reward Design

| # | Method | What it is | Pros | Cons | Fit |
|---|--------|-----------|------|------|-----|
| 98 | Potential-based reward shaping (Ng et al.) | Add F=γΦ(s')−Φ(s) to rewards | Policy-invariant WITHIN states; dense signal | Shaped VALUES not sign-comparable ACROSS states — CONFIRMED failure here (see autopsy §below) | D as-implemented / **A** as residual baseline (see shortlist) |
| 99 | RUDDER | LSTM redistributes terminal reward to key steps | Directly attacks sparse-terminal credit assignment | Notoriously hard to tune; contribution analysis brittle | C |
| 100 | Hindsight Experience Replay | Relabel failures as successes for alternate goals | Great for goal-reaching | No natural goal relabeling in W/L games | C |
| 101 | Successor features | Decompose value into features × reward weights | Transfer across reward functions | Single fixed reward here | C |
| 102 | Auxiliary tasks (UNREAL/GVFs) | Predict extra signals (pixel change, next state, etc.) | Better representations from same data | Task choice matters | B (predict opponent's next revealed card / prize race as aux heads) |
| 103 | Two-hot / HL-Gauss value targets | Regress value as categorical two-hot distribution | Fixes saturation + gradient pathologies of MSE on ±1; MuZero/Dreamer standard | Minor implementation lift | **A** (pre-approved; apply to replay-value net) |
| 104 | Reward clipping/normalization schemes | Standardize return scale | Stability | Trivial here (±1 already) | C |

---

## Family 9 — Exploration

| # | Method | What it is | Pros | Cons | Fit |
|---|--------|-----------|------|------|-----|
| 105 | ε-greedy / Boltzmann | Random/temperature action noise | Trivial | Undirected | B (already used in collection; temperature=1.0 finding stands) |
| 106 | UCB / Thompson sampling | Optimism / posterior sampling bandits | Principled per-state exploration | Bandit regime, not deep-RL bottleneck here | C |
| 107 | RND | Novelty via random-net prediction error | Simple, effective curiosity | Exploration is not our binding constraint | C |
| 108 | ICM | Curiosity from forward-model error | Same | Same; noisy-TV issues | C |
| 109 | Noisy Nets | Learned parametric exploration | Replaces ε cleanly | Marginal here | C |

---

## Family 10 — Representation, Sequence & Auxiliary Learning

| # | Method | What it is | Pros | Cons | Fit |
|---|--------|-----------|------|------|-----|
| 110 | SPR / CURL (self-supervised RL reps) | Contrastive/predictive state representations | Sample efficiency on pixels | Our state is already symbolic/structured | C |
| 111 | Data augmentation (RAD-style) | Augment states during training | Cheap regularizer | Card-game symmetries limited (bench permutation IS one) | B (bench-permutation augmentation of replay corpus — free data ×k) |
| 112 | Transformer policy over game history | Full-history sequence model instead of state MLP | Implicit belief tracking; Metamon used this | GPU training; inference latency on CPU | B (bounded: encoder for the VALUE net only) |
| 113 | Hand/deck prediction auxiliary head | Supervised head predicting hidden zones | Direct belief signal; labels free in replays (outcomes reveal cards) | Partial labels only (unrevealed cards stay unknown) | **A** (aux head on replay-value net; belief model already proves signal exists) |
| 114 | Set-/permutation-invariant encoders (Deep Sets) | Encode hand/bench as sets not vectors | Correct inductive bias; fewer params | Rewrite of encode.py | B (pairs with #111) |
| 115 | LLM-as-policy | Prompt an LLM to play | Zero training | Latency/clock fatal; rejected (stands) | C |
| 116 | Preference-based RL (RLHF-style) | Learn reward from trajectory comparisons | No hand-crafted reward needed | We HAVE the true reward (W/L) | C |

---

## Family 11 — Evolutionary & Direct Search

| # | Method | What it is | Pros | Cons | Fit |
|---|--------|-----------|------|------|-----|
| 117 | OpenAI-ES | Perturb params, estimate gradient from returns | Embarrassingly parallel; no backprop | Thousands of episodes per step | C |
| 118 | CMA-ES | Covariance-adapted evolution strategy | Best-in-class for ≤100-dim search | Game returns are noisy fitness | B (better optimizer for the HEURISTIC's ~dozens of weights than the 2026-07-02 SPSA run that failed to clear its gate) |
| 119 | NEAT / neuroevolution | Evolve network topology+weights | No gradient plumbing | Sample-hungry; dominated | C |
| 120 | Genetic programming over rules | Evolve symbolic policy rules | Interpretable output; report-friendly | Search space huge; crossover on code brittle | C |

---

## Shortlist (A-rated) and Proposed Alterations

Chosen for: consuming an EXTERNAL information source (the confirmed
requirement), composing with the one confirmed positive (endgame-gated
rollout search), bounded compute, and clock safety. Each alteration names
the con it negates.

| Pick | Core method(s) | The con | The alteration that negates it |
|------|----------------|---------|-------------------------------|
| **1. Residual value learning on Φ** | #98 + #33/#103 | Shaped targets sign-incomparable across states (the autopsy finding) | Keep `outcome − Φ(s)` as the REGRESSION target but define the deployed value as `V(s) = Φ(s) + resid(s)` — adding Φ back restores cross-state comparability exactly, while the net only has to learn the (easier, lower-variance) residual. Use two-hot head to avoid ±1 saturation. Gate: same replay sign-acc protocol; Φ+resid must beat both Φ v2 (0.624) and replay_value epoch-0 (0.635). |
| **2. Replay-value net as leaf evaluator** | #14/#39/#44 on real replays | Rollout leaf values are noise outside the endgame (ISMCTS closure); replay data lacks behavior probs for off-policy corrections | Train V on the 1,400-replay corpus with tree-backup/λ-returns (no importance weights needed), two-hot head, bench-permutation augmentation (#111), and hand-prediction aux head (#113). Consume it as the leaf eval that extends the search gate EARLIER than rollouts can reach (≤3–4 prizes, mid-game). External data source ⇒ the self-play ceiling argument doesn't apply. |
| **3. Gumbel root selection in the endgame searcher** | #64 | Gumbel assumes a trustworthy root Q; ours are rollout-estimated and noisy | Use sequential halving over determinization-AVERAGED Q (each candidate action evaluated across the same belief-sampled worlds — variance-matched comparisons), so halving eliminates on paired samples rather than raw noisy Q. Guarantees the sim budget concentrates instead of PUCT-collapsing. |
| **4. Expert iteration where the EXPERT is the shipped endgame searcher** | #62 + #12 | ExIt only works if the expert beats the base policy — previously false, now TRUE (59%±4.8%) | Distill searcher decisions (endgame states only) into (a) heuristic rule candidates for the report's error-analysis loop and (b) a small endgame policy net; the distilled policy then serves as the ROLLOUT policy inside the searcher — a genuine closed improvement loop whose signal originates from search-reached real terminals, not from the net's own priors. |
| **5. One-step policy improvement over the heuristic** | #20 + #9 | Iterated offline RL amplifies value errors; full RL self-play is a confirmed dead end | Do exactly ONE improvement step: fit Q̂ on replays (pick 2's net), then at inference act greedily w.r.t. Q̂ ONLY where it disagrees with the heuristic AND the advantage gap clears a confidence threshold (τ-quantile from FQE #26 calibration). Everything else falls through to the heuristic — bounded downside, clock-safe, measurable via the standard 400-game A/B. |
| **6. Exploiter rig as a permanent flaw-miner** | #94 + #84-lite | Weak exploiters only find shallow flaws; strong ones cost compute | Keep exploiters CHEAP and rule-based/archetype-based (opponent_pool.py already exists), but rotate the pool per the observed ladder meta shares; treat every exploiter win-cluster as a replay-mining lead (the v27 workflow, systematized) rather than as training data. |

Explicitly deprioritized despite pre-approval: Gumbel EXPERT ITERATION on
self-play value targets (#61/#64 as a training loop) — until picks 1–2
produce a value net that beats Φ v2 on real replays, any self-play training
loop re-enters the closed "no external signal" regime. Suphx annealed
oracle (#89) stays slack-only, per the five-family plan.

---

## Cross-References

- Φ-shaping failure autopsy: `docs/report-log.md` 2026-07-05 "Phase 0
  ablation grid" entry + `training/nn/dmc_nstep.py` docstring.
- Five-family gap analysis this survey extends: `docs/report-log.md`
  2026-07-07 "Training-methods gap analysis" entry.
- Closed-line evidence: `docs/nn-training.md` §AlphaZero-Style Push →
  Resume Here.
