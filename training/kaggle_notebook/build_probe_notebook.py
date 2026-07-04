"""Builds mcts-probe.ipynb — two cheap pre-build probes for the MCTS-at-inference
fork (per advisor guidance 2026-07-04, before committing to building the tree):

1. Branching semantics: does search_step(parent_id, [actionA]) and
   search_step(parent_id, [actionB]) from the SAME parent_id produce two
   independent child search states (both steppable, non-interfering), or is
   search_step destructive/linear on its search_id? This determines whether
   MCTS can share a single search_begin root across all children of a node
   (cheap) or needs a fresh search_begin per rollout (expensive).
2. Timing: search_begin and search_step latency, extrapolated to a per-decision
   sim budget under the 10-minute/match clock (~150 decisions/game observed).

Run from repo root: python training/kaggle_notebook/build_probe_notebook.py
"""
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO_ROOT, "training", "kaggle_notebook", "mcts-probe.ipynb")


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def writefile_cell(repo_path, out_name=None):
    with open(os.path.join(REPO_ROOT, repo_path), encoding="utf-8") as f:
        content = f.read()
    return code(f"%%writefile {out_name or os.path.basename(repo_path)}\n" + content)


cells = [
    md("# MCTS Probe — branching semantics + inference-time budget\n\n"
       "Two cheap checks before committing engineering to `training/nn/mcts.py`, "
       "per advisor guidance: (1) can one `search_begin` root be branched into "
       "multiple independent children via repeated `search_step` calls from the "
       "same parent search_id, and (2) does search fit inside the 10-minute "
       "match clock at a useful sims/decision count. "
       "**Accelerator: CPU. Attach dataset `kiyotah/cg-lib`.**\n"),
    writefile_cell("main.py"),
    md("## Step 1 — real game, pick a mid-game MAIN decision with 3+ options\n"
       "(need branching room — a decision with only 1-2 legal options can't "
       "test independent children)"),
    code(
        "import kaggle_environments\n"
        "from kaggle_environments import make\n"
        "import main\n\n"
        "DECK = main.DECK\n"
        "env = make('cabt', configuration={'decks': [DECK, DECK]})\n"
        "env.run([main.agent, main.agent])\n"
        "steps = env.steps\n"
        "print('total steps:', len(steps))\n\n"
        "def extract_decisions(steps, seat):\n"
        "    out = []\n"
        "    for i in range(1, len(steps)):\n"
        "        prev = steps[i - 1][seat].get('observation', {})\n"
        "        act = steps[i][seat].get('action')\n"
        "        sel = prev.get('select')\n"
        "        if sel is None or act is None:\n"
        "            continue\n"
        "        if not sel.get('option'):\n"
        "            continue\n"
        "        out.append({'obs': prev, 'action': act})\n"
        "    return out\n\n"
        "decs = extract_decisions(steps, 0)\n"
        "target = None\n"
        "for d in decs:\n"
        "    sel = d['obs']['select']\n"
        "    cur = d['obs'].get('current') or {}\n"
        "    if sel.get('type') == 0 and len(sel.get('option', [])) >= 3 and (cur.get('turn') or 0) > 5:\n"
        "        target = d\n"
        "        break\n"
        "if target is None:\n"
        "    target = max(decs, key=lambda d: len(d['obs']['select'].get('option', [])))\n"
        "n_opts = len(target['obs']['select']['option'])\n"
        "print('picked decision at turn', (target['obs'].get('current') or {}).get('turn'), 'n_opts', n_opts)\n"
    ),
    md("## Step 2 — search_begin, then branch into all children from the SAME parent"),
    code(
        "import sys, glob, time\n"
        "for _pat in ['/kaggle/input/**/cg-lib', '/kaggle/input/cg-lib']:\n"
        "    _paths = glob.glob(_pat, recursive=True)\n"
        "    if _paths:\n"
        "        sys.path.insert(0, _paths[0])\n"
        "        break\n\n"
        "from cg.api import to_observation_class, search_begin, search_step, search_end, search_release\n\n"
        "obs_dict = target['obs']\n"
        "obs = to_observation_class(obs_dict)\n"
        "state = obs.current\n"
        "me = state.yourIndex\n"
        "my_p = state.players[me]\n"
        "opp_p = state.players[1 - me]\n\n"
        "def filler(n, pool=DECK):\n"
        "    return [pool[i % len(pool)] for i in range(n)]\n\n"
        "your_deck = filler(my_p.deckCount)\n"
        "your_prize = filler(len(my_p.prize))\n"
        "opponent_deck = filler(opp_p.deckCount)\n"
        "opponent_prize = filler(len(opp_p.prize))\n"
        "opponent_hand = filler(opp_p.handCount)\n"
        "opponent_active = []\n\n"
        "t0 = time.perf_counter()\n"
        "root = search_begin(obs, your_deck, your_prize, opponent_deck,\n"
        "                     opponent_prize, opponent_hand, opponent_active,\n"
        "                     manual_coin=True)\n"
        "t_begin = time.perf_counter() - t0\n"
        "print('search_begin took', t_begin, 's. searchId=', root.searchId)\n"
        "print('root n_opts:', len(root.observation.select.option))\n\n"
        "children = []\n"
        "for i in range(min(3, n_opts)):\n"
        "    t0 = time.perf_counter()\n"
        "    child = search_step(root.searchId, [i])\n"
        "    dt = time.perf_counter() - t0\n"
        "    children.append((i, child, dt))\n"
        "    sel = child.observation.select\n"
        "    print(f'child from action {i}: searchId={child.searchId} dt={dt:.4f}s '\n"
        "          f'stype={sel.type if sel else None} n_opts={len(sel.option) if sel else 0}')\n\n"
        "ids = [c[1].searchId for c in children]\n"
        "print('child searchIds distinct from root and each other:', len(set(ids + [root.searchId])) == len(ids) + 1)\n\n"
        "print('re-stepping child 0 again to confirm root/child0 unaffected by child1/child2 branching:')\n"
        "sel0 = children[0][1].observation.select\n"
        "if sel0 and len(sel0.option) > 0:\n"
        "    again = search_step(children[0][1].searchId, [0])\n"
        "    print('  ok, searchId=', again.searchId, 'result=', again.observation.current.result)\n"
    ),
    md("## Step 3 — timing: search_step latency over many calls (naive argmax-0 rollout)\n"
       "Extrapolates sims/decision achievable under the 10-min/match clock "
       "(~150 decisions/game observed elsewhere in this repo)."),
    code(
        "import time\n\n"
        "N_ROLLOUTS = 20\n"
        "MAX_DEPTH = 15\n"
        "step_times = []\n"
        "terminal_hits = 0\n"
        "for r in range(N_ROLLOUTS):\n"
        "    cur = search_step(root.searchId, [0]) if r == 0 else None\n"
        "    # fresh branch per rollout from root to avoid exhausting one child's tree\n"
        "    cur = search_step(root.searchId, [r % n_opts])\n"
        "    for d in range(MAX_DEPTH):\n"
        "        sel = cur.observation.select\n"
        "        if sel is None or len(sel.option) == 0:\n"
        "            terminal_hits += 1\n"
        "            break\n"
        "        t0 = time.perf_counter()\n"
        "        cur = search_step(cur.searchId, [0])\n"
        "        step_times.append(time.perf_counter() - t0)\n\n"
        "import statistics\n"
        "print('total search_step calls:', len(step_times))\n"
        "print('mean step latency (s):', statistics.mean(step_times))\n"
        "print('p95 step latency (s):', sorted(step_times)[int(0.95 * len(step_times))])\n"
        "print('terminal/max-depth hits:', terminal_hits, '/', N_ROLLOUTS)\n\n"
        "mean_step = statistics.mean(step_times)\n"
        "avg_depth = len(step_times) / max(1, N_ROLLOUTS)\n"
        "sim_cost = mean_step * avg_depth\n"
        "print(f'avg depth per rollout: {avg_depth:.1f}, avg cost per rollout: {sim_cost:.4f}s')\n\n"
        "DECISIONS_PER_GAME = 150\n"
        "CLOCK_SECONDS = 600\n"
        "budget_per_decision = CLOCK_SECONDS / DECISIONS_PER_GAME\n"
        "print(f'budget per decision under 10-min clock / {DECISIONS_PER_GAME} decisions: {budget_per_decision:.3f}s')\n"
        "print(f'=> sims/decision affordable at depth {avg_depth:.0f}: {budget_per_decision / sim_cost:.1f}')\n"
        "print('(net policy/value inference cost NOT included above — add separately from local net_agent.py timing)')\n"
    ),
    code(
        "search_end()\n"
        "print('search_end() called')\n"
    ),
    md("## Findings (fill in after running, then paste into docs/engine-api.md)\n\n"
       "- Are child searchIds distinct and independently steppable (branching confirmed)?\n"
       "- Mean/p95 `search_step` latency.\n"
       "- Sims/decision affordable under the 10-min clock at the observed depth.\n"
       "- Decision: does inference-time search fit? If yes, prioritize wiring "
       "`prior_blend.py` + a heuristic-dominant prior into a real tree over "
       "`mcts_collect.py`/training-loop MCTS.\n"),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {OUT}")
