"""Builds mcts-spike.ipynb — the Part 3a SearchState confirmation spike
(see the approved plan at C:\\Users\\jande\\.claude\\plans\\when-do-we-start-eager-mountain.md).

Runs one real game locally (kaggle_environments' bundled engine, same as the
rest of this project), pulls a mid-game decision, then calls the Kaggle-only
cg.api.search_begin/search_step/search_end for the first time to confirm (not
guess) how it behaves in practice — the signature and SearchState shape are
already known from reading cg-lib's actual source earlier this session; what's
unconfirmed is runtime behavior against a real state.

Run from repo root: python training/kaggle_notebook/build_spike_notebook.py
"""
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO_ROOT, "training", "kaggle_notebook", "mcts-spike.ipynb")


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
    md("# MCTS Part 3a — SearchState Confirmation Spike\n\n"
       "See `docs/engine-api.md` and the approved plan "
       "(`when-do-we-start-eager-mountain.md`). `cg.api.search_begin`'s "
       "signature and `SearchState`'s shape are already known from reading "
       "cg-lib's actual source this session — this notebook confirms the "
       "**runtime behavior** against a real mid-game decision, which was not "
       "yet verified. **Accelerator: CPU (no GPU needed — pure engine "
       "mechanics, no torch).** Attach dataset `kiyotah/cg-lib`.\n"),
    writefile_cell("main.py"),
    md("## Step 1 — run one real game with the current heuristic (v22), no cg-lib yet\n\n"
       "Deliberately done BEFORE touching cg-lib's `cg` package import, so "
       "`kaggle_environments`' own bundled engine (used here to generate a "
       "real mid-game decision) can't collide with the cg-lib copy we import "
       "later."),
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
        "    # same logic as training/bc_collect.py's extract_decisions\n"
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
        "print('decisions for seat 0:', len(decs))\n\n"
        "target = None\n"
        "for d in decs:\n"
        "    sel = d['obs']['select']\n"
        "    cur = d['obs'].get('current') or {}\n"
        "    if sel.get('type') == 0 and (cur.get('turn') or 0) > 5:\n"
        "        target = d\n"
        "        break\n"
        "if target is None:\n"
        "    target = decs[len(decs) // 2]\n"
        "print('picked decision at turn', (target['obs'].get('current') or {}).get('turn'))\n"
        "print('original select:', target['obs']['select'])\n"
    ),
    md("## Step 2 — attach cg-lib, convert the observation, call search_begin\n\n"
       "`search_begin` takes the `Observation` dataclass "
       "(`to_observation_class(obs_dict)`), not the raw dict. `your_deck` / "
       "`your_prize` / `opponent_deck` / `opponent_prize` / `opponent_hand` "
       "only need to match the real **counts** (per the docstrings) — since "
       "this is a mirror game, `main.DECK` is a legitimate source of "
       "count-matched filler card IDs for both sides."),
    code(
        "import sys, glob\n"
        "for _pat in ['/kaggle/input/**/cg-lib', '/kaggle/input/cg-lib']:\n"
        "    _paths = glob.glob(_pat, recursive=True)\n"
        "    if _paths:\n"
        "        sys.path.insert(0, _paths[0])\n"
        "        break\n\n"
        "from cg.api import to_observation_class, search_begin, search_step, search_end\n\n"
        "obs_dict = target['obs']\n"
        "obs = to_observation_class(obs_dict)\n"
        "state = obs.current\n"
        "me = state.yourIndex\n"
        "my_p = state.players[me]\n"
        "opp_p = state.players[1 - me]\n"
        "print('yourIndex', me, 'my deckCount', my_p.deckCount, 'opp deckCount', opp_p.deckCount)\n"
        "print('my prize', len(my_p.prize), 'opp prize', len(opp_p.prize), 'opp handCount', opp_p.handCount)\n"
        "print('opp active face-down?', opp_p.active and opp_p.active[0] is None)\n\n"
        "def filler(n, pool=DECK):\n"
        "    return [pool[i % len(pool)] for i in range(n)]\n\n"
        "your_deck = filler(my_p.deckCount)\n"
        "your_prize = filler(len(my_p.prize))\n"
        "opponent_deck = filler(opp_p.deckCount)\n"
        "opponent_prize = filler(len(opp_p.prize))\n"
        "opponent_hand = filler(opp_p.handCount)\n"
        "opponent_active = []  # only needed if opp active is face-down (checked above)\n\n"
        "try:\n"
        "    ss = search_begin(obs, your_deck, your_prize, opponent_deck,\n"
        "                      opponent_prize, opponent_hand, opponent_active,\n"
        "                      manual_coin=True)\n"
        "    print('search_begin OK. searchId=', ss.searchId)\n"
        "    print('returned select:', ss.observation.select)\n"
        "    ret_state = ss.observation.current\n"
        "    print('returned yourIndex', ret_state.yourIndex, 'result', ret_state.result)\n"
        "    ret_opp = ret_state.players[1 - ret_state.yourIndex]\n"
        "    print('returned opp hand (placeholder check):', ret_opp.hand)\n"
        "except Exception as e:\n"
        "    import traceback; traceback.print_exc()\n"
        "    ss = None\n"
    ),
    md("## Step 3 — a few search_step calls, then search_end\n\n"
       "Deliberately naive (always picks option 0) — this is just to observe "
       "whose-turn/terminal/select-shape behavior across a couple of steps, "
       "not to make good decisions."),
    code(
        "if ss is not None:\n"
        "    cur_ss = ss\n"
        "    for i in range(4):\n"
        "        sel = cur_ss.observation.select\n"
        "        cur = cur_ss.observation.current\n"
        "        if sel is None:\n"
        "            print(f'step {i}: select is None (terminal?) result={cur.result if cur else None}')\n"
        "            break\n"
        "        n = len(sel.option)\n"
        "        print(f'step {i}: stype={sel.type} ctx={sel.context} n_opts={n} '\n"
        "              f'yourIndex={cur.yourIndex} result={cur.result}')\n"
        "        if n == 0:\n"
        "            break\n"
        "        try:\n"
        "            cur_ss = search_step(cur_ss.searchId, [0])\n"
        "        except Exception as e:\n"
        "            import traceback; traceback.print_exc()\n"
        "            break\n"
        "    search_end()\n"
        "    print('search_end() called')\n"
        "else:\n"
        "    print('skipped — search_begin failed above')\n"
    ),
    md("## Findings (fill in after running, then paste into docs/engine-api.md)\n\n"
       "- Did `search_begin` succeed with count-matched filler IDs? \n"
       "- Shape of `ss.observation.select` right after `search_begin` vs the "
       "original real `select` — same decision point, or re-rooted?\n"
       "- Did `manual_coin=True` change anything observable?\n"
       "- What did `opp_p.hand` look like in the returned observation (raw "
       "`[1072]*n` placeholder, or something else)?\n"
       "- Whose-turn / terminal behavior across the `search_step` loop.\n"),
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
