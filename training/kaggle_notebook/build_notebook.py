"""Builds ptcg-bc-training.ipynb by embedding training/nn/*.py as %%writefile
cells, so the notebook source stays in sync with the locally-tested code
instead of being hand-transcribed. Run from repo root: python
training/kaggle_notebook/build_notebook.py
"""
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NN_DIR = os.path.join(REPO_ROOT, "training", "nn")
OUT = os.path.join(REPO_ROOT, "training", "kaggle_notebook", "ptcg-bc-training.ipynb")


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def writefile_cell(fname):
    with open(os.path.join(NN_DIR, fname), encoding="utf-8") as f:
        content = f.read()
    return code(f"%%writefile {fname}\n" + content)


cells = [
    md("# PTCG Alakazam v22 — BC Warmup Training\n\n"
       "Behavior cloning from v22 self-play (547,796 decisions, 2,000 games). "
       "See `docs/nn-training.md` in the repo for the full plan. This notebook:\n\n"
       "1. Writes out `encode.py` / `model.py` / `dataset.py` / `train_bc.py` "
       "(kept in sync with `training/nn/` in the repo — smoke-tested locally on CPU "
       "before this notebook was built).\n"
       "2. Trains the BC policy/value net (10 epochs, LR 1e-4).\n"
       "3. Reports held-out top-1 accuracy against v22's chosen actions "
       "(the direct imitation-quality metric) and saves the checkpoint.\n\n"
       "**Data:** attach dataset `jander6364/ptcg-alakazam-v22-bc-data` as input.\n"
       "**Accelerator:** GPU T4 x1 (Settings → Accelerator).\n"),
    code("import torch\nprint('torch', torch.__version__, 'cuda:', torch.cuda.is_available())\n"
         "!echo '--- /kaggle/input ---'; ls -la /kaggle/input/ 2>&1\n"
         "!echo '--- recursive (depth 3) ---'; find /kaggle/input -maxdepth 3 2>&1\n"),
    writefile_cell("encode.py"),
    writefile_cell("model.py"),
    writefile_cell("dataset.py"),
    writefile_cell("train_bc.py"),
    md("## Train\n\n"
       "Gate from `docs/nn-training.md`: this notebook reports **held-out top-1 "
       "action-match accuracy** against v22 (a direct imitation-quality proxy). "
       "The competition-relevant gates — 65%+ vs random, ~50% vs v22 — require "
       "actually playing games with the trained net, which needs an agent "
       "wrapper (`training/README.md` next steps) run outside this notebook, "
       "e.g. locally via `training/ab_test.py`."),
    code("!python train_bc.py --data \"/kaggle/input/**/bc_data*.pkl\" "
         "--epochs 10 --batch-size 256 --lr 1e-4 --out /kaggle/working/ptcg_bc_v1.pth\n"),
    md("## Next steps\n\n"
       "- Download `/kaggle/working/ptcg_bc_v1.pth`.\n"
       "- Write a thin `agent(obs_dict)` wrapper around the checkpoint (load "
       "`PTCGNet`, encode the obs with `encode.py`, argmax the masked logits, "
       "map back to `select.option` indices) so it satisfies the same contract "
       "as `main.py` / `opponents/*.py`.\n"
       "- Evaluate locally: `python training/ab_test.py <net_agent.py> main.py 400` "
       "(random-vs-net and v22-vs-net gates from `docs/nn-training.md`).\n"
       "- If it clears ~50% vs v22, ship it to the ladder (single forward pass — "
       "no MCTS latency risk) and start self-play Phase 1."),
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
