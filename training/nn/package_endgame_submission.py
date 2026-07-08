"""Package endgame_agent.py (2026-07-07, docs/report-log.md "Endgame-gated
search") as a standalone multi-file Kaggle submission.

The shipped entrypoint must be named main.py, but the repo's main.py is the
plain v28 heuristic that mcts.py/ismcts_agent.py/determinize.py/
endgame_agent.py already import as `main` (dev convention, unchanged here).
So this script builds a STAGING copy where:
  - the current main.py is copied to heuristic.py
  - the search-stack files are copied with `main` -> `heuristic` import
    references rewritten (staged copies only; repo files untouched)
  - mcts.py's net_common (torch) import is made lazy (rollout leaf-eval,
    endgame_agent's only mode, never touches it -- avoids an unnecessary
    torch dependency at ship time)
  - a new main.py wrapper is written: sys.path setup for the staged
    subtree, a try/except import of the search stack (falls back to pure
    heuristic play if anything is missing -- Design Principle #2), and the
    endgame gate itself

Confirmed via kaggle_environments/agent.py (get_last_callable): the
competition loader appends the submitted main.py's own directory to
sys.path before exec'ing it, so sibling files/subdirectories in the same
tar ARE importable -- this does not need to be a single monolithic file.

Usage: python training/nn/package_endgame_submission.py [--out-dir DIR]
Then: tar the DIR's contents (main.py at root) into submission.tar.gz.
"""
import argparse
import os
import shutil

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAIN_WRAPPER = '''"""Endgame-gated search agent (shipped entrypoint).

Plays the v28 heuristic (heuristic.py, this tar's copy of the repo's
main.py) verbatim, except when either player is <=2 prizes from winning,
where it runs belief-determinized rollout search instead (see
docs/report-log.md 2026-07-07 "Endgame-gated search gate 1" for the
confirmed 59.0% +/- 4.8% result vs plain heuristic play, n=400).

Falls back to pure heuristic play if the search stack fails to import for
any reason (missing dependency, unavailable cg.api, etc.) -- never allowed
to break agent() itself.
"""
import os
import sys

# NOTE: this file has NO __file__ at runtime on the real ladder -- Kaggle's
# get_last_callable() execs the submitted main.py from a raw string into a
# bare {} namespace (kaggle_environments/agent.py), which never sets
# __file__. (Confirmed via the v29 validation-episode failure: NameError:
# name '__file__' is not defined -- the first ship's clean-room test used
# harness.py's importlib.util.spec_from_file_location, a real module load
# that DOES set __file__, so it didn't catch this.) Anything imported BY
# main.py (heuristic, ismcts_agent, mcts, ...) goes through the normal
# import system and gets a real __file__ -- so borrow heuristic's.
from heuristic import DECK, agent as heuristic_agent  # noqa: E402
import heuristic as _heuristic_mod  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(_heuristic_mod.__file__))
for _sub in ("training/nn", "training/belief", "opponents"):
    _p = os.path.join(_HERE, *_sub.split("/"))
    if _p not in sys.path:
        sys.path.insert(0, _p)

_HAS_SEARCH = False
try:
    from ismcts_agent import BeliefMCTSSearcher  # noqa: E402
    _SIMS = int(os.environ.get("ENDGAME_SIMS", "60"))
    _PRIZES = int(os.environ.get("ENDGAME_PRIZES", "2"))
    _C_PUCT = float(os.environ.get("MCTS_C_PUCT", "1.4"))
    _PRIOR_TEMP = float(os.environ.get("MCTS_PRIOR_TEMP", "2.0"))
    _HAS_SEARCH = True
except Exception as _e:
    print(f"[main] search stack unavailable, playing pure heuristic: {_e!r}",
          file=sys.stderr)

_fallback_count = 0


def _is_endgame(obs_dict):
    try:
        players = obs_dict["current"]["players"]
        return min(len(p.get("prize") or []) for p in players) <= _PRIZES
    except Exception:
        return False


def agent(obs_dict: dict) -> list:
    global _fallback_count
    sel = obs_dict.get("select")
    if sel is None:
        return DECK
    if not sel.get("option"):
        return []
    if not _HAS_SEARCH or len(sel.get("option", [])) <= 1 or not _is_endgame(obs_dict):
        return heuristic_agent(obs_dict)
    try:
        searcher = BeliefMCTSSearcher(sims=_SIMS, c_puct=_C_PUCT,
                                      prior_temp=_PRIOR_TEMP, leaf_eval="rollout")
        return searcher.choose(obs_dict)
    except Exception as e:
        _fallback_count += 1
        print(f"[main] FALLBACK #{_fallback_count} to heuristic: {e!r}",
              file=sys.stderr)
        return heuristic_agent(obs_dict)
'''


def _rewrite_main_refs(text):
    return (text
            .replace("import main as heuristic", "import heuristic")
            .replace("from main import DECK, agent as heuristic_agent",
                      "from heuristic import DECK, agent as heuristic_agent")
            .replace("from main import DECK", "from heuristic import DECK")
            .replace("from main import agent as heuristic_agent",
                      "from heuristic import agent as heuristic_agent"))


def _make_net_common_lazy(text):
    """mcts.py's net_common (torch) import is only needed by _net_leaf_value
    (leaf_eval='net'), which endgame_agent's rollout-only ship never uses.
    Move the import into that method so its absence can't break the rollout
    path this submission actually exercises."""
    text = text.replace("from net_common import load_model, encode_batch  # noqa: E402\n", "")
    return text.replace(
        "                    model = load_model(self.net_ckpt)\n",
        "                    from net_common import load_model, encode_batch\n"
        "                    model = load_model(self.net_ckpt)\n",
    )


def build(out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    shutil.copy(os.path.join(_REPO_ROOT, "main.py"), os.path.join(out_dir, "heuristic.py"))
    shutil.copy(os.path.join(_REPO_ROOT, "deck.csv"), os.path.join(out_dir, "deck.csv"))

    with open(os.path.join(out_dir, "main.py"), "w") as f:
        f.write(MAIN_WRAPPER)

    os.makedirs(os.path.join(out_dir, "training", "nn"))
    os.makedirs(os.path.join(out_dir, "training", "belief"))
    os.makedirs(os.path.join(out_dir, "opponents"))

    for name in ("mcts.py", "ismcts_agent.py"):
        src = os.path.join(_REPO_ROOT, "training", "nn", name)
        with open(src) as f:
            text = f.read()
        text = _rewrite_main_refs(text)
        if name == "mcts.py":
            text = _make_net_common_lazy(text)
        with open(os.path.join(out_dir, "training", "nn", name), "w") as f:
            f.write(text)

    src = os.path.join(_REPO_ROOT, "training", "belief", "determinize.py")
    with open(src) as f:
        text = _rewrite_main_refs(f.read())
    with open(os.path.join(out_dir, "training", "belief", "determinize.py"), "w") as f:
        f.write(text)

    with open(os.path.join(_REPO_ROOT, "training", "archetype_decks.json")) as f:
        recon_json = f.read()
    with open(os.path.join(out_dir, "training", "archetype_decks.json"), "w") as f:
        f.write(recon_json)

    # Ship all four exact-decklist opponent bots BeliefDeterminizer draws on
    # (lucario/dragapult/abomasnow/starmie) -- the same stack the 400-game
    # gate ran against; dropping any would make this an unvalidated variant.
    for name in ("lucario_agent.py", "dragapult_agent.py", "abomasnow_agent.py",
                 "starmie_agent.py"):
        shutil.copy(os.path.join(_REPO_ROOT, "opponents", name),
                    os.path.join(out_dir, "opponents", name))

    print(f"staged submission at {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(_REPO_ROOT, "training", "endgame_submission_stage"))
    args = ap.parse_args()
    build(args.out_dir)
