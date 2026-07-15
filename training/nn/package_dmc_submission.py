"""Package dmc_agent.py (greedy, NET_EPS=0) as a standalone Kaggle submission —
the first-ever live-ladder read for any learned model in this project (all
prior "beats/loses to X%" numbers are offline-vs-v29d only, which Design
Principle #1 warns can lie). Ships training/ptcg_dmc_r6_checkpoint1.pth
(round-6 study's checkpoint-1, 8.25% +/- 2.7% offline vs v29d, n=400 — the
project's best DMC checkpoint as of 2026-07-12).

Same staging trick as package_endgame_submission.py: the repo's main.py
(the v29d heuristic) is copied to heuristic.py inside the staged dir,
because encode.py depends on it directly (`import main as _heuristic` for
belief-posterior/census/PH_DMG_PER_CARD -- the DMC net's own state features
include the heuristic's belief model). dmc_agent.py's `from main import
DECK` is rewritten to `from heuristic import DECK` in the staged copy only;
the repo's dmc_agent.py is untouched.

No heuristic fallback wrapper here (unlike the endgame agent) -- this ships
the raw DMC policy so the ladder read is uncontaminated. dmc_agent.py's own
try/except already returns a safe legal default action if inference ever
throws (Design Principle #2), it just doesn't fall back to heuristic play.

Usage: python training/nn/package_dmc_submission.py [--out-dir DIR] [--ckpt PATH]
Then: tar the DIR's contents (main.py at root) into submission.tar.gz.
"""
import argparse
import os
import shutil

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAIN_WRAPPER = '''"""DMC (Deep Monte Carlo) Q-network agent -- greedy inference (NET_EPS=0),
shipped for the first-ever live-ladder read of a learned model in this
project. See training/nn/package_dmc_submission.py for provenance;
docs/report-log.md 2026-07-11/12 "round-6 data-scaling study" for the
offline gate this checkpoint cleared (8.25% +/- 2.7% vs v29d, n=400).
"""
import os
import sys

os.environ.setdefault("NET_EPS", "0")
os.environ.setdefault("NET_BIG", "1")

# NOTE: this file has NO __file__ at runtime on the real ladder -- Kaggle's
# get_last_callable() execs the submitted main.py from a raw string into a
# bare {} namespace, which never sets __file__ (confirmed via the v29
# validation-episode NameError, docs/report-log.md 2026-07-07). Borrow
# heuristic.py's __file__ instead, same fix as the endgame-agent ship.
import heuristic as _heuristic_mod  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(_heuristic_mod.__file__))
os.environ["NET_CKPT"] = os.path.join(_HERE, "training", "nn", "ptcg_dmc_r6_checkpoint1.pth")

for _sub in ("training/nn",):
    _p = os.path.join(_HERE, *_sub.split("/"))
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dmc_agent import agent  # noqa: E402,F401
'''


def _rewrite_main_refs(text):
    return (text
            .replace("import main as _heuristic", "import heuristic as _heuristic")
            .replace("from main import DECK", "from heuristic import DECK"))


def build(out_dir, ckpt):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    shutil.copy(os.path.join(_REPO_ROOT, "main.py"), os.path.join(out_dir, "heuristic.py"))
    shutil.copy(os.path.join(_REPO_ROOT, "deck.csv"), os.path.join(out_dir, "deck.csv"))

    with open(os.path.join(out_dir, "main.py"), "w") as f:
        f.write(MAIN_WRAPPER)

    nn_out = os.path.join(out_dir, "training", "nn")
    os.makedirs(nn_out)

    for name in ("encode.py", "model.py", "model_big.py", "threat.py",
                 "net_common.py", "tempo_features.py"):
        src = os.path.join(_REPO_ROOT, "training", "nn", name)
        with open(src) as f:
            text = f.read()
        text = _rewrite_main_refs(text)
        with open(os.path.join(nn_out, name), "w") as f:
            f.write(text)

    # Kaggle's agent sandbox has no cg module (the 54624481 validation
    # failure) — threat.py falls back to this bundled dump there.
    shutil.copy(os.path.join(_REPO_ROOT, "training", "nn", "card_tables.json"),
                os.path.join(nn_out, "card_tables.json"))

    src = os.path.join(_REPO_ROOT, "training", "nn", "dmc_agent.py")
    with open(src) as f:
        text = _rewrite_main_refs(f.read())
    with open(os.path.join(nn_out, "dmc_agent.py"), "w") as f:
        f.write(text)

    shutil.copy(ckpt, os.path.join(nn_out, "ptcg_dmc_r6_checkpoint1.pth"))

    print(f"staged submission at {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(_REPO_ROOT, "training", "dmc_submission_stage"))
    ap.add_argument("--ckpt", default=os.path.join(_REPO_ROOT, "training", "ptcg_dmc_r6_checkpoint1.pth"))
    args = ap.parse_args()
    build(args.out_dir, args.ckpt)
