"""v31 unreachable-tank fix gates (pre-registered docs/report-log.md
2026-07-18): G1 grimm bot n=400, G2 mirror vs pre-fix frozen n=400,
G3 anchors n=300 each. Reuses wsearch._wr_pair (seat-alternated)."""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from wsearch import ANCHORS, FROZEN, _wr_pair  # noqa: E402

MAIN = os.path.join(_REPO, "main.py")
GRIMM = os.path.join(_REPO, "opponents", "grimmsnarl_agent.py")


def main():
    out = {}
    wr, n, e = _wr_pair(MAIN, GRIMM, 400, 15)
    out["G1_grimm"] = {"wr": wr, "n": n, "errors": e}
    print(f"G1 grimm: {wr:.4f} n={n} errors={e}", flush=True)
    wr, n, e = _wr_pair(MAIN, FROZEN, 400, 15)
    out["G2_mirror"] = {"wr": wr, "n": n, "errors": e}
    print(f"G2 mirror: {wr:.4f} n={n} errors={e}", flush=True)
    for name in ("lucario", "abomasnow", "starmie", "dragapult"):
        wr, n, e = _wr_pair(MAIN, ANCHORS[name], 300, 15)
        out[f"G3_{name}"] = {"wr": wr, "n": n, "errors": e}
        print(f"G3 {name}: {wr:.4f} n={n} errors={e}", flush=True)
    with open(os.path.join(_HERE, "v31_gate_results.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
