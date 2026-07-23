"""Extract public competition agents from their Kaggle notebooks into
`opponents/public/` so they can be used as OFFLINE SPARRING PARTNERS.

Why: these agents have KNOWN live ladder scores (933.8 down to 509.6). Playing
our candidates against a panel of known-strength opponents is the only way this
project has ever had to (a) get a discriminating offline opponent -- every
reference anchor reads <=6% against our champion -- and (b) fit offline strength
to publicScore directly, instead of guessing at Design Principle #1.

LICENCE / USE BOUNDARY. These are Apache-2.0 public notebooks by other
competitors. They are used here ONLY as local opponents and as diff targets for
our own development. Nothing extracted here is ever submitted, in whole or in
part, as our agent. Attribution is written into every extracted file.

Two embedding styles are handled:
  * `%%writefile main.py` cells (prvsiyan, aristophanivan, raunakdey07)
  * base64+zlib `AGENT_PAYLOADS` blobs (lucifer19/battlecore)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import tarfile
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "opponents" / "public"

# ref -> (output stem, publicScore at extraction time, short label)
KNOWN = {
    "aristophanivan/probablity-v2":                        ("probability_v2", 933.8, "Mega Lucario ex heuristic"),
    "lucifer19/battlecore-compact-agent":                   ("battlecore", 846.8, "Archaludon metal tempo"),
    "raunakdey07/pok-mon-tcg-advanced-heuristic-agent":     ("advanced_heuristic", 796.8, "Expectimax+UCB1 heuristic"),
    "prvsiyan/ptcg-ai-battle-search-audited-alakazam-v9":   ("alakazam_v9", 778.2, "Search-audited Alakazam"),
    "prvsiyan/ptcg-ai-battle-field-audited-alakazam-v8":    ("alakazam_v8", 739.7, "Field-audited Alakazam"),
}

HEADER = '''"""{label} -- THIRD-PARTY PUBLIC AGENT, used as an offline opponent only.

Source notebook : https://www.kaggle.com/code/{ref}
Licence         : Apache-2.0 (public Kaggle notebook)
publicScore     : {score} (read {date})

Extracted by tools/extract_public_agents.py for use as a KNOWN-STRENGTH sparring
partner in our offline panel (docs/endgame-plan.md P1). This file is NOT ours and
is NEVER submitted, in whole or in part. Local-path and Kaggle-path shims were
added at the marked lines so it runs under training/harness.py; the decision
logic is untouched.
"""
'''

# Kaggle-only side effects that must not fire when we import the module locally.
SHIMS = r'''
# --- harness shim (added by tools/extract_public_agents.py, not by the author) ---
import os as _os, sys as _sys, glob as _glob
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_LOCAL_CG = _os.path.join(_HERE, "..", "..", "training", "local_cg")
if _os.path.isdir(_LOCAL_CG) and _LOCAL_CG not in _sys.path:
    _sys.path.insert(0, _LOCAL_CG)

# The originals write deck.csv into the CWD at import time and then read it back.
# Redirect both to a private temp dir so importing an opponent never touches the
# repo working tree.
import tempfile as _tempfile
_SCRATCH = _tempfile.mkdtemp(prefix="ptcg_public_agent_")
_real_open = open
class _PathShim:
    pass
# --- end harness shim ---
'''


def _cells(nb_path: Path):
    d = json.loads(nb_path.read_text(encoding="utf-8"))
    for c in d["cells"]:
        yield c


def from_writefile(nb_path: Path) -> str | None:
    """Pull the body of a `%%writefile main.py` cell."""
    for c in _cells(nb_path):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        m = re.match(r"\s*%%writefile\s+\S*main\.py\s*\n", s)
        if m:
            return s[m.end():]
    return None


def from_payloads(nb_path: Path, prefer=("A", "B")) -> str | None:
    """Decode battlecore's base64+zlib AGENT_PAYLOADS blob without executing it."""
    src = "\n".join("".join(c["source"]) for c in _cells(nb_path) if c["cell_type"] == "code")
    blobs = dict(re.findall(r'"(\w+)"\s*:\s*\{[^}]*?"payload"\s*:\s*"([A-Za-z0-9+/=\\\n\s]+?)"',
                            src, re.S))
    if not blobs:
        blobs = {k: v for k, v in re.findall(r'(\w+)\s*=\s*"""([A-Za-z0-9+/=\s]{400,})"""', src)}
    for key in list(prefer) + sorted(blobs):
        raw = blobs.get(key)
        if not raw:
            continue
        b = base64.b64decode(re.sub(r"\s+", "", raw))
        for dec in (lambda x: zlib.decompress(x), lambda x: zlib.decompress(x, 16 + zlib.MAX_WBITS)):
            try:
                data = dec(b)
            except Exception:
                continue
            # payload may be a tarball or the raw source
            try:
                with tarfile.open(fileobj=io.BytesIO(data)) as t:
                    for n in t.getnames():
                        if n.endswith("main.py"):
                            return t.extractfile(n).read().decode("utf-8")
            except Exception:
                pass
            text = data.decode("utf-8", errors="replace")
            if "def agent(" in text:
                return text
    return None


def patch(source: str) -> str:
    """Make a Kaggle-shaped agent importable and runnable under our harness."""
    out = source
    # 1. deck.csv writes at import time -> route to a temp dir
    out = re.sub(r'Path\(\s*["\']deck\.csv["\']\s*\)\.write_text',
                 'Path(_os.path.join(_SCRATCH, "deck.csv")).write_text', out)
    out = re.sub(r'open\(\s*["\']deck\.csv["\']',
                 'open(_os.path.join(_SCRATCH, "deck.csv")', out)
    # 2. deck path lookup -> our temp copy
    out = re.sub(r'DECK_PATH\s*=\s*["\']deck\.csv["\']',
                 'DECK_PATH = _os.path.join(_SCRATCH, "deck.csv")', out)
    # 3. guarantee the module exposes DECK for harness.load_agent
    if not re.search(r"^\s*DECK\s*=", out, re.M):
        for alt in ("my_deck", "MY_DECK", "deck"):
            if re.search(rf"^\s*{alt}\s*=", out, re.M):
                out += f"\n\n# harness shim: expose the module deck under the name our harness expects\nDECK = list({alt})\n"
                break
    else:
        out += "\n\n# harness shim: DECK already defined by the author above.\n"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels-dir", required=True,
                    help="directory holding the pulled *.ipynb notebooks")
    ap.add_argument("--date", default="2026-07-23")
    args = ap.parse_args()

    kd = Path(args.kernels_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "__init__.py").write_text("", encoding="utf-8")

    manifest = []
    for ref, (stem, score, label) in KNOWN.items():
        slug = ref.split("/")[1]
        cands = list(kd.glob(f"**/{slug}.ipynb"))
        if not cands:
            print(f"  SKIP {ref}: notebook not found under {kd}")
            continue
        nb = cands[0]
        src = from_writefile(nb) or from_payloads(nb)
        if not src:
            print(f"  SKIP {ref}: no agent source found in {nb.name}")
            continue
        # `from __future__` must precede every statement, so hoist any such
        # lines out of the author's source and place them directly after our
        # docstring, ahead of the shim imports.
        future = re.findall(r"^from __future__ import .*$", src, re.M)
        src_nofuture = re.sub(r"^from __future__ import .*$\n?", "", src, flags=re.M)
        head = HEADER.format(label=label, ref=ref, score=score, date=args.date)
        if future:
            head += "\n".join(dict.fromkeys(future)) + "\n"
        body = head + SHIMS + patch(src_nofuture)
        dest = OUT / f"{stem}.py"
        dest.write_text(body, encoding="utf-8")
        manifest.append(dict(ref=ref, file=dest.name, publicScore=score, label=label))
        print(f"  OK   {dest.name:24s} {score:7.1f}  ({len(src)} chars)  <- {ref}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {len(manifest)} agents + manifest.json to {OUT}")


if __name__ == "__main__":
    main()
