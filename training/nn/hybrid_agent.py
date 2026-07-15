"""Hybrid champion agent (issue #3 Phase C/D): plays main.agent (v29d)
verbatim EXCEPT at in-regime single-select decisions (regime_detector's
pre-registered rule), where a Q-net trained on from-state regime self-play
picks argmax Q(s, option). Any error anywhere falls back to the heuristic's
own choice — the subpolicy can never crash or stall a game (Design
Principle #2: timeout = instant loss).

Env vars: REGIME_CKPT (Q-net checkpoint), REGIME_BIG (default 1 = model_big),
REGIME_LOG (optional JSONL: one line per overridden decision, for audit).
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import main as heuristic  # noqa: E402
from regime_detector import regime_fires  # noqa: E402

DECK = heuristic.DECK  # harness.load_agent requires the module to export DECK

_CKPT = os.environ.get("REGIME_CKPT") or os.path.join(
    _REPO_ROOT, "training", "regime_qnet.pth")
# default "1": the pre-registered recipe trains with --big; loading a big
# checkpoint into the small net would silently drop every mismatched key
_BIG = os.environ.get("REGIME_BIG", "1") == "1"
_LOG = os.environ.get("REGIME_LOG")

overridden = 0  # introspection for gates


def _qnet_choice(obs_dict, sel):
    import torch
    from net_common import load_model, encode_batch
    if _BIG:
        from model_big import PTCGNetBig
        global _big_model
        try:
            model = _big_model
        except NameError:
            model = PTCGNetBig()
            model.load_state_dict(torch.load(_CKPT, map_location="cpu"), strict=True)
            model.eval()
            _big_model = model
    else:
        model = load_model(_CKPT)
    batch, n_actions = encode_batch(obs_dict, sel)
    with torch.no_grad():
        logits, _ = model(*batch)
    return int(logits[0, :n_actions].argmax().item())


def agent(obs_dict: dict) -> list:
    fallback = heuristic.agent(obs_dict)
    try:
        sel = obs_dict.get("select") or {}
        opts = sel.get("option") or []
        if len(opts) <= 1 or (sel.get("minCount", 0) or 0) >= 2:
            return fallback
        cur = obs_dict.get("current") or {}
        if not regime_fires(cur, cur.get("yourIndex")):
            return fallback
        pick = _qnet_choice(obs_dict, sel)
        global overridden
        overridden += 1
        if _LOG:
            with open(_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"turn": cur.get("turn"),
                                    "n_opts": len(opts), "qnet": pick,
                                    "heuristic": fallback[:1]}) + "\n")
        return [pick]
    except Exception:
        return fallback
