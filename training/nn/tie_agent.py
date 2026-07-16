"""Learned tie-breaker agent (RL track, post-issue-#3): plays main.agent
(v29d) verbatim EXCEPT when the MAIN-phase (stype==0) argmax has an EXACT
score tie — measured at 31.7% of multi-option MAIN-phase decisions, where
v29d currently picks by array order (the v29c retreat bugs were this class).
Within the tied set ONLY, picks argmax Q from the round-6 DMC checkpoint.
By construction this cannot override any decision the heuristic actually
expresses a preference about — plan coherence is preserved; the bar is
"beat array order", not "beat v29d".

Env vars: TIE_CKPT (default training/ptcg_dmc_r6_checkpoint1.pth),
TIE_BIG (default 1). Any exception -> heuristic fallback.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import main as heuristic  # noqa: E402

DECK = heuristic.DECK  # harness.load_agent requires the module to export DECK

_CKPT = os.environ.get("TIE_CKPT") or os.path.join(
    _REPO_ROOT, "training", "ptcg_dmc_r6_checkpoint1.pth")
_BIG = os.environ.get("TIE_BIG", "1") == "1"

tie_breaks = 0  # introspection for gates
_model = None


def _load():
    global _model
    if _model is None:
        import torch
        if _BIG:
            from model_big import PTCGNetBig
            m = PTCGNetBig()
        else:
            from model import PTCGNet
            m = PTCGNet()
        m.load_state_dict(torch.load(_CKPT, map_location="cpu"), strict=True)
        m.eval()
        _model = m
    return _model


def agent(obs_dict: dict) -> list:
    fallback = heuristic.agent(obs_dict)
    try:
        sel = obs_dict.get("select")
        if not sel or sel.get("type") != 0:
            return fallback
        opts = sel.get("option") or []
        if len(opts) <= 1 or not fallback:
            return fallback
        s = heuristic.score_options_main(obs_dict, sel)
        m = max(s)
        tie = [i for i, v in enumerate(s) if v == m]
        # only act on true argmax ties that the heuristic resolved by order;
        # if some later gate moved the pick off the tie set, respect it
        if len(tie) <= 1 or fallback[0] not in tie:
            return fallback
        import torch
        from net_common import encode_batch
        model = _load()
        batch, n = encode_batch(obs_dict, sel)
        with torch.no_grad():
            logits, _ = model(*batch)
        q = logits[0]
        pick = max(tie, key=lambda i: q[i].item() if i < n else float("-inf"))
        global tie_breaks
        tie_breaks += 1
        return [pick]
    except Exception:
        return fallback
