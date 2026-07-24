"""ACAN inference agent — the learned conservative override (S4).

Same structure that made the search work (heuristic drives; override only on a
clear margin), with the 2-ply belief-determinized search replaced by ONE forward
pass of the Action-Conditioned Advantage Net.

Consequences that matter:
  * No search at runtime -> no `search_begin` dependency, no per-decision time
    budget, no Kaggle search-RNG exposure.
  * An A/B of this agent makes ZERO search calls, so unlike every other agent in
    our search family it is cleanly evaluable in the local harness (immune to the
    -39pp sham-search placebo contamination).

Candidate construction below is copied verbatim from twoply_agent's `_search_decide`
so the net sees the same candidate sets at serve time that it was trained on.

Env: ACAN_CKPT (default training/nn/acan.pth), ACAN_THRESH (default: the
rate-matched threshold stored in the checkpoint).
"""
import os
import sys
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "training", "local_cg"),
           os.path.join(_REPO, "training", "belief"),
           os.path.join(_REPO, "training", "nn"),
           os.path.join(_REPO, "training"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location("h_main", os.path.join(_REPO, "main.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
DECK = list(H.DECK)

# These MUST match twoply_agent/twoply_collect exactly -- they define the
# candidate set, and any drift means the net is served candidate sets it was
# never trained on.
MAIN = 0
_OPT_ATTACK, _OPT_END = 13, 14
MAX_OPTS = int(os.environ.get("TWOPLY_MAXOPTS", "24"))

_OK = True
try:
    import numpy as np
    import torch
    from encode import numeric_feats, NUM_FEATS
    from acan_model import ACAN, act_index

    _CKPT = torch.load(os.environ.get("ACAN_CKPT", os.path.join(_HERE, "acan.pth")),
                       map_location="cpu", weights_only=False)
    _CARDS = {int(k): int(v) for k, v in _CKPT["cards"].items()}
    _ATKS = {int(k): int(v) for k, v in _CKPT["atks"].items()}
    _MODEL = ACAN(_CKPT["n_state"], len(_CARDS), len(_ATKS))
    _MODEL.load_state_dict(_CKPT["state_dict"])
    _MODEL.eval()
    _CLIP = float(_CKPT["clip"])
    _MU, _SD = _CKPT["mu"], _CKPT["sd"]
    _BASE_SCALE = float(_CKPT.get("base_scale", 1.0))
    _THRESH = float(os.environ.get("ACAN_THRESH", _CKPT.get("threshold", _CKPT["margin"])))
except Exception:
    _OK = False

_STATS = {"scored": 0, "overrides": 0}


def _act_desc(opts, cur, me_i, i):
    o = opts[i]
    t = o.get("type")
    cardid = -1
    if t in (7, 8, 9) and o.get("area") == 2:
        hnd = cur.get("players", [{}])[me_i].get("hand") or []
        ix = o.get("index")
        if ix is not None and 0 <= ix < len(hnd):
            cardid = (hnd[ix] or {}).get("id", -1)
    return [int(t) if t is not None else -1, int(cardid),
            int(o.get("attackId") or -1)]


def _acan_decide(obs_dict):
    if not _OK:
        return None
    sel = obs_dict.get("select")
    cur = obs_dict.get("current") or {}
    if not sel or sel.get("context") != MAIN:
        return None
    opts = sel.get("option") or []
    n = len(opts)
    if n < 3 or n > MAX_OPTS or (cur.get("turn") or 0) < 2:
        return None

    me_i = cur.get("yourIndex", 0)
    base = H.score_options_main(obs_dict, sel)
    if not base or len(base) != n:
        return None
    base_order = sorted(range(n), key=lambda i: -base[i])
    heur_top = base_order[0]

    cand = [heur_top]
    for i in base_order[1:]:
        t = opts[i].get("type")
        if t in (_OPT_ATTACK, _OPT_END):
            continue
        if base[i] < 0:
            continue
        cand.append(i)
        if len(cand) >= 8:
            break
    if len(cand) < 2:
        return None

    feats = numeric_feats({"current": cur})
    if feats is None or len(feats) != NUM_FEATS:
        return None
    _STATS["scored"] += 1

    hb = float(base[heur_top])
    s = np.tile(np.asarray(list(feats), dtype=np.float32), (len(cand), 1))
    s = torch.from_numpy(((s - _MU) / _SD).astype(np.float32))
    a = torch.tensor([act_index(_act_desc(opts, cur, me_i, i), _CARDS, _ATKS)
                      for i in cand], dtype=torch.long)
    f = torch.tensor([[1.0 if i == heur_top else 0.0,
                       (float(base[i]) - hb) / _BASE_SCALE] for i in cand],
                     dtype=torch.float32)
    with torch.no_grad():
        p = _MODEL(s, a, f).numpy() * _CLIP

    j = int(np.argmax(p))
    if cand[j] == heur_top or p[j] < _THRESH:
        return None
    _STATS["overrides"] += 1
    return cand[j]


def agent(obs_dict: dict):
    try:
        sel = obs_dict.get("select")
        override = _acan_decide(obs_dict)
        if override is not None:
            rest = [i for i in range(len(sel.get("option") or []))
                    if i != override]
            return H._safe_return([override] + rest, sel)
    except Exception:
        pass
    return H.agent(obs_dict)
