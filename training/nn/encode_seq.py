"""Per-decision encoding for the sequence-policy experiment: encode.py's
per-state encode_sample() plus the 11 Φ v4 antisymmetric features appended
(docs/next-session-plan.md Phase 1 item 3 — the calculated-values thesis,
proven twice by the eval ladder). No new tokenization; this only adds a
fixed-size numeric side-channel per step.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from encode import encode_sample  # noqa: E402
from eval_v4 import features_v4  # noqa: E402

PHI4_DIM = 11


def encode_sample_seq(obs, sel):
    """Returns encode_sample()'s dict plus a "phi4" key: list of 11 floats,
    or 11 zeros if the state is malformed (mirrors encode.py's own
    fail-soft-to-zero pattern for belief features)."""
    enc = encode_sample(obs, sel)
    cur = obs.get("current") or {}
    me_idx = cur.get("yourIndex", 0)
    try:
        phi = features_v4(cur, me_idx)
        enc["phi4"] = phi.tolist() if phi is not None else [0.0] * PHI4_DIM
    except Exception:
        enc["phi4"] = [0.0] * PHI4_DIM
    return enc
