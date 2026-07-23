"""Agent subprocess for the process-isolated arena (training/iso_arena.py).

Loads ONE agent in its own OS process — its own cg.dll, its own global RNG — so
its search_begin/step calls perturb THIS process's RNG, never the referee's live
battle. This is what makes search-agent measurement valid.

IPC: length-framed PICKLE over stdin/stdout (binary). Pickle, not JSON, because
the kaggle_environments observation is a `Struct` and JSON round-trips it lossily
(int keys -> str, tuples -> lists), which changes the agent's decisions — pickle
preserves the exact object the in-process agent would see. Frame = 4-byte
big-endian length + pickle payload. Empty/EOF => exit.
"""
import sys
import os
import struct
import pickle
import importlib.util

# The IPC channel is fd 0/1 (binary). The agent, cg.api, and the cg.dll C library
# can all write to stdout; C-level printf bypasses Python redirects. Dup the real
# fd 1 for protocol use, then point fd 1 at stderr so nothing pollutes the pipe.
_PROTO_FD = os.dup(1)
os.dup2(2, 1)
sys.stdout = sys.stderr
_IN = os.fdopen(0, "rb")
_OUT = os.fdopen(_PROTO_FD, "wb")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO, "training", "local_cg"),
           os.path.join(_REPO, "training", "belief"),
           os.path.join(_REPO, "training"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _read_frame():
    hdr = _IN.read(4)
    if not hdr or len(hdr) < 4:
        return None
    (n,) = struct.unpack(">I", hdr)
    buf = _IN.read(n)
    return pickle.loads(buf)


def _write_frame(obj):
    payload = pickle.dumps(obj)
    _OUT.write(struct.pack(">I", len(payload)))
    _OUT.write(payload)
    _OUT.flush()


def _load(path):
    path = os.path.abspath(path)
    name = "iso_agent_" + str(abs(hash(path)) % 10**8)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def main():
    agent = _load(sys.argv[1])
    while True:
        obs = _read_frame()
        if obs is None:
            break
        try:
            action = agent(obs)
            if not isinstance(action, list):
                action = [0]
            action = [int(x) for x in action]
        except Exception:
            action = [0]
        _write_frame(action)


if __name__ == "__main__":
    main()
