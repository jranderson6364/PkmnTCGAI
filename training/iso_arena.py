"""Process-isolated arena that stays FAITHFUL to the real engine.

The contamination (placebo, 2026-07-23): a search agent's search_begin/step calls
perturb the shared cg.dll global RNG, corrupting the live battle by up to -39pp
vs asymmetric opponents. But driving the battle with raw cg.game does NOT
reproduce kaggle_environments' outcomes (pure heuristic vs dragapult: 40% raw vs
99% via kaggle_environments) — the faithful engine is the cabt env, not a
hand-rolled loop.

So: keep the battle in kaggle_environments (the exact engine Kaggle runs, main
process), but replace each agent with a lightweight PROXY that forwards the
observation to a SUBPROCESS worker and returns its action. The worker runs the
real agent — including any search — in its OWN process, hence its own loaded
cg.dll with its own global RNG. The agent's search calls can no longer touch the
main-process battle's RNG. This is Kaggle's isolation model (one process per
agent) layered on the faithful battle.

Validation: a PURE heuristic must score the SAME here as in the in-process
harness (isolation changes nothing when there are no search calls).

Run:  python training/iso_arena.py <agentA.py> <agentB.py> --games 100
"""
import argparse
import math
import os
import pickle
import struct
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO, "training"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_WORKER = os.path.join(_HERE, "iso_worker.py")


def _load_deck(path):
    import importlib.util
    p = os.path.abspath(path)
    name = "iso_deck_" + str(abs(hash(p)) % 10**8)
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return list(mod.DECK)


class Worker:
    """One agent, running in its own OS process (own cg.dll / own RNG).
    IPC = length-framed pickle (see iso_worker.py for why not JSON)."""
    def __init__(self, agent_path):
        self.p = subprocess.Popen(
            [sys.executable, _WORKER, os.path.abspath(agent_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=0, env=dict(os.environ))

    def act(self, obs):
        try:
            payload = pickle.dumps(dict(obs))
            self.p.stdin.write(struct.pack(">I", len(payload)))
            self.p.stdin.write(payload)
            self.p.stdin.flush()
            hdr = self.p.stdout.read(4)
            if not hdr or len(hdr) < 4:
                return [0]
            (n,) = struct.unpack(">I", hdr)
            buf = self.p.stdout.read(n)
            return pickle.loads(buf)
        except Exception:
            return [0]

    def close(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        try:
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def play_game_isolated(pathA, deckA, pathB, deckB, a_seat, max_steps=None):
    """One kaggle_environments cabt game; each agent runs in its own subprocess.
    a_seat = physical slot (0/1) agentA occupies. Returns 'A'/'B'/'T'."""
    from kaggle_environments import make

    workers = {a_seat: Worker(pathA), 1 - a_seat: Worker(pathB)}
    decks = [None, None]
    decks[a_seat] = deckA
    decks[1 - a_seat] = deckB

    def proxy(seat):
        def _fn(obs):
            return workers[seat].act(obs)
        return _fn

    try:
        config = {"decks": decks}
        if max_steps:
            config["episodeSteps"] = max_steps
        env = make("cabt", configuration=config)
        env.run([proxy(0), proxy(1)])
        last = env.steps[-1]
        r0, r1 = last[0].get("reward"), last[1].get("reward")
    finally:
        for w in workers.values():
            w.close()

    if r0 is None or r1 is None or r0 == r1:
        return "T"
    winner_seat = 0 if r0 > r1 else 1
    return "A" if winner_seat == a_seat else "B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agentA")
    ap.add_argument("agentB")
    ap.add_argument("--games", type=int, default=60)
    args = ap.parse_args()

    deckA = _load_deck(args.agentA)
    deckB = _load_deck(args.agentB)

    aw = bw = t = 0
    t0 = time.time()
    for g in range(args.games):
        r = play_game_isolated(args.agentA, deckA, args.agentB, deckB, g % 2)
        if r == "A":
            aw += 1
        elif r == "B":
            bw += 1
        else:
            t += 1
        if (g + 1) % 10 == 0:
            print(f"  {g+1}/{args.games}  A{aw}-B{bw}-T{t}  "
                  f"({100*aw/max(1,aw+bw):.1f}%)  {round(time.time()-t0)}s", flush=True)

    n = aw + bw
    wr = aw / max(1, n)
    se = math.sqrt(wr * (1 - wr) / max(1, n))
    print(f"\n=== ISOLATED ARENA (faithful cabt + subprocess agents): "
          f"{os.path.basename(args.agentA)} vs {os.path.basename(args.agentB)} ===")
    print(f"A {aw}W  B {bw}W  T {t}  (n={args.games})")
    print(f"A win rate: {wr:.3f} +/- {1.96*se:.3f} (95% CI)")


if __name__ == "__main__":
    main()
