"""Local game harness for the cabt engine (ships inside kaggle_environments).

Runs full games locally (~0.5-1s each) — no Kaggle session needed. Used by
ab_test.py (A/B evaluation), bc_collect.py (BC data), weight_search.py (weight
tuning). Works on any machine with `pip install kaggle_environments --no-deps`
(--no-deps avoids a Windows long-path failure in an unrelated dependency).

Agent modules must expose `agent(obs_dict) -> list[int]` and `DECK` (60 ints).
"""
import importlib.util
import logging
import os
import sys
import time

# silence kaggle_environments' noisy env-registration logging before import
logging.disable(logging.INFO)
os.environ.setdefault("PYTHONWARNINGS", "ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_agent(path):
    """Import an agent module from a file path. Returns (agent_fn, deck, module)."""
    path = os.path.abspath(path)
    name = "agent_" + os.path.splitext(os.path.basename(path))[0] + "_" + str(abs(hash(path)) % 10**8)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.agent, list(mod.DECK), mod


def play_game(agent0, deck0, agent1, deck1, keep_steps=False):
    """Run one full game. Returns dict with rewards, steps, wall time, and
    (optionally) the full step trace for BC collection / analysis."""
    from kaggle_environments import make

    t0 = time.time()
    env = make("cabt", configuration={"decks": [deck0, deck1]})
    env.run([agent0, agent1])
    wall = time.time() - t0
    last = env.steps[-1]
    result = {
        "rewards": [last[0].get("reward"), last[1].get("reward")],
        "statuses": [last[0].get("status"), last[1].get("status")],
        "n_steps": len(env.steps),
        "wall_s": wall,
    }
    if keep_steps:
        result["steps"] = env.steps
    return result


def _worker(job):
    """Multiprocessing worker: loads agents fresh in each process (module-level
    caches like main._STALL_MEMO stay isolated per game batch)."""
    path0, path1, keep_steps = job
    a0, d0, _ = load_agent(path0)
    a1, d1, _ = load_agent(path1)
    try:
        return play_game(a0, d0, a1, d1, keep_steps=keep_steps)
    except Exception as e:
        return {"error": repr(e)}


def run_matches(path0, path1, n_games, workers=None, keep_steps=False, progress=True):
    """Play n_games of path0 vs path1 (seat order fixed — caller alternates).
    Returns list of result dicts."""
    import multiprocessing as mp

    jobs = [(path0, path1, keep_steps)] * n_games
    results = []
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    if workers <= 1:
        for i, job in enumerate(jobs):
            results.append(_worker(job))
            if progress and (i + 1) % 10 == 0:
                print(f"  {i+1}/{n_games}", file=sys.stderr)
    else:
        with mp.Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker, jobs)):
                results.append(r)
                if progress and (i + 1) % 25 == 0:
                    print(f"  {i+1}/{n_games}", file=sys.stderr)
    return results


def summarize(results, name0="A", name1="B"):
    w = l = t = err = 0
    total_wall = 0.0
    for r in results:
        if "error" in r:
            err += 1
            continue
        total_wall += r["wall_s"]
        r0, r1 = r["rewards"][0], r["rewards"][1]
        # A crashed agent gets reward None while its opponent gets 1 (kaggle_environments
        # doesn't symmetrize this to -1/1) -- checking r0 alone silently miscounts those
        # as ties whenever the crash lands in slot 0. Confirmed via opponents/dragapult_agent.py,
        # which crashes on every local game (missing cg.api, Kaggle-dataset-only import) and
        # was showing up as ~50% ties instead of the real ~100% win rate.
        if r0 == 1 or r1 == -1:
            w += 1
        elif r0 == -1 or r1 == 1:
            l += 1
        else:
            t += 1
    n = w + l + t
    wr = (w + 0.5 * t) / n if n else 0.0
    return {
        "n": n, "errors": err,
        f"{name0}_wins": w, f"{name1}_wins": l, "ties": t,
        f"{name0}_winrate": round(wr, 4),
        "avg_game_s": round(total_wall / n, 2) if n else None,
    }
