"""Local game harness for the cabt engine (ships inside kaggle_environments).

Runs full games locally (~0.5-1s each) — no Kaggle session needed. Used by
ab_test.py (A/B evaluation), gauntlet.py, and all training/nn collectors.
Works on any machine with `pip install kaggle_environments --no-deps`
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


def play_game(agent0, deck0, agent1, deck1, keep_steps=False, max_steps=None):
    """Run one full game. Returns dict with rewards, steps, wall time, and
    (optionally) the full step trace for BC collection / analysis.
    max_steps caps runaway games (cabt's default episodeSteps is 10M, so two
    passive agents are otherwise bounded only by deck-out); a capped game ends
    as a tie."""
    from kaggle_environments import make

    t0 = time.time()
    config = {"decks": [deck0, deck1]}
    if max_steps:
        config["episodeSteps"] = max_steps
    env = make("cabt", configuration=config)
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
    caches like main._STALL_MEMO stay isolated per game batch).

    job's optional 4th element (extra_env, a dict or None) is applied via
    os.environ.update() BEFORE loading either agent -- lets a caller assign
    a per-job identifier (e.g. a game_id an agent module can read at import
    time and embed in its own side-channel logging) without changing the
    job tuple shape for existing callers (extra_env defaults to None,
    meaning "no change," so every pre-existing 3-tuple job still works).

    job's optional 5th element (deck_override, a (deck0, deck1) tuple of
    list[int]-or-None) overrides load_agent's own DECK per side -- lets a
    caller vary the OPPONENT's deck per game (e.g. mcts_collect.py's
    opponent_pool.py sampling a different real-meta archetype deck per
    game for a deck-agnostic pilot like generic_pilot.py) without changing
    the job tuple shape for existing callers (defaults to None, meaning
    "use each agent's own module DECK," identical to prior behavior)."""
    path0, path1, keep_steps = job[0], job[1], job[2]
    extra_env = job[3] if len(job) > 3 else None
    deck_override = job[4] if len(job) > 4 else None
    if extra_env:
        os.environ.update(extra_env)
    a0, d0, _ = load_agent(path0)
    a1, d1, _ = load_agent(path1)
    if deck_override:
        d0 = deck_override[0] if deck_override[0] is not None else d0
        d1 = deck_override[1] if deck_override[1] is not None else d1
    try:
        result = play_game(a0, d0, a1, d1, keep_steps=keep_steps)
    except Exception as e:
        result = {"error": repr(e)}
    if extra_env:
        # imap_unordered returns results in completion order, not submission
        # order -- echo the job's own extra_env back so a caller using it to
        # tag jobs (e.g. a game_id) can tell which job a given result is for.
        result["extra_env"] = extra_env
    return result


def run_matches(path0, path1, n_games, workers=None, keep_steps=False, progress=True,
                 extra_envs=None, decks=None):
    """Play n_games of path0 vs path1 (seat order fixed — caller alternates).
    Returns list of result dicts.

    extra_envs: optional list of length n_games, each a dict of env vars (or
    None) applied inside that specific job's worker process before agents
    load -- e.g. mcts_collect.py uses this to assign a unique per-game
    MCTS_GAME_ID so a search-based collect log can be correlated to game
    outcomes correctly even when workers>1 interleaves multiple games'
    decisions across processes. None (default) preserves prior behavior
    exactly for every other caller.

    decks: optional list of length n_games, each a (deck0, deck1) tuple of
    list[int]-or-None (or None for "no override this game") -- overrides
    load_agent's own DECK per side, e.g. for a deck-agnostic pilot playing a
    specific real-meta archetype deck sampled by opponent_pool.py. None
    (default) preserves prior behavior exactly for every other caller."""
    import multiprocessing as mp

    if extra_envs is None:
        extra_envs = [None] * n_games
    if decks is None:
        decks = [None] * n_games
    jobs = [(path0, path1, keep_steps, extra_envs[i], decks[i]) for i in range(n_games)]
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
