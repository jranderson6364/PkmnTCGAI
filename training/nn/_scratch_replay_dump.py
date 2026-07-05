"""One-off: re-run net-exploiter vs main.py with keep_steps=True, dump winning
games (net wins) as JSON compatible with tools/analyze_replay.py, with
team_names set so 'Jason Anderson' == main.py's seat (the side we want to
read for piloting mistakes)."""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "training"))
sys.path.insert(0, REPO_ROOT)

from harness import run_matches  # noqa: E402

NET_AGENT = os.path.join(_HERE, "selfplay_agent.py")
MAIN = os.path.join(REPO_ROOT, "main.py")
OUT_DIR = os.path.join(REPO_ROOT, "replays", "exploiter_wins")
os.makedirs(OUT_DIR, exist_ok=True)

os.environ["NET_CKPT"] = os.path.abspath(os.path.join(REPO_ROOT, "training", "ptcg_exploiter_r1.pth"))
os.environ["NET_TEMP"] = "1.0"

def main():
    BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    WANT_WINS = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    found = 0
    batch_num = 0
    while found < WANT_WINS and batch_num < 15:
        batch_num += 1
        # alternate net seat each batch
        net_seat = batch_num % 2
        paths = (NET_AGENT, MAIN) if net_seat == 0 else (MAIN, NET_AGENT)
        results = run_matches(paths[0], paths[1], BATCH, workers=6, keep_steps=True, progress=False)
        for gi, r in enumerate(results):
            if "error" in r or "steps" not in r:
                continue
            outcome = r["rewards"][net_seat]
            if outcome != 1:
                continue
            main_seat = 1 - net_seat
            team_names = [None, None]
            team_names[main_seat] = "Jason Anderson"
            team_names[net_seat] = "NetExploiter"
            doc = {
                "info": {"TeamNames": team_names},
                "rewards": r["rewards"],
                "steps": r["steps"],
            }
            found += 1
            outpath = os.path.join(OUT_DIR, f"win_{found:03d}_b{batch_num}g{gi}.json")
            with open(outpath, "w", encoding="utf-8") as f:
                json.dump(doc, f)
            print(f"wrote {outpath}", file=sys.stderr)
            if found >= WANT_WINS:
                break
        print(f"batch {batch_num}: {found}/{WANT_WINS} wins found so far", file=sys.stderr)

    print(f"DONE found={found}")


if __name__ == "__main__":
    main()
