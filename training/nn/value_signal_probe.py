"""Value-signal probe (2026-07-04, per advisor guidance): the load-bearing
question under every remaining "beat v25c" path. AWR's value head saturated
near +-1 on self-play (mirror) data, and the PIMC rollout's simulated
mirror-opponent won 90/90 — the SAME underlying fact (mirror games are
near-deterministic, so nothing ever taught the value head to discriminate
contested states). This probe checks whether that's a genuine property of
the GAME, or an artifact of only ever training on near-decisive mirror data:
does the value head show real spread on CONTESTED states from games against
genuinely different opponents (the 4 anchor bots), bucketed by whether that
game's outcome was close or a blowout?

- Saturated even on contested cross-opponent states -> the game is close to
  decisive from most states; no value-based method has headroom here.
- Real spread on contested states -> the saturation was a self-play-data
  artifact; retraining the value head on diverse/contested data (anchor
  games, or the 680 real ladder replays) is the untried, cheap fix.

Usage: python training/nn/value_signal_probe.py --ckpt ../ptcg_dagger_r2.pth --games 12
"""
import argparse
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "training"))

from net_common import load_model, value_estimate  # noqa: E402
from harness import load_agent, play_game  # noqa: E402

ANCHORS = ["opponents/lucario_agent.py", "opponents/dragapult_agent.py",
           "opponents/abomasnow_agent.py", "opponents/starmie_agent.py"]


def our_decisions(steps, our_seat, sample_every=6):
    """Every `sample_every`-th real decision point for our seat (main-phase
    select, so states are comparable across games)."""
    out = []
    for i in range(1, len(steps)):
        prev = steps[i - 1][our_seat].get("observation", {})
        act = steps[i][our_seat].get("action")
        sel = prev.get("select")
        if sel is None or act is None or not sel.get("option"):
            continue
        if sel.get("type") != 0:
            continue
        out.append(prev)
    return out[::sample_every]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_REPO_ROOT, "training", "ptcg_dagger_r2.pth"))
    ap.add_argument("--games", type=int, default=12, help="games per anchor")
    args = ap.parse_args()

    model = load_model(args.ckpt)
    main_agent, main_deck, _ = load_agent(os.path.join(_REPO_ROOT, "main.py"))

    game_records = []  # (n_decisions, our_result, sampled_values)
    for anchor_path in ANCHORS:
        opp_agent, opp_deck, _ = load_agent(os.path.join(_REPO_ROOT, anchor_path))
        for g in range(args.games):
            our_seat = g % 2
            if our_seat == 0:
                result = play_game(main_agent, main_deck, opp_agent, opp_deck, keep_steps=True)
            else:
                result = play_game(opp_agent, opp_deck, main_agent, main_deck, keep_steps=True)
            steps = result["steps"]
            decs = our_decisions(steps, our_seat)
            values = []
            for d in decs:
                sel = d["select"]
                try:
                    v = value_estimate(model, d, sel)
                    values.append(v)
                except Exception:
                    continue
            reward = result["rewards"][our_seat]
            game_records.append({
                "anchor": anchor_path, "n_steps": result["n_steps"],
                "our_reward": reward, "values": values,
            })
            print(f"{anchor_path:<32} seat={our_seat} n_steps={result['n_steps']:>4} "
                  f"reward={reward} n_sampled={len(values)}")

    lengths = [r["n_steps"] for r in game_records]
    median_len = statistics.median(lengths)
    print(f"\nmedian game length (n_steps): {median_len}")

    close_vals, blowout_vals, all_vals = [], [], []
    for r in game_records:
        all_vals.extend(r["values"])
        if r["n_steps"] >= median_len:
            close_vals.extend(r["values"])
        else:
            blowout_vals.extend(r["values"])

    def report(label, vals):
        if not vals:
            print(f"{label}: no samples")
            return
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        p25 = vals_sorted[int(0.25 * n)]
        p50 = vals_sorted[int(0.50 * n)]
        p75 = vals_sorted[int(0.75 * n)]
        print(f"{label}: n={n} mean={statistics.mean(vals):.3f} "
              f"std={statistics.pstdev(vals):.3f} p25={p25:.3f} p50={p50:.3f} p75={p75:.3f} "
              f"frac_extreme(|v|>0.95)={sum(1 for v in vals if abs(v) > 0.95) / n:.1%}")

    print()
    report("ALL states (vs 4 diverse anchors)", all_vals)
    report("LONGER-than-median games (more contested)", close_vals)
    report("SHORTER-than-median games (more decisive)", blowout_vals)


if __name__ == "__main__":
    main()
