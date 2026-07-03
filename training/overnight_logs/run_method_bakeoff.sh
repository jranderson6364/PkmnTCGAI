#!/bin/sh
# Method bake-off (pre-registered docs/report-log.md 2026-07-03):
# every row through the same gauntlet.py, full 8-anchor panel (incl. repaired
# dragapult), 200 games/anchor, one recorded run id per row.
cd "C:/Users/jande/Downloads/Projects/PkmnTCGAI"
SEED=method-run1
G=200
W=10
python training/gauntlet.py --candidate training/random_agent.py  --name method-random         --games $G --workers $W --seed $SEED
python training/gauntlet.py --candidate training/generic_pilot.py --name method-generic-greedy --games $G --workers $W --seed $SEED
python training/gauntlet.py --candidate main.py                   --name method-heuristic-v25c --games $G --workers $W --seed $SEED
NET_CKPT=C:/Users/jande/Downloads/Projects/PkmnTCGAI/training/ptcg_bc_v2.pth     python training/gauntlet.py --candidate training/nn/net_agent.py --name method-bc-v2      --games $G --workers $W --seed $SEED
NET_CKPT=C:/Users/jande/Downloads/Projects/PkmnTCGAI/training/ptcg_dagger_r2.pth python training/gauntlet.py --candidate training/nn/net_agent.py --name method-dagger-r2  --games $G --workers $W --seed $SEED
echo "METHOD BAKE-OFF COMPLETE"
