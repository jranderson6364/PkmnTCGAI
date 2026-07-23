# Public competition agents — offline sparring partners

Third-party agents extracted from public Apache-2.0 Kaggle notebooks by
`tools/extract_public_agents.py`, each with its **published live ladder score**.

**These are not ours and are never submitted, in whole or in part.** They exist
so we can (a) play against opponents that actually beat our champion — every
reference anchor in `opponents/` reads <=6% against v29d — and (b) calibrate
offline strength against real publicScore (`training/calibrate_panel.py`).

Attribution and source URL are in the header of every file. Only local-path and
Kaggle-path shims were added; decision logic is untouched.

| file | publicScore | archetype |
|---|---|---|
| `probability_v2.py` | 933.8 | Mega Lucario ex heuristic |
| `advanced_heuristic.py` | 796.8 | Expectimax + UCB1 heuristic |
| `alakazam_v9.py` | 778.2 | Search-audited Alakazam |
| `alakazam_v8.py` | 739.7 | Field-audited Alakazam |

Regenerate with:

```
python tools/extract_public_agents.py --kernels-dir <dir of pulled .ipynb>
```
