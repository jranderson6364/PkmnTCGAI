"""Belief-weighted determinization sampler (Phase C consumer 2,
docs/belief-model.md).

Given OUR observation of a live decision, produce a plausible full-state
sample of every hidden zone, suitable for `search_begin(...)`:

  - OUR hidden zones exactly: our decklist is known (main.DECK), so the
    unseen remainder (deck order + face-down prizes) is the true multiset,
    dealt randomly — not mirror filler.
  - OPPONENT hidden zones by belief: archetype sampled from main.py's
    shipped `_belief_posterior` (5-class softmax + the v28-calibrated 0.97
    confidence threshold + the crustle-line override), that archetype's
    60-card list minus everything publicly observed of theirs, dealt into
    hand / prizes / deck / face-down active. Unknown/low-confidence reads
    fall back to the pre-existing placeholder behavior (mirror-deck filler),
    matching the honest-`unknown` rule in docs/belief-model.md.

Deck lists: exact for lucario/dragapult/abomasnow/starmie
(opponents/*_agent.py DECK) and alakazam (main.DECK); reconstructed from
real replay evidence for the rest (training/archetype_decks.json), padded
to 60 with FILLER_ID where evidence is thin (e.g. kyogre's honest 30/60).

Consumers: ISMCTS-style search (fresh sample() per simulation — sampling
once and reusing across sims is the strategy-fusion bug already documented
in training/nn/mcts.py's postmortem, don't reintroduce it).
"""
import importlib
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "opponents")):
    if p not in sys.path:
        sys.path.insert(0, p)

import main as heuristic  # noqa: E402  (shipped agent: _belief_posterior + DECK)

# The placeholder id search_begin has always accepted as hidden-zone filler.
FILLER_ID = 1072

# posterior class -> exact-decklist module (opponents/) or None for main.DECK
_EXACT_SOURCES = {
    "lucario": "lucario_agent",
    "dragapult": "dragapult_agent",
    "abomasnow": "abomasnow_agent",
    "starmie": "starmie_agent",
    "alakazam": None,
}


def _load_decklists():
    decks = {}
    for name, mod_name in _EXACT_SOURCES.items():
        if mod_name is None:
            decks[name] = list(heuristic.DECK)
        else:
            try:
                decks[name] = list(importlib.import_module(mod_name).DECK)
            except Exception:
                pass  # missing bot module: archetype falls back to filler
    try:
        with open(os.path.join(REPO_ROOT, "training", "archetype_decks.json")) as f:
            recon = json.load(f)
        for name, cards in recon.items():
            if name in decks:  # exact list wins over reconstruction
                continue
            flat = []
            for c in cards:
                flat.extend([c["cardId"]] * int(c.get("copies", 1)))
            flat = flat[:60] + [FILLER_ID] * max(0, 60 - len(flat))
            decks[name] = flat
    except Exception:
        pass
    return decks


def _observed_opponent_cards(opp):
    """Multiset of every opponent card publicly out of their hidden zones:
    board pokemon + preEvolution chain + tools + attached energy cards,
    discard pile, and any face-up prize. Mirrors the visibility rules
    main._belief_posterior relies on."""
    seen = []
    for pk in list(opp.get("active") or []) + list(opp.get("bench") or []):
        if not pk:
            continue
        if pk.get("id"):
            seen.append(pk["id"])
        for pre in pk.get("preEvolution") or []:
            if (pre or {}).get("id"):
                seen.append(pre["id"])
        for t in pk.get("tools") or []:
            if (t or {}).get("id"):
                seen.append(t["id"])
        for ec in pk.get("energyCards") or []:
            if (ec or {}).get("id"):
                seen.append(ec["id"])
    for c in opp.get("discard") or []:
        if (c or {}).get("id"):
            seen.append(c["id"])
    for c in opp.get("prize") or []:
        if (c or {}).get("id"):
            seen.append(c["id"])
    return seen


def _remove_multiset(pool, remove):
    pool = list(pool)
    for cid in remove:
        try:
            pool.remove(cid)
        except ValueError:
            pass  # observed card not in the (possibly reconstructed) list
    return pool


class BeliefDeterminizer:
    def __init__(self, conf_threshold=0.97):
        self.conf_threshold = conf_threshold
        self.decks = _load_decklists()

    def _pick_archetype(self, opp, turn, rng):
        """Returns (decklist or None, label). None decklist = fall back to
        placeholder filler (honest-unknown rule)."""
        post, _wall, crustle_seen = heuristic._belief_posterior(opp, turn)
        if crustle_seen and "crustle" in self.decks:
            return self.decks["crustle"], "crustle"
        if not post:
            return None, "unknown"
        conf = max(post.values())
        if conf < self.conf_threshold:
            return None, "unknown"
        label = rng.choices(list(post.keys()), weights=list(post.values()))[0]
        return self.decks.get(label), label

    def sample(self, obs_dict, our_seat, rng=None):
        """One fresh determinization of every hidden zone. Returns a dict
        whose first six values are the positional args search_begin expects
        after `observation`: your_deck, your_prize, opponent_deck,
        opponent_prize, opponent_hand, opponent_active — plus archetype
        metadata for logging/ablation."""
        rng = rng or random
        cur = obs_dict["current"]
        me = cur["players"][our_seat]
        opp = cur["players"][1 - our_seat]
        turn = cur.get("turn") or 0

        # --- our side: exact unseen multiset from our known decklist ---
        my_visible = []
        for pk in list(me.get("active") or []) + list(me.get("bench") or []):
            if not pk:
                continue
            if pk.get("id"):
                my_visible.append(pk["id"])
            for pre in pk.get("preEvolution") or []:
                if (pre or {}).get("id"):
                    my_visible.append(pre["id"])
            for t in pk.get("tools") or []:
                if (t or {}).get("id"):
                    my_visible.append(t["id"])
            for ec in pk.get("energyCards") or []:
                if (ec or {}).get("id"):
                    my_visible.append(ec["id"])
        for c in me.get("discard") or []:
            if (c or {}).get("id"):
                my_visible.append(c["id"])
        for c in me.get("hand") or []:
            if (c or {}).get("id"):
                my_visible.append(c["id"])
        for c in me.get("prize") or []:
            if (c or {}).get("id"):
                my_visible.append(c["id"])
        my_unseen = _remove_multiset(heuristic.DECK, my_visible)
        rng.shuffle(my_unseen)
        my_prize_slots = list(me.get("prize") or [])
        your_deck = self._deal(my_unseen, me.get("deckCount") or 0, rng)
        your_prize = [
            (slot or {}).get("id") or self._deal(my_unseen, 1, rng)[0]
            for slot in my_prize_slots
        ]

        # --- opponent side: belief-sampled archetype list, else filler ---
        decklist, label = self._pick_archetype(opp, turn, rng)
        opp_prize_slots = list(opp.get("prize") or [])
        opp_active_facedown = bool(opp.get("active")) and opp["active"][0] is None
        need = ((opp.get("deckCount") or 0) + (opp.get("handCount") or 0)
                + sum(1 for s in opp_prize_slots if s is None)
                + (1 if opp_active_facedown else 0))
        if decklist is None:
            opp_unseen = [FILLER_ID] * need
        else:
            opp_unseen = _remove_multiset(decklist, _observed_opponent_cards(opp))
            rng.shuffle(opp_unseen)
        opponent_hand = self._deal(opp_unseen, opp.get("handCount") or 0, rng)
        opponent_prize = [
            (slot or {}).get("id") or self._deal(opp_unseen, 1, rng)[0]
            for slot in opp_prize_slots
        ]
        opponent_active = self._deal(opp_unseen, 1, rng) if opp_active_facedown else []
        opponent_deck = self._deal(opp_unseen, opp.get("deckCount") or 0, rng)

        return {
            "your_deck": your_deck, "your_prize": your_prize,
            "opponent_deck": opponent_deck, "opponent_prize": opponent_prize,
            "opponent_hand": opponent_hand, "opponent_active": opponent_active,
            "archetype": label,
        }

    @staticmethod
    def _deal(pool, n, rng):
        """Pop n cards from the shuffled pool, padding with FILLER_ID if the
        (possibly reconstructed/imperfect) list runs short of the engine's
        required counts."""
        out = []
        for _ in range(n):
            out.append(pool.pop() if pool else FILLER_ID)
        return out
