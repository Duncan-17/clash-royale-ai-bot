"""Clean board-state representation -- Phase 5 step D1.

Assembles the RELIABLE perception outputs into one structured object the policy
reasons over. Deliberately only DIRECT, trustworthy reads (Duncan 2026-09-10 --
don't hardcode fragile vision-derived inference like opponent elixir / card cycle):
  - OUR elixir + hand: direct pixel reads (elixir_counter / card_detector).
  - units: the detector -> thin_tracker tracks (side / class / hp / level / bbox).
  - towers: alive + hp per slot.
No opponent-elixir belief, no cycle reconstruction -- the learned policy develops
that later.

Card elixir costs come from config/cards.json (the official roster). Card names in
the hand are the project's slugs (from assets/hand_templates/); cards.json uses
display names, mapped to the same slug via _slugify + card_detector.canonical_slug.
"""

import json
import os
from dataclasses import dataclass, field

CARDS_JSON = "config/cards.json"
MIN_HAND_CONF = 0.55  # below this a hand slot reads "unknown" (don't act on a guess)


def _slugify(name: str) -> str:
    """'Mini P.E.K.K.A' -> 'mini_pekka', 'X-Bow' -> 'x-bow', 'Goblin Gang' ->
    'goblin_gang'. Matches assets/hand_templates/ slugs."""
    from perception.card_detector import canonical_slug
    return canonical_slug(name.strip().lower().replace(".", "").replace(" ", "_"))


_cost_cache = None


def card_costs() -> dict:
    """{card_slug: elixir_cost}. Loaded once from config/cards.json."""
    global _cost_cache
    if _cost_cache is None:
        _cost_cache = {}
        if os.path.isfile(CARDS_JSON):
            with open(CARDS_JSON, encoding="utf-8") as f:
                for c in json.load(f):
                    cost = c.get("elixirCost")
                    if cost is not None:
                        _cost_cache[_slugify(c["name"])] = cost
    return _cost_cache


@dataclass
class HandSlot:
    index: int          # 0-3, left to right
    card: str           # card slug, or "unknown"
    confidence: float
    cost: int = None    # elixir cost, or None if unknown card / no cost data

    @property
    def known(self) -> bool:
        return self.card != "unknown" and self.confidence >= MIN_HAND_CONF


@dataclass
class GameState:
    elixir: float                  # current elixir (int value from the number read)
    elixir_fraction: float         # 0-1 progress toward the next point
    hand: list                     # 4 HandSlot
    units: list = field(default_factory=list)   # tracked unit dicts (INDIVIDUALS)
    groups: list = field(default_factory=list)  # swarm-grouped view (blobs + non-swarm
    # units): a Skeleton Army reads as one 'skeleton' entity with count. Non-lossy --
    # `units` still holds every individual; the policy uses whichever view fits.
    towers: dict = field(default_factory=dict)  # name -> {"alive": bool, "hp": float|None}

    def enemy_units(self):
        return [u for u in self.units if u.get("side") == "enemy"]

    def ally_units(self):
        return [u for u in self.units if u.get("side") == "ally"]

    def enemy_groups(self):
        return [u for u in self.groups if u.get("side") == "enemy"]

    def ally_groups(self):
        return [u for u in self.groups if u.get("side") == "ally"]

    def affordable(self):
        """Known hand cards we can pay for right now, cheapest first."""
        aff = [s for s in self.hand if s.known and s.cost is not None and s.cost <= self.elixir]
        return sorted(aff, key=lambda s: s.cost)


def build_game_state(frame, tracked_units, tower_info=None,
                     hand_templates=None, elixir_templates=None) -> GameState:
    """Assemble a GameState from one frame + the tracker's current units.

    frame           : BGR screenshot (for the elixir + hand reads)
    tracked_units   : list of unit dicts from thin_tracker (already side-resolved)
    tower_info      : {name: {"alive", "hp"}} from detector_view.TowerPanel-style logic
    *_templates     : preloaded template dicts (pass them in a loop -- don't reload/frame)
    """
    from perception.card_detector import identify_hand, load_templates as load_hand
    from perception.elixir_counter import read_elixir, load_templates as load_elixir
    from perception.swarm_grouper import group_swarms

    if hand_templates is None:
        hand_templates = load_hand()
    if elixir_templates is None:
        elixir_templates = load_elixir()

    el = read_elixir(frame, elixir_templates)
    costs = card_costs()
    hand = []
    for i, (card, conf) in enumerate(identify_hand(frame, hand_templates)):
        hand.append(HandSlot(index=i, card=card, confidence=conf, cost=costs.get(card)))

    return GameState(
        elixir=el["value"],
        elixir_fraction=el["fraction"],
        hand=hand,
        units=list(tracked_units),
        groups=group_swarms(tracked_units),   # swarm blobs + non-swarm units (see field note)
        towers=dict(tower_info or {}),
    )
