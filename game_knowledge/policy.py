"""Rule-based baseline policy -- Phase 5 step D2, v0.

Deliberately SIMPLE. Its job is to close the perception -> decision -> action loop
and start generating self-play data, NOT to play well:
  - if enemy troops are on OUR half, drop the cheapest affordable troop on the most
    advanced one (defend);
  - else, when elixir is high enough to not leak, push a troop up a bridge lane
    (alternating lanes);
  - otherwise wait.

Spells are excluded (a spell isn't a defender/pusher; spell targeting is its own
problem, left for later). Everything here is a knob a learned policy replaces --
this is the bootstrap, per Duncan's direction (don't over-invest in hand rules).
"""

from dataclasses import dataclass

from config.red_zones import ARENA_X_MAX, ARENA_X_MIN, RIVER_Y_BOTTOM

# Cards that aren't troops/buildings to "place on" a threat -- skip in the baseline.
SPELL_CARDS = {
    "fireball", "arrows", "zap", "the_log", "rocket", "lightning", "poison",
    "earthquake", "freeze", "rage", "clone", "tornado", "mirror", "giant_snowball",
    "goblin_barrel", "graveyard", "royal_delivery", "goblin_curse", "void",
    "barbarian_barrel",
}

LEFT_LANE_X, RIGHT_LANE_X = 228, 700   # ally/enemy tower lane centres (config tower geometry)
PUSH_Y = 725                            # just below the river, our side -> the troop walks up
MIN_DEPLOY_Y = RIVER_Y_BOTTOM + 12      # 723: first legal row on our half
MAX_DEPLOY_Y = 1300                     # above the hand tray (~1331)
PUSH_ELIXIR = 7                         # push only with a buffer, so we don't leak near 10


@dataclass
class Action:
    slot: int      # hand slot 0-3 to play
    x: int         # placement, arena pixels
    y: int
    card: str      # card slug (for record_own_play + logging)
    reason: str    # human-readable why, for the dry-run log


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _center(bbox):
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


class RuleBasedPolicy:
    def __init__(self):
        self._next_lane = 0  # alternate push lanes

    def decide(self, state):
        """Return an Action, or None to wait."""
        troops = [s for s in state.affordable() if s.card not in SPELL_CARDS]
        if not troops:
            return None
        pick = troops[0]  # cheapest affordable troop

        # 1) DEFEND: any enemy troop on our half -> drop a defender on the most advanced one.
        threats = [u for u in state.enemy_units() if _center(u["bbox"])[1] > RIVER_Y_BOTTOM]
        if threats:
            t = max(threats, key=lambda u: _center(u["bbox"])[1])  # closest to our towers
            tx, ty = _center(t["bbox"])
            return Action(pick.index,
                          int(_clamp(tx, ARENA_X_MIN + 20, ARENA_X_MAX - 20)),
                          int(_clamp(ty, MIN_DEPLOY_Y, MAX_DEPLOY_Y)),
                          pick.card, f"defend {t.get('name')}")

        # 2) PUSH: only when flush with elixir (avoid leaking), alternating lanes.
        if state.elixir >= PUSH_ELIXIR:
            lane_x = LEFT_LANE_X if self._next_lane == 0 else RIGHT_LANE_X
            self._next_lane ^= 1
            return Action(pick.index, lane_x, PUSH_Y, pick.card,
                          f"push {'left' if lane_x == LEFT_LANE_X else 'right'}")

        return None
