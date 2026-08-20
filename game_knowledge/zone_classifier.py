"""
Per-lane deterministic zone classification -- the "spawn location
legality" system from Duncan's "perception fusion architecture" design
doc (2026-08-19/20; full doc content is in conversation history, not
reproduced here). Given known princess-tower alive/destroyed state
(perception/tower_reader.py's TowerStateTracker), the board splits into
exactly 3 zone types per lane -- and in two of them, a spawn's location
is DETERMINISTIC PROOF of team, not corroborating evidence to be weighed
against other signals:

- ALLY-only zone: only the ally player can legally have a unit here right
  now. A unit detected here IS the ally's, with the same certainty as any
  other hard-coded game rule.
- ENEMY-only zone: symmetric, deterministic proof of the enemy.
- NEUTRAL zone: opens once a princess tower on that lane is destroyed --
  confirmed directly by Duncan (2026-08-20, real gameplay explanation):
  the specific pocket where a destroyed princess tower stood becomes
  legally placeable by BOTH players, not just whoever destroyed it.
  Location alone cannot disambiguate team here -- this is the ONLY case
  that should fall through to other evidence (badge, spawn icon).

This is what perception/unit_tracker.py's own _side_from_position was
always meant to be corrected into once tower state existed (see that
function's own docstring for the gap it explicitly flagged) -- this
module is the real fix, not an addition alongside it.

Geometry is all real (config/red_zones.py, click-calibrated against a
real capture) except the ally-side neutral zones, which are DERIVED by
vertical symmetry from the one real (enemy-side) measurement, not
independently confirmed -- see config/red_zones.py's own comment.
"""

from config.red_zones import (
    ALLY_LEFT_NEUTRAL_ZONE,
    ALLY_RIGHT_NEUTRAL_ZONE,
    BOARD_CENTER_X,
    ENEMY_LEFT_NEUTRAL_ZONE,
    ENEMY_RIGHT_NEUTRAL_ZONE,
    RIVER_Y_BOTTOM,
    RIVER_Y_TOP,
    point_in_zone,
)


def _lane(x: float) -> str:
    return "left" if x < BOARD_CENTER_X else "right"


def classify_zone(x: float, y: float, tower_state: dict) -> str:
    """Returns "ally" / "enemy" / "neutral" / "unknown" for board position
    (x, y), given `tower_state` (perception.tower_reader.TowerStateTracker
    .update()'s own return dict -- only the 4 princess towers' "destroyed"
    flags are used; king towers don't affect placement legality).

    "unknown" covers positions inside the river band itself (no zone,
    ground troops literally cannot stand there) as well as anywhere no
    rule below applies -- deliberately conservative, same "err toward
    unknown rather than guess wrong" precedent perception/unit_tracker.py's
    original _side_from_position already established.

    Ordering matters: neutral-zone membership is checked BEFORE the plain
    river split, since a neutral pocket can sit entirely on what would
    otherwise read as the opponent's exclusive half."""
    lane = _lane(x)
    enemy_zone = ENEMY_LEFT_NEUTRAL_ZONE if lane == "left" else ENEMY_RIGHT_NEUTRAL_ZONE
    ally_zone = ALLY_LEFT_NEUTRAL_ZONE if lane == "left" else ALLY_RIGHT_NEUTRAL_ZONE

    enemy_tower_destroyed = tower_state.get(f"enemy_{lane}", {}).get("destroyed", False)
    ally_tower_destroyed = tower_state.get(f"ally_{lane}", {}).get("destroyed", False)

    if enemy_tower_destroyed and point_in_zone(x, y, enemy_zone):
        return "neutral"
    if ally_tower_destroyed and point_in_zone(x, y, ally_zone):
        return "neutral"

    if y > RIVER_Y_BOTTOM:
        return "ally"
    if y < RIVER_Y_TOP:
        return "enemy"
    return "unknown"
