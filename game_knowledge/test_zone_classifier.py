"""
Tests for game_knowledge/zone_classifier.py -- verifies the deterministic
polarized/neutral zone logic against the real, click-calibrated geometry
in config/red_zones.py.

Run: python -m game_knowledge.test_zone_classifier
"""

from game_knowledge.zone_classifier import classify_zone

TOWERS_ALL_ALIVE = {
    "ally_left": {"destroyed": False}, "ally_right": {"destroyed": False},
    "enemy_left": {"destroyed": False}, "enemy_right": {"destroyed": False},
}


def _with(**destroyed):
    state = {k: {"destroyed": v["destroyed"]} for k, v in TOWERS_ALL_ALIVE.items()}
    for name, is_destroyed in destroyed.items():
        state[name] = {"destroyed": is_destroyed}
    return state


def test_deep_ally_half_is_ally_regardless_of_towers():
    assert classify_zone(300, 1200, TOWERS_ALL_ALIVE) == "ally"
    print("test 1 OK: deep ally half classifies ally with all towers alive")


def test_deep_enemy_half_is_enemy_when_towers_alive():
    assert classify_zone(300, 300, TOWERS_ALL_ALIVE) == "enemy"
    print("test 2 OK: deep enemy half classifies enemy with all towers alive")


def test_river_band_is_unknown():
    assert classify_zone(300, 680, TOWERS_ALL_ALIVE) == "unknown"
    print("test 3 OK: river band position is unknown, not guessed")


def test_enemy_left_zone_is_neutral_only_once_that_tower_is_destroyed():
    point = (200, 550)  # inside ENEMY_LEFT_NEUTRAL_ZONE (65,501,450,711), above the river
    assert classify_zone(*point, TOWERS_ALL_ALIVE) == "enemy", "should be plain enemy territory while the tower stands"
    assert classify_zone(*point, _with(enemy_left={"destroyed": True})) == "neutral"
    print("test 4 OK: the enemy-left pocket is enemy-exclusive while alive, neutral once destroyed")


def test_enemy_right_zone_mirrors_correctly():
    point = (700, 550)  # inside the mirrored ENEMY_RIGHT_NEUTRAL_ZONE (450,501,835,711)
    assert classify_zone(*point, TOWERS_ALL_ALIVE) == "enemy"
    assert classify_zone(*point, _with(enemy_right={"destroyed": True})) == "neutral"
    # destroying the LEFT tower must not affect the right lane's classification
    assert classify_zone(*point, _with(enemy_left={"destroyed": True})) == "enemy"
    print("test 5 OK: the mirrored right-lane zone works independently of the left lane")


def test_ally_left_zone_is_neutral_only_once_that_tower_is_destroyed():
    point = (200, 780)  # inside ALLY_LEFT_NEUTRAL_ZONE (65,650,450,860), below the river
    assert classify_zone(*point, TOWERS_ALL_ALIVE) == "ally"
    assert classify_zone(*point, _with(ally_left={"destroyed": True})) == "neutral"
    print("test 6 OK: the ally-left pocket is ally-exclusive while alive, neutral once destroyed")


def test_neutral_zone_destruction_does_not_affect_the_other_side():
    """Destroying the enemy's tower must not make the ALLY's own symmetric
    pocket neutral, and vice versa -- each zone is gated by its own
    tower's state only."""
    ally_point = (200, 780)
    assert classify_zone(*ally_point, _with(enemy_left={"destroyed": True})) == "ally", \
        "enemy tower falling shouldn't open the ally's own pocket"
    print("test 7 OK: a lane's neutral zone is gated by its OWN tower only, not the opposite side's")


if __name__ == "__main__":
    test_deep_ally_half_is_ally_regardless_of_towers()
    test_deep_enemy_half_is_enemy_when_towers_alive()
    test_river_band_is_unknown()
    test_enemy_left_zone_is_neutral_only_once_that_tower_is_destroyed()
    test_enemy_right_zone_mirrors_correctly()
    test_ally_left_zone_is_neutral_only_once_that_tower_is_destroyed()
    test_neutral_zone_destruction_does_not_affect_the_other_side()
    print("\nALL TESTS PASSED")
