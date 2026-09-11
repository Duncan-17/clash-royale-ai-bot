"""Tests for the Phase 5 baseline: game_state assembly helpers + RuleBasedPolicy.

Run: python -m game_knowledge.test_policy
"""

from game_knowledge.game_state import GameState, HandSlot, card_costs
from game_knowledge.policy import RuleBasedPolicy, PUSH_ELIXIR, LEFT_LANE_X, RIGHT_LANE_X

C = card_costs()


def _hand(*cards):
    return [HandSlot(i, name, 0.9, C.get(name)) for i, name in enumerate(cards)]


def _enemy(name, cx, cy):
    return {"id": 1, "name": name, "side": "enemy", "bbox": (cx - 20, cy - 30, cx + 20, cy + 30)}


def main():
    # cost data loaded and sane
    assert C.get("knight") == 3 and C.get("goblins") == 2 and C.get("pekka") == 7
    print("test 1 OK: card costs load from cards.json")

    # affordable() = known, priced, cost<=elixir, cheapest first
    s = GameState(4, 0.0, _hand("knight", "pekka", "goblins", "musketeer"), [])
    aff = [x.card for x in s.affordable()]
    assert aff[0] == "goblins" and "pekka" not in aff, aff  # pekka(7) unaffordable at 4
    print("test 2 OK: affordable() filters by cost, cheapest first")

    pol = RuleBasedPolicy()
    # DEFEND: enemy troop on our half -> place cheapest troop on the most advanced threat
    far = _enemy("giant", 300, 760)
    near = _enemy("hog_rider", 700, 950)  # more advanced (bigger y)
    a = pol.decide(GameState(4, 0.0, _hand("knight", "fireball", "goblins", "musketeer"), [far, near]))
    assert a and a.card == "goblins" and a.reason.startswith("defend") and a.y >= 723
    assert abs(a.x - 700) < 30, "should defend the most-advanced (right-lane) threat"
    print("test 3 OK: defends the most advanced threat with the cheapest troop")

    # spells are never chosen as the placement card
    a = pol.decide(GameState(10, 0.0, _hand("fireball", "rocket", "zap", "arrows"), []))
    assert a is None
    print("test 4 OK: spell-only hand -> no play (spells excluded)")

    # PUSH only when flush, and lanes alternate
    quiet = GameState(PUSH_ELIXIR, 0.0, _hand("knight", "fireball", "goblins", "musketeer"), [])
    a1 = pol.decide(quiet)
    a2 = pol.decide(GameState(PUSH_ELIXIR, 0.0, _hand("knight", "fireball", "goblins", "musketeer"), []))
    assert a1.reason.startswith("push") and a2.reason.startswith("push")
    assert {a1.x, a2.x} == {LEFT_LANE_X, RIGHT_LANE_X}, "push lanes must alternate"
    print("test 5 OK: pushes only when flush, alternating lanes")

    # below PUSH_ELIXIR with no threat -> wait
    assert pol.decide(GameState(PUSH_ELIXIR - 1, 0.0, _hand("knight", "fireball", "goblins", "musketeer"), [])) is None
    print("test 6 OK: low elixir + no threat -> wait")

    # an UNKNOWN / low-confidence hand is never played
    lowconf = [HandSlot(i, "knight", 0.2, 3) for i in range(4)]
    assert pol.decide(GameState(10, 0.0, lowconf, [])) is None
    print("test 7 OK: low-confidence hand slots are not played")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
