"""Tests for perception/swarm_grouper.py -- collapsing same-class+side swarm
clusters into one group entity (a Skeleton Army as 'skeleton x8', etc.).

Run: python -m perception.test_swarm_grouper
"""

from perception.swarm_grouper import group_swarms


def _u(tid, name, cx, cy, side="enemy", w=34, h=44, hp=None, level=None):
    return {"id": tid, "name": name, "side": side, "confidence": 0.8,
            "hp": hp, "level": level,
            "bbox": (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)}


def _find_group(out, name, side):
    return next((u for u in out if u.get("group") and u["name"] == name
                and u["side"] == side), None)


def _packed(name, cx, cy, side, n, step=30, **kw):
    """n same-class units packed in a row starting at (cx,cy), ~one box apart."""
    return [_u(1000 + i, name, cx + i * step, cy, side=side, **kw) for i in range(n)]


def main():
    # 1) a Skeleton Army: 8 packed enemy skeletons -> one group, count 8, side enemy.
    out = group_swarms(_packed("skeleton", 300, 500, "enemy", 8))
    g = _find_group(out, "skeleton", "enemy")
    assert g is not None and g["count"] == 8, out
    assert len([u for u in out if u["name"] == "skeleton"]) == 1, "collapsed to one entity"
    # enclosing bbox spans all members
    assert g["bbox"][0] <= 300 - 17 and g["bbox"][2] >= 300 + 7 * 30 + 17
    print("test 1 OK: a packed skeleton swarm -> one 'skeleton' group with count")

    # 2) two separate clusters (two lanes) -> two groups, not one.
    left = _packed("goblin", 200, 400, "enemy", 4)
    right = _packed("goblin", 700, 400, "enemy", 3)
    out = group_swarms(left + right)
    goblin_groups = [u for u in out if u.get("group") and u["name"] == "goblin"]
    assert len(goblin_groups) == 2, f"two lanes -> two groups, got {len(goblin_groups)}"
    assert sorted(g["count"] for g in goblin_groups) == [3, 4]
    print("test 2 OK: two separated clusters stay two groups")

    # 3) a lone swarm unit stays an individual (below MIN_GROUP), untouched.
    lone = _u(7, "skeleton", 800, 900, "ally")
    out = group_swarms([lone])
    assert out == [lone], "a single skeleton passes through unchanged"
    print("test 3 OK: a lone swarm unit is left as an individual")

    # 4) non-swarm classes are NEVER grouped, even two adjacent ones.
    out = group_swarms([_u(1, "giant", 400, 500, "enemy"), _u(2, "giant", 420, 505, "enemy")])
    assert all(not u.get("group") for u in out) and len(out) == 2
    print("test 4 OK: non-swarm classes (two giants) never merged")

    # 5) different SIDES don't merge (enemy vs ally skeletons meeting at the bridge).
    out = group_swarms(_packed("skeleton", 400, 780, "enemy", 3)
                       + _packed("skeleton", 400, 800, "ally", 3))
    assert _find_group(out, "skeleton", "enemy") and _find_group(out, "skeleton", "ally")
    print("test 5 OK: enemy and ally swarms stay separate groups")

    # 6) mixed frame: a skeleton swarm + a knight nearby -> swarm groups, knight intact.
    knight = _u(50, "knight", 320, 520, "enemy")
    out = group_swarms(_packed("skeleton", 300, 500, "enemy", 5) + [knight])
    assert _find_group(out, "skeleton", "enemy")["count"] == 5
    assert knight in out, "the knight passes through untouched"
    print("test 6 OK: swarm grouped, the lone non-swarm unit passes through")

    # 7) aggregates: count, union bbox, mean hp, stable-ish id = min member id.
    members = _packed("bat", 500, 300, "enemy", 4, hp=0.5, level=3)
    members[2]["hp"] = 1.0
    out = group_swarms(members)
    g = _find_group(out, "bat", "enemy")
    assert g["count"] == 4 and g["id"] == 1000            # min of ids 1000..1003
    assert abs(g["hp"] - (0.5 * 3 + 1.0) / 4) < 1e-6      # mean of member hps
    assert g["level"] == 3 and set(g["member_ids"]) == {1000, 1001, 1002, 1003}
    print("test 7 OK: group aggregates count / union bbox / mean hp / min-id")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
