"""Tests for game_knowledge/play_tracker.py -- own-play ground truth + opponent
play detection + byproduct filtering + unit->card mapping.

Run: python -m game_knowledge.test_play_tracker
"""

from game_knowledge.play_tracker import PlayTracker, unit_to_card


def _t(tid, name, side, cx, cy, w=40, h=60):
    return {"id": tid, "name": name, "side": side,
            "bbox": (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)}


def main():
    # 1) unit -> card mapping (mismatches mapped, 1:1 identity)
    assert unit_to_card("spear_goblin") == "spear_goblins"
    assert unit_to_card("goblin") == "goblins"
    assert unit_to_card("knight") == "knight"          # identity fallback
    assert unit_to_card("mini_pekka") == "mini_pekka"  # identity fallback
    print("test 1 OK: unit->card maps plurals, leaves 1:1 names alone")

    # 2) a new ENEMY track is an opponent play, mapped to a card, logged ONCE
    pt = PlayTracker()
    evs = pt.update([_t(1, "goblin", "enemy", 400, 300)], now=100.0)
    assert [e.kind for e in evs] == ["opponent"] and evs[0].card == "goblins"
    evs2 = pt.update([_t(1, "goblin", "enemy", 405, 305)], now=100.1)  # same track next frame
    assert evs2 == [] and len(pt.opponent_plays) == 1, "opponent play must log once per track"
    print("test 2 OK: new enemy track -> one opponent play (card-mapped)")

    # 3) OWN play binds a matching new ally track: authoritative ally + card, no opp log
    pt = PlayTracker()
    pt.record_own_play("giant", (500, 700), now=10.0)
    track = _t(7, "giant", "unknown", 508, 712)       # appears near placement, side not yet known
    evs = pt.update([track], now=11.0)
    assert [e.kind for e in evs] == ["own"] and evs[0].card == "giant"
    assert track["side"] == "ally" and track["side_source"] == "own_play" and track["card"] == "giant"
    assert len(pt.opponent_plays) == 0
    print("test 3 OK: own play binds a new ally track -> authoritative ally + card")

    # 4) a bound (our) track can NEVER be mislabeled enemy, and never logs as opponent
    bad = _t(7, "giant", "enemy", 520, 620)           # detector wrongly says enemy one frame
    pt.update([bad], now=12.0)
    assert bad["side"] == "ally" and bad["side_source"] == "own_play"
    assert len(pt.opponent_plays) == 0, "our own unit must not become an opponent play"
    print("test 4 OK: bound own unit stays ally (can't be misread enemy / logged as opp)")

    # 5) byproduct guard: a skeleton next to an enemy witch is NOT an opponent play;
    #    a skeleton away from any spawner IS.
    pt = PlayTracker()
    witch = _t(1, "witch", "enemy", 400, 300)
    near_skel = _t(2, "skeleton", "enemy", 430, 320)   # within SPAWN_RADIUS of the witch
    evs = pt.update([witch, near_skel], now=50.0)
    kinds = sorted(e.card for e in evs)
    assert "skeletons" not in [e.card for e in evs], "witch's skeleton must not be a play"
    assert any(e.card == "witch" for e in evs), "the witch itself is a play"
    far_skel = _t(3, "skeleton", "enemy", 700, 900)    # nowhere near a spawner
    evs2 = pt.update([witch, near_skel, far_skel], now=50.1)
    assert [e.card for e in evs2] == ["skeletons"], "a lone skeleton IS an opponent play"
    print("test 5 OK: spawner byproducts filtered; a lone spawnable unit still counts")

    # 6) an own play that never gets a matching track expires (no false bind, no crash)
    pt = PlayTracker()
    pt.record_own_play("knight", (100, 100), now=0.0)
    evs = pt.update([_t(9, "knight", "ally", 800, 1400)], now=2.0)  # far away, and later
    assert evs == [] or evs[0].kind != "own", "a far/late track must not bind to the play"
    evs2 = pt.update([], now=10.0)   # long after -> pending play expired, no error
    assert evs2 == []
    print("test 6 OK: unmatched own play expires cleanly (no false bind)")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
