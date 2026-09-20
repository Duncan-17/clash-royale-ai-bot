"""Tests for game_knowledge/play_localizer.py -- TV Royale play<->unit binding.

Covers the two handoff gaps (#1 shared-card side, #2 class flicker) plus the
confidence gate (abstain when unclear), spells (no bind), detector-gap troops
(bind by position/uniqueness with no class), placement canonicalization, and the
deferred-bind path (play fires before the sprite is detected).

Run: python -m game_knowledge.test_play_localizer
"""

from game_knowledge.play_localizer import (
    PlayLocalizer, spawns_trackable_unit, detected_classes, card_category)

W, H = 900, 1600


def _det(name, cx, cy, side="unknown", side_source=None, struggle=False,
         off_deck=False, conf=0.9, w=40, h=60):
    """A detector board-unit dict at pixel centre (cx, cy)."""
    d = {"name": name, "confidence": conf, "side": side,
         "bbox": (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2),
         "struggle": struggle, "off_deck": off_deck, "level": None,
         "hp_fraction": None, "has_badge": False}
    if side_source is not None:
        d["side_source"] = side_source
    return d


def _play(side, card, t):
    return {"side": side, "card": card, "t": t}


def _up(loc, dets, now):
    """Advance the localizer at the fixed test frame size."""
    loc.update(dets, now, W, H)


def _find(units, name):
    return next((u for u in units if u["name"] == name), None)


# top(enemy) ~ y<800, bottom(friendly) ~ y>800 in a 1600-tall frame; the bridge ~800.
TOP_Y, BOT_Y, BRIDGE_Y = 500, 1100, 780


def main():
    # -- card metadata sanity --------------------------------------------------
    assert card_category("knight") == "troop" and card_category("fireball") == "spell"
    assert spawns_trackable_unit("knight") and spawns_trackable_unit("cannon")
    assert not spawns_trackable_unit("fireball") and not spawns_trackable_unit("goblin_barrel")
    assert not spawns_trackable_unit("unknown_3") and not spawns_trackable_unit(None)
    assert detected_classes("berserker") == set()          # detector-gap troop, no class
    assert detected_classes("goblins") == {"goblin"}
    print("test 1 OK: card metadata (troop/building bind; spell/gap/unknown handled)")

    # -- #1 SHARED-CARD SIDE: an enemy dart_goblin at the bridge whose detector side
    #    came out 'ally' (wrong). The enemy play stamps it enemy. --------------------
    loc = PlayLocalizer()
    d = _det("dart_goblin", 450, BRIDGE_Y, side="ally", side_source="position")
    _up(loc, [d], now=10.0)                              # unit appears (mis-sided)
    r = loc.bind(_play("enemy", "dart_goblin", 10.3), now=10.3)
    assert r is not None and r["det_side"] == "enemy"
    u = _find(loc.resolved_units(), "dart_goblin")
    assert u["side"] == "enemy" and u["side_source"] == "play" and u["bound"]
    print("test 2 OK: #1 shared-card side -- play stamps the true owner over the mislabel")

    # -- #2 CLASS FLICKER: a friendly berserker (gap card) the detector calls
    #    'baby_dragon'. The play stamps 'berserker' and clears the struggle flag. -----
    loc = PlayLocalizer()
    d = _det("baby_dragon", 500, BOT_Y, side="ally", struggle=True, off_deck=True)
    _up(loc, [d], now=20.0)
    r = loc.bind(_play("friendly", "berserker", 20.2), now=20.2)
    assert r is not None
    u = _find(loc.resolved_units(), "berserker")
    assert u is not None and u["side"] == "ally" and u["bound"]
    assert u["struggle"] is False and u["off_deck"] is False   # a played card is real
    assert _find(loc.resolved_units(), "baby_dragon") is None  # renamed, not duplicated
    print("test 3 OK: #2 class flicker -- gap troop stamped from the play, struggle cleared")

    # -- STICKS through flicker: next frames the detector wobbles the class/side, the
    #    stamp holds (same track id kept by IoU). ------------------------------------
    d2 = _det("bandit", 505, BOT_Y + 5, side="enemy")      # detector now says bandit+enemy
    _up(loc, [d2], now=20.4)
    u = _find(loc.resolved_units(), "berserker")
    assert u is not None and u["side"] == "ally", "stamp must survive detector wobble"
    print("test 4 OK: the stamp sticks across class/side flicker on the same track")

    # -- CONFIDENCE GATE: two enemy units of the played class appear together -> which
    #    one is the play is ambiguous -> abstain (no bind, no guess). ----------------
    loc = PlayLocalizer()
    _up(loc, [_det("skeleton", 300, TOP_Y), _det("skeleton", 600, TOP_Y)], now=30.0)
    assert loc.bind(_play("enemy", "skeletons", 30.2), now=30.2) is None
    assert all(not u["bound"] for u in loc.resolved_units())
    print("test 5 OK: ambiguous (two matching new units) -> abstain")

    # -- SPELL: a fireball play never binds to a coincidental new unit. --------------
    loc = PlayLocalizer()
    _up(loc, [_det("knight", 500, TOP_Y, side="enemy")], now=40.0)
    assert loc.bind(_play("enemy", "fireball", 40.1), now=40.1) is None
    print("test 6 OK: spell play does not bind to a stray unit")

    # -- WRONG HALF: a friendly play must not grab a unit sitting deep in the enemy
    #    half (a lone new unit there is not this bottom play's placement). ------------
    loc = PlayLocalizer()
    _up(loc, [_det("knight", 450, 200)], now=50.0)       # deep top, friendly plays knight
    assert loc.bind(_play("friendly", "knight", 50.1), now=50.1) is None
    print("test 7 OK: a unit in the opponent's half is not bound to this play")

    # -- PLACEMENT canonicalization: enemy(top) placement is mirrored to self-bottom. -
    loc = PlayLocalizer()
    _up(loc, [_det("hog_rider", 450, 400, side="enemy")], now=60.0)  # cy=0.25 (top)
    r = loc.bind(_play("enemy", "hog_rider", 60.1), now=60.1)
    assert r is not None and abs(r["placement"]["y"] - 0.75) < 0.02      # mirrored 0.25->0.75
    assert abs(r["placement"]["x"] - 0.5) < 0.02
    print("test 8 OK: enemy placement canonicalized to the acting player's own bottom")

    # -- DEFERRED bind: the play fires a beat before the sprite is detected; a later
    #    update() finds the new unit and binds it. -----------------------------------
    loc = PlayLocalizer()
    _up(loc, [], now=70.0)                               # nothing on the board yet
    assert loc.bind(_play("enemy", "giant", 70.0), now=70.0) is None    # queued, not bound
    _up(loc, [_det("giant", 450, TOP_Y, side="enemy")], now=70.5)     # sprite appears
    u = _find(loc.resolved_units(), "giant")
    assert u is not None and u["bound"] and u["side"] == "enemy"
    print("test 9 OK: deferred bind -- play fires before the sprite, binds on appearance")

    # -- DEFERRED expiry: if the unit never shows, the pending play is dropped cleanly.
    loc = PlayLocalizer()
    loc.bind(_play("friendly", "valkyrie", 80.0), now=80.0)
    _up(loc, [], now=82.0)                               # past bind_forward, no unit
    _up(loc, [_det("valkyrie", 500, BOT_Y, side="ally")], now=82.5)   # a LATER valkyrie
    assert _find(loc.resolved_units(), "valkyrie")["bound"] is False, "expired play must not bind late"
    print("test 10 OK: an unmatched play expires (no false late bind)")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
