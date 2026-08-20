"""
Regression tests for perception/unit_tracker.py's side-resolution logic
(_resolve_side / SIDE_LOCK_HITS / spawn-icon seeding), added 2026-08-18
alongside the fix itself -- same "verify with a dedicated test before
trusting it" rigor as every other fix in this project (see
game_knowledge/test_opponent_play_tracker.py for the established pattern).

Run: python -m perception.test_unit_tracker
"""

from unittest.mock import patch

from perception.unit_tracker import MAX_MISSED_FRAMES, RECENT_VANISH_MAX_FRAMES, UnitTracker


def _bbox_at(cx, cy, w=50, h=50):
    return (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)


def _det(name="giant", side="unknown", bbox=(100, 630, 150, 680), confidence=0.9):
    # Default bbox's bottom edge (680) sits INSIDE the river band
    # (RIVER_Y_TOP=650..RIVER_Y_BOTTOM=711) on purpose -- _side_from_position
    # returns "unknown" there, so tests that don't care about position stay
    # isolated to whatever signal (badge debounce / spawn icon) they're
    # actually exercising. Tests that DO care about position use their own
    # explicit bbox instead (see test_position_* below).
    return {"name": name, "side": side, "bbox": bbox, "confidence": confidence}


def test_single_bad_frame_does_not_lock_side():
    """The exact real bug this was built to fix: a track sees many
    "unknown" reads, then ONE stray known reading -- that single frame must
    NOT lock the track's side."""
    tracker = UnitTracker()
    tracker.update([_det(side="unknown")])
    tracker.update([_det(side="unknown")])
    result = tracker.update([_det(side="enemy")])  # one stray read
    assert result[0]["side"] == "unknown", f"a single frame locked the side: {result[0]['side']}"
    print("test 1 OK: a single stray known reading does not lock a track's side")


def test_two_consecutive_matching_reads_lock_side():
    """The debounce's positive case: two agreeing reads in a row should
    lock, so a real unit's side is still usable quickly."""
    tracker = UnitTracker()
    tracker.update([_det(side="unknown")])
    tracker.update([_det(side="enemy")])
    result = tracker.update([_det(side="enemy")])
    assert result[0]["side"] == "enemy", f"two agreeing reads failed to lock: {result[0]['side']}"
    print("test 2 OK: two consecutive matching reads lock the side")


def test_disagreeing_read_resets_the_candidate():
    """A reading that disagrees with the pending candidate must reset the
    counter, not accumulate toward locking the wrong/old candidate."""
    tracker = UnitTracker()
    tracker.update([_det(side="unknown")])
    tracker.update([_det(side="enemy")])   # candidate=enemy, hits=1
    tracker.update([_det(side="ally")])    # disagreement -> candidate=ally, hits=1 (not locked)
    result = tracker.update([_det(side="ally")])  # second ally in a row -> locks ally
    assert result[0]["side"] == "ally", f"expected ally after reset+re-agree, got {result[0]['side']}"
    print("test 3 OK: a disagreeing read resets the candidate instead of averaging in")


def test_spawn_icon_seed_locks_immediately_but_badge_can_still_correct_it():
    """A found spawn icon should set the side on its very first sighting
    (bypassing the debounce) -- but per Duncan's 2026-08-18 correction,
    the icon is only tier ICON, not the final word: a single conflicting
    badge frame must NOT override it (still needs its own debounce), but
    two CONSECUTIVE conflicting badge frames must win, since the badge is
    the strongest of the three signals."""
    tracker = UnitTracker()
    with patch("perception.unit_tracker.find_spawn_icon", return_value=("ally", 500.0, (10, 10, 40, 40))):
        tracker.update([_det(side="unknown")], screenshot="fake_screenshot")  # frame 1: icon seeds tier=ICON, side="ally"
        result = tracker.update([_det(side="unknown")], screenshot="fake_screenshot")  # frame 2: confirms (hits=2)
    assert result[0]["side"] == "ally", f"spawn icon failed to seed immediately: {result[0]['side']}"

    with patch("perception.unit_tracker.find_spawn_icon", return_value=(None, 0.0, None)):
        # One conflicting badge frame -- _side_compatible now allows this to
        # match (track is only tier ICON, not yet TIER_BADGE), but the
        # debounce means it shouldn't flip yet on just one frame.
        result = tracker.update([_det(side="enemy")], screenshot="fake_screenshot")
        assert result[0]["side"] == "ally", f"a single conflicting badge frame flipped the side too early: {result[0]['side']}"
        # A second consecutive conflicting frame -- badge debounce now
        # satisfied, and badge always wins regardless of the icon's tier.
        result = tracker.update([_det(side="enemy")], screenshot="fake_screenshot")
    assert result[0]["side"] == "enemy", f"badge failed to override the icon-seeded side after 2 consistent reads: {result[0]['side']}"
    print("test 4 OK: an icon seed sets the side immediately, but 2 consistent conflicting badge reads still override it")


def test_badge_corrects_a_wrong_position_lock_across_the_river():
    """THE core regression this session's fix addresses (2026-08-18 live
    report): a unit whose first-seen position is past the river gets
    locked "enemy" (tier POSITION) -- but if it then shows a real,
    repeated blue ally badge (e.g. it's actually an ally unit that had
    already crossed, or a reacquired track after a tracking gap with no
    vanish-match available), the badge must be able to correct it back to
    "ally" once its own debounce is satisfied, instead of staying wrongly
    locked forever the way the original position-priority design did."""
    tracker = UnitTracker()
    enemy_side_bbox = (300, 300, 350, 350)  # bottom=350 -> position locks "enemy" (tier POSITION)
    tracker.update([_det(side="unknown", bbox=enemy_side_bbox)])
    result = tracker.update([_det(side="unknown", bbox=enemy_side_bbox)])
    assert result[0]["side"] == "enemy", f"setup failed, expected position to lock enemy first: {result[0]['side']}"

    # Now two consecutive real "ally" badge reads arrive (a clean, visible
    # blue level badge + health bar) -- the badge must win.
    tracker.update([_det(side="ally", bbox=enemy_side_bbox)])
    result = tracker.update([_det(side="ally", bbox=enemy_side_bbox)])
    assert result[0]["side"] == "ally", f"badge failed to correct a wrong position lock: {result[0]['side']}"
    print("test 4b OK: 2 consistent ally badge reads correct an earlier wrong 'enemy' position lock")


def test_badge_cannot_flip_a_stationary_ally_king_tower_unit_to_enemy():
    """Real bug from a live match log (2026-08-18, 4th round): a Giant sat
    essentially stationary behind the ally king tower (correctly "ally"
    for 6 straight frames), then a false badge misread flipped it to
    "enemy" for the rest of its life -- even though it never moved
    anywhere near the river. A debounced "enemy" badge candidate must be
    rejected while position still says "ally", since that's game-rule
    impossible; a debounced "ally" candidate is NOT gated the same way
    (see test_badge_corrects_a_wrong_position_lock_across_the_river)."""
    tracker = UnitTracker()
    bbox = (468, 1055, 595, 1181)  # deep behind the ally king tower, from the real log
    tracker.update([_det(side="unknown", bbox=bbox)])
    tracker.update([_det(side="unknown", bbox=bbox)])  # confirmed, "ally" via position

    # Two consecutive false "enemy" badge reads at essentially the same
    # (still unambiguously ally) position -- must NOT flip the side.
    tracker.update([_det(side="enemy", bbox=bbox)])
    result = tracker.update([_det(side="enemy", bbox=bbox)])
    assert result[0]["side"] == "ally", f"a positionally-impossible 'enemy' badge read flipped the side: {result[0]['side']}"
    print("test 4c OK: a debounced 'enemy' badge candidate is rejected while the unit is still unambiguously on the ally half")


def test_position_locks_ally_for_a_placement_deep_on_the_ally_half():
    """The exact real bug this was built to fix (2026-08-18 live match):
    a Giant first detected deep on the ally half (bbox bottom well past
    RIVER_Y_BOTTOM=711) must resolve "ally" from position alone, even
    though its very first badge reading says "enemy" -- reproduces the
    real log case (track first seen at bbox [309,1057,458,1195], badge
    flipped to "enemy" one frame after appearing)."""
    tracker = UnitTracker()
    bbox = (309, 1057, 458, 1195)  # bottom=1195, deep past RIVER_Y_BOTTOM
    # Frame 1: a bad badge read ("enemy") arrives on a track whose position
    # is unambiguously ours -- position is checked BEFORE the badge value
    # is ever looked at, so it locks "ally" (tier POSITION) immediately
    # regardless of what the badge says this same frame.
    tracker.update([_det(side="enemy", bbox=bbox)])
    # Frame 2: side="unknown" -- just confirms the position-set lock held
    # for a non-conflicting frame (see
    # test_badge_corrects_a_wrong_position_lock_across_the_river for the
    # separate case of 2 consistent CONFLICTING badge reads actually
    # overriding it, which is real, correct, and intended new behavior).
    result = tracker.update([_det(side="unknown", bbox=bbox)])
    assert result[0]["side"] == "ally", f"position failed to override a bad early badge read: {result[0]['side']}"
    print("test 6 OK: a placement deep on the ally half locks 'ally' even when the badge says 'enemy'")


def test_position_locks_enemy_for_a_placement_deep_on_the_enemy_half():
    tracker = UnitTracker()
    bbox = (300, 300, 350, 350)  # bottom=350, well above RIVER_Y_TOP=650
    tracker.update([_det(side="unknown", bbox=bbox)])
    result = tracker.update([_det(side="unknown", bbox=bbox)])
    assert result[0]["side"] == "enemy", f"expected position to lock enemy: {result[0]['side']}"
    print("test 7 OK: a placement deep on the enemy half locks 'enemy' from position alone")


def test_position_gives_no_info_inside_the_river_band():
    """A unit whose bbox bottom sits inside the river's own Y-band must
    fall through to the badge-debounce path instead of guessing a side
    from position — this is the "neutral zone tells you nothing" case
    Duncan's own framing anticipated."""
    tracker = UnitTracker()
    bbox = (300, 630, 350, 680)  # bottom=680, inside RIVER_Y_TOP..RIVER_Y_BOTTOM (650..711)
    tracker.update([_det(side="unknown", bbox=bbox)])
    result = tracker.update([_det(side="unknown", bbox=bbox)])
    assert result[0]["side"] == "unknown", f"river-band position should give no info: {result[0]['side']}"
    print("test 8 OK: a river-band position gives no info and falls through unchanged")


def test_dropped_ally_track_reappearing_nearby_inherits_ally_side():
    """The exact scenario Duncan reported live (2026-08-18): a walking
    ally unit's track gets lost mid-crossing (a real, already-documented
    tracking gap -- Project_Status "troop-tracking-gap double-counting")
    and reacquired under a brand-new track_id already past the river. The
    reacquired track must inherit "ally" from the dropped one instead of
    being freshly locked "enemy" by _side_from_position."""
    tracker = UnitTracker()
    near_bbox = (300, 700, 350, 750)  # bottom=750, past RIVER_Y_BOTTOM=711 -> position="ally"
    tracker.update([_det(side="unknown", bbox=near_bbox)])
    tracker.update([_det(side="unknown", bbox=near_bbox)])  # confirmed, side="ally" via position

    # Simulate the tracking gap: enough consecutive empty frames to exceed
    # the grace period and actually drop the track (not just miss a frame).
    for _ in range(MAX_MISSED_FRAMES + 1):
        tracker.update([])

    # Reacquired ~100px away, but now on the enemy side by raw position
    # (bottom=625, well above RIVER_Y_TOP=650) -- without the vanish-bridge,
    # this would freshly lock "enemy" from position alone.
    far_bbox = (310, 600, 360, 650)
    tracker.update([_det(side="unknown", bbox=far_bbox)])
    result = tracker.update([_det(side="unknown", bbox=far_bbox)])
    assert result[0]["side"] == "ally", f"reacquired track should have inherited 'ally': {result[0]['side']}"
    print("test 9 OK: a dropped ally track reappearing nearby (past the river) inherits 'ally', not a fresh 'enemy' position lock")


def test_vanish_match_expires_after_the_window():
    """A track reappearing well after RECENT_VANISH_MAX_FRAMES must NOT
    inherit the old side -- it should resolve fresh (via position here)."""
    tracker = UnitTracker()
    near_bbox = (300, 700, 350, 750)
    tracker.update([_det(side="unknown", bbox=near_bbox)])
    tracker.update([_det(side="unknown", bbox=near_bbox)])
    for _ in range(MAX_MISSED_FRAMES + 1):
        tracker.update([])

    # Let the vanish record expire before the unit reappears.
    for _ in range(RECENT_VANISH_MAX_FRAMES + 1):
        tracker.update([])

    far_bbox = (310, 550, 360, 600)  # bottom=600, clearly < RIVER_Y_TOP=650 -> position="enemy" (fresh read)
    tracker.update([_det(side="unknown", bbox=far_bbox)])
    result = tracker.update([_det(side="unknown", bbox=far_bbox)])
    assert result[0]["side"] == "enemy", f"expired vanish record should not have been used: {result[0]['side']}"
    print("test 10 OK: an expired vanish record is not used -- the track resolves fresh instead")


def test_single_misclassified_frame_does_not_spawn_a_phantom_track():
    """The exact bug the class accumulator fixes: an established track hit
    by ONE misclassified frame (the documented Valkyrie/spear_goblin<->
    archer hallucination bugs) must stay the SAME track_id and keep its
    real reported name, not fail to match and spin up a phantom."""
    tracker = UnitTracker()
    bbox = _bbox_at(300, 630)  # river band -> side stays "unknown", isolates this test to class
    tracker.update([_det(name="knight", bbox=bbox)])
    result = tracker.update([_det(name="knight", bbox=bbox)])
    track_id = result[0]["track_id"]

    # One bad frame: a different class at essentially the same position.
    result = tracker.update([_det(name="archer", bbox=bbox)])
    assert result[0]["track_id"] == track_id, "a single misclassified frame spawned a new track instead of matching the existing one"
    assert result[0]["name"] == "knight", f"a single misclassified frame relabeled the track early: {result[0]['name']}"

    # Reverts back -- confirms the track survived clean, not just coincidentally still there.
    result = tracker.update([_det(name="knight", bbox=bbox)])
    assert result[0]["track_id"] == track_id and result[0]["name"] == "knight"
    print("test 11 OK: a single misclassified frame matches the existing track instead of spawning a phantom, and doesn't relabel it")


def test_sustained_different_class_eventually_relabels_the_track():
    """2 CONSECUTIVE frames of a different class should actually relabel
    the track (mirrors SIDE_LOCK_HITS's debounce) -- the accumulator
    isn't a permanent lock, just protection against a single bad frame."""
    tracker = UnitTracker()
    bbox = _bbox_at(300, 630)
    tracker.update([_det(name="knight", bbox=bbox)])
    tracker.update([_det(name="knight", bbox=bbox)])
    tracker.update([_det(name="archer", bbox=bbox)])          # 1st archer read
    result = tracker.update([_det(name="archer", bbox=bbox)])  # 2nd consecutive -> relabels
    assert result[0]["name"] == "archer", f"2 consistent reads should have relabeled the track: {result[0]['name']}"
    print("test 12 OK: 2 consecutive consistent reads of a different class do relabel the track")


def test_building_targeter_reacquires_far_along_its_own_heading():
    """A building-targeting unit (Giant) walking a straight line, missing
    several frames, and reappearing FAR beyond the old flat
    MATCH_DISTANCE_PX radius but consistent with its own extrapolated
    heading must still reacquire the SAME track_id -- the direct fix for
    the documented troop-tracking-gap double-counting bug."""
    tracker = UnitTracker()
    cx, cy = 300, 900
    step = 40
    result = None
    for _ in range(8):  # build up a converged velocity estimate (~40px/frame in +x)
        result = tracker.update([_det(name="giant", bbox=_bbox_at(cx, cy))])
        cx += step
    track_id = result[0]["track_id"]

    for _ in range(5):  # miss 5 frames -- predicted position moves ~200px further in +x
        tracker.update([])

    far_cx = cx + step * 5  # reappears where the extrapolated heading actually predicts,
    # ~200px past the last real detection -- well beyond the old flat MATCH_DISTANCE_PX=90
    result = tracker.update([_det(name="giant", bbox=_bbox_at(far_cx, cy))])
    assert result and result[0]["track_id"] == track_id, "building-targeter failed to reacquire along its own predictable heading"
    print("test 13 OK: a building-targeting unit reacquires the same track far along its own predicted heading")


def test_building_targeter_does_not_reacquire_sideways_of_its_heading():
    """The other half of the same fix: a building-targeter is NOT expected
    to suddenly veer sideways (real building-targeters don't retarget), so
    a reappearance well to the SIDE of its predicted heading should fail
    the tight directional test and start a separate track."""
    tracker = UnitTracker()
    cx, cy = 300, 900
    step = 40
    for _ in range(8):
        result = tracker.update([_det(name="giant", bbox=_bbox_at(cx, cy))])
        cx += step
    track_id = result[0]["track_id"]

    for _ in range(5):
        tracker.update([])

    # Same predicted x (roughly where the heading says it should be), but
    # displaced 150px sideways in y -- well past DIRECTIONAL_LATERAL_RADIUS_PX=60.
    result = tracker.update([_det(name="giant", bbox=_bbox_at(cx, cy + 150))])
    assert not result or result[0]["track_id"] != track_id, "building-targeter wrongly reacquired far to the side of its own predictable heading"
    print("test 14 OK: a building-targeter reappearing sideways of its own heading does NOT reacquire the same track")


def test_non_building_unit_reacquires_sideways_via_the_wide_fallback():
    """The contrast case: a NON-building-targeter (Knight) making the same
    kind of sideways deviation (simulating a real retarget) should still
    reacquire via the generous WIDE circular search, since its direction
    was never trusted in the first place."""
    tracker = UnitTracker()
    cx, cy = 300, 900
    step = 40
    for _ in range(8):
        result = tracker.update([_det(name="knight", bbox=_bbox_at(cx, cy))])
        cx += step
    track_id = result[0]["track_id"]

    for _ in range(5):
        tracker.update([])

    result = tracker.update([_det(name="knight", bbox=_bbox_at(cx, cy + 150))])
    assert result and result[0]["track_id"] == track_id, "non-building unit failed to reacquire via the wide fallback search"
    print("test 15 OK: a non-building unit reacquires the same track via the wide fallback even when it deviated sideways")


def test_tower_state_makes_a_neutral_pocket_unknown_instead_of_a_hard_guess():
    """The actual payoff of wiring game_knowledge.zone_classifier in
    2026-08-20: a position that would deterministically read "enemy"
    while the enemy's left princess tower stands must read "unknown"
    (neutral, no info) once that tower is confirmed destroyed -- and the
    SAME position must still read the old deterministic "enemy" when no
    tower_state is passed at all (backward compatibility)."""
    bbox = (175, 525, 225, 575)  # bottom edge y2=575, inside ENEMY_LEFT_NEUTRAL_ZONE
    # (65,501,450,711) and above the river (y<650) -- would read "enemy" under the old
    # plain-river approximation.
    towers_alive = {
        "ally_left": {"destroyed": False}, "ally_right": {"destroyed": False},
        "enemy_left": {"destroyed": False}, "enemy_right": {"destroyed": False},
    }
    towers_enemy_left_down = {**towers_alive, "enemy_left": {"destroyed": True}}

    tracker_no_tower_state = UnitTracker()
    tracker_no_tower_state.update([_det(side="unknown", bbox=bbox)])
    result = tracker_no_tower_state.update([_det(side="unknown", bbox=bbox)])
    assert result[0]["side"] == "enemy", f"no tower_state should fall back to the old plain-river read: {result[0]['side']}"

    tracker_alive = UnitTracker()
    tracker_alive.update([_det(side="unknown", bbox=bbox)], tower_state=towers_alive)
    result = tracker_alive.update([_det(side="unknown", bbox=bbox)], tower_state=towers_alive)
    assert result[0]["side"] == "enemy", f"tower still standing should still read enemy: {result[0]['side']}"

    tracker_neutral = UnitTracker()
    tracker_neutral.update([_det(side="unknown", bbox=bbox)], tower_state=towers_enemy_left_down)
    result = tracker_neutral.update([_det(side="unknown", bbox=bbox)], tower_state=towers_enemy_left_down)
    assert result[0]["side"] == "unknown", f"a confirmed-destroyed tower should make this position neutral (unknown): {result[0]['side']}"
    print("test 16 OK: tower_state correctly turns a deterministic-while-alive position into 'unknown' once that tower is confirmed destroyed")


def test_conflicting_side_frame_does_not_split_an_established_track():
    """The real bug Duncan hit (traced in logs/debug_live_20260820_013026.jsonl):
    an ally Giant tracked cleanly for 134 consecutive frames, then on the VERY
    NEXT frame -- zero missed frames, only 22px of movement -- stopped being
    tracked and a new track_id appeared instead. Cause: side was a hard MATCHING
    gate, so one conflicting side reading made the real detection fail to match
    anything and spawn a fresh track. Downstream that reads as a brand-new unit,
    which is how phantom "played" events get manufactured.

    Side is an ATTRIBUTE; it must not decide IDENTITY (this project's own fusion-
    architecture doc says identity resolves first). It's now a soft ranking
    penalty instead, so the track survives and the debounce owns the side."""
    tracker = UnitTracker()
    bbox = _bbox_at(300, 900)
    # Establish a confident ally track (badge-tier, well past the debounce).
    for _ in range(6):
        result = tracker.update([_det(name="giant", side="ally", bbox=bbox)])
    track_id = result[0]["track_id"]
    assert result[0]["side"] == "ally"

    # One conflicting frame, essentially the same position (22px, like the real log).
    result = tracker.update([_det(name="giant", side="enemy", bbox=_bbox_at(322, 900))])
    assert result, "the detection vanished entirely instead of matching"
    assert result[0]["track_id"] == track_id, (
        f"a single conflicting side frame split the track "
        f"({track_id} -> {result[0]['track_id']}) -- this is the phantom-unit bug"
    )
    assert result[0]["side"] == "ally", "one conflicting frame should not flip a confirmed side"
    print("test 17 OK: one conflicting side frame keeps the same track_id (no phantom split) and does not flip the side")


def test_no_screenshot_falls_through_to_badge_debounce():
    """screenshot=None (the default, e.g. an existing caller that hasn't
    been updated) must not error, and must fall straight through to the
    ordinary badge-debounce path."""
    tracker = UnitTracker()
    tracker.update([_det(side="unknown")])
    tracker.update([_det(side="enemy")])
    result = tracker.update([_det(side="enemy")])
    assert result[0]["side"] == "enemy", f"no-screenshot path broke the badge debounce: {result[0]['side']}"
    print("test 5 OK: screenshot=None falls through to the badge-debounce path unchanged")


if __name__ == "__main__":
    test_single_bad_frame_does_not_lock_side()
    test_two_consecutive_matching_reads_lock_side()
    test_disagreeing_read_resets_the_candidate()
    test_spawn_icon_seed_locks_immediately_but_badge_can_still_correct_it()
    test_badge_corrects_a_wrong_position_lock_across_the_river()
    test_badge_cannot_flip_a_stationary_ally_king_tower_unit_to_enemy()
    test_position_locks_ally_for_a_placement_deep_on_the_ally_half()
    test_position_locks_enemy_for_a_placement_deep_on_the_enemy_half()
    test_position_gives_no_info_inside_the_river_band()
    test_tower_state_makes_a_neutral_pocket_unknown_instead_of_a_hard_guess()
    test_dropped_ally_track_reappearing_nearby_inherits_ally_side()
    test_vanish_match_expires_after_the_window()
    test_single_misclassified_frame_does_not_spawn_a_phantom_track()
    test_sustained_different_class_eventually_relabels_the_track()
    test_building_targeter_reacquires_far_along_its_own_heading()
    test_building_targeter_does_not_reacquire_sideways_of_its_heading()
    test_non_building_unit_reacquires_sideways_via_the_wide_fallback()
    test_conflicting_side_frame_does_not_split_an_established_track()
    test_no_screenshot_falls_through_to_badge_debounce()
    print("\nALL TESTS PASSED")
