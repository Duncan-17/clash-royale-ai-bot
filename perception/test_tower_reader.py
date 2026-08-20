"""
Regression tests for perception/tower_reader.py -- both a real-capture
check (against assets/tower_reader_calibration/left_princess_towers_
destroyed.png, a real match capture with both left princess towers
destroyed and both right ones at full health) and synthetic tests for
TowerStateTracker's destroyed-debounce timing, same "verify with real
data + a dedicated test before trusting it" rigor as every other
calibrated module in this project.

Run: python -m perception.test_tower_reader
"""

from unittest.mock import patch

import cv2
import numpy as np

from perception.tower_reader import (
    DESTROYED_CONFIRM_FRAMES,
    TowerStateTracker,
    read_all_towers,
)

REAL_CAPTURE = "assets/tower_reader_calibration/left_princess_towers_destroyed.png"
DAMAGED_CAPTURE = "assets/tower_reader_calibration/ally_left_princess_damaged.png"
ENEMY_DAMAGED_CAPTURE = "assets/tower_reader_calibration/enemy_left_princess_damaged.png"


def test_real_capture_hp_fractions():
    img = cv2.imread(REAL_CAPTURE)
    assert img is not None, f"couldn't load {REAL_CAPTURE}"
    hp = read_all_towers(img)
    assert hp["ally_left"] is None, f"expected ally_left unreadable (destroyed), got {hp['ally_left']}"
    assert hp["enemy_left"] is None, f"expected enemy_left unreadable (destroyed), got {hp['enemy_left']}"
    assert hp["ally_right"] is not None and hp["ally_right"] > 0.9, f"expected ally_right near full: {hp['ally_right']}"
    assert hp["enemy_right"] is not None and hp["enemy_right"] > 0.9, f"expected enemy_right near full: {hp['enemy_right']}"
    # King towers -- ground truth per Duncan at capture time: ally ~69%, enemy ~25%.
    assert hp["ally_king"] is not None and abs(hp["ally_king"] - 0.69) < 0.05, f"expected ally_king ~0.69: {hp['ally_king']}"
    assert hp["enemy_king"] is not None and abs(hp["enemy_king"] - 0.25) < 0.05, f"expected enemy_king ~0.25: {hp['enemy_king']}"
    print("test 1 OK: real capture -- left towers unreadable (destroyed), right towers near-full HP, king towers match Duncan's reported ~69%/~25%")


def test_never_activated_king_tower_is_not_confused_with_destroyed():
    """The real regression this session caught: a king tower that's
    simply never been activated yet (no damage taken, no princess tower
    down on that side) reads identically to a destroyed one ("no clean
    row") -- TowerStateTracker must NOT confirm it destroyed just because
    it's never successfully been read, unlike a princess tower."""
    blank = np.zeros((1600, 900, 3), dtype=np.uint8)  # no tower UI anywhere -- simulates
    # a king tower that's simply never rendered a bar yet
    tracker = TowerStateTracker()
    for _ in range(DESTROYED_CONFIRM_FRAMES + 5):
        state = tracker.update(blank)
    assert not state["ally_king"]["destroyed"], "a never-activated king tower was wrongly confirmed destroyed"
    assert not state["enemy_king"]["destroyed"]
    assert state["ally_king"]["hp_fraction"] is None
    # A princess tower with the exact same "never successfully read" history SHOULD still
    # confirm destroyed -- no equivalent "not yet activated" state exists for it.
    assert state["ally_left"]["destroyed"], "a princess tower with no bar from frame 1 should still confirm destroyed"
    print("test 5 OK: a never-activated king tower is NOT confused with destroyed, but a princess tower in the same situation still correctly confirms destroyed")


def test_tracker_confirms_destroyed_after_the_debounce_window():
    img = cv2.imread(REAL_CAPTURE)
    tracker = TowerStateTracker()
    for i in range(DESTROYED_CONFIRM_FRAMES - 1):
        state = tracker.update(img)
        assert not state["ally_left"]["destroyed"], f"destroyed fired too early, frame {i+1}"
    state = tracker.update(img)  # the DESTROYED_CONFIRM_FRAMES-th consecutive miss
    assert state["ally_left"]["destroyed"], "destroyed should be confirmed by now"
    assert state["enemy_left"]["destroyed"]
    assert not state["ally_right"]["destroyed"] and not state["enemy_right"]["destroyed"]
    print("test 2 OK: TowerStateTracker only confirms destroyed after DESTROYED_CONFIRM_FRAMES consecutive misses")


def test_a_single_miss_does_not_confirm_destroyed():
    """A lone unreadable frame (transient noise) must not be trusted as destroyed."""
    img = cv2.imread(REAL_CAPTURE)
    tracker = TowerStateTracker()
    state = tracker.update(img)
    assert not state["ally_left"]["destroyed"], "a single miss should not confirm destroyed"
    print("test 3 OK: a single unreadable frame alone does not confirm destroyed")


def test_destroyed_is_permanent_even_if_a_bar_reappears():
    """Once confirmed destroyed, a tower stays destroyed even if a later
    frame somehow finds a clean row again (shouldn't happen for real, but
    the tracker's own permanence guarantee is worth locking down directly
    rather than just assumed)."""
    real_img = cv2.imread(REAL_CAPTURE)
    full_health_img = real_img.copy()
    # Splice ally_right's own real (full-health) bar pixels onto ally_left's region,
    # simulating "a bar reappeared" for a track that's already been confirmed destroyed.
    from perception.tower_reader import TOWER_BAR_REGIONS, REGION_PAD_Y
    lx1, ly1, lx2, ly2 = TOWER_BAR_REGIONS["ally_left"]
    rx1, ry1, rx2, ry2 = TOWER_BAR_REGIONS["ally_right"]
    ly1p, ly2p = max(0, ly1 - REGION_PAD_Y), ly2 + REGION_PAD_Y
    ry1p, ry2p = max(0, ry1 - REGION_PAD_Y), ry2 + REGION_PAD_Y
    full_health_img[ly1p:ly2p, lx1:lx2] = real_img[ry1p:ry2p, rx1:rx2]

    tracker = TowerStateTracker()
    for _ in range(DESTROYED_CONFIRM_FRAMES):
        tracker.update(real_img)  # confirm destroyed using the real (rubble) capture
    state = tracker.update(full_health_img)  # now feed it a frame where ally_left LOOKS full-health
    assert state["ally_left"]["destroyed"], "destroyed flipped back to alive -- should be permanent"
    print("test 4 OK: destroyed stays permanent even if a later frame's region looks like a full bar again")


def test_destroyed_tower_region_is_never_read_again():
    """Duncan's explicit ask: once a tower is confidently destroyed, stop
    checking its region at all -- a troop walking through that now-empty
    patch of ground is exactly the kind of thing that could produce a
    false reading if the region kept getting scanned. Verifies the actual
    MECHANISM (read_tower is never called), not just the resulting state
    (already covered by test_destroyed_is_permanent_even_if_a_bar_reappears)."""
    img = cv2.imread(REAL_CAPTURE)
    tracker = TowerStateTracker()
    for _ in range(DESTROYED_CONFIRM_FRAMES):
        tracker.update(img)  # confirms ally_left/enemy_left destroyed
    assert tracker._destroyed["ally_left"] and tracker._destroyed["enemy_left"]

    from perception.tower_reader import read_tower as _real_read_tower
    with patch("perception.tower_reader.read_tower", wraps=_real_read_tower) as mock_read:
        tracker.update(img)
        checked_towers = {call.args[1] for call in mock_read.call_args_list}
    assert "ally_left" not in checked_towers, "read_tower was still called for a confirmed-destroyed tower"
    assert "enemy_left" not in checked_towers
    assert "ally_right" in checked_towers and "enemy_right" in checked_towers, "still-alive towers should keep being checked"
    print("test 6 OK: a confirmed-destroyed tower's region is never read again, still-alive towers keep being checked")


def test_real_damaged_princess_tower_reads_the_true_partial_fill():
    """The real bug this fixes (2026-08-20): a healthy, ~90%+ princess
    tower read exactly 0% the instant it took any damage at all, in a
    real live match. Traced to a wrong EMPTY_BGR guess for princess-ally
    (borrowed from side_reader.py's badge colors by analogy) that
    happened to closely match a dark BORDER pixel row present at every HP
    level, fooling _best_row into preferring that border row over the
    real bar row the moment the bar stopped being 100% solid FILLED_BGR.
    Fixed with a real measurement from this exact capture (a clean,
    unambiguous fill/empty transition at ~90-92% width, confirmed
    consistent across two separate rows) -- this test locks that
    real-world value in against regression."""
    img = cv2.imread(DAMAGED_CAPTURE)
    assert img is not None, f"couldn't load {DAMAGED_CAPTURE}"
    hp = read_all_towers(img)
    assert hp["ally_left"] is not None and abs(hp["ally_left"] - 0.91) < 0.05, \
        f"expected ally_left ~0.91 (Duncan's own '~90+%' estimate): {hp['ally_left']}"
    # Untouched towers in the same real capture should still read correctly, confirming
    # the EMPTY_BGR fix didn't regress anything else.
    assert hp["ally_right"] is not None and hp["ally_right"] > 0.9
    assert hp["enemy_left"] is not None and hp["enemy_left"] > 0.9
    assert hp["enemy_right"] is not None and hp["enemy_right"] > 0.9
    print(f"test 9 OK: a real ~90%-damaged princess tower reads its true partial fill ({hp['ally_left']:.2f}), not a false 0%")


def test_zero_flash_around_each_hit_does_not_overwrite_the_prior_value():
    """A second, distinct real live-match pattern (2026-08-20): a damaged
    princess tower flashed to 0.0 for ~2 CONSECUTIVE frames around EVERY
    new hit (long enough to satisfy the normal 2-frame HP_CONFIRM_FRAMES)
    before settling on the real new, lower-but-nonzero value -- e.g. a
    real observed sequence 0.734 -> [0.0, 0.0] -> 0.433 -> [0.0, 0.0] ->
    0.358 -> ... Reproduces that exact pattern: the false zero must NOT
    overwrite the last confirmed (correct) value, but the real new value
    must still confirm normally once it repeats."""
    img = cv2.imread(REAL_CAPTURE)
    tracker = TowerStateTracker()
    # 0.734 confirmed, then a 2-frame false-zero flash, then 0.433 confirmed.
    readings = iter([0.734, 0.734, 0.0, 0.0, 0.433, 0.433])
    with patch("perception.tower_reader.read_tower",
               side_effect=lambda shot, name: next(readings) if name == "ally_left" else None):
        results = [tracker.update(img) for _ in range(6)]
    hp_over_time = [r["ally_left"]["hp_fraction"] for r in results]
    assert abs(hp_over_time[1] - 0.734) < 0.01, f"0.734 should have confirmed: {hp_over_time}"
    assert abs(hp_over_time[3] - 0.734) < 0.01, f"the 2-frame zero flash should NOT have overwritten 0.734: {hp_over_time}"
    assert abs(hp_over_time[5] - 0.433) < 0.01, f"the real new value should still confirm once repeated: {hp_over_time}"
    shown = [round(h, 3) if h is not None else None for h in hp_over_time]
    print(f"test 10 OK: a 2-frame zero flash around a hit doesn't overwrite the prior value, but the real new value still confirms ({shown})")


def test_real_damaged_enemy_princess_tower_reads_partial_fill():
    """Same bug as the ally side, found separately (2026-08-20): the
    princess-ENEMY EMPTY_BGR had also been sampled from a full-health
    capture, so it captured the dark border/bezel row rather than any real
    empty-fill colour, and the tower snapped to 0% the instant it took any
    damage. Corrected from a real damaged-enemy column scan (filled
    BGR(93,37,224) -> empty BGR(73,49,92)). This locks in that a damaged
    enemy tower reads a real partial value rather than 0."""
    img = cv2.imread(ENEMY_DAMAGED_CAPTURE)
    assert img is not None, f"couldn't load {ENEMY_DAMAGED_CAPTURE}"
    hp = read_all_towers(img)
    assert hp["enemy_left"] is not None and 0.4 < hp["enemy_left"] < 0.8,         f"expected enemy_left a real partial value (~0.6), got {hp['enemy_left']}"
    assert hp["enemy_right"] is not None and hp["enemy_right"] > 0.9,         f"undamaged enemy_right should still read near-full: {hp['enemy_right']}"
    print(f"test 11 OK: a real damaged ENEMY princess tower reads its true partial fill ({hp['enemy_left']:.2f}), not 0%")


def test_hp_value_debounced_against_a_single_spurious_reading():
    """The real bug this fixes (2026-08-20 live match): a fully-healthy
    tower's bar occasionally reads a wildly wrong single-frame value
    (confirmed via a real log: a steady 1.0 flashed to 0.0 for exactly one
    frame, then immediately reverted -- almost certainly a hit-flash VFX
    artifact). A lone spurious reading must not update the reported
    hp_fraction; two consecutive agreeing readings should."""
    img = cv2.imread(REAL_CAPTURE)
    tracker = TowerStateTracker()
    readings = iter([1.0, 1.0, 1.0, 0.0, 1.0, 1.0])  # a lone spurious 0.0 blip mid-stream
    with patch("perception.tower_reader.read_tower",
               side_effect=lambda shot, name: next(readings) if name == "ally_right" else None):
        results = [tracker.update(img) for _ in range(6)]
    hp_over_time = [r["ally_right"]["hp_fraction"] for r in results]
    assert hp_over_time[2] == 1.0, f"should have confirmed 1.0 by the 2nd agreeing read: {hp_over_time}"
    assert hp_over_time[3] == 1.0, f"a single spurious 0.0 reading should not have updated the reported value: {hp_over_time}"
    print(f"test 7 OK: a lone spurious reading doesn't update the reported HP value ({hp_over_time})")


def test_short_miss_streak_no_longer_falsely_confirms_destroyed():
    """The exact real bug this session found: a real match had BOTH enemy
    princess towers confirmed destroyed by frame 5 (well under a second
    in) -- almost certainly a startup transient, not real damage. A short
    (old-threshold-sized) miss streak followed by real successful reads
    must NOT confirm destroyed with the new, much longer window."""
    call_count = [0]

    def fake_read(shot, name):
        if name != "enemy_right":
            return None
        call_count[0] += 1
        return None if call_count[0] <= 5 else 1.0  # 5 misses (the OLD threshold), then recovers

    img = cv2.imread(REAL_CAPTURE)
    tracker = TowerStateTracker()
    with patch("perception.tower_reader.read_tower", side_effect=fake_read):
        state = None
        for _ in range(10):
            state = tracker.update(img)
    assert not state["enemy_right"]["destroyed"], "a 5-frame miss streak should no longer falsely confirm destroyed"
    print("test 8 OK: a short (old-threshold-sized) miss streak no longer falsely confirms destroyed")


if __name__ == "__main__":
    test_real_capture_hp_fractions()
    test_real_damaged_princess_tower_reads_the_true_partial_fill()
    test_real_damaged_enemy_princess_tower_reads_partial_fill()
    test_tracker_confirms_destroyed_after_the_debounce_window()
    test_a_single_miss_does_not_confirm_destroyed()
    test_destroyed_is_permanent_even_if_a_bar_reappears()
    test_never_activated_king_tower_is_not_confused_with_destroyed()
    test_destroyed_tower_region_is_never_read_again()
    test_hp_value_debounced_against_a_single_spurious_reading()
    test_short_miss_streak_no_longer_falsely_confirms_destroyed()
    test_zero_flash_around_each_hit_does_not_overwrite_the_prior_value()
    print("\nALL TESTS PASSED")
