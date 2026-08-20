"""
Synthetic tests for perception/frame_diff.py — no live capture needed, same
"buildable and testable now" split described in the TV Royale pipeline plan
(Project_Status / clash-royale-ai-pipeline-explained.md section 5).

Usage:
    python -m perception.test_frame_diff
"""

import numpy as np

from perception.frame_diff import closest_blob_to, find_new_blobs


def blank(h=400, w=400):
    return np.zeros((h, w, 3), dtype=np.uint8)


def draw_square(img, x1, y1, size, color=(255, 255, 255)):
    out = img.copy()
    out[y1:y1 + size, x1:x1 + size] = color
    return out


def test_1_no_change_no_blobs():
    before = draw_square(blank(), 50, 50, 30)
    after = before.copy()
    assert find_new_blobs(before, after) == []
    print("test 1 OK: identical frames -> no blobs")


def test_2_new_square_detected():
    before = blank()
    after = draw_square(before, 100, 100, 40)
    boxes = find_new_blobs(before, after)
    assert len(boxes) == 1, boxes
    x1, y1, x2, y2 = boxes[0]
    assert 95 <= x1 <= 105 and 95 <= y1 <= 105 and 135 <= x2 <= 145 and 135 <= y2 <= 145, boxes
    print("test 2 OK: new square detected at", boxes[0])


def test_3_tiny_noise_filtered_out():
    before = blank()
    after = before.copy()
    after[10:12, 10:12] = (255, 255, 255)  # 2x2 = 4px, well under MIN_BLOB_AREA
    assert find_new_blobs(before, after) == []
    print("test 3 OK: tiny noise blob correctly filtered out")


def test_4_existing_object_moving_is_still_a_diff():
    """Documents the module's own stated limitation: it can't tell 'new
    unit' from 'moved unit' by itself -- moving an existing square still
    produces a diff blob at both its old and new position. Callers must
    supply frames that bracket a single known play event."""
    before = draw_square(blank(), 50, 50, 30)
    after = draw_square(blank(), 90, 50, 30)  # same square, moved right
    boxes = find_new_blobs(before, after)
    assert len(boxes) >= 1, boxes
    print("test 4 OK (documents limitation): a moved object still yields diff blob(s):", boxes)


def test_5_closest_blob_to():
    boxes = [(10, 10, 30, 30), (200, 200, 220, 220), (100, 100, 120, 120)]
    assert closest_blob_to(boxes, (105, 105)) == (100, 100, 120, 120)
    assert closest_blob_to([], (0, 0)) is None
    print("test 5 OK: closest_blob_to picks the nearest box, None on empty input")


if __name__ == "__main__":
    test_1_no_change_no_blobs()
    test_2_new_square_detected()
    test_3_tiny_noise_filtered_out()
    test_4_existing_object_moving_is_still_a_diff()
    test_5_closest_blob_to()
    print()
    print("ALL TESTS PASSED")
