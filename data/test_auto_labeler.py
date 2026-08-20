"""
Synthetic tests for data/auto_labeler.py's dataset-writing half
(YoloDatasetWriter + localize_play, class-only taxonomy -- see that
module's docstring for why side isn't part of it) — no live TV Royale
access needed, same
"buildable and testable now" split described in the pipeline plan
(Project_Status / clash-royale-ai-pipeline-explained.md section 5). Writes
into a temp directory, never the real data/unit_training/ collected
dataset.

Usage:
    python -m data.test_auto_labeler
"""

import shutil
import tempfile

import numpy as np

from data.auto_labeler import CLASS_INDEX, MAX_EXAMPLES_PER_CLASS, YoloDatasetWriter, localize_play


def blank_image(h=200, w=300):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_1_write_and_read_back():
    tmp = tempfile.mkdtemp()
    try:
        w = YoloDatasetWriter(output_dir=tmp)
        example_id = w.write(blank_image(200, 300), [("knight", (30, 40, 90, 160))])
        assert example_id is not None

        import os
        assert os.path.isfile(os.path.join(tmp, "images", f"{example_id}.jpg"))
        label_path = os.path.join(tmp, "labels", f"{example_id}.txt")
        assert os.path.isfile(label_path)
        with open(label_path) as f:
            line = f.read().strip()
        class_id, cx, cy, bw, bh = line.split()
        assert int(class_id) == CLASS_INDEX["knight"]
        # bbox (30,40,90,160) in a 300x200 image -> center (60,100), size (60,120)
        assert abs(float(cx) - 60 / 300) < 1e-4
        assert abs(float(cy) - 100 / 200) < 1e-4
        assert abs(float(bw) - 60 / 300) < 1e-4
        assert abs(float(bh) - 120 / 200) < 1e-4
        print("test 1 OK: wrote and correctly normalized", line)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_2_empty_objects_writes_nothing():
    tmp = tempfile.mkdtemp()
    try:
        w = YoloDatasetWriter(output_dir=tmp)
        result = w.write(blank_image(), [])
        assert result is None
        import os
        assert os.listdir(os.path.join(tmp, "images")) == []
        print("test 2 OK: no objects -> nothing written")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_3_per_class_cap_enforced():
    tmp = tempfile.mkdtemp()
    try:
        w = YoloDatasetWriter(output_dir=tmp)
        # Write one over the cap for a single class, confirming the writer
        # stops counting/labeling that class once at MAX_EXAMPLES_PER_CLASS,
        # without needing to actually write thousands of files to prove it --
        # directly manipulate the internal counter to simulate "already at cap".
        w._class_counts["archer"] = MAX_EXAMPLES_PER_CLASS
        result = w.write(blank_image(), [("archer", (10, 10, 20, 20))])
        assert result is None, "an image with only an at-cap class should write nothing"
        print("test 3 OK: at-cap class correctly skipped")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_4_mixed_frame_keeps_rare_class_drops_capped_one():
    tmp = tempfile.mkdtemp()
    try:
        w = YoloDatasetWriter(output_dir=tmp)
        w._class_counts["archer"] = MAX_EXAMPLES_PER_CLASS
        example_id = w.write(blank_image(), [
            ("archer", (10, 10, 20, 20)),   # at cap -- should be dropped
            ("knight", (30, 30, 50, 50)),  # not at cap -- should be kept
        ])
        assert example_id is not None
        with open(f"{tmp}/labels/{example_id}.txt") as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        assert len(lines) == 1, lines
        assert int(lines[0].split()[0]) == CLASS_INDEX["knight"]
        print("test 4 OK: capped object dropped, rare object still written")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_5_resumes_counts_from_existing_files():
    tmp = tempfile.mkdtemp()
    try:
        w1 = YoloDatasetWriter(output_dir=tmp)
        w1.write(blank_image(), [("giant", (0, 0, 50, 50))])
        w1.write(blank_image(), [("giant", (0, 0, 50, 50))])
        # Fresh writer instance, same directory -- must pick up existing counts
        w2 = YoloDatasetWriter(output_dir=tmp)
        assert w2._class_counts["giant"] == 2, w2._class_counts
        assert w2._next_id == 2
        print("test 5 OK: a new writer instance resumes counts from disk")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_6_localize_play_prefers_highest_age_track():
    tracks = [
        {"track_id": 1, "age": 2, "bbox": (0, 0, 10, 10)},
        {"track_id": 2, "age": 8, "bbox": (20, 20, 30, 30)},
        {"track_id": 3, "age": 4, "bbox": (40, 40, 50, 50)},
    ]
    box = localize_play(tracks)
    assert box == (20, 20, 30, 30), box
    print("test 6 OK: localize_play picked the highest-age (most established) track")


def test_7_localize_play_falls_back_to_frame_diff():
    before = blank_image(100, 100)
    after = before.copy()
    after[40:60, 40:60] = (255, 255, 255)
    box = localize_play([], before_crop=before, after_crop=after)
    assert box is not None
    x1, y1, x2, y2 = box
    assert 35 <= x1 <= 45 and 55 <= x2 <= 65, box
    print("test 7 OK: localize_play fell back to frame_diff and found", box)


def test_8_localize_play_returns_none_when_nothing_found():
    assert localize_play([]) is None
    before = blank_image(50, 50)
    after = before.copy()
    assert localize_play([], before_crop=before, after_crop=after) is None
    print("test 8 OK: localize_play correctly returns None when nothing plausible exists")


if __name__ == "__main__":
    test_1_write_and_read_back()
    test_2_empty_objects_writes_nothing()
    test_3_per_class_cap_enforced()
    test_4_mixed_frame_keeps_rare_class_drops_capped_one()
    test_5_resumes_counts_from_existing_files()
    test_6_localize_play_prefers_highest_age_track()
    test_7_localize_play_falls_back_to_frame_diff()
    test_8_localize_play_returns_none_when_nothing_found()
    print()
    print("ALL TESTS PASSED")
