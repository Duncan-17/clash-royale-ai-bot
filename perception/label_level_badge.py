"""
Adds a level-badge digit crop as a labeled reference example, built
directly from a real in-game screenshot — same pattern as label_hand.py /
label_elixir.py. Re-running for an already-known value ADDS another
example rather than overwriting (near-duplicate frames are skipped
automatically, same dedup logic as elixir_counter.py's add_example()).

Unlike label_elixir.py (which reads a fixed screen region), a badge moves
with whichever unit it belongs to, so this needs that unit's own detected
bbox to know where to look — get it from `python -m perception.unit_detector`
first (it prints each detection's bbox), then pass that bbox here. This
reuses perception/side_reader.py's own search/localization (_search()) to
find the actual badge blob within that unit's search window, rather than
needing you to hand-pick exact badge pixel coordinates.

Usage:
    python -m perception.unit_detector                              # find a unit's bbox first
    python -m perception.label_level_badge <value> <x1> <y1> <x2> <y2>

Give the level value (e.g. "3") you can see on the badge in
test_screenshot.png, and the unit's bbox exactly as unit_detector.py
printed it. Normal range is 1-17 (16 is the usual cap; a Mirrored card
copies at +1, so 17 is real too) -- values up to 25 are still accepted
without complaint since Hero Mini P.E.K.K.A. can reach 25 via a specific
mechanic (not otherwise handled anywhere yet, per Duncan 2026-08-17), but
anything past that is almost certainly a typo.
"""

import sys

import cv2

from perception.side_reader import LEVEL_MAX_VALUE, LEVEL_MIN_VALUE, add_level_example, candidate_badge_crop


def main():
    if len(sys.argv) != 6:
        print("Usage: python -m perception.label_level_badge <value> <x1> <y1> <x2> <y2>")
        print("Run `python -m perception.unit_detector` first to get a unit's bbox.")
        sys.exit(1)

    value = sys.argv[1]
    if not value.isdigit() or not (LEVEL_MIN_VALUE <= int(value) <= LEVEL_MAX_VALUE):
        print(f"Value must be an integer {LEVEL_MIN_VALUE}-{LEVEL_MAX_VALUE}.")
        sys.exit(1)
    bbox = tuple(int(v) for v in sys.argv[2:6])

    screenshot = cv2.imread("test_screenshot.png")
    if screenshot is None:
        raise FileNotFoundError("test_screenshot.png not found — capture one first.")

    crop = candidate_badge_crop(screenshot, bbox)
    if crop is None:
        print(f"No badge-shaped blob found near bbox={bbox} — "
              f"nothing to label. Check the bbox is right and the badge is actually visible.")
        sys.exit(1)

    path = add_level_example(value, crop)
    if path:
        print(f"Added new example for level={value}: {path}")
    else:
        print(f"Skipped — a near-identical example for level={value} is already on file, "
              f"or MAX_EXAMPLES_PER_VALUE reached")


if __name__ == "__main__":
    main()
