"""
Seeds/adds hand-slot crops as labeled reference examples, built directly
from real in-game screenshots. Re-running this for an already-known card
ADDS another example (more real-world variation) rather than overwriting
the old one — near-duplicate frames are automatically skipped, see
card_detector.py's add_example().

Usage:
    python -m perception.label_hand fireball minions knight mini_pekka

Give the 4 correct card names, left to right, matching what's currently
in test_screenshot.png's hand. Use lowercase, underscore-separated names
(e.g. mini_pekka) to stay consistent with other tooling.
"""

import sys

import cv2

from config.hand_region import hand_slot_regions
from perception.card_detector import add_example


def main():
    if len(sys.argv) != 5:
        print("Usage: python -m perception.label_hand <slot1> <slot2> <slot3> <slot4>")
        sys.exit(1)

    names = sys.argv[1:5]
    screenshot = cv2.imread("test_screenshot.png")
    if screenshot is None:
        raise FileNotFoundError("test_screenshot.png not found — capture one first.")

    for name, (x1, y1, x2, y2) in zip(names, hand_slot_regions()):
        crop = screenshot[y1:y2, x1:x2]
        if add_example(name, crop):
            print(f"Added new example for {name}")
        else:
            print(f"Skipped {name} — a near-identical example is already on file")


if __name__ == "__main__":
    main()
