"""
Seeds/adds an elixir-number crop as a labeled reference example, built
directly from a real in-game screenshot — same pattern as label_hand.py.
Re-running this for an already-known value ADDS another example rather
than overwriting (near-duplicate frames are skipped automatically).

Usage:
    python -m perception.label_elixir <value>

Give the elixir value (0-10) currently shown in test_screenshot.png's
number readout.
"""

import sys

import cv2

from perception.elixir_counter import NUMBER_REGION, add_example


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m perception.label_elixir <value>")
        sys.exit(1)

    value = sys.argv[1]
    if not value.isdigit() or not (0 <= int(value) <= 10):
        print("Value must be an integer 0-10.")
        sys.exit(1)

    screenshot = cv2.imread("test_screenshot.png")
    if screenshot is None:
        raise FileNotFoundError("test_screenshot.png not found — capture one first.")

    x1, y1, x2, y2 = NUMBER_REGION
    crop = screenshot[y1:y2, x1:x2]
    path = add_example(value, crop)
    if path:
        print(f"Added new example for elixir={value}: {path}")
    else:
        print(f"Skipped — a near-identical example for elixir={value} is already on file")


if __name__ == "__main__":
    main()
