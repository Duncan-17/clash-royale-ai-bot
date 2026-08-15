"""
Combines detection + labeling into one step: captures the current hand,
runs the detector, and for any slot that comes back UNKNOWN, prompts you
right there for the correct name and saves it. Known cards are silently
skipped (no prompt) unless you pass --relabel, which re-checks every slot
and lets you add another example even for a card already recognized.

For fully hands-off collection during play instead, use auto_collect.py —
this one is for a quick manual top-up when you're not running that.

Usage:
    python -m perception.collect_cards
    python -m perception.collect_cards --relabel   # also re-check known slots
"""

import sys

import cv2

from config.hand_region import hand_slot_regions
from control.actions import EmulatorController
from perception.capture import ScreenCapture
from perception.card_detector import CONFIDENCE_THRESHOLD, add_example, best_match, load_templates


def main():
    relabel = "--relabel" in sys.argv

    print("Capturing current hand...")
    controller = EmulatorController()
    capture = ScreenCapture(controller)
    screenshot = capture.screenshot()
    cv2.imwrite("test_screenshot.png", screenshot)

    templates = load_templates()
    regions = hand_slot_regions()

    added, skipped = 0, 0
    for i, (x1, y1, x2, y2) in enumerate(regions, start=1):
        crop = screenshot[y1:y2, x1:x2]
        name, score = best_match(crop, templates)
        known = score >= CONFIDENCE_THRESHOLD

        if known and not relabel:
            print(f"Slot {i}: {name} (already known, confidence={score:.3f})")
            skipped += 1
            continue

        if known:
            print(f"Slot {i}: currently recognized as {name} (confidence={score:.3f})")
        else:
            print(f"Slot {i}: UNKNOWN (best guess was {name}, confidence={score:.3f})")

        typed = input(
            f"  Enter the correct name for slot {i} (lowercase_with_underscores), "
            f"or press Enter to skip: "
        ).strip()

        if not typed:
            print("  Skipped.")
            skipped += 1
            continue

        if add_example(typed, crop):
            print(f"  Added new example for {typed}")
            added += 1
        else:
            print(f"  Skipped — a near-identical example of {typed} is already on file")
            skipped += 1

    total_examples = sum(len(v) for v in load_templates().values())
    print(f"\nDone — {added} example(s) added, {skipped} left as-is.")
    print(f"Template library now has {total_examples} example(s) total.")


if __name__ == "__main__":
    main()
