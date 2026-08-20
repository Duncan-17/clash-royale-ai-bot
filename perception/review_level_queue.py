"""
Batch-labels whatever auto_collect.py queued up as unrecognized level
badges. Shows each queued badge crop one at a time (magnified — the real
crops are tiny, ~20-45px); type the digit value shown, or press Enter to
discard it (e.g. a false-positive blob that isn't really a badge at all).
Same pattern as perception/review_queue.py, for perception/side_reader.py's
digit-template library instead of card templates.

Usage:
    python -m perception.review_level_queue
"""

import os

import cv2

from perception.side_reader import LEVEL_MAX_VALUE, LEVEL_MIN_VALUE, add_level_example

QUEUE_DIR = "assets/unlabeled_level_queue"
DISPLAY_SCALE = 8  # badge crops are tiny -- magnify for a human to actually read the digit


def main():
    if not os.path.isdir(QUEUE_DIR):
        print("No queue directory found — nothing to review.")
        return

    files = sorted(f for f in os.listdir(QUEUE_DIR) if f.endswith(".png"))
    if not files:
        print("Queue is empty — nothing to review.")
        return

    print(f"{len(files)} item(s) queued for review.\n")
    labeled, discarded = 0, 0

    for i, filename in enumerate(files, start=1):
        path = os.path.join(QUEUE_DIR, filename)
        img = cv2.imread(path)
        if img is None:
            os.remove(path)
            continue

        zoomed = cv2.resize(img, (img.shape[1] * DISPLAY_SCALE, img.shape[0] * DISPLAY_SCALE),
                             interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Review (check this window, then type the answer in your terminal)", zoomed)
        cv2.waitKey(1)  # render the window without blocking on it

        typed = input(
            f"[{i}/{len(files)}] Enter level value (e.g. 3), or press Enter to discard "
            f"(not a real badge): "
        ).strip()

        if typed:
            if not typed.isdigit() or not (LEVEL_MIN_VALUE <= int(typed) <= LEVEL_MAX_VALUE):
                print(f"  Not a valid level ({LEVEL_MIN_VALUE}-{LEVEL_MAX_VALUE}) — discarding instead.")
                typed = ""
        if typed:
            add_level_example(typed, img)
            labeled += 1
            print(f"  Filed as level={typed}.")
        else:
            discarded += 1
            print("  Discarded.")

        os.remove(path)

    cv2.destroyAllWindows()
    print(f"\nDone — {labeled} labeled, {discarded} discarded.")


if __name__ == "__main__":
    main()
