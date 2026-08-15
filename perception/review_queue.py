"""
Batch-labels whatever auto_collect.py queued up as unrecognized. Shows
each queued crop one at a time; type the correct name to file it as a new
training example for that card, or press Enter to discard it (e.g. it's a
capture glitch, or a transitional frame you don't want to keep).

Usage:
    python -m perception.review_queue
"""

import os

import cv2

from perception.card_detector import add_example

QUEUE_DIR = "assets/unlabeled_queue"


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

        cv2.imshow("Review (check this window, then type the answer in your terminal)", img)
        cv2.waitKey(1)  # render the window without blocking on it

        typed = input(
            f"[{i}/{len(files)}] Enter card name (lowercase_with_underscores), "
            f"or press Enter to discard: "
        ).strip()

        if typed:
            add_example(typed, img)
            labeled += 1
            print(f"  Filed as {typed}.")
        else:
            discarded += 1
            print("  Discarded.")

        os.remove(path)

    cv2.destroyAllWindows()
    print(f"\nDone — {labeled} labeled, {discarded} discarded.")


if __name__ == "__main__":
    main()
