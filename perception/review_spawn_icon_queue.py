"""
Batch-labels whatever auto_collect.py queued up as spawn-icon detections.
Shows each queued crop one at a time (magnified — real crops are tiny,
~35-45px), along with what perception.spawn_icon_reader.find_spawn_icon()
itself guessed (encoded in the filename, "_claimed_ally"/"_claimed_enemy")
as a STARTING SUGGESTION only, not a trusted label — Duncan makes the
actual call. Type "a" (ally), "e" (enemy), or press Enter to discard
(e.g. it isn't really a spawn icon at all — a false-positive detection).

Every detection from perception/auto_collect.py's spawn-icon collection
goes through this queue — nothing is ever auto-filed directly into
assets/spawn_icon_templates/ without this review step (see CLAUDE.md and
that collection loop's own comments for why: a new, unvalidated detector
writing straight into a permanent asset library with no human check
already produced one genuinely ambiguous, wrongly-unreviewed example).

Usage:
    python -m perception.review_spawn_icon_queue
"""

import os

import cv2

from perception.spawn_icon_reader import add_spawn_icon_example

QUEUE_DIR = "assets/unlabeled_spawn_icon_queue"
DISPLAY_SCALE = 8  # icon crops are tiny -- magnify for a human to actually read the fill color


def _claimed_side(filename: str):
    for side in ("ally", "enemy"):
        if filename.endswith(f"_claimed_{side}.png"):
            return side
    return None


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

        claimed = _claimed_side(filename)
        zoomed = cv2.resize(img, (img.shape[1] * DISPLAY_SCALE, img.shape[0] * DISPLAY_SCALE),
                             interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Review (check this window, then type the answer in your terminal)", zoomed)
        cv2.waitKey(1)  # render the window without blocking on it

        typed = input(
            f"[{i}/{len(files)}] detector guessed '{claimed}' -- "
            f"type 'a' (ally), 'e' (enemy), or press Enter to discard (not a real icon): "
        ).strip().lower()

        if typed in ("a", "ally"):
            add_spawn_icon_example("ally", img)
            labeled += 1
            print("  Filed as ally.")
        elif typed in ("e", "enemy"):
            add_spawn_icon_example("enemy", img)
            labeled += 1
            print("  Filed as enemy.")
        else:
            discarded += 1
            print("  Discarded.")

        os.remove(path)

    cv2.destroyAllWindows()
    print(f"\nDone — {labeled} labeled, {discarded} discarded.")


if __name__ == "__main__":
    main()
