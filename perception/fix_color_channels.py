"""
One-time repair for the red/blue channel-swap bug (see capture.py's
updated docstring for the root cause): every file that was saved via
add_example() before the fix — everything in assets/hand_templates/ except
catalog_seed.png files (those came straight from the downloaded Supercell
icons, untouched by the buggy pipeline), plus everything in
assets/unlabeled_queue/ — has its red and blue channels swapped.

This corrects them IN PLACE. No data is lost or needs to be re-collected;
a channel swap is fully reversible.

Usage:
    python -m perception.fix_color_channels
"""

import os

import cv2

TARGET_DIRS = ["assets/hand_templates", "assets/unlabeled_queue"]
SKIP_FILENAMES = {"catalog_seed.png"}  # never touched by the buggy pipeline — already correct


def fix_file(path: str) -> bool:
    img = cv2.imread(path)
    if img is None:
        return False
    fixed = img[:, :, ::-1]  # reverses R<->B, undoing the swap
    cv2.imwrite(path, fixed)
    return True


def main():
    fixed_count, skipped_count = 0, 0

    for base_dir in TARGET_DIRS:
        if not os.path.isdir(base_dir):
            continue
        for root, _dirs, files in os.walk(base_dir):
            for filename in files:
                if not filename.endswith(".png"):
                    continue
                if filename in SKIP_FILENAMES:
                    skipped_count += 1
                    continue
                path = os.path.join(root, filename)
                if fix_file(path):
                    fixed_count += 1
                    print(f"Fixed {path}")

    print(f"\nDone — fixed {fixed_count} file(s), skipped {skipped_count} catalog seed(s) (already correct).")
    print("Open a few fixed files to confirm colors look right now.")


if __name__ == "__main__":
    main()
