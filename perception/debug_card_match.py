"""
Visual debugging for card_detector.py: saves a side-by-side image showing
each hand slot's actual crop next to whichever specific stored example it
matched against, so mismatches are visible instead of just a wrong name
and a number.

Usage:
    python -m perception.debug_card_match
Writes debug_hand_match.png — open it to see what's actually being compared.
"""

import os

import cv2
import numpy as np

from config.hand_region import hand_slot_regions
from perception.card_detector import MATCH_SIZE, load_templates, to_gray

TEMPLATE_DIR = "assets/hand_templates"


def best_match_with_source(crop, templates):
    """Like card_detector.best_match, but also returns the winning
    example's original color image (for display) rather than just its
    grayscale in-memory version — the grayscale versions in `templates`
    don't carry a file path back to disk, so this re-reads whichever file
    scored best."""
    resized_gray = to_gray(crop)
    best_name, best_score, best_idx = None, -1.0, None
    for name, examples in templates.items():
        for i, example in enumerate(examples):
            result = cv2.matchTemplate(resized_gray, example, cv2.TM_CCOEFF_NORMED)
            score = float(result[0][0])
            if score > best_score:
                best_name, best_score, best_idx = name, score, i

    if best_name is None:
        return None, 0.0, None

    # Re-find the actual file on disk for the winning (name, index) pair,
    # so we can show it in real color rather than the stored grayscale.
    card_dir = os.path.join(TEMPLATE_DIR, best_name)
    files = sorted(f for f in os.listdir(card_dir) if f.endswith(".png"))
    winning_file = os.path.join(card_dir, files[best_idx])
    winning_img = cv2.imread(winning_file)

    return best_name, best_score, winning_img


def main():
    screenshot = cv2.imread("test_screenshot.png")
    templates = load_templates()

    rows = []
    for i, (x1, y1, x2, y2) in enumerate(hand_slot_regions(), start=1):
        crop = screenshot[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, MATCH_SIZE)

        name, score, matched_img = best_match_with_source(crop, templates)
        matched_resized = cv2.resize(matched_img, MATCH_SIZE) if matched_img is not None else np.zeros_like(crop_resized)

        label = np.full((30, MATCH_SIZE[0] * 2, 3), 40, dtype=np.uint8)
        cv2.putText(label, f"slot{i}: {name} ({score:.2f})", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        pair = np.hstack([crop_resized, matched_resized])
        row = np.vstack([label, pair])
        rows.append(row)

    montage = np.vstack(rows)
    cv2.imwrite("debug_hand_match.png", montage)
    print("Saved debug_hand_match.png — left = actual hand slot crop, right = matched example.")


if __name__ == "__main__":
    main()
