"""
Identifies which card is in each of the 4 hand slots by comparing a crop
of each slot against reference examples in assets/hand_templates/.

Templates are stored as MULTIPLE example images per card
(assets/hand_templates/<card_name>/*.png), not one frozen file per card —
matching against many real examples is what makes this robust to
real-world variation (different animation moments, the low-elixir grey
state, the in-between wipe transition).

Files are saved and loaded as TRUE single-channel grayscale (not color
converted-to-look-grayscale) — this is deliberate, not just an optimization:
an earlier version stored full-color crops, and a mismatch between PIL's
RGB channel order and OpenCV's BGR convention meant every saved file had
its red and blue channels swapped ("blue and yellow" cards). Storing
genuine single-channel data removes the possibility of that bug recurring,
since there's no channel order left to get wrong, and it also means an
opened file shows exactly what the matching algorithm is comparing —
no separate "for display" vs "for matching" representation to drift apart.

Usage:
    python -m perception.card_detector
Reads test_screenshot.png, prints the best-guess card name + confidence
for each of the 4 hand slots.
"""

import os
import time

import cv2
import numpy as np

from config.hand_region import hand_slot_regions

TEMPLATE_DIR = "assets/hand_templates"
MATCH_SIZE = (64, 64)  # both crop and every example get resized to this before comparing.
# Lowered from (128,128) on 2026-08-15 once capture stopped being the bottleneck (see
# perception/capture.py) and card matching itself became the dominant per-frame cost
# (~145ms at 128x128 against ~210 stored examples). Verified against real stored examples
# before changing: discriminative margin between the correct card and the next-best-wrong
# one stays ~0.14-0.30 at every size tested down to 32x32 — no accuracy tradeoff at 64x64,
# just ~4x less matching time (~35ms for 4 slots).
CONFIDENCE_THRESHOLD = 0.85  # below this, report UNKNOWN rather than a bad guess
DUPLICATE_THRESHOLD = 0.995  # above this, a new example is treated as a near-duplicate and skipped
MAX_EXAMPLES_PER_CARD = 40  # once a card has this many REAL examples on file, stop adding more


def to_full_gray(img: np.ndarray) -> np.ndarray:
    """Converts a color (BGR) crop to single-channel grayscale at its
    native resolution — no resizing. This is what gets saved to disk, so
    stored files keep full detail (useful if a future phase wants higher-
    res crops) and are unambiguous single-channel data."""
    if img.ndim == 2:
        return img  # already grayscale
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Resize to MATCH_SIZE for comparison. Accepts either a color (BGR)
    crop straight from a screenshot, or an already-grayscale image loaded
    from disk — handles both so the same function works for live crops and
    stored templates alike."""
    gray = to_full_gray(img)
    return cv2.resize(gray, MATCH_SIZE)


def load_templates() -> dict:
    """Returns {card_name: [gray_example, gray_example, ...]}."""
    templates = {}
    if not os.path.isdir(TEMPLATE_DIR):
        return templates
    for name in os.listdir(TEMPLATE_DIR):
        card_dir = os.path.join(TEMPLATE_DIR, name)
        if not os.path.isdir(card_dir):
            continue  # skip stray files
        examples = []
        for filename in os.listdir(card_dir):
            if not filename.endswith(".png"):
                continue
            img = cv2.imread(os.path.join(card_dir, filename), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            examples.append(cv2.resize(img, MATCH_SIZE))
        if examples:
            templates[name] = examples
    return templates


def best_match(crop: np.ndarray, templates: dict):
    """Returns (card_name, confidence 0-1): the best score against ANY
    stored example of the winning card. `crop` should be a color (BGR)
    image straight from a screenshot — converted to grayscale here."""
    if not templates:
        return None, 0.0
    resized_gray = to_gray(crop)
    best_name, best_score = None, -1.0
    for name, examples in templates.items():
        for example in examples:
            result = cv2.matchTemplate(resized_gray, example, cv2.TM_CCOEFF_NORMED)
            score = float(result[0][0])
            if score > best_score:
                best_name, best_score = name, score
    return best_name, best_score


def identify_hand(screenshot: np.ndarray, templates: dict) -> list:
    """
    Returns a list of (card_name, confidence) for each of the 4 hand slots.
    card_name is "UNKNOWN" when the best match falls below CONFIDENCE_THRESHOLD.
    """
    results = []
    for (x1, y1, x2, y2) in hand_slot_regions():
        crop = screenshot[y1:y2, x1:x2]
        name, score = best_match(crop, templates)
        if score < CONFIDENCE_THRESHOLD:
            name = "UNKNOWN"
        results.append((name, score))
    return results


def add_example(name: str, crop: np.ndarray):
    """
    Saves `crop` as a new grayscale example for card `name`, deduped
    against whatever examples that card already has on file. `crop` can be
    a color (BGR) live capture or an already-grayscale image (e.g. a
    catalog icon) — either way, what's written to disk is single-channel
    grayscale. Returns the saved file path on success, or None if it was
    skipped (either a near-duplicate, or the card already has
    MAX_EXAMPLES_PER_CARD real examples on file).

    The cap exists so that cards you already play a lot don't quietly grow
    into folders of hundreds of near-redundant images while a card you
    just added to your deck is still trying to build up its first handful
    of examples — auto_collect.py calls this every ~1s during a match for
    every recognized slot, so without a cap, well-known cards would just
    keep accumulating indefinitely for as long as you keep playing.
    catalog_seed.png doesn't count against the cap (it's a placeholder
    that gets naturally out-scored once real examples exist, not part of
    the "real coverage" count).
    """
    card_dir = os.path.join(TEMPLATE_DIR, name)
    os.makedirs(card_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(card_dir) if f.endswith(".png")]
    real_example_count = sum(1 for f in existing_files if f != "catalog_seed.png")
    if real_example_count >= MAX_EXAMPLES_PER_CARD:
        return None

    gray_full = to_full_gray(crop)
    gray_small = to_gray(crop)

    for filename in existing_files:
        existing = cv2.imread(os.path.join(card_dir, filename), cv2.IMREAD_GRAYSCALE)
        if existing is None:
            continue
        result = cv2.matchTemplate(gray_small, cv2.resize(existing, MATCH_SIZE), cv2.TM_CCOEFF_NORMED)
        if float(result[0][0]) > DUPLICATE_THRESHOLD:
            return None

    out_path = os.path.join(card_dir, f"{int(time.time() * 1000)}.png")
    cv2.imwrite(out_path, gray_full)
    return out_path


if __name__ == "__main__":
    screenshot = cv2.imread("test_screenshot.png")
    if screenshot is None:
        raise FileNotFoundError("test_screenshot.png not found — capture one first.")

    templates = load_templates()
    if not templates:
        print("No templates in assets/hand_templates/ yet — run label_hand.py or seed_deck_from_catalog.py first.")
    else:
        total_examples = sum(len(v) for v in templates.values())
        print(f"Loaded {total_examples} example(s) across {len(templates)} card(s).\n")
        for i, (name, score) in enumerate(identify_hand(screenshot, templates), start=1):
            print(f"Slot {i}: {name:20s} confidence={score:.3f}")
