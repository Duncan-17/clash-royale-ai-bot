"""Full-catalog bootstrap for the TV Royale pipeline: seeds EVERY card in
assets/card_icons/ into assets/hand_templates/<name>/catalog_seed.png, so
the hand reader can recognize arbitrary opponent decks — not just our own 8
(config/deck.txt), which is all perception/seed_deck_from_catalog.py covers.

Same weak-but-present catalog-seed idea as that script (see its docstring):
a catalog icon frames the card differently than the live hand render, so on
its own it's a low-confidence match — but TV Royale hands are read with
matching RESTRICTED to the 16 cards actually in the two decks (Duncan reads
both decks off the TV Royale menu before the match and supplies them, the
two-deck analogue of deck.txt), and against a 16-way choice "weak but
present" steers the first real occurrence to the right card. Confirmed
in-hand crops then get collected and queued for review to REPLACE the seed
with real TV-Royale-native examples over time.

Leaves any card that already has real examples alone (our 8 deck cards keep
their 40 real examples each). Idempotent — re-running only fills gaps.

Usage:
    python -m perception.seed_all_cards_from_catalog
"""

import os

import cv2
import numpy as np

from perception.card_detector import canonical_slug

CATALOG_DIR = "assets/card_icons"
TEMPLATE_DIR = "assets/hand_templates"
SEED_NAME = "catalog_seed.png"

# The catalog icons are full framed cards with transparency ONLY in the
# rounded-corner cutouts (identical corner mask on every card). A plain
# imread drops alpha and flattens those corners to WHITE, which dominates
# the TM_CCOEFF_NORMED correlation and made the seeds useless (Duncan's
# catch, 2026-08-25). Fix: alpha-composite the card over a neutral GREY
# background (his choice among black/grey/white — closest to the dark
# in-hand tray without white's over-strong signal) and crop to the art's
# bounding box so the seed isn't framed wider than the real in-hand crops.
SEED_BG_GRAY = 128
ALPHA_THRESHOLD = 16  # alpha above this counts as "art" for the bbox crop


def clean_icon_to_seed(icon_rgba: np.ndarray) -> np.ndarray:
    """RGBA catalog icon -> single-channel grayscale seed: alpha-composited
    over grey, cropped to the non-transparent art. Falls back gracefully if
    the icon has no alpha channel (treats it as fully opaque)."""
    if icon_rgba.ndim == 3 and icon_rgba.shape[2] == 4:
        alpha = icon_rgba[:, :, 3]
        rgb = icon_rgba[:, :, :3].astype(float)
        a = (alpha.astype(float) / 255.0)[:, :, None]
        composited = (rgb * a + SEED_BG_GRAY * (1 - a)).astype(np.uint8)
        ys, xs = np.where(alpha > ALPHA_THRESHOLD)
        if len(ys):
            composited = composited[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    else:
        composited = icon_rgba[:, :, :3] if icon_rgba.ndim == 3 else icon_rgba
    return cv2.cvtColor(composited, cv2.COLOR_BGR2GRAY) if composited.ndim == 3 else composited


def main():
    icons = sorted(f for f in os.listdir(CATALOG_DIR) if f.endswith(".png"))
    seeded, already_has_real, reseeded, failed = [], [], [], []

    for filename in icons:
        name = canonical_slug(filename[:-4])  # strip .png, then readable-slug the awkward ones
        card_dir = os.path.join(TEMPLATE_DIR, name)
        real_examples = []
        if os.path.isdir(card_dir):
            real_examples = [f for f in os.listdir(card_dir)
                             if f.endswith(".png") and f != SEED_NAME]
        if real_examples:
            already_has_real.append(name)
            continue

        icon = cv2.imread(os.path.join(CATALOG_DIR, filename), cv2.IMREAD_UNCHANGED)
        if icon is None:
            failed.append(name)
            continue
        gray = clean_icon_to_seed(icon)
        os.makedirs(card_dir, exist_ok=True)
        seed_path = os.path.join(card_dir, SEED_NAME)
        (reseeded if os.path.isfile(seed_path) else seeded).append(name)
        cv2.imwrite(seed_path, gray)

    print(f"Catalog icons found: {len(icons)}")
    print(f"Newly seeded: {len(seeded)}")
    print(f"Seed refreshed (already had a seed): {len(reseeded)}")
    print(f"Left alone (already have real examples): {len(already_has_real)} -> {already_has_real}")
    if failed:
        print(f"FAILED to read icon for: {failed}")


if __name__ == "__main__":
    main()
