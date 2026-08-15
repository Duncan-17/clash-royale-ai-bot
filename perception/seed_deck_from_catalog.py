"""
One-time-per-card bootstrap: for every card in config/deck.txt that has NO
real examples yet in assets/hand_templates/<name>/, seeds that folder with
its Supercell catalog icon (assets/card_icons/<name>.png, already
downloaded by download_card_icons.py) as a starting reference.

This is a weak match on its own (see debug_card_match.py's original
findings — catalog icons use different framing than the live hand render),
but with auto_collect.py restricting matching to just the 8 deck cards
instead of the full ~122-card library, "weak but present" is usually
enough to steer the first real occurrence toward the right guess. Any
misfire gets caught in the normal post-run pruning pass.

The catalog icon is downloaded as a normal color PNG, but everything in
assets/hand_templates/ is now stored as single-channel grayscale (see
card_detector.py's docstring for why — this eliminates the RGB/BGR
channel-swap bug that previously corrupted stored templates). So this
script converts the icon to grayscale before saving it, instead of copying
the color file directly — a plain shutil.copy would silently reintroduce a
color file into an otherwise all-grayscale folder, and card_detector.py's
load_templates() would then load it as grayscale via cv2.IMREAD_GRAYSCALE
anyway, just inconsistently from how it was stored on disk.

The seed is named catalog_seed.png specifically so it's easy to spot and
delete later once real examples exist for that card (optional cleanup —
harmless to leave in place either way, since matching takes the best score
across all of a card's examples, and a real example will typically
outscore the catalog seed once one exists).

Usage:
    python -m perception.seed_deck_from_catalog
"""

import os

import cv2

DECK_FILE = "config/deck.txt"
CATALOG_DIR = "assets/card_icons"
TEMPLATE_DIR = "assets/hand_templates"


def load_deck() -> list:
    if not os.path.isfile(DECK_FILE):
        raise FileNotFoundError(f"{DECK_FILE} not found — create it first (one card name per line).")
    names = []
    with open(DECK_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    return names


def main():
    deck = load_deck()
    if len(deck) != 8:
        print(f"Warning: {DECK_FILE} has {len(deck)} card(s), not 8 — continuing anyway.")

    seeded, already_has_examples, no_catalog_icon = [], [], []

    for name in deck:
        card_dir = os.path.join(TEMPLATE_DIR, name)
        has_real_examples = os.path.isdir(card_dir) and any(
            f.endswith(".png") for f in os.listdir(card_dir)
        )
        if has_real_examples:
            already_has_examples.append(name)
            continue

        catalog_icon = os.path.join(CATALOG_DIR, f"{name}.png")
        if not os.path.isfile(catalog_icon):
            no_catalog_icon.append(name)
            continue

        icon = cv2.imread(catalog_icon)  # color, straight from the downloaded catalog file
        if icon is None:
            no_catalog_icon.append(name)
            continue
        gray = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)

        os.makedirs(card_dir, exist_ok=True)
        cv2.imwrite(os.path.join(card_dir, "catalog_seed.png"), gray)
        seeded.append(name)

    print(f"Seeded {len(seeded)} card(s) from the catalog: {seeded}")
    if already_has_examples:
        print(f"Already had real examples, left alone: {already_has_examples}")
    if no_catalog_icon:
        print(
            f"No catalog icon found for: {no_catalog_icon} — check the name matches "
            f"config/cards.json exactly (see fetch_card_roster.py's output), or these "
            f"will need a manual first example via label_hand.py."
        )


if __name__ == "__main__":
    main()
