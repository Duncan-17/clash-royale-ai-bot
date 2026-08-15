"""
One-time cleanup for files saved before the RGB/BGR channel-swap bug was
fixed (see capture.py and card_detector.py's docstrings for the full
history). Those files aren't just cosmetically wrong — a straight grayscale
conversion of a channel-swapped image gives the WRONG luminance value for
every pixel (grayscale = 0.299*R + 0.587*G + 0.114*B; swap R and B first
and you get a different number, not just a different-looking image), so
they can't simply be converted to grayscale in place and trusted. Some
happened to look "fine" by eye (e.g. a card whose true red and blue channel
values were already close together), but that's luck, not evidence the
underlying data is correct — no per-file visual check is reliable enough to
sort good from bad.

This deletes the two categories of pre-fix, not-safely-repairable files:

1. Every file literally named "seed.png" under assets/hand_templates/<card>/
   — these came from migrate_templates.py's very first run, before
   capture.py was fixed. (catalog_seed.png files are NOT touched — those
   are written by seed_deck_from_catalog.py directly from the downloaded
   catalog icon, never touched by the buggy live-capture path, so they're
   fine as-is.)
2. Everything in assets/unlabeled_queue/ — crops queued during pre-fix
   auto_collect.py runs, same exposure.

Nothing here is data you'll miss: run seed_deck_from_catalog.py afterward
to refill any card left with zero examples (fresh, correctly-grayscale
catalog seeds), then run auto_collect.py during a match to rebuild real
examples — faster than the manual labeling that produced these files the
first time around, and everything it saves from now on is genuinely
single-channel grayscale from the start.

Usage:
    python -m perception.cleanup_pre_fix_files
"""

import os

TEMPLATE_DIR = "assets/hand_templates"
QUEUE_DIR = "assets/unlabeled_queue"


def main():
    removed = []

    if os.path.isdir(TEMPLATE_DIR):
        for name in os.listdir(TEMPLATE_DIR):
            card_dir = os.path.join(TEMPLATE_DIR, name)
            if not os.path.isdir(card_dir):
                continue
            seed_path = os.path.join(card_dir, "seed.png")
            if os.path.isfile(seed_path):
                os.remove(seed_path)
                removed.append(seed_path)

    queue_removed = 0
    if os.path.isdir(QUEUE_DIR):
        for filename in os.listdir(QUEUE_DIR):
            if filename.endswith(".png"):
                os.remove(os.path.join(QUEUE_DIR, filename))
                queue_removed += 1

    print(f"Removed {len(removed)} pre-fix seed.png file(s):")
    for p in removed:
        print(f"  {p}")
    print(f"\nRemoved {queue_removed} pre-fix queued crop(s) from {QUEUE_DIR}.")
    print(
        "\nNext steps:\n"
        "  1. python -m perception.seed_deck_from_catalog   (refill any card now at 0 examples)\n"
        "  2. Play a match, run python -m perception.auto_collect during it, to rebuild real examples."
    )


if __name__ == "__main__":
    main()
