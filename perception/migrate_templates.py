"""
One-time migration: converts the old flat assets/hand_templates/<name>.png
layout to the new assets/hand_templates/<name>/<file>.png layout (multiple
examples per card, needed for auto_collect.py to add more over time).

Safe to run once; does nothing if there's nothing left to migrate.

Usage:
    python -m perception.migrate_templates
"""

import os
import shutil

TEMPLATE_DIR = "assets/hand_templates"


def main():
    if not os.path.isdir(TEMPLATE_DIR):
        print("No template directory found — nothing to migrate.")
        return

    moved = 0
    for filename in os.listdir(TEMPLATE_DIR):
        full_path = os.path.join(TEMPLATE_DIR, filename)
        if not os.path.isfile(full_path) or not filename.endswith(".png"):
            continue
        name = filename[:-4]
        card_dir = os.path.join(TEMPLATE_DIR, name)
        os.makedirs(card_dir, exist_ok=True)
        shutil.move(full_path, os.path.join(card_dir, "seed.png"))
        moved += 1
        print(f"Migrated {filename} -> {name}/seed.png")

    if moved == 0:
        print("Nothing to migrate — already on the new folder layout (or no templates yet).")
    else:
        print(f"\nDone — migrated {moved} card(s) to the new folder format.")


if __name__ == "__main__":
    main()
