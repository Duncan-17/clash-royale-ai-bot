"""
Downloads reference card icon images (Supercell's own art assets, from the
iconUrls field the Clash Royale API already gave us in config/cards.json)
into assets/card_icons/. These become the templates card_detector.py will
match hand slots against.

Usage:
    python -m perception.download_card_icons
"""

import json
import os
import re

import requests

CARDS_JSON = "config/cards.json"
OUT_DIR = "assets/card_icons"


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name.lower())


def main():
    with open(CARDS_JSON) as f:
        cards = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)

    downloaded, skipped = 0, 0
    for card in cards:
        name = card.get("name", "unknown")
        icon_urls = card.get("iconUrls", {})
        url = icon_urls.get("medium") or icon_urls.get("evolutionMedium")

        if not url:
            print(f"  SKIP  {name} — no icon URL in this card's data")
            skipped += 1
            continue

        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"  FAIL  {name} — HTTP {resp.status_code}")
            skipped += 1
            continue

        out_path = os.path.join(OUT_DIR, f"{safe_filename(name)}.png")
        with open(out_path, "wb") as f:
            f.write(resp.content)
        downloaded += 1

    print(f"\nDownloaded {downloaded} icons to {OUT_DIR}/ ({skipped} skipped)")


if __name__ == "__main__":
    main()
