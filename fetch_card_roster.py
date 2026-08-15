"""
Phase 2 smoke test — pulls the full card roster (name, elixir cost, rarity)
from the Clash Royale API and saves it to config/cards.json.

Usage:
    python fetch_card_roster.py
"""

import json

from data.clash_royale_api import ClashRoyaleAPI


def main():
    print("Connecting to the Clash Royale API...")
    api = ClashRoyaleAPI()

    print("Fetching card roster...")
    cards = api.get_cards()

    out_path = "config/cards.json"
    with open(out_path, "w") as f:
        json.dump(cards, f, indent=2)

    print(f"Saved {len(cards)} cards to {out_path}")
    print("\nSample:")
    for card in cards[:5]:
        print(f"  {card['name']:20s} elixir={card.get('elixirCost', '?'):<3} rarity={card['rarity']}")


if __name__ == "__main__":
    main()
