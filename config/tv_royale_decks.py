"""Both players' 8-card decks for the CURRENT TV Royale match, read off the
TV Royale menu before the match starts (Duncan can see both full decks
there). Filling these in restricts hand-card matching to just these 16
cards instead of the whole ~122-card catalog — the two-deck analogue of
config/deck.txt for live play, and what turns the weak catalog seeds
(assets/hand_templates/<card>/catalog_seed.png) into reliable reads.

Edit ENEMY_DECK (top player) and FRIENDLY_DECK (bottom player) before each
match. Card names are forgiving — normalize() lowercases, turns spaces into
underscores, and applies ALIASES — so "Mini P.E.K.K.A", "mini pekka" and
"mini_pekka" all resolve to the catalog slug. restricted_templates() then
filters a loaded template library down to these decks, and warns about any
name that doesn't match a real assets/hand_templates/<slug>/ folder so a
typo surfaces immediately instead of silently dropping a card.

Leave a deck empty ([]) to fall back to matching against the full library
for that player (useful before you've entered the decks, or for a quick
smoke test).
"""

import os

TEMPLATE_DIR = "assets/hand_templates"

# Fill these in per match (top player / bottom player). 8 slugs each.
# Set 2026-09-07 for Duncan's test match.
ENEMY_DECK = [       # top player
    "skeletons", "mega_minion", "goblin_cage", "arrows",
    "fireball", "battle_ram", "wizard", "fire_spirit",
]
FRIENDLY_DECK = [    # bottom player
    "wizard", "knight", "prince", "valkyrie",
    "fireball", "balloon", "bomb_tower", "mini_pekka",
]

# Friendly-name -> catalog slug for the cards whose folder name isn't just
# the lowercased spaced name. Extend as needed.
ALIASES = {
    # pekka/mini_pekka are the canonical readable slugs now (2026-09-07); the
    # awkward card_icons spellings are folded to them at icon-load time, so
    # nothing here needs to point at p_e_k_k_a anymore.
    "minipekka": "mini_pekka",
    "babydragon": "baby_dragon",
    "giants": "giant",
    "xbow": "x-bow",
    "x_bow": "x-bow",
    "log": "the_log",
    "ewiz": "electro_wizard",
    "ebarbs": "elite_barbarians",
    "mk": "mega_knight",
}


def normalize(name: str) -> str:
    """A human-typed card name -> its assets/hand_templates/ folder slug."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    key = key.replace(".", "")  # "p.e.k.k.a" -> "pekka" before alias lookup
    if key in ALIASES:
        return ALIASES[key]
    # default: spaces->underscores, keep as-is (covers goblins, hog_rider, ...)
    return name.strip().lower().replace(" ", "_")


def combined_cards() -> list:
    """The up-to-16 normalized card slugs across both decks (order preserved,
    de-duplicated in case both players run the same card)."""
    seen, out = set(), []
    for name in list(ENEMY_DECK) + list(FRIENDLY_DECK):
        slug = normalize(name)
        if slug not in seen:
            seen.add(slug)
            out.append(slug)
    return out


def _validate(slugs):
    missing = [s for s in slugs if not os.path.isdir(os.path.join(TEMPLATE_DIR, s))]
    if missing:
        print(f"[tv_royale_decks] WARNING: no hand_templates folder for {missing} "
              f"— check the name (run perception/seed_all_cards_from_catalog.py "
              f"names, or config/cards.json).")
    return missing


def restricted_templates(all_templates: dict, deck=None) -> dict:
    """Filter a loaded {card: [examples]} library down to the given deck
    (default: both decks combined). Returns the full library unchanged if
    the deck is empty, so an unfilled config falls back to full matching."""
    slugs = [normalize(n) for n in deck] if deck is not None else combined_cards()
    if not slugs:
        return all_templates
    _validate(slugs)
    return {name: ex for name, ex in all_templates.items() if name in slugs}
