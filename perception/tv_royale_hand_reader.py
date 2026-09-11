"""Reads both players' hands from a TV Royale spectator frame, handling the
selected/raised-card state, for the auto-labeling pipeline (see
data/tv_royale_watcher.py and clash-royale-ai-pipeline-explained.md section 5).

Why this exists on top of card_detector.identify_hand: in TV Royale a player
can select any hand card (it lifts ~SELECTED_SHIFT_Y px toward the arena to
show it's picked up) and un-select it freely; the card is only PLAYED when
they then tap the arena, at which point it leaves the hand and the queue's
next card fills the slot. Two consequences this reader handles:

1. A selected card's art is shifted OUT of its resting box, so reading the
   resting box alone returns a bad/blank match (measured: the selected
   fireball scored 0.40 at rest, 0.85 at the +25 shift). So each slot is
   read at BOTH its resting box and the shifted box, and the better match
   wins — a selected card still reads correctly.
2. Whether a slot is currently selected is itself a useful signal: it means
   that card is POISED to be played. It does NOT mean it was played —
   selection is reversible and spends no elixir. The authoritative "played"
   signal stays with game_knowledge/cycle_tracker.py: a slot's confirmed
   card being REPLACED by a new one, cross-checked against an elixir drop.
   is_selected is a corroborating/timing hint layered on top of that, not a
   replacement for it.

This reader reports the RAW best match per slot (card + score) plus the
selection info; it does NOT force "UNKNOWN" below a threshold — the consumer
(each CycleTracker, constructed with its own confidence_threshold) decides
what to trust. The TVR threshold is lower than live play's 0.85 because
matching is against weak catalog SEEDS restricted to the 16 known deck cards
(config/tv_royale_decks.py) — tune it against real footage.

Usage:
    from perception.card_detector import load_templates
    from config.tv_royale_decks import restricted_templates
    tpl = restricted_templates(load_templates())
    enemy, friendly = read_both_hands(screenshot, tpl)
    # feed enemy/friendly (each a list of 4 SlotRead) to two CycleTrackers
"""

from collections import namedtuple

from config.tv_royale_regions import (
    enemy_hand_slots, friendly_hand_slots, selected_shift,
)
from perception.card_detector import best_match

# result[0]=card, result[1]=score keeps this drop-in compatible with
# cycle_tracker.update's slot_results (it reads the first two fields).
SlotRead = namedtuple("SlotRead", "card score is_selected resting_score shifted_score")

# How much better the shifted read must score than the resting read to call a
# slot "selected". A non-selected card reads well at rest and only slightly
# worse when the box is nudged toward the arena (still mostly on the art), so
# shifted rarely beats resting; a selected card's art has actually moved, so
# shifted wins clearly. First-pass margin — tune against real footage.
SELECT_MARGIN = 0.10


def _read_slot(screenshot, box, shift, templates, lazy_conf=None):
    h = screenshot.shape[0]
    x1, y1, x2, y2 = box
    name_r, score_r = best_match(screenshot[y1:y2, x1:x2], templates)

    # Perf: a card that isn't selected reads WELL at its resting box, and a
    # selected one reads poorly there (its art is shifted out). So when the
    # resting match is already confident, skip the second (shifted) match
    # entirely -- it can't have been selected. Roughly halves the common-case
    # cost (most slots aren't selected). lazy_conf=None keeps the old
    # always-both-positions behavior.
    if lazy_conf is not None and score_r >= lazy_conf:
        return SlotRead(name_r, score_r, False, round(score_r, 3), None)

    sy1, sy2 = max(0, y1 + shift), min(h, y2 + shift)
    name_s, score_s = best_match(screenshot[sy1:sy2, x1:x2], templates)

    is_selected = score_s > score_r + SELECT_MARGIN
    if is_selected:
        card, score = name_s, score_s
    else:
        card, score = name_r, score_r
    return SlotRead(card, score, is_selected, round(score_r, 3), round(score_s, 3))


def read_hand(screenshot, templates, is_enemy: bool, lazy_conf=None):
    """List of 4 SlotRead (left->right) for one player's hand. is_enemy=True
    reads the TOP player, False the BOTTOM player. lazy_conf (see _read_slot)
    skips the shifted read when the resting match already clears it -- a big
    speedup for the live view, at no cost to selection detection."""
    slots = enemy_hand_slots() if is_enemy else friendly_hand_slots()
    shift = selected_shift(is_enemy)
    return [_read_slot(screenshot, box, shift, templates, lazy_conf) for box in slots]


def read_both_hands(screenshot, templates, lazy_conf=None):
    """(enemy_reads, friendly_reads) — one list of 4 SlotRead per player."""
    return (read_hand(screenshot, templates, True, lazy_conf),
            read_hand(screenshot, templates, False, lazy_conf))


if __name__ == "__main__":
    import cv2
    from perception.card_detector import load_templates
    from config.tv_royale_decks import restricted_templates, combined_cards

    screenshot = cv2.imread("test_screenshot.png")
    if screenshot is None:
        raise FileNotFoundError("test_screenshot.png not found — capture a TV Royale frame first.")

    # Restrict to the 16 deck cards if config/tv_royale_decks.py is filled;
    # otherwise fall back to the full catalog library.
    templates = restricted_templates(load_templates())
    deck = combined_cards()
    print(f"Matching against {len(templates)} card(s)"
          f"{' (restricted to both decks)' if deck else ' (full library - decks not filled in)'}.\n")

    def show(label, reads):
        print(f"{label}:")
        for i, r in enumerate(reads, 1):
            star = "  <-- SELECTED" if r.is_selected else ""
            print(f"  slot {i}: {r.card:14s} score={r.score:.2f} "
                  f"(rest={r.resting_score} shift={r.shifted_score}){star}")

    enemy, friendly = read_both_hands(screenshot, templates)
    show("ENEMY (top player)", enemy)
    show("FRIENDLY (bottom player)", friendly)
