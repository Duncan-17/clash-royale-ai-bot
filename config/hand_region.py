"""
Pixel geometry of the 4-card hand row during a live match, calibrated
against a real Training Camp screenshot (portrait resolution, ~900x1600).

Calibrated from two clicks (card 1's corners, card 4's corners) via
find_coords.py — cards 2 and 3 are derived, not independently measured,
since the row is evenly spaced.

Re-calibrate if LDPlayer's resolution setting ever changes — these are
raw pixel coordinates, not proportional/relative ones.
"""

CARD_LEFT_START = 202
CARD_TOP = 1331
CARD_WIDTH = 155      # average of card 1 (153) and card 4 (156)
CARD_HEIGHT = 189
CARD_SPACING = 169    # left-edge-to-left-edge distance between adjacent slots
HAND_SIZE = 4


def hand_slot_regions():
    """Returns 4 (x1, y1, x2, y2) boxes, left to right, for the hand row."""
    regions = []
    for i in range(HAND_SIZE):
        x1 = CARD_LEFT_START + i * CARD_SPACING
        y1 = CARD_TOP
        x2 = x1 + CARD_WIDTH
        y2 = y1 + CARD_HEIGHT
        regions.append((x1, y1, x2, y2))
    return regions
