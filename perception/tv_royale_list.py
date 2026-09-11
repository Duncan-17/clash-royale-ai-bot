"""Parses the TV Royale list screen: finds each game's Watch button and reads
both players' full 8-card decks off the row. This is what makes the macro
(data/tv_royale_watcher.py) able to pick a game, KNOW both decks before
watching (no API needed -- see Project_Status), and click in.

Calibrated 2026-09-07 against temp_img/tvr_list.png (900x1600), confirmed by
Duncan. The list is a scrollable column of game rows; each row shows the left
player's 8 cards (4x2) on the left, the right player's 8 (4x2) on the right,
and a green Watch button at a fixed x. Watch buttons are found by template
match (assets/tv_royale/watch_button.png), so rows are located dynamically
wherever they scroll to -- only the button's X and the card COLUMNS are fixed;
the row's Y comes from the detected button.

Deck cards are matched against assets/card_icons/ (the shop icons), NOT the
in-hand hand_templates: the list uses the same framed-icon art as the catalog,
so it matches cleanly there. This is a different, easier read than the in-hand
one, and it's how decks get auto-detected per game.
"""

import os

import cv2
import numpy as np

from perception.card_detector import canonical_slug

WATCH_TEMPLATE = "assets/tv_royale/watch_button.png"
CARD_ICON_DIR = "assets/card_icons"
MATCH_SIZE = (64, 64)  # both list crop and icon template resized to this before matching

WATCH_MATCH_THRESHOLD = 0.6   # TM_CCOEFF_NORMED; real buttons score ~0.77-0.80
WATCH_MIN_Y_GAP = 200         # rows are ~439px apart -- suppress double-hits on one button

# Deck-card geometry relative to a detected Watch button's top-left (bx, by).
# Only the row Y shifts per game; the columns are fixed. (from tvr_list.png)
LEFT_COLS = [53, 148, 243, 338]     # left player's 4 card-column left edges
RIGHT_COLS = [460, 555, 650, 745]   # right player's 4 card-column left edges
CARD_W, CARD_H = 92, 100
ROW_DYS = [-223, -111]              # top-edge of the top / bottom card row, rel. to button top y


def _clean_icon(icon):
    """RGBA/BGR catalog icon -> COLOR crop of its art: alpha-composited over
    grey and cropped to the non-transparent bounding box. Color is kept on
    purpose -- Clash Royale cards are strongly color-coded, and color matching
    is what got the list read to ~near-perfect (validated 2026-09-07); the
    grayscale version confused e.g. mega_minion with magic_archer."""
    if icon.ndim == 3 and icon.shape[2] == 4:
        a = (icon[:, :, 3:4].astype(float)) / 255.0
        comp = (icon[:, :, :3].astype(float) * a + 128 * (1 - a)).astype(np.uint8)
        al = icon[:, :, 3]
        ys, xs = np.where(al > 16)
        if len(ys):
            comp = comp[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return comp
    return icon[:, :, :3] if icon.ndim == 3 else cv2.cvtColor(icon, cv2.COLOR_GRAY2BGR)


def load_icon_templates(card_dir=CARD_ICON_DIR):
    """{card_name: color_template(64x64)} from assets/card_icons/."""
    templates = {}
    for fname in os.listdir(card_dir):
        if not fname.endswith(".png"):
            continue
        icon = cv2.imread(os.path.join(card_dir, fname), cv2.IMREAD_UNCHANGED)
        if icon is None:
            continue
        templates[canonical_slug(fname[:-4])] = cv2.resize(_clean_icon(icon), MATCH_SIZE)
    return templates


def _match_card(crop, templates):
    """(card_name, score) best color match of a list-card crop."""
    a = cv2.resize(crop, MATCH_SIZE)
    best_name, best_score = None, -2.0
    for name, tmpl in templates.items():
        score = float(cv2.matchTemplate(a, tmpl, cv2.TM_CCOEFF_NORMED)[0][0])
        if score > best_score:
            best_name, best_score = name, score
    return best_name, best_score


def detect_watch_buttons(screenshot, template=None):
    """Returns detected Watch buttons as (top_left_x, top_left_y), top to
    bottom. Dynamic: works wherever rows have scrolled to."""
    if template is None:
        template = cv2.imread(WATCH_TEMPLATE)
    res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= WATCH_MATCH_THRESHOLD)
    kept = []
    for y, x in sorted(zip(ys, xs), key=lambda p: p[0]):
        if all(abs(y - ky) > WATCH_MIN_Y_GAP for ky, kx in kept):
            kept.append((y, x))
    th, tw = template.shape[:2]
    return [(x, y) for (y, x) in kept], (tw, th)


def _read_deck(screenshot, button_y, cols, templates):
    """8 (card_name, score) for one player, reading-order (top row L->R, then
    bottom row)."""
    deck = []
    for dy in ROW_DYS:
        for x in cols:
            y1 = button_y + dy
            crop = screenshot[y1:y1 + CARD_H, x:x + CARD_W]
            deck.append(_match_card(crop, templates))
    return deck


def deck_signature(game):
    """A hashable fingerprint of a game for dedup across scrolls/re-opens:
    the frozenset of both decks' card names. Two list rows with the same 16
    cards are the same game (good enough -- the exact same matchup appearing
    twice with identical decks is vanishingly unlikely)."""
    names = [c for c, _ in game["left_deck"]] + [c for c, _ in game["right_deck"]]
    return frozenset(names)


def parse_list(screenshot, templates=None):
    """Returns a list of games currently visible, top to bottom:
        {"watch_xy": (cx, cy),           # center of the Watch button to tap
         "left_deck":  [(card, score)*8],
         "right_deck": [(card, score)*8]}
    """
    if templates is None:
        templates = load_icon_templates()
    buttons, (tw, th) = detect_watch_buttons(screenshot)
    games = []
    for (bx, by) in buttons:
        games.append({
            "watch_xy": (int(bx + tw // 2), int(by + th // 2)),
            "left_deck": _read_deck(screenshot, by, LEFT_COLS, templates),
            "right_deck": _read_deck(screenshot, by, RIGHT_COLS, templates),
        })
    return games


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "temp_img/tvr_list.png"
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"{path} not found.")
    for i, g in enumerate(parse_list(img), 1):
        print(f"\nGame {i}  (watch at {g['watch_xy']}):")
        print("  left :", ", ".join(f"{c}({s:.2f})" for c, s in g["left_deck"]))
        print("  right:", ", ".join(f"{c}({s:.2f})" for c, s in g["right_deck"]))
