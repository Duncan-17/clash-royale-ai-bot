"""
Pixel geometry of the TV Royale spectator layout, at this project's
portrait resolution (900x1600), calibrated 2026-08-25 against a real
TV Royale capture with Duncan (manual find_coords.py clicks — see
temp_img/tvr_calib_proposal.py for the confirmation overlay).

This is THE GATE the whole TV Royale auto-labeling pipeline waited on
(see data/tv_royale_watcher.py, data/auto_labeler.py, and Project_Status
"NEXT PHASE: TV ROYALE DATA COLLECTION"). TV Royale shows BOTH players'
hands + elixir at once, so game_knowledge/cycle_tracker.py — already
proven on our own side — can run as two instances to produce
authoritative card + player + timestamp for every play, with no board
inference. These regions are what feed those two instances.

Layout notes worth keeping:
- The two players are laid out VERTICALLY MIRRORED about the screen
  center (y=800). Confirmed numerically: enemy card top y=76 mirrors
  friendly card bottom y=1525 (76+1525 ~= 1600), and enemy card bottom
  208 mirrors friendly card top 1392 (208+1392 = 1600).
- "enemy" / "friendly" here are just labels for the TOP and BOTTOM
  spectated players respectively (Duncan's own terms during calibration).
  In TV Royale neither is really "ours" — TOP player owns the top towers,
  BOTTOM player owns the bottom towers. Each player's plays land in their
  own half of the arena.
- The pink badge on each hand card is its ELIXIR COST, not a level.
- A SELECTED card (one being placed) shifts ~SELECTED_SHIFT_Y px TOWARD
  the arena (down for the top player, up for the bottom player), same
  size — measured directly: the enemy fireball, selected in the
  calibration frame, sat at (359,101,465,234) vs its resting
  (358,76,464,208). Play detection must tolerate this transient shift on
  the slot that's mid-play, or a slot briefly reads empty/mismatched.

These are RAW pixel coordinates, not proportional — re-calibrate (same
find_coords.py process) if LDPlayer's resolution ever changes.
"""

# ---------------------------------------------------------------------------
# Hand rows — 4 evenly-spaced slots each. Slot 1 and slot 4 corners were
# clicked directly; slots 2 and 3 are derived by even spacing between them
# (same method config/hand_region.py uses for the live single-player hand).
# ---------------------------------------------------------------------------
HAND_SIZE = 4

# ENEMY = top player. slot1 (122,76,227,208), slot4 (476,76,584,208) measured.
ENEMY_HAND_LEFT_START = 122
ENEMY_HAND_TOP = 76
ENEMY_HAND_CARD_WIDTH = 106      # avg of slot1 (105) and slot4 (108)
ENEMY_HAND_CARD_HEIGHT = 132
ENEMY_HAND_SPACING = (476 - 122) / 3  # left-edge-to-left-edge = 118.0

# FRIENDLY = bottom player. slot1 (122,1392,227,1525), slot4 (480,1393,584,1525).
FRIENDLY_HAND_LEFT_START = 122
FRIENDLY_HAND_TOP = 1392
FRIENDLY_HAND_CARD_WIDTH = 104    # avg of slot1 (105) and slot4 (104)
FRIENDLY_HAND_CARD_HEIGHT = 133
FRIENDLY_HAND_SPACING = (480 - 122) / 3  # = 119.33

# "Next:" preview card (the single upcoming card, far left of each hand row).
ENEMY_NEXT = (30, 117, 83, 184)
FRIENDLY_NEXT = (30, 1406, 81, 1472)

# Elixir bars. Both bars share the SAME horizontal geometry (same UI asset,
# vertically mirrored) — only their y-band differs. Fill-scan anchors were
# calibrated 2026-08-25 by solving both bars simultaneously against ground
# truth (top=6 elixir, bottom=3): the fill boundaries at x=555 (6.0) and
# x=323 (3.0) give LEFT=91, RIGHT=864 for both — and RIGHT=864 matches the
# live single-player bar's own BAR_RIGHT_X (elixir_counter.py) exactly.
# Validated with elixir_counter.read_bar_fill_region: top reads 6.55, bottom
# 3.51 (integer part = ground truth; the fraction is real regen glow toward
# the next point). Duncan confirmed both visually (~6.5 / ~3.5).
ELIXIR_BAR_LEFT_X = 91.0    # 0-elixir mark (left edge of segment 1), both bars
ELIXIR_BAR_RIGHT_X = 864.0  # 10-elixir mark (right edge of segment 10), both bars
# Sample rows through each bar's flat body (avoiding the top/bottom bevel and,
# crucially, the colored clan banner just outside each bar).
ENEMY_ELIXIR_ROWS = [36, 38, 40, 42, 44]           # top bar body, y~31-48
FRIENDLY_ELIXIR_ROWS = [1561, 1563, 1565, 1567, 1569]  # bottom bar body, y~1560-1573

# Bounding boxes (for drawing/debug overlays). The fill READING uses the
# anchors above, not these boxes.
ENEMY_ELIXIR_BAR = (91, 30, 864, 50)
FRIENDLY_ELIXIR_BAR = (91, 1558, 864, 1574)

# Arena playfield crop (training-image region + frame_diff area). Includes the
# wooden border for full context; tighten to the green field if ever needed.
ARENA_CROP = (24, 210, 878, 1462)

# A selected/being-placed card translates this many px toward the arena
# (down for enemy/top, up for friendly/bottom), same size. See module docstring.
SELECTED_SHIFT_Y = 25


def _hand_slots(left_start, top, width, height, spacing):
    return [
        (round(left_start + i * spacing), top,
         round(left_start + i * spacing) + width, top + height)
        for i in range(HAND_SIZE)
    ]


def enemy_hand_slots():
    """4 (x1,y1,x2,y2) boxes, left to right, for the TOP player's hand."""
    return _hand_slots(ENEMY_HAND_LEFT_START, ENEMY_HAND_TOP,
                       ENEMY_HAND_CARD_WIDTH, ENEMY_HAND_CARD_HEIGHT,
                       ENEMY_HAND_SPACING)


def friendly_hand_slots():
    """4 (x1,y1,x2,y2) boxes, left to right, for the BOTTOM player's hand."""
    return _hand_slots(FRIENDLY_HAND_LEFT_START, FRIENDLY_HAND_TOP,
                       FRIENDLY_HAND_CARD_WIDTH, FRIENDLY_HAND_CARD_HEIGHT,
                       FRIENDLY_HAND_SPACING)


def selected_shift(is_enemy: bool) -> int:
    """Signed y-shift (px) of a card while it's selected/being placed:
    +down for the top (enemy) player, -up for the bottom (friendly) one."""
    return SELECTED_SHIFT_Y if is_enemy else -SELECTED_SHIFT_Y


def read_enemy_elixir(screenshot):
    """Continuous 0-10 elixir estimate for the TOP player. Thin wrapper over
    perception.elixir_counter.read_bar_fill_region with this bar's geometry."""
    from perception.elixir_counter import read_bar_fill_region
    return read_bar_fill_region(screenshot, ELIXIR_BAR_LEFT_X, ELIXIR_BAR_RIGHT_X,
                                ENEMY_ELIXIR_ROWS)


def read_friendly_elixir(screenshot):
    """Continuous 0-10 elixir estimate for the BOTTOM player."""
    from perception.elixir_counter import read_bar_fill_region
    return read_bar_fill_region(screenshot, ELIXIR_BAR_LEFT_X, ELIXIR_BAR_RIGHT_X,
                                FRIENDLY_ELIXIR_ROWS)


BAR_PRESENT_MIN = 0.5  # fraction of the bar row that must be bar-colored to count as "present"


def _bar_colored_frac(screenshot, rows, x0=ELIXIR_BAR_LEFT_X, x1=ELIXIR_BAR_RIGHT_X):
    """Mean fraction (over the sample rows) of the elixir-bar span that is
    bar-colored -- magenta FILL or navy EMPTY. This detects the bar's STRUCTURE
    regardless of how full it is, which is the whole point: an earlier version
    keyed on magenta at the bar's LEFT segment and false-negatived every time a
    player dipped below ~2 elixir (the bug that made the macro try to leave a
    live match, 2026-09-07). Menu/list/results screens have neither color here."""
    import numpy as np
    x0, x1 = int(x0), int(x1)
    fracs = []
    for y in rows:
        if y >= screenshot.shape[0]:
            continue
        seg = screenshot[y, x0:x1].astype(int)
        b, g, r = seg[:, 0], seg[:, 1], seg[:, 2]
        s = b + g + r
        magenta = (b > 150) & (r > 150) & (np.abs(b - r) < 50)
        navy = (s > 90) & (s < 215) & (r >= b)  # dark reddish-navy empty segment
        fracs.append(float((magenta | navy).mean()))
    return sum(fracs) / len(fracs) if fracs else 0.0


def is_in_match(screenshot) -> bool:
    """True when the frame is live TV Royale gameplay: at least one elixir bar's
    full-width structure is present (bar-colored on a majority of its span),
    independent of elixir level. Menu / list / results / loading screens have no
    such bar. This is the gate that (a) starts collection only on real gameplay
    and (b) detects a replay ended when it goes False. Uses EITHER bar (the top
    one reads ~0.95 on gameplay vs ~0.04 on the list), so a low-elixir player
    can't trip it; a brief dropout is absorbed by the caller's frame debounce."""
    return (_bar_colored_frac(screenshot, ENEMY_ELIXIR_ROWS) > BAR_PRESENT_MIN
            or _bar_colored_frac(screenshot, FRIENDLY_ELIXIR_ROWS) > BAR_PRESENT_MIN)
