"""Tap coordinates and timings for navigating the TV Royale UI, for the
unattended macro (data/tv_royale_watcher.py). Calibrated 2026-09-07 with
Duncan against real screens (900x1600), same find_coords.py process as every
other calibration here.

CONFIRMED (Duncan clicked these):
- the main-menu hamburger (3-line) icon that opens the menu popup,
- the "TV Royale" entry on that popup,
- the arena browse arrows (prev/next) and the close (X) on the TV Royale list.
Clicking a game's Watch button drops STRAIGHT into the replay (no intermediate
screen); a WATCH_SETTLE_SECONDS pause after the tap skips the entry transition
so the collector never sees a loading/wipe frame.

STILL TO CALIBRATE (sensible defaults below, marked): how a replay ENDS and how
to get back to the list, and the scroll gesture. The macro is written so these
are the only unknowns and each has a safe fallback (re-open TV Royale from the
menu to guarantee a clean list, whatever the end screen looks like).
"""

# --- confirmed taps (x, y) ---
MENU_ICON = (821, 181)       # main-menu hamburger
TV_ROYALE_OPTION = (551, 357)  # "TV Royale" on the menu popup
ARENA_PREV = (72, 277)       # ◀ previous arena
ARENA_NEXT = (821, 273)      # ▶ next arena
LIST_CLOSE = (840, 186)      # ✕ close the TV Royale list

# --- timings (seconds) ---
WATCH_SETTLE_SECONDS = 8.0   # Duncan's call: pause after tapping Watch to clear the entry transition
MENU_SETTLE_SECONDS = 1.2    # after a menu/list tap, let the screen settle before reading it
POLL_INTERVAL_SECONDS = 0.25  # how often to check the match-active gate while watching

# --- scroll (TO CALIBRATE) ---
# The list scrolls; row pitch is ~439px. A swipe up moves it down the list.
# First-pass gesture -- verify the travel against a scrolled capture.
SCROLL_SWIPE = (450, 1200, 450, 500)  # (x1, y1, x2, y2): drag up ~700px
SCROLL_DURATION_MS = 400

# --- returning to the list after a replay (CONFIRMED 2026-09-07) ---
# A finished replay shows an end/results screen with an "OK" button; tapping it
# returns to the TV Royale list. Crucially the list comes back at the SAME
# scroll position as before, with the game you just watched now CENTERED -- so
# the macro does NOT re-open (which would reset to the top); it just taps OK and
# keeps going down the list from where it was.
REPLAY_EXIT_TAP = (446, 1452)   # the "OK" button on the replay-ended screen
