# Debug-view conventions (MUST follow for every debug/overlay panel)

Rules for any live debug UI in this project (detector_view, debug_live, future
ones). These come from real pain during live testing — follow them.

## RULE 1 — FIXED LAYOUT. Categories never move. (2026-09-09, Duncan)

Adding or removing an item in one category MUST NOT shift the position of any
other info. Every section and every row within it has a **reserved, fixed
screen position**; empty slots stay blank rather than collapsing.

Why: the detector jitters frame-to-frame (a unit flickers in/out, a giant dies
and its rows vanish). If rows are laid out by "current y, increment per item,"
the whole panel jumps every frame and becomes unreadable/unusable. The eye must
be able to rely on "friendly units are always *here*, tower health is always
*there*."

Concretely:
- Reserve a fixed y and a fixed max-row count per section (ENEMY, FRIENDLY,
  TOWERS, PLAYS, ...). Render items into that fixed block; overflow is truncated,
  underflow leaves blank rows. Never compute the next section's y from the
  previous section's item count.
- Fixed-position also means fixed columns — align fields (name / hp / level) at
  fixed x so values don't jitter horizontally either.

## RULE 2 — prefer the DETECTOR over flaky pixel readers for state

When a debug panel shows game state (tower alive/destroyed, side, spawn), drive
it from the (now good) YOLO detector, not the old calibrated-pixel modules,
which false-fire (e.g. the pixel tower_reader flagged a princess tower destroyed
after one hit while the detector still clearly saw the tower + its bar). Use the
detector as the source of truth; pixel logic only to read a VALUE inside a box
the detector already located (e.g. HP fill inside a detected bar).
