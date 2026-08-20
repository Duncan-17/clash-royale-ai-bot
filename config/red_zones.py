"""
Two DIFFERENT kinds of "troops can't be here" zones — deliberately kept
separate (Duncan's distinction, 2026-08-15), not merged into one concept:

- FOOTPRINT zones answer "where can a troop physically stand" — a fact
  about the GAME, true regardless of how vision is implemented. This is
  what card-placement validation, tile-grid reasoning, and general board
  understanding should eventually use. Calibrated tight, to just the
  tower's own base structure (where Duncan originally clicked via
  find_coords.py).
- DETECTION EXCLUSION zones answer "where does the sprite visually
  overlap enough to fool the unit detector" — a fact about THIS
  PARTICULAR MODEL's failure mode, not about the game. Deliberately
  larger than the footprint, extended upward to cover the defending
  figure/apparatus standing on each tower (which sticks up well above
  the tower's base — confirmed by measuring a live capture with pixel
  gridlines, see conversation history), since that's what was actually
  getting misdetected as a troop.

Why the distinction matters, concretely: a troop CAN legitimately be
placed directly behind a tower, close to it — that's real, walkable
board area. If a future placement-validation feature used the (larger)
detection-exclusion zones instead of the (tighter) footprint zones, it
would wrongly reject legal placements. unit_detector.py's filtering uses
DETECTION EXCLUSION (it's solving a vision problem); anything reasoning
about the actual game board later should use FOOTPRINT instead.

Water is simpler — no known equivalent "sprite bigger than hitbox"
problem, so there's only one WATER_ZONES definition, used for both
purposes: the river band, split into 3 segments with gaps at the two
bridges (ground troops can cross there), only applies to GROUND-
transport units (see unit_metadata.py) since flying units legitimately
cross the water itself, not just the bridges.

Calibrated 2026-08-15 against a real 900x1600 capture: Duncan clicked
top-left/bottom-right corners of each tower and the river/bridges via
find_coords.py (giving the FOOTPRINT zones directly), then the DETECTION
EXCLUSION zones were derived by measuring how far the actual defending
figure/apparatus extends beyond that footprint on a fresh live capture.
Both verified visually by drawing them over real screenshots before
trusting either. Re-calibrate the same way (same as hand_region.py's own
note) if LDPlayer's resolution setting ever changes.
"""

# Where a troop can physically stand — never occupies these tiles,
# regardless of any vision-model quirk. Use for game-logic reasoning
# (placement validation, tile-grid work), NOT detection filtering.
TOWER_FOOTPRINT_ZONES = [
    (391, 216, 506, 277),    # enemy king
    (167, 302, 263, 385),    # enemy left princess
    (638, 305, 736, 385),    # enemy right princess
    (387, 1088, 506, 1168),  # ally king
    (167, 954, 266, 1036),   # ally left princess
    (638, 956, 736, 1036),   # ally right princess
]

# Where unit_detector.py should suppress a detection — larger than the
# footprint on purpose, to also cover the defending figure/apparatus
# standing on each tower (king towers needed ~75px more headroom above
# the footprint, princess towers ~55px). A vision-pipeline patch, not a
# game fact — do not reuse this set for anything besides filtering
# detector output.
TOWER_DETECTION_EXCLUSION_ZONES = [
    (391, 141, 506, 277),    # enemy king
    (167, 247, 263, 385),    # enemy left princess
    (638, 250, 736, 385),    # enemy right princess
    (387, 1013, 506, 1168),  # ally king
    (167, 899, 266, 1036),   # ally left princess
    (638, 901, 736, 1036),   # ally right princess
]

# River band, split into 3 segments with gaps at the two bridges. Used
# for both footprint and detection purposes — no known sprite/hitbox
# mismatch here the way towers have.
_RIVER_Y = (650, 711)
_LEFT_BRIDGE_X = (179, 252)
_RIGHT_BRIDGE_X = (648, 720)
_RIVER_X_SPAN = (64, 887)

# Public names for the river band's own Y-extent -- exposed directly (not
# just left as the private _RIVER_Y tuple) because more than one caller
# outside this module now needs "which half of the board is this y-value
# on" as its own question, separate from the water/footprint zone lists
# above. Added 2026-08-18 alongside perception/unit_tracker.py's new
# position-based side signal (see that module's _side_from_position) --
# game_knowledge/opponent_play_tracker.py previously derived its own
# RIVER_Y_TOP = WATER_ZONES[0][1] rather than have a real name to import;
# that ad-hoc derivation is replaced by importing this directly.
RIVER_Y_TOP, RIVER_Y_BOTTOM = _RIVER_Y

WATER_ZONES = [
    (_RIVER_X_SPAN[0], _RIVER_Y[0], _LEFT_BRIDGE_X[0], _RIVER_Y[1]),
    (_LEFT_BRIDGE_X[1], _RIVER_Y[0], _RIGHT_BRIDGE_X[0], _RIVER_Y[1]),
    (_RIGHT_BRIDGE_X[1], _RIVER_Y[0], _RIVER_X_SPAN[1], _RIVER_Y[1]),
]

# Per-lane NEUTRAL zones -- opens once a princess tower is destroyed. Confirmed directly
# by Duncan (2026-08-20, real gameplay explanation, not a guess): the specific pocket
# where a destroyed princess tower stood becomes legally placeable by BOTH players, not
# just whoever destroyed it -- "each player can only play cards on their side of the
# river... once a princess tower is destroyed that zone becomes neutral." This is the
# geometry piece of the "perception fusion architecture" design doc's zone-classification
# system (see game_knowledge/zone_classifier.py, which consumes this).
#
# BOARD_CENTER_X calibrated directly: Duncan's own click methodology traced the boundary
# with points "directly in front of the king towers in a straight line," i.e. the board's
# horizontal center -- confirmed by the two x-values landing within 2px of each other
# (449, 451) despite being clicked at very different y (503 vs 711).
BOARD_CENTER_X = 450

# The ONE real, click-calibrated zone (via find_coords.py against a real capture with the
# ENEMY's own left princess tower destroyed): (65,501)-(450,711). Everything else below is
# DERIVED from this by symmetry, not independently measured -- the horizontal (left/right
# lane) mirror was Duncan's own explicit instruction ("this should let you flip it to find
# the right princess tower zone as well"); the vertical (ally/enemy) mirror is a reasonable
# inference from the board's own top/bottom symmetry that Duncan has NOT separately
# confirmed with a real ally-tower-destroyed capture -- worth verifying if a real capture
# ever becomes available, same "flag it, don't hide it" pattern used throughout this
# project for first-pass-but-unverified calibration.
ENEMY_LEFT_NEUTRAL_ZONE = (65, 501, 450, 711)
ENEMY_RIGHT_NEUTRAL_ZONE = (
    2 * BOARD_CENTER_X - ENEMY_LEFT_NEUTRAL_ZONE[2], ENEMY_LEFT_NEUTRAL_ZONE[1],
    2 * BOARD_CENTER_X - ENEMY_LEFT_NEUTRAL_ZONE[0], ENEMY_LEFT_NEUTRAL_ZONE[3],
)
_RIVER_MID_Y = (RIVER_Y_TOP + RIVER_Y_BOTTOM) / 2  # mirror axis for the ally/enemy flip below


def _mirror_y(zone):
    x1, y1, x2, y2 = zone
    return (x1, 2 * _RIVER_MID_Y - y2, x2, 2 * _RIVER_MID_Y - y1)


ALLY_LEFT_NEUTRAL_ZONE = _mirror_y(ENEMY_LEFT_NEUTRAL_ZONE)
ALLY_RIGHT_NEUTRAL_ZONE = _mirror_y(ENEMY_RIGHT_NEUTRAL_ZONE)


# Horizontal limits of the PLAYABLE arena -- outside these is decorative scenery
# (trees, rocks, grass), where no unit can ever legitimately be. Added 2026-08-20
# after Duncan reported the new detector flagging units "out of bounds on the
# edges": real, repeatable false positives were landing on a rock at cx~861 and
# in the trees at cx~30 (an 'ice_spirit' and an 'archer_queen', both recurring at
# the SAME coordinates across separate frames -- the signature of scenery being
# misread, not a transient glitch).
#
# Deliberately NOT reusing _RIVER_X_SPAN (64, 887) even though it's tempting: the
# river GRAPHIC extends past the playable tile area on the right, so 887 would
# have let the cx~861 rock false-positive straight through. Derived instead from
# a measured left edge (~66, which does happen to agree with the river's own left
# bound) mirrored about BOARD_CENTER_X -- the arena is symmetric, independently
# confirmed here by the two ally princess towers' footprint centers averaging
# 451.75 against a BOARD_CENTER_X of 450. Verified visually by drawing both lines
# over a real capture and checking they land on the checkered-tile boundary.
ARENA_X_MIN = 66
ARENA_X_MAX = 2 * BOARD_CENTER_X - ARENA_X_MIN  # 834

# Vertical bounds are deliberately NOT defined here: perception/unit_detector.py's
# own UNIT_Y_START/UNIT_Y_END arena crop already bounds detections vertically, and
# both detectors run against that same crop.


def in_arena_bounds(x, y) -> bool:
    """Game fact: is (x, y) inside the playable arena horizontally. False means
    decorative scenery where a unit can never legitimately be."""
    return ARENA_X_MIN <= x <= ARENA_X_MAX


def point_in_zone(x, y, zone) -> bool:
    x1, y1, x2, y2 = zone
    return x1 <= x <= x2 and y1 <= y <= y2


def in_tower_footprint(x, y) -> bool:
    """Game fact: is (x, y) on a tower's actual footprint. For placement/
    board-logic reasoning, not detection filtering."""
    return any(point_in_zone(x, y, z) for z in TOWER_FOOTPRINT_ZONES)


def in_tower_detection_exclusion(x, y) -> bool:
    """Vision-pipeline fact: is (x, y) somewhere unit_detector.py should
    suppress output, due to a known false-positive source (the tower's
    own defending figure). NOT a claim about where troops can stand —
    see module docstring."""
    return any(point_in_zone(x, y, z) for z in TOWER_DETECTION_EXCLUSION_ZONES)


def in_water_zone(x, y) -> bool:
    return any(point_in_zone(x, y, z) for z in WATER_ZONES)


def should_suppress_detection(name: str, bbox: tuple) -> bool:
    """Single source of truth for "this detection can't be a real unit here",
    shared by perception/unit_detector.py and perception/katacr_detector.py so
    both detectors are held to identical rules (added 2026-08-20 -- previously
    this logic lived privately in unit_detector._should_suppress, and the new
    KataCR path had no red-zone filtering at all, which is what let river and
    out-of-bounds false positives through).

    Three independent reasons to reject, of two different KINDS -- see this
    module's own docstring for why that distinction matters:
      - outside the playable arena horizontally (a GAME FACT)
      - in the river, for ground units only (a GAME FACT; flying units
        legitimately cross water, hence the transport check)
      - inside a tower's detection-exclusion zone (a VISION-PIPELINE patch for
        a known false-positive source, NOT a claim about where units can stand)
    `name` is this project's underscore convention ('spear_goblin')."""
    from perception.unit_metadata import is_air  # local import: avoids a config->perception
    # import at module load, keeping config/ importable on its own.
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    if not in_arena_bounds(cx, cy):
        return True
    if in_tower_detection_exclusion(cx, cy):
        return True
    if not is_air(name) and in_water_zone(cx, cy):
        return True
    return False
