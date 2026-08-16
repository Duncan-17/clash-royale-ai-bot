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

WATER_ZONES = [
    (_RIVER_X_SPAN[0], _RIVER_Y[0], _LEFT_BRIDGE_X[0], _RIVER_Y[1]),
    (_LEFT_BRIDGE_X[1], _RIVER_Y[0], _RIGHT_BRIDGE_X[0], _RIVER_Y[1]),
    (_RIGHT_BRIDGE_X[1], _RIVER_Y[0], _RIVER_X_SPAN[1], _RIVER_Y[1]),
]


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
