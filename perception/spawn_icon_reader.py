"""
Reads the deploy-delay "spawn icon" — a small circular gold-ringed gauge,
shown briefly on a troop right as it's placed, filling BLUE (ally) or RED
(enemy) clockwise as the deploy delay counts down. Confirmed directly by
Duncan (2026-08-17, with two clean reference screenshots — see
assets/spawn_icon_templates/) after this same icon had already been seen
several times this project as a FALSE POSITIVE for perception/side_reader.py's
badge detector (same red/blue color family, similar size) requiring a
dedicated shape exclusion (MAX_STANDALONE_BADGE_PX) to filter out.

Why this exists as its own module rather than folding into side_reader.py:
different job entirely. side_reader.py answers "what side is THIS
already-tracked unit" continuously over its whole life, gated by damage for
allies. This answers "was a card JUST played, and which side" at the single
moment of placement — reliably, for BOTH sides, regardless of damage state,
since the icon shows for a fresh full-health ally troop too (unlike its
badge). Session context (2026-08-17): a live-match review of
game_knowledge/opponent_play_tracker.py's output found several real bug
categories (a persistence-lock false side-read, troop-tracking-gap double-
counting, multi-unit burst-count struggles) that a reliable spawn-moment
signal could address at the source, instead of continuing to patch
inference built on fragile raw unit-detection grouping. See
temp_img/opponent_plays_review_notes.md for the full review this grew out of.

Color calibration (BGR), sampled directly from Duncan's two reference
screenshots (assets/spawn_icon_templates/ally/bootstrap_1.png,
.../enemy/bootstrap_1.png) — NOT reused blindly from side_reader.py's own
badge calibration, though they turned out nearly identical (a real,
confirmed finding, not an assumption):
  ally  (blue) fill: BGR mean (251.4, 200.5, 119.2), n=457px
  enemy (red)  fill: BGR mean (50.5, 54.0, 248.4), n=427px
  gold ring (same both sides): BGR ~(50-57, 181-185, 222-226)
Both fill colors match perception/side_reader.py's own badge fill colors
closely enough that this module reuses side_reader's _red_mask/_blue_mask
functions directly rather than re-deriving separate thresholds — the real
new work here is shape/geometry (a ring, not a rounded square) and a
completely different search-window question (see below), not color.

Position, per Duncan's direct explanation (2026-08-17) — the icon marks
the troop's TRUE BOARD/TILE position, not a fixed offset from its sprite's
bounding box, which is why its position relative to a detected bbox varies
so much between examples already seen this session (near a Giant's own
FEET in one real capture, near the HEAD in an earlier one that turned out
to be a different, shorter-troop case):
  - tall troops (Giant): near the bbox's own BOTTOM edge (its feet)
  - short troops (Mini P.E.K.K.A.): closer to the bbox's own CENTER
  - flying troops: BELOW the sprite, at the shadow's position (their true
    ground position, not their visual altitude)
  - multi-unit swarms: at the base of ONE representative unit, generally
    the lead-most one — NOT one icon per unit in the group. Not handled by
    this module itself (a single-bbox search); the caller (eventually
    game_knowledge/opponent_play_tracker.py) will need to search near
    multiple group members rather than assume which one is "the" leader.
SEARCH_ABOVE_FACTOR/SEARCH_BELOW_FACTOR below are chosen to span from
roughly a unit's own center down to well below its bbox (covering the
flying-shadow case), deliberately NEVER reaching up toward the head/upper
body — a real, informed choice from Duncan's explanation, not a blind
generous guess the way early side_reader.py search windows started out
(see that module's own SEARCH_ABOVE_FACTOR history for the contrast).

Status (2026-08-17): color calibration done, module structure built,
tested against real captures available so far. NOT YET integrated into
opponent_play_tracker.py's new-play detection — that's the next step once
this module's own accuracy is confirmed against more real examples.
"""

import math
import os
import time

import cv2
import numpy as np

from config.red_zones import WATER_ZONES
from perception.side_reader import _blue_mask, _red_mask

TEMPLATE_DIR = "assets/spawn_icon_templates"
MATCH_SIZE = (48, 48)  # real icon size measured so far: ~35-45px per side
MATCH_THRESHOLD = 0.5  # first-pass, NOT tuned against real data yet -- only
# two bootstrap templates (one per side) exist so far. Loosened relative to
# side_reader.py's LEVEL_MATCH_THRESHOLD=0.65 for the same reason that one
# started loose: nothing to tune against until more real examples exist.

GOLD_RING_MIN_R = 180
GOLD_RING_MIN_G = 140
GOLD_RING_MAX_G = 230
GOLD_RING_MAX_B = 100

# Search window, scaled to the unit's own bbox -- see module docstring's
# "Position" section for why these specific bounds (never near the head,
# spans center-of-bbox down through well-below-bbox).
SEARCH_ABOVE_FACTOR = 0.4   # how far above bbox BOTTOM to allow (i.e. as
# far up as roughly bbox center for a short troop) -- expressed as a
# fraction subtracted from the bottom, not the top, since "near the head"
# is never correct per Duncan and shouldn't be in the window at all.
SEARCH_BELOW_FACTOR = 0.7   # how far below bbox bottom to search (covers
# a flying unit's shadow, which sits outside/below its own visual sprite).
# Widened from 0.4 (2026-08-18) per Duncan's live report: icons were
# consistently missed for cards placed directly behind the ally king
# tower, right at the back edge of the legal placement area -- his own
# read is that the icon there renders further below the unit's detected
# bbox than usual (little vertical room is left for the sprite/bbox itself
# that close to the board's own bottom edge, so the icon ends up
# relatively further from bbox bottom than for a mid-board placement).
# First-pass widening, not yet re-verified against a fresh real capture of
# this specific case -- re-check on the next live test.
SEARCH_SIDE_FACTOR = 0.6    # horizontal margin, same convention as
# side_reader.py's own SEARCH_SIDE_FACTOR.

MIN_ICON_PX = 25   # real icons measured ~35-45px; floor set well below that
MAX_ICON_PX = 60   # ceiling with margin above the measured range
MIN_ICON_ASPECT = 0.7   # roughly circular/square, not a thin sliver
MAX_ICON_ASPECT = 1.4

MORPH_KERNEL = np.ones((3, 3), np.uint8)


def _gold_ring_mask(bgr: np.ndarray) -> np.ndarray:
    b = bgr[..., 0].astype(np.int16)
    g = bgr[..., 1].astype(np.int16)
    r = bgr[..., 2].astype(np.int16)
    return (r > GOLD_RING_MIN_R) & (g > GOLD_RING_MIN_G) & (g < GOLD_RING_MAX_G) & (b < GOLD_RING_MAX_B)


def _suppress_water_zones(region: np.ndarray, sx1: int, sy1: int):
    """Zeroes out (in place) any pixels of `region` that fall inside a
    WATER_ZONES river segment (config/red_zones.py). Added per Duncan's
    2026-08-18 report of real false-positive spawn icons on the river --
    the same category of fix perception/side_reader.py already applied for
    tower UI pixels (_suppress_tower_zones) and unit_detector.py applies
    for troop detections (_should_suppress), just not yet here. Unlike
    side_reader.py's tower fix, this reuses WATER_ZONES rather than a
    unit-transport-aware check (is_air()/in_water_zone()) -- a spawn icon
    is never itself a flying unit's true position (see module docstring's
    "flying troops: BELOW the sprite, at the shadow" note), so river
    pixels are never a legitimate icon location regardless of the
    triggering unit's transport type. `region` must already be a COPY
    (screenshot[y1:y2, x1:x2] is a view, not a copy -- mutating it in
    place would corrupt the original frame)."""
    rh, rw = region.shape[:2]
    ex1, ey1, ex2, ey2 = sx1, sy1, sx1 + rw, sy1 + rh
    for zx1, zy1, zx2, zy2 in WATER_ZONES:
        ix1, iy1 = max(zx1, ex1) - sx1, max(zy1, ey1) - sy1
        ix2, iy2 = min(zx2, ex2) - sx1, min(zy2, ey2) - sy1
        if ix2 > ix1 and iy2 > iy1:
            region[iy1:iy2, ix1:ix2] = 0


def _icon_shape_candidates(mask: np.ndarray) -> list:
    """Same style as side_reader.py's _blob_candidates, but sized/shaped
    for the icon (roughly circular, ~35-45px) instead of a badge+bar."""
    m = (mask.astype(np.uint8)) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, MORPH_KERNEL)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if not (MIN_ICON_PX <= w <= MAX_ICON_PX and MIN_ICON_PX <= h <= MAX_ICON_PX):
            continue
        aspect = w / h
        if not (MIN_ICON_ASPECT <= aspect <= MAX_ICON_ASPECT):
            continue
        candidates.append(((x, y, x + w, y + h), area))
    candidates.sort(key=lambda c: -c[1])
    return candidates


def _best_icon_candidate(mask: np.ndarray, anchor: tuple):
    """Returns (bbox, area, distance_to_anchor) for the shape-valid
    candidate (see _icon_shape_candidates) in `mask` whose own center is
    CLOSEST to `anchor` (both in the same region-local coords) -- or None
    if no shape-valid candidate exists. `anchor` should be roughly where
    the TARGET unit's own icon is expected (see find_spawn_icon).

    Added 2026-08-18, replacing a straight "return the single largest
    candidate" approach, after widening SEARCH_BELOW_FACTOR (same day, to
    fix icons being missed for cards placed at the very back of the legal
    placement area) exposed a real bug: a wider window can now also catch
    a DIFFERENT, nearby unit's own icon, which may have a larger pixel
    area purely because of where its fill animation happens to be at that
    instant -- unrelated to whether it's actually the target unit's icon.
    Confirmed on a real case Duncan caught: a Giant's search window
    started also catching an ally Knight's icon placed just below it, and
    the Knight's (larger, further away) icon was winning over the Giant's
    own (smaller at that moment, but much closer) real icon. Proximity to
    the unit's own expected anchor is a far more principled tie-breaker
    than raw pixel area, which has nothing to do with whose icon it is."""
    best = None
    for bbox, area in _icon_shape_candidates(mask):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        dist = math.hypot(cx - anchor[0], cy - anchor[1])
        if best is None or dist < best[2]:
            best = (bbox, area, dist)
    return best


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, MATCH_SIZE)


def load_templates() -> dict:
    """Returns {"ally": [gray_example, ...], "enemy": [...]}."""
    templates = {"ally": [], "enemy": []}
    for side in templates:
        side_dir = os.path.join(TEMPLATE_DIR, side)
        if not os.path.isdir(side_dir):
            continue
        for filename in os.listdir(side_dir):
            if not filename.endswith(".png"):
                continue
            img = cv2.imread(os.path.join(side_dir, filename), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            templates[side].append(cv2.resize(img, MATCH_SIZE))
    return templates


def best_match(crop: np.ndarray, templates: dict):
    """Returns (side_str_or_None, confidence)."""
    resized_gray = _to_gray(crop)
    best_side, best_score = None, -1.0
    for side, examples in templates.items():
        for example in examples:
            result = cv2.matchTemplate(resized_gray, example, cv2.TM_CCOEFF_NORMED)
            score = float(result[0][0])
            if score > best_score:
                best_side, best_score = side, score
    return best_side, best_score


def find_spawn_icon(screenshot: np.ndarray, bbox: tuple):
    """screenshot: full-frame BGR image. bbox: (x1, y1, x2, y2) of a unit,
    same coordinates perception.unit_detector.detect() produces. Returns
    (side, confidence, full_frame_icon_bbox) -- side is "ally"/"enemy"/None
    (None if no icon-shaped red or blue candidate was found at all).

    Side comes directly from COLOR (whichever of red/blue produced the
    icon-shaped candidate CLOSEST to this unit's own expected icon
    position -- see _best_icon_candidate, changed 2026-08-18 from "largest
    candidate wins" after a real multi-unit false-positive), the same way
    perception.side_reader.detect_side() determines side from its own
    red/blue blob areas -- NOT from template matching, despite
    `templates`/best_match() existing in this module. Real reason, found
    2026-08-17 while testing against the one confirmed real icon capture
    available: unlike a badge's DIGIT (constant shape, only the separate
    bar's fill % varies -- see side_reader.py's _badge_crop, which crops
    the bar out specifically so it doesn't corrupt matching), the icon's
    entire visible content IS a continuously-changing pie-fill animation
    with no constant part to isolate. A single reference screenshot at one
    fill level scored 0.03 against a real icon at a different fill level --
    correlation naturally craters between two very differently-shaped fill
    wedges, not a threshold-tuning problem. `templates`/best_match() are
    kept for a LATER verification role (once enough real examples across
    many fill levels exist to make that meaningful) but don't gate the
    result here. `confidence` returned is the winning candidate's pixel
    AREA, not a template score or a distance -- comparable to how
    side_reader.py reports area too.
    """
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None, 0.0, None
    full_h, full_w = screenshot.shape[:2]

    sx1 = max(0, int(x1 - SEARCH_SIDE_FACTOR * bw))
    sx2 = min(full_w, int(x2 + SEARCH_SIDE_FACTOR * bw))
    sy1 = max(0, int(y2 - SEARCH_ABOVE_FACTOR * bh))
    sy2 = min(full_h, int(y2 + SEARCH_BELOW_FACTOR * bh))
    if sx2 <= sx1 or sy2 <= sy1:
        return None, 0.0, None

    region = screenshot[sy1:sy2, sx1:sx2].copy()  # .copy() -- a screenshot slice is a
    # VIEW, not a copy; _suppress_water_zones mutates in place and must not corrupt the caller's frame.
    _suppress_water_zones(region, sx1, sy1)
    # Red and blue candidates are found SEPARATELY, never unioned before
    # shape-filtering -- corrected 2026-08-17 after a real bug: an early
    # version unioned red_mask|blue_mask|gold_ring_mask together first,
    # and in a real search window that happened to include part of the
    # river, blue_mask's real (correct) hits on the river water merged
    # with/crowded out the icon's own correctly-shaped red blob during
    # the same morphological-close + contour step, causing the true icon
    # to be missed entirely. side_reader.py already avoids exactly this by
    # keeping its own red/blue candidate search fully separate -- this
    # follows the same pattern. Gold-ring localization is measured but NOT
    # used here (see _gold_ring_mask's own docstring) -- a real check found
    # it hit only ~7% of a confirmed real ring's pixels in this same
    # screenshot, too weak/noisy to help localize yet; kept defined in case
    # a future recalibration with more real examples makes it useful.
    # Anchor: this unit's own expected icon position (bbox bottom-center,
    # region-local coords) -- used to pick among candidates by PROXIMITY
    # rather than raw area, see _best_icon_candidate's docstring for why.
    anchor = ((x1 + x2) / 2 - sx1, y2 - sy1)
    red_best = _best_icon_candidate(_red_mask(region), anchor)
    blue_best = _best_icon_candidate(_blue_mask(region), anchor)

    if red_best is None and blue_best is None:
        return None, 0.0, None
    if blue_best is None or (red_best is not None and red_best[2] <= blue_best[2]):
        side, best = "enemy", red_best
    else:
        side, best = "ally", blue_best
    bx1, by1, bx2, by2 = best[0]
    full_bbox = (bx1 + sx1, by1 + sy1, bx2 + sx1, by2 + sy1)
    return side, float(best[1]), full_bbox


MAX_EXAMPLES_PER_SIDE = 40  # matches perception/auto_collect.py's own
# MAX_EXAMPLES_PER_CARD=40 -- a reasonable volume target, not derived from
# anything specific to icons.
DUPLICATE_THRESHOLD = 0.995  # matches perception/card_detector.py's own


def add_spawn_icon_example(side: str, crop: np.ndarray):
    """Saves `crop` (color BGR, already isolated to just the icon -- see
    find_spawn_icon()'s returned icon_bbox) as a new example under
    assets/spawn_icon_templates/<side>/, deduped against near-identical
    examples already on file. Returns the saved path, or None if skipped.

    Saves in COLOR, unlike elixir_counter.py's add_example()/side_reader.py's
    add_level_example() (which this originally mirrored too closely) --
    corrected 2026-08-17 after Duncan caught it directly: those two save
    grayscale because a digit's SHAPE is all that matters for matching, but
    a spawn icon's entire signal IS its color (red vs blue), so grayscale
    conversion at save time was destroying the one thing this asset library
    exists to preserve. The DEDUP comparison below still converts to
    grayscale internally (fine -- shape/position alone is enough to catch a
    near-identical repeat crop), but the FILE WRITTEN TO DISK keeps full color.

    Called ONLY from perception/review_spawn_icon_queue.py, after Duncan has
    personally confirmed the side -- perception/auto_collect.py's own
    collection loop deliberately does NOT call this directly, unlike the
    card/elixir/level-badge collectors' auto-file paths. A first version did
    call this straight from auto_collect.py (side coming from color felt
    "confident enough" to skip review) and a genuinely ambiguous real
    example went into the permanent library unreviewed as a result -- see
    CLAUDE.md's "never write to a permanent asset library without review"
    rule, added directly because of this incident. `side` here is
    Duncan's own confirmed answer, not this module's own guess."""
    side_dir = os.path.join(TEMPLATE_DIR, side)
    os.makedirs(side_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(side_dir) if f.endswith(".png")]
    if len(existing_files) >= MAX_EXAMPLES_PER_SIDE:
        return None

    gray_small = _to_gray(crop)  # dedup comparison only -- NOT what gets written to disk

    for filename in existing_files:
        existing = cv2.imread(os.path.join(side_dir, filename), cv2.IMREAD_GRAYSCALE)
        if existing is None:
            continue
        result = cv2.matchTemplate(gray_small, cv2.resize(existing, MATCH_SIZE), cv2.TM_CCOEFF_NORMED)
        if float(result[0][0]) > DUPLICATE_THRESHOLD:
            return None

    out_path = os.path.join(side_dir, f"{int(time.time() * 1000)}.png")
    cv2.imwrite(out_path, crop)
    return out_path


if __name__ == "__main__":
    templates = load_templates()
    total = sum(len(v) for v in templates.values())
    counts = {k: len(v) for k, v in templates.items()}
    print(f"Loaded {total} template(s): {counts}")

    print("\n=== Against the real Giant 'played' snapshot (confirmed real icon, near the feet) ===")
    path = "temp_img/opponent_plays/20260817_141624_961306_played_giant.png"
    if os.path.isfile(path):
        img = cv2.imread(path)
        bbox = (606, 508, 779, 654)  # from that event's own JSON sidecar
        side, area, icon_bbox = find_spawn_icon(img, bbox)
        print(f"  bbox={bbox} -> side={side} area={area:.0f} icon_bbox={icon_bbox}  (expected: enemy, icon_bbox=(672,614,708,652))")
        if icon_bbox:
            display = img.copy()
            cv2.rectangle(display, bbox[:2], bbox[2:], (150, 150, 150), 1)
            cv2.rectangle(display, icon_bbox[:2], icon_bbox[2:], (0, 255, 255), 3)
            cv2.putText(display, f"{side} area={area:.0f}", (icon_bbox[0], max(15, icon_bbox[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            os.makedirs("temp_img", exist_ok=True)
            cv2.imwrite("temp_img/debug_spawn_icon_reader.png", display)
            print("  saved temp_img/debug_spawn_icon_reader.png")
    else:
        print(f"  {path} not found, skipping")
