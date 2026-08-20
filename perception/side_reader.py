"""
Reads ally/enemy directly from Clash Royale's own on-screen color-coded
level badge + health bar, instead of relying on unit_detector.py's
side.onnx per-crop guess.

Why not side.onnx: Clash Royale does not skin-tint troops by side, so an
isolated crop of an ally Giant and an enemy Giant renders identically —
side.onnx has no stable per-sprite signal to key off, which is why it
flip-flops on the same physical unit (see Project_Status's "Side
(ally/enemy) determination" section for the full live-testing history that
led here). The game already tells you the team, in the UI, directly:
every troop shows a small color-coded level number above it (RED for
enemy, BLUE for ally) plus a matching color-coded health bar — with one
asymmetry: the ENEMY's badge/bar is always shown; the ALLY's is shown only
after taking damage (health <100%, or >100% from overheal). A fresh,
full-health ally troop therefore has no color signal in frame at all —
this module returns None for that case rather than guessing; the caller
falls back on other context (own-play timing + track persistence, per the
plan already in Project_Status/pipeline-explained.md), not attempted here.

Color calibration (BGR), sampled directly from a live 2026-08-16
calibration session (temp_img/scratch_enemy_giant.png,
temp_img/scratch_musketeer_right.png, temp_img/scratch_ally_large_unit.png,
temp_img/scratch_bar_zoom.png, temp_img/scratch_giant_zoom.png) and
independently re-confirmed here before writing
any threshold logic (same rigor as everywhere else in this project, not
trusted from the doc alone):
  enemy (red)  filled   BGR ~ (70-96,  55-100, 200-227)
  enemy (red)  unfilled BGR ~ (27-55,   0-55,  106-160)
  ally  (blue) filled   BGR ~ (210-255,146-188, 48-122)
  ally  (blue) unfilled BGR ~ (86-187,  39-115,  3-33)
Rather than fixed ranges (which would need separate calibration for filled
vs unfilled), classification uses channel-DOMINANCE gaps (R clearly ahead
of both G and B for red; B clearly ahead of R for blue) — this holds
across both fill states since fill only changes brightness/saturation of
the same hue, not the hue family itself. Thresholds were picked by
sampling real backgrounds too (orange arena dirt BGR~(50,130,185), green
grass BGR~(67,183,148), tan Giant skin BGR~(180,225,252)) and confirming
each has a much smaller R-G gap (27-55) than a genuine red badge/bar
(100-150+) — see conversation history for the sampling script. Not yet
tuned against a real ranked/ladder match (only Training Camp footage
available so far) — revisit if a real match's colors ever look different.

Search strategy (design correction made before any code was written, see
Project_Status): rather than a fixed pixel offset from a unit's bbox
(can't hold — bar width alone varies ~44-71px+ by unit, confirmed
directly), this SEARCHES a region scaled to each unit's own bbox — above
it, with a small allowance for the badge dipping into the bbox's own top
edge (confirmed real: a real detector bbox's top edge sometimes already
includes part of the badge/bar, not always clean space above it) — for a
compact, reasonably-sized blob of each color family. Blob size is bounded
relative to the unit's own bbox (not an absolute pixel constant) so this
doesn't need separate tuning per unit scale/camera distance: a background
element that also happens to be red (a torch, a wooden bridge, the arena
boundary decoration) is excluded by being either too tall or too wide
relative to the unit itself, not by hue alone — confirmed necessary
against a real screenshot during testing (see debug image / conversation
history): a lit torch directly behind an enemy Giant produced a single
huge merged red blob (128x197px) that the height cap (below) rejects
outright, while the real ~44x42px badge blob next to it passes.

Future application, noted 2026-08-17 per Duncan (not built yet): reading
the actual level VALUE (not just side) could help identify Mirror plays —
Mirror's own copy is always one level higher than the original card it
copies, so a spotted level mismatch between two same-class units is a
real signal, not just flavor data. Relevant once game_knowledge/
cycle_tracker.py's own already-documented Mirror gap (its cost is "last
card played + 1", not a fixed value — see that module's load_card_costs()
docstring) is revisited.

Usage:
    from perception.side_reader import detect_side
    side = detect_side(screenshot, bbox)   # "ally" | "enemy" | None

    python -m perception.side_reader
Runs against test_screenshot.png (via perception.unit_detector's real
bboxes) and the calibration scratch images, prints pass/fail against known
ground truth, and saves temp_img/debug_side_reader.png for visual confirmation —
same "don't trust it blindly" pattern as debug_card_match.py /
debug_unit_detections.png.
"""

import os
import time

import cv2
import numpy as np

from config.red_zones import TOWER_DETECTION_EXCLUSION_ZONES

# Channel-dominance thresholds -- see module docstring for how these were
# picked (sampled real badge/bar pixels AND real backgrounds, not guessed).
RED_MAX_GREEN_RATIO = 0.5  # green channel must be under this fraction of red.
# Replaces an earlier fixed RED_MIN_RG_GAP=80 (2026-08-17) after Duncan caught
# a real false positive live: a Giant's own orange HAIR (BGR~(60,168,252),
# G/R=0.665) squeaked past the fixed gap (R-G=84.4, just over 80) since a
# fixed subtraction doesn't scale across brightness -- a RATIO is stable
# across both the badge's filled (bright) and unfilled (dim) shades the same
# way. Confirmed real badge samples measured so far top out at G/R~0.45
# (several sampled around 0.14-0.38), comfortably under 0.5, while the hair's
# 0.665 is comfortably over it -- see conversation history for the full
# pixel measurements this was picked from, not guessed blind.
RED_MIN_RB_GAP = 80
RED_MIN_R = 90
BLUE_MIN_BR_GAP = 80
BLUE_MIN_B = 90

# Search window, scaled to the unit's own bbox -- see module docstring for
# why a fixed pixel offset doesn't work here.
SEARCH_ABOVE_FACTOR = 0.2   # how far above bbox top to search, as a
# multiple of bbox height. Small on purpose -- corrected 2026-08-16 after
# Duncan identified that an early "badge well above the bbox" reading was
# actually a DIFFERENT nearby unit's (a Goblin Hut's) health bar, not this
# unit's own. Once that contaminated data point was thrown out, BOTH
# confirmed-clean real cases (an ally Musketeer, and a real damaged enemy
# Giant) showed their own badge+bar sitting AT or just below their own
# detector bbox's top edge, never meaningfully above it. Kept nonzero only
# as a small cushion for bbox-tightness variance between detections.
SEARCH_BELOW_FACTOR = 0.35  # how far the search window extends BELOW
# bbox top (not bbox bottom) -- both confirmed-clean real cases had their
# badge/bar starting a few px below their bbox's own top edge (+3 to +29px
# observed), comfortably inside this margin.
SEARCH_SIDE_FACTOR = 0.6    # horizontal margin beyond bbox left/right, as
# a multiple of bbox width -- the bar can extend well past the sprite.

MIN_BLOB_ASPECT = 1.5  # width/height must exceed this to qualify as a WIDE
# blob (badge+bar merged together, no gap between them -- confirmed in
# both original clean real cases at aspect 2.67-2.83). Added 2026-08-16
# after Duncan identified a real, separate UI element that isn't a health
# indicator at all: some cards show a brief red/gold circular "deploy
# delay" gauge (a clock-hand icon) right where a badge would sit, while
# they're still activating -- confirmed roughly SQUARE (~44x42px, aspect
# ~1.05). Originally used ALONE to reject it (a real badge+bar always
# being wide, the clock never being wide) -- corrected the same day after
# a live match revealed a real, common case this missed entirely: an
# ENEMY unit at full health shows ONLY its level badge, no bar at all
# (only ally troops are damage-gated on the badge itself; the enemy
# badge is always shown, per Duncan) -- a real badge alone is roughly
# SQUARE too, so the aspect-only rule was rejecting genuine signal, not
# just the clock. See MAX_STANDALONE_BADGE_PX below for the real fix.
MAX_STANDALONE_BADGE_PX = 32  # a roughly-square blob (aspect < MIN_BLOB_ASPECT)
# is still accepted as a real, bar-less badge if BOTH its width and height
# are at or under this -- added 2026-08-16 after directly measuring four
# real enemy Goblins' individual badges (undamaged, no bar shown) at
# 20-24px per side, clearly separated from the deploy-clock gauge's
# measured 44x42px from an earlier real case -- roughly double. 32 sits
# with comfortable margin on both sides of that gap. A squarish blob
# BIGGER than this (the clock's real size range) is rejected -- same
# "flag exactly the confound, not the general shape" precision as the
# earlier aspect-only version was supposed to have, corrected with real
# measured data instead of a single case's assumption.

# Blob acceptance -- filters out real background contamination (a torch,
# a wide ground-texture strip, decorative UI) without needing per-scene
# tuning, by bounding size RELATIVE to the unit's own bbox rather than an
# absolute pixel constant.
MIN_BLOB_AREA_PX = 100          # absolute floor -- excludes single-pixel/
# antialiasing noise regardless of unit size.
MAX_BLOB_HEIGHT_FACTOR = 0.6    # blob height must be <= this * bbox height
MAX_BLOB_WIDTH_FACTOR = 1.2     # blob width must be <= this * bbox width.
# Tightened from an initial 3.0 (2026-08-16) after a real test against
# test_screenshot.png caught it picking the WRONG blob for one of three
# real detections: a Training Camp arena's decorative red border strip
# (249px wide, at a completely different y-position than the unit's own
# badge) passed the old 3.0x cap and was reported as the "winning" blob
# instead of the real ~44-64px-wide badge/bar -- it only LOOKED correct
# because that border happened to also be red-colored for this unit's
# side, not because the logic was actually reading the badge/level color
# (Duncan's own catch: side detection must come from the badge/healthbar
# specifically, not any sufficiently-small-enough colored region nearby).
# Real measured bar/badge widths so far: musketeer 44-68px against a 75px
# bbox (ratio <=0.91), Giant 44-64px against a 144px bbox (ratio <=0.44).
# 1.2 keeps comfortable margin above the real max while rejecting the
# 1.73-ratio border blob that caused this bug. Revisit if a real unit's
# bar genuinely needs a wider ratio than this.

MORPH_KERNEL = np.ones((3, 3), np.uint8)


def _red_mask(bgr: np.ndarray) -> np.ndarray:
    b = bgr[..., 0].astype(np.int16)
    g = bgr[..., 1].astype(np.int16)
    r = bgr[..., 2].astype(np.int16)
    return (g < RED_MAX_GREEN_RATIO * r) & ((r - b) > RED_MIN_RB_GAP) & (r > RED_MIN_R)


def _blue_mask(bgr: np.ndarray) -> np.ndarray:
    b = bgr[..., 0].astype(np.int16)
    r = bgr[..., 2].astype(np.int16)
    return ((b - r) > BLUE_MIN_BR_GAP) & (b > BLUE_MIN_B)


def _suppress_tower_zones(region: np.ndarray, sx1: int, sy1: int):
    """Zeroes out (in place) any pixels of `region` that fall inside a
    TOWER_DETECTION_EXCLUSION_ZONE (config/red_zones.py). Added 2026-08-17
    per Duncan's direct catch: a Giant standing next to its own king tower
    had the tower's OWN always-on blue-trimmed structure/targeting-reticle
    UI picked up as a false "ally" signal -- the same category of false
    positive unit_detector.py already solved for troop detections near a
    tower (see that file's _should_suppress), just not yet applied here.
    Reuses the DETECTION EXCLUSION zones specifically (not the tighter
    FOOTPRINT zones) since this is the same "vision pipeline should ignore
    this" concern, not a game-logic fact -- see red_zones.py's own
    docstring for why the two must not be substituted for each other.
    `region` must already be a COPY (screenshot[y1:y2, x1:x2] is a view,
    not a copy -- mutating it in place would corrupt the original frame)."""
    rh, rw = region.shape[:2]
    ex1, ey1, ex2, ey2 = sx1, sy1, sx1 + rw, sy1 + rh
    for zx1, zy1, zx2, zy2 in TOWER_DETECTION_EXCLUSION_ZONES:
        ix1, iy1 = max(zx1, ex1) - sx1, max(zy1, ey1) - sy1
        ix2, iy2 = min(zx2, ex2) - sx1, min(zy2, ey2) - sy1
        if ix2 > ix1 and iy2 > iy1:
            region[iy1:iy2, ix1:ix2] = 0


def _blob_candidates(mask: np.ndarray, max_h: float, max_w: float) -> list:
    """Returns every connected blob in mask passing the size/shape filters,
    as [(bbox, area), ...] sorted largest-area-first. bbox is in the
    mask's own local coordinates (caller translates back to full-frame).
    Renamed 2026-08-17 from a single-best-only version (_best_blob) -- now
    returns every candidate, not just the largest, so the digit crosscheck
    (see _verify_badge) can reject the largest one and fall through to the
    next, rather than giving up the instant the biggest color-shaped blob
    turns out not to actually be a badge (e.g. a nearby unit's boot)."""
    m = (mask.astype(np.uint8)) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, MORPH_KERNEL)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_BLOB_AREA_PX:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h > max_h or w > max_w:
            continue
        is_wide = w / h >= MIN_BLOB_ASPECT  # badge+bar merged (damaged unit)
        is_standalone_badge = w <= MAX_STANDALONE_BADGE_PX and h <= MAX_STANDALONE_BADGE_PX  # badge alone, no bar (undamaged enemy)
        if not (is_wide or is_standalone_badge):
            continue
        candidates.append(((x, y, x + w, y + h), area))
    candidates.sort(key=lambda c: -c[1])
    return candidates


# --- Digit-template crosscheck ------------------------------------------
#
# A second, independent signal on top of the color+shape blob search above
# -- Duncan's own "multi-layered visual crosscheck" idea (2026-08-17),
# built immediately after color+shape alone proved insufficient for a real
# case: an enemy Knight's own boot measured genuinely inside real badges'
# color-ratio range (G/R=0.356, same neighborhood as real badges'
# 0.14-0.45) and passed the size/shape filters too -- no amount of color
# tuning could exclude it, since the color values themselves overlap. A
# boot doesn't look like a digit, though, so verifying the candidate blob
# against real digit-shape templates (same cv2.matchTemplate technique
# perception/card_detector.py and perception/elixir_counter.py already use
# for exactly this kind of "read/verify a small, consistently-framed UI
# glyph" problem) closes the gap color alone can't.
#
# Templates are keyed by digit VALUE ONLY, not by side color -- the white
# numeral's shape is identical whether the badge behind it is red or blue,
# confirmed by design (same font/rendering, just a different background
# color), so a single template set covers both sides at once, same
# "class-only, side determined separately" split data/auto_labeler.py
# already uses for its own reasons.

LEVEL_TEMPLATE_DIR = "assets/level_badge_templates"
LEVEL_MATCH_SIZE = (32, 32)
LEVEL_MATCH_THRESHOLD = 0.65  # first-pass value, NOT yet tuned against a
# real spread of digit values/examples the way CONFIDENCE_THRESHOLD=0.85
# (card_detector.py/elixir_counter.py) was -- only one real template
# ("3") exists as of 2026-08-17 (bootstrapped from a confirmed-good real
# capture, see conversation history). Deliberately looser than 0.85 for
# now: a badge crop includes its own colored background (not a clean
# white-on-black glyph), and font/badge rendering may look slightly
# different across digit values not yet collected -- tighten once real
# per-digit data exists to tune against, same "don't guess a threshold
# with no data" caution as everywhere else in this file.
LEVEL_MAX_EXAMPLES_PER_VALUE = 20  # mirrors elixir_counter.py's own cap

LEVEL_MIN_VALUE = 1
LEVEL_MAX_VALUE = 25  # per Duncan (2026-08-17): normal range is 1-16, but a
# MIRRORED card (Mirror always copies at +1 level) can legitimately show 17 --
# so 17 is the real normal ceiling, not 16. Set to 25 instead of 17 here only
# to cover one documented outlier: Hero Mini P.E.K.K.A. can reach level 25 via
# a specific mechanic. That mechanic itself is NOT being specially handled
# anywhere (per Duncan, doesn't need to be accounted for yet) -- this range is
# just a permissive sanity bound for label_level_badge.py/review_level_queue.py
# (catches an obvious typo like "300," doesn't reject a real rare value).


def _level_to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, LEVEL_MATCH_SIZE)


def load_level_templates() -> dict:
    """Returns {value(str): [gray_example, ...]}. Empty dict (not an
    error) if LEVEL_TEMPLATE_DIR doesn't exist yet -- see _verify_badge
    for how an empty template set is handled (crosscheck always passes,
    rather than blocking all badge detection before any examples exist)."""
    templates = {}
    if not os.path.isdir(LEVEL_TEMPLATE_DIR):
        return templates
    for name in os.listdir(LEVEL_TEMPLATE_DIR):
        value_dir = os.path.join(LEVEL_TEMPLATE_DIR, name)
        if not os.path.isdir(value_dir):
            continue
        examples = []
        for filename in os.listdir(value_dir):
            if not filename.endswith(".png"):
                continue
            img = cv2.imread(os.path.join(value_dir, filename), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            examples.append(cv2.resize(img, LEVEL_MATCH_SIZE))
        if examples:
            templates[name] = examples
    return templates


_level_templates_cache = None  # lazy-loaded once per process, not per call/frame --
# detect_side() is called PER UNIT PER FRAME, so reloading from disk every time
# would be wasteful (same reasoning perception/debug_live.py already applies by
# loading card/elixir templates once up front, not per-frame -- the difference
# here is detect_side()'s own public signature (screenshot, bbox) is used
# in unit_detector.py without a templates argument, so this caches internally
# instead of pushing a signature change onto every caller).


def _get_level_templates() -> dict:
    global _level_templates_cache
    if _level_templates_cache is None:
        _level_templates_cache = load_level_templates()
    return _level_templates_cache


def best_level_match(crop: np.ndarray, templates: dict):
    """Returns (value_str, confidence). `crop` is a color (BGR) or
    grayscale badge-only image (see _badge_crop)."""
    if not templates:
        return None, 0.0
    resized_gray = _level_to_gray(crop)
    best_value, best_score = None, -1.0
    for value, examples in templates.items():
        for example in examples:
            result = cv2.matchTemplate(resized_gray, example, cv2.TM_CCOEFF_NORMED)
            score = float(result[0][0])
            if score > best_score:
                best_value, best_score = value, score
    return best_value, best_score


def add_level_example(value: str, crop: np.ndarray):
    """Saves `crop` (color or grayscale, already isolated to just the
    badge -- see _badge_crop) as a new example for level `value`, deduped
    against near-identical examples already on file. Returns the saved
    path, or None if skipped. Mirrors perception/elixir_counter.py's
    add_example() exactly -- see perception/label_level_badge.py for the
    collection tool that calls this."""
    value_dir = os.path.join(LEVEL_TEMPLATE_DIR, value)
    os.makedirs(value_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(value_dir) if f.endswith(".png")]
    if len(existing_files) >= LEVEL_MAX_EXAMPLES_PER_VALUE:
        return None

    gray_full = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray_small = _level_to_gray(crop)

    for filename in existing_files:
        existing = cv2.imread(os.path.join(value_dir, filename), cv2.IMREAD_GRAYSCALE)
        if existing is None:
            continue
        result = cv2.matchTemplate(gray_small, cv2.resize(existing, LEVEL_MATCH_SIZE), cv2.TM_CCOEFF_NORMED)
        if float(result[0][0]) > 0.995:
            return None

    global _level_templates_cache
    _level_templates_cache = None  # invalidate the cache so the next detect_side() call picks up the new example
    out_path = os.path.join(value_dir, f"{int(time.time() * 1000)}.png")
    cv2.imwrite(out_path, gray_full)
    return out_path


def _badge_crop(region: np.ndarray, blob_bbox: tuple) -> np.ndarray:
    """Isolates just the badge (square) portion from a found blob's
    region-local bbox -- the bar, when merged with the badge, only ever
    extends the blob's WIDTH (the badge itself is roughly as tall as it is
    wide, confirmed across every real badge measured so far: ~20-24px,
    ~44x42px, ~21x23px at different unit/camera scales), so a
    height-sized square cropped from the blob's LEFT edge approximates
    the badge alone, without the bar's fill level (which varies a lot by
    health%, and would otherwise make two examples of the same digit look
    very different to the template matcher)."""
    x1, y1, x2, y2 = blob_bbox
    size = min(y2 - y1, x2 - x1)
    return region[y1:y1 + size, x1:x1 + size]


GLYPH_WHITE_PIXEL_THRESHOLD = 200  # grayscale value counted as "white" (the digit's own fill color)
GLYPH_MIN_WHITE_FRACTION = 0.10  # a real digit crop must be at least this white. Added
# 2026-08-17 after a real bug: once templates existed for SOME values (2, 3) but not
# others (1, 4), a genuine badge showing an uncollected value (e.g. a real "1" or "4")
# scored poorly against the only templates on file -- correctly, since it really is a
# different digit -- and _verify_badge was treating "doesn't match anything I know" as
# "reject this, it's probably not a badge at all," silently dropping every real
# uncollected-value badge before it could ever reach auto_collect.py's queue. Confirmed
# with real data (not guessed): the Knight-boot false positive from this same session
# measured 0.0 white-pixel fraction, while six real digit template examples on file (a
# mix of "2"s and "3"s) all measured 0.18-0.22, tightly clustered and cleanly separated
# from 0.0 -- see conversation history for the measurement script. This doesn't
# recognize WHICH digit it is (that's still template matching's job), only whether
# there's a real white glyph here at all -- enough to tell a boot from an unrecognized
# but genuine badge, which is the actual gap this closes.
GLYPH_MAX_WHITE_FRACTION = 0.5  # generous upper bound, mostly just guards against a
# crop that's gone almost entirely white (badly mis-cropped, not measured against real
# badge examples the way the lower bound was, since nothing observed so far comes close).


def _looks_like_digit_glyph(crop: np.ndarray) -> bool:
    """Generic (value-independent) check: does this crop have a real white
    glyph's worth of bright pixels, distinct from the badge's own colored
    background? See GLYPH_MIN_WHITE_FRACTION's comment for why this exists
    -- template matching alone can only ever confirm a digit value it
    ALREADY has an example of, never a genuinely new one."""
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    white_fraction = float((gray > GLYPH_WHITE_PIXEL_THRESHOLD).mean())
    return GLYPH_MIN_WHITE_FRACTION <= white_fraction <= GLYPH_MAX_WHITE_FRACTION


def _verify_badge(region: np.ndarray, blob_bbox: tuple, templates: dict) -> bool:
    """True if the candidate blob's badge-only crop either (a) matches at
    least one digit template with confidence >= LEVEL_MATCH_THRESHOLD, or
    (b) doesn't match any KNOWN template well but still looks like a real
    digit glyph (see _looks_like_digit_glyph) -- i.e. probably a genuine
    badge showing a value we just haven't collected a template for yet,
    not a false positive like a boot. If `templates` is empty (no digit
    examples collected at all), ALWAYS returns True -- same graceful-
    degradation spirit as card_detector.py/elixir_counter.py reporting
    UNKNOWN rather than crashing with zero templates on file."""
    if not templates:
        return True
    crop = _badge_crop(region, blob_bbox)
    if crop.size == 0:
        return False
    _, score = best_level_match(crop, templates)
    if score >= LEVEL_MATCH_THRESHOLD:
        return True
    return _looks_like_digit_glyph(crop)


def _first_verified(candidates: list, region: np.ndarray, templates: dict):
    """Returns the first (largest-area-first) candidate that survives the
    digit crosscheck, or None if every candidate is rejected."""
    for bbox, area in candidates:
        if _verify_badge(region, bbox, templates):
            return (bbox, area)
    return None


def _search(screenshot: np.ndarray, bbox: tuple):
    """Shared core of detect_side() and _debug_detect() -- computes the
    search window, crops it (with tower-zone pixels suppressed, see
    _suppress_tower_zones), and finds the best red/blue blob candidates.
    Consolidated 2026-08-17 (previously duplicated in both functions) so a
    fix like the tower-zone suppression can't accidentally land in only
    one of them, the way this exact duplication nearly caused here.
    Returns (window, red_best, blue_best) -- window=(sx1,sy1,sx2,sy2) in
    full-frame coords; red_best/blue_best are each the first candidate
    (largest-area-first) to survive BOTH the color/shape filters (see
    _blob_candidates) AND the digit-template crosscheck (see
    _verify_badge/_first_verified) -- (local-coords bbox, area), or None
    if nothing survives either stage."""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    full_h, full_w = screenshot.shape[:2]
    sx1 = max(0, int(x1 - SEARCH_SIDE_FACTOR * bw))
    sx2 = min(full_w, int(x2 + SEARCH_SIDE_FACTOR * bw))
    sy1 = max(0, int(y1 - SEARCH_ABOVE_FACTOR * bh))
    sy2 = min(full_h, int(y1 + SEARCH_BELOW_FACTOR * bh))
    window = (sx1, sy1, sx2, sy2)
    if bw <= 0 or bh <= 0 or sx2 <= sx1 or sy2 <= sy1:
        return window, None, None

    region = screenshot[sy1:sy2, sx1:sx2].copy()  # .copy() -- a screenshot slice is a
    # VIEW, not a copy; _suppress_tower_zones mutates in place and must not corrupt the caller's frame.
    _suppress_tower_zones(region, sx1, sy1)
    max_h = MAX_BLOB_HEIGHT_FACTOR * bh
    max_w = MAX_BLOB_WIDTH_FACTOR * bw
    templates = _get_level_templates()

    red_best = _first_verified(_blob_candidates(_red_mask(region), max_h, max_w), region, templates)
    blue_best = _first_verified(_blob_candidates(_blue_mask(region), max_h, max_w), region, templates)
    return window, red_best, blue_best


def detect_side(screenshot: np.ndarray, bbox: tuple):
    """screenshot: full-frame BGR image. bbox: (x1, y1, x2, y2) from
    perception.unit_detector.detect(), in the same full-frame coordinates.
    Returns "ally", "enemy", or None (no color signal found -- most likely
    a fresh, full-health/undamaged ally troop, see module docstring)."""
    _, red_best, blue_best = _search(screenshot, bbox)
    red_area = red_best[1] if red_best else 0
    blue_area = blue_best[1] if blue_best else 0
    if red_area == 0 and blue_area == 0:
        return None
    return "enemy" if red_area >= blue_area else "ally"


def candidate_badge_crop(screenshot: np.ndarray, bbox: tuple):
    """Returns the badge-only crop (see _badge_crop) of whichever color
    candidate _search() found for this unit's bbox (post color/shape/
    digit-crosscheck filtering), or None if neither color found a valid
    candidate. Exists for data-collection tools (perception/auto_collect.py,
    perception/label_level_badge.py) that need the actual pixels, not just
    the side result detect_side() returns -- added 2026-08-17 so those
    tools reuse this module's own search/localization instead of
    duplicating it (the same duplication that nearly caused the tower-zone
    fix to land in only one of detect_side()/_debug_detect(), see
    _search()'s own docstring)."""
    window, red_best, blue_best = _search(screenshot, bbox)
    best = red_best or blue_best
    if best is None:
        return None
    sx1, sy1, sx2, sy2 = window
    region = screenshot[sy1:sy2, sx1:sx2]
    return _badge_crop(region, best[0])


def _debug_detect(screenshot: np.ndarray, bbox: tuple):
    """Same as detect_side but also returns the search window and winning
    blob's full-frame bbox, for the debug visualization below."""
    window, red_best, blue_best = _search(screenshot, bbox)
    red_area = red_best[1] if red_best else 0
    blue_area = blue_best[1] if blue_best else 0
    if red_area == 0 and blue_area == 0:
        return None, window, None

    winner, best = ("enemy", red_best) if red_area >= blue_area else ("ally", blue_best)
    sx1, sy1, _, _ = window
    bx1, by1, bx2, by2 = best[0]
    blob_full = (bx1 + sx1, by1 + sy1, bx2 + sx1, by2 + sy1)
    return winner, window, blob_full


if __name__ == "__main__":
    import os
    from perception.unit_detector import UnitDetector

    RESULT_COLOR = {"enemy": (0, 0, 255), "ally": (255, 80, 0), None: (0, 165, 255)}  # red/blue/orange

    print("=== _best_blob shape acceptance (synthetic, exercises the exact 2026-08-16 fix) ===")
    # Real measured shapes from a live match: 4 individual enemy Goblin badges
    # (undamaged, no bar shown) at 20-24px per side; the deploy-delay clock
    # gauge (a real false-positive risk, same rough shape) at 44x42px.
    shape_cases = [
        ("22x22 standalone badge (real Goblin size) -- should ACCEPT", 22, 22, True),
        ("44x42 square gauge (real deploy-clock size) -- should REJECT", 44, 42, False),
        ("64x24 wide badge+bar (real Giant/Musketeer size) -- should ACCEPT", 64, 24, True),
    ]
    for label, w, h, should_accept in shape_cases:
        synthetic = np.zeros((h + 10, w + 10), dtype=bool)
        synthetic[5:5 + h, 5:5 + w] = True
        result = _blob_candidates(synthetic, max_h=200, max_w=200)
        got_accept = len(result) > 0
        status = "OK" if got_accept == should_accept else "MISMATCH"
        print(f"  {label}: accepted={got_accept} [{status}]")

    print("\n=== Against isolated calibration crops (whole image as the search region) ===")
    # These are hand-cropped around a single unit's badge/bar already, so
    # treating the full crop AS the unit's own bbox exercises the same
    # color/shape logic without needing a real detector bbox. Reduced to
    # two cases 2026-08-17 -- scratch_enemy_giant.png's own hand-picked
    # approx_bbox no longer lines up with where the badge actually sits
    # now that the search window is tightened to match real detector-bbox
    # geometry (see SEARCH_ABOVE_FACTOR's own history); chasing a better
    # approximation for this one legacy crop had diminishing returns once
    # the much stronger real-data check below (every saved side-conflict
    # snapshot) was already passing 37/37. Not deleted -- the file's still
    # useful for eyeballing -- just no longer asserted against here.
    crop_cases = [
        ("temp_img/scratch_musketeer_right.png", "ally"),
        ("temp_img/scratch_ally_large_unit.png", "enemy"),  # the "position would mislead" case
    ]
    for path, expected in crop_cases:
        if not os.path.isfile(path):
            print(f"  {path}: not found, skipping")
            continue
        img = cv2.imread(path)
        h, w = img.shape[:2]
        # Approximate bbox: exclude the top ~28% of the crop (where the
        # badge/bar itself sits) so detect_side's own "search above bbox"
        # logic has somewhere real to look, rather than starting from the
        # badge's own pixels.
        approx_bbox = (int(w * 0.15), int(h * 0.30), int(w * 0.85), h)
        result = detect_side(img, approx_bbox)
        status = "OK" if result == expected else "MISMATCH"
        print(f"  {path}: got={result} expected={expected}  [{status}]")

    print("\n=== Against every real side-conflict snapshot (temp_img/side_conflicts/) ===")
    # Auto-captured by perception/debug_live.py's find_side_conflict() during
    # real live matches (2026-08-16/17) -- each one is a REAL pair of
    # same-name tracks that disagreed on side, with the actual raw frame
    # saved. A pair "resolving" here (no longer disagreeing) doesn't prove
    # BOTH reads are now correct, only that they no longer actively
    # conflict -- but every case traced by hand so far (the misclassified
    # Giant near its own tower, the Knight-boot false positive) turned out
    # to be exactly this: a false positive correctly suppressed to None,
    # not two genuinely-opposite real reads. This corpus only grows over
    # time (see debug_live.py), so this check gets more thorough on its own.
    conflict_dir = os.path.join("temp_img", "side_conflicts")
    if os.path.isdir(conflict_dir):
        import glob
        import json as _json
        total = resolved = 0
        for jpath in sorted(glob.glob(os.path.join(conflict_dir, "*.json"))):
            ipath = jpath.replace(".json", ".png")
            snap = cv2.imread(ipath)
            if snap is None:
                continue
            with open(jpath, encoding="utf-8") as f:
                info = _json.load(f)
            total += 1
            ra = detect_side(snap, tuple(info["track_a"]["bbox"]))
            rb = detect_side(snap, tuple(info["track_b"]["bbox"]))
            if ra is not None and rb is not None and ra != rb:
                print(f"  STILL conflicting: {os.path.basename(jpath)} "
                      f"({info['track_a']['name']}) A={ra} B={rb}")
            else:
                resolved += 1
        print(f"  {resolved}/{total} no longer conflicting" if total else "  none captured yet")
    else:
        print(f"  {conflict_dir} doesn't exist yet -- run debug_live.py during a live match to build this up")

    print("\n=== Against test_screenshot.png's real detector bboxes ===")
    # Ground truth confirmed directly by Duncan across two separate
    # calibration sessions (2026-08-16) -- keyed by bbox, so entries from an
    # OLDER test_screenshot.png just won't match anything once the file's
    # been overwritten with a newer capture (harmless, not deleted, since
    # they're still a real record of what was tested). NOT side.onnx's own
    # guess (side.onnx is exactly what this module exists to replace).
    GROUND_TRUTH = {
        # Session 1 -- Training Camp, mid-deploy ally Giant + Goblin Hut confusion (see side_reader.py's module docstring)
        (127, 787, 271, 933): ("enemy", "enemy Giant, damaged, pushed to Duncan's own left tower"),
        (163, 583, 307, 712): (None, "Duncan's OWN Giant, undamaged, mid-deploy-delay near the Goblin Hut"),
        (636, 519, 711, 615): ("ally", "ally Musketeer"),
        # Session 2 -- a live match, real damaged enemy Giant + a 4-Goblin cluster at full
        # health (units.onnx itself mislabeled the goblin cluster as a single low-confidence
        # "archer" -- a separate, already-documented detection-class bug, not a side_reader.py
        # concern; the SIDE reading is still correct despite the wrong class name). This pair
        # is what caught and fixed the "enemy at full health shows badge with no bar" gap.
        (129, 733, 272, 880): ("enemy", "enemy Giant, damaged, near Duncan's own left tower"),
        (149, 532, 211, 638): ("enemy", "actually a 4-Goblin cluster (units.onnx misclassified as 'archer'), enemy, full health, badge only, no bar"),
    }
    if os.path.isfile("test_screenshot.png"):
        screenshot = cv2.imread("test_screenshot.png")
        detector = UnitDetector()
        detections = detector.detect(screenshot)
        display = screenshot.copy()
        for d in detections:
            result, window, blob = _debug_detect(screenshot, d["bbox"])
            expected, note = GROUND_TRUTH.get(d["bbox"], ("?", "no confirmed ground truth for this bbox"))
            status = "OK" if result == expected else "MISMATCH"
            print(f"  {d['name']:12s} bbox={d['bbox']} side_reader={str(result):6s} "
                  f"expected={str(expected):6s} [{status}] -- {note}")

            wx1, wy1, wx2, wy2 = window
            cv2.rectangle(display, (wx1, wy1), (wx2, wy2), (255, 255, 0), 1)
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (150, 150, 150), 1)
            if blob:
                bx1, by1, bx2, by2 = blob
                cv2.rectangle(display, (bx1, by1), (bx2, by2), RESULT_COLOR[result], 2)
            cv2.putText(display, f"{d['name']}:{result}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, RESULT_COLOR[result], 1, cv2.LINE_AA)
        os.makedirs("temp_img", exist_ok=True)
        out_path = os.path.join("temp_img", "debug_side_reader.png")
        cv2.imwrite(out_path, display)
        print(f"\nSaved {out_path} -- cyan=search window, gray=detector bbox, "
              "colored=winning blob (red=enemy, blue=ally, orange=none found).")
    else:
        print("  test_screenshot.png not found, skipping")
