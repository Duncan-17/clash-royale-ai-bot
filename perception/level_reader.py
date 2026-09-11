"""Level-badge digit reading -- the small red/blue level number rendered above a
unit ("bar-level" in the detector's taxonomy).

EXTRACTED 2026-09-09 from perception/side_reader.py so it survives that module's
retirement (side_reader is #1 on the "reduce" list -- its colour-blob SEARCH is
the compensation the YOLO detector replaces; but its digit-TEMPLATE library and
matcher are still useful and are what live here now). side_reader.py re-exports
these names so its own remaining code and the label/review tools keep working.

Two eras use this:
  - OLD (side_reader): finds the badge blob by colour search, crops it, reads it.
  - NEW (this project's direction): the YOLO detector emits `bar-level` as a
    first-class box, so there is nothing to search for -- read_level() takes that
    detected box directly. perception/bar_reader.associate_level_badges() binds a
    bar-level box to its unit; perception/yolo_detector attaches the read value.

Templates are keyed by digit VALUE ONLY, not by side colour -- the white numeral's
shape is identical whether the badge behind it is red (enemy) or blue (ally), so
one template set covers both sides (same "class-only, side determined separately"
split used across the project).

WHAT NEEDS REAL DATA (flagged, not guessed -- RULE #1): as of extraction the
template library only has examples for levels 1-4, and those were cut by
side_reader's blob search, whose framing may differ slightly from a YOLO
`bar-level` box. So read_level reliably reports only values it has templates for
and returns None ("unknown, don't guess") otherwise. Growing coverage is the
collection path: queue_level_crop() banks real YOLO-box crops to
assets/unlabeled_level_queue/ for Duncan to label via review_level_queue.py
(RULE #2 -- never auto-file into the permanent library).
"""

import os
import time

import cv2
import numpy as np

LEVEL_TEMPLATE_DIR = "assets/level_badge_templates"
LEVEL_MATCH_SIZE = (32, 32)
LEVEL_MATCH_THRESHOLD = 0.65  # first-pass value, NOT yet tuned against a
# real spread of digit values/examples the way CONFIDENCE_THRESHOLD=0.85
# (card_detector.py/elixir_counter.py) was. Deliberately looser than 0.85 for
# now: a badge crop includes its own colored background (not a clean
# white-on-black glyph), and font/badge rendering may look slightly
# different across digit values not yet collected -- tighten once real
# per-digit data exists to tune against, same "don't guess a threshold
# with no data" caution as everywhere else in this project.
LEVEL_MAX_EXAMPLES_PER_VALUE = 20  # mirrors elixir_counter.py's own cap

LEVEL_MIN_VALUE = 1
LEVEL_MAX_VALUE = 25  # per Duncan (2026-08-17): normal range is 1-16, but a
# MIRRORED card (Mirror always copies at +1 level) can legitimately show 17 --
# so 17 is the real normal ceiling, not 16. Set to 25 instead of 17 here only
# to cover one documented outlier: Hero Mini P.E.K.K.A. can reach level 25 via
# a specific mechanic. That mechanic itself is NOT being specially handled
# anywhere (per Duncan, doesn't need to be accounted for yet) -- this range is
# just a permissive sanity bound (catches an obvious typo like "300," doesn't
# reject a real rare value).

QUEUE_DIR = "assets/unlabeled_level_queue"  # RULE #2: crops land here for Duncan
# to label with review_level_queue.py; NEVER auto-filed into the permanent library.
QUEUE_MAX = 400            # stop banking once the queue is this deep (review it)
QUEUE_DEDUP_THRESHOLD = 0.95  # skip a crop near-identical to one already queued
QUEUE_MIN_BADGE_PX = 8     # ignore boxes too small to hold a readable digit


def _level_to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, LEVEL_MATCH_SIZE)


def load_level_templates() -> dict:
    """Returns {value(str): [gray_example, ...]}. Empty dict (not an error) if
    LEVEL_TEMPLATE_DIR doesn't exist yet -- callers treat an empty set as "read
    nothing / crosscheck always passes" rather than crashing."""
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


_level_templates_cache = None  # lazy-loaded once per process, not per call/frame
# -- read_level()/detect_side() run PER UNIT PER FRAME, so reloading from disk
# every time would be wasteful. add_level_example() invalidates it.


def _get_level_templates() -> dict:
    global _level_templates_cache
    if _level_templates_cache is None:
        _level_templates_cache = load_level_templates()
    return _level_templates_cache


def best_level_match(crop: np.ndarray, templates: dict):
    """Returns (value_str, confidence). `crop` is a color (BGR) or grayscale
    badge image; it is greyed + resized to LEVEL_MATCH_SIZE before matching."""
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
    """Saves `crop` (color or grayscale, already isolated to the badge) as a new
    example for level `value`, deduped against near-identical examples already on
    file. Returns the saved path, or None if skipped. Mirrors
    perception/elixir_counter.py's add_example() exactly -- see
    perception/review_level_queue.py for the review tool that calls this."""
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
    _level_templates_cache = None  # invalidate so the next read picks up the new example
    out_path = os.path.join(value_dir, f"{int(time.time() * 1000)}.png")
    cv2.imwrite(out_path, gray_full)
    return out_path


GLYPH_WHITE_PIXEL_THRESHOLD = 200  # grayscale value counted as "white" (the digit's own fill color)
GLYPH_MIN_WHITE_FRACTION = 0.10  # a real digit crop must be at least this white. A
# genuine badge showing an uncollected value scores poorly against the only
# templates on file (correctly -- it's a different digit), so this value-independent
# check is what tells "a real but un-templated digit" from "not a badge at all"
# (a Knight boot measured 0.0 here; six real digit examples measured 0.18-0.22).
GLYPH_MAX_WHITE_FRACTION = 0.5  # generous upper bound; guards against a crop gone
# almost entirely white (badly mis-cropped).


def _looks_like_digit_glyph(crop: np.ndarray) -> bool:
    """Generic (value-independent) check: does this crop have a real white glyph's
    worth of bright pixels, distinct from the badge's own colored background?
    Template matching alone can only ever confirm a digit value it ALREADY has an
    example of, never a genuinely new one -- this catches the new ones."""
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    white_fraction = float((gray > GLYPH_WHITE_PIXEL_THRESHOLD).mean())
    return GLYPH_MIN_WHITE_FRACTION <= white_fraction <= GLYPH_MAX_WHITE_FRACTION


def _crop(screenshot: np.ndarray, bbox: tuple) -> np.ndarray:
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    h, w = screenshot.shape[:2]
    return screenshot[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]


def read_level(screenshot: np.ndarray, level_bbox: tuple, templates: dict = None):
    """Read the digit from a detected `bar-level` box.

    Returns (value_str, confidence) when the crop matches a known digit template
    at >= LEVEL_MATCH_THRESHOLD, else (None, confidence). None means "don't guess"
    -- either not a real glyph, or a genuine level we have no template for yet
    (collect it via queue_level_crop -> review_level_queue). It deliberately does
    NOT fall back to _looks_like_digit_glyph for a VALUE: that check only says "a
    digit is here," not which one, so returning a value from it would be a guess.
    """
    if templates is None:
        templates = _get_level_templates()
    crop = _crop(screenshot, level_bbox)
    if crop.size == 0 or min(crop.shape[:2]) < QUEUE_MIN_BADGE_PX:
        return None, 0.0
    value, score = best_level_match(crop, templates)
    if value is not None and score >= LEVEL_MATCH_THRESHOLD:
        return value, score
    return None, score


def queue_level_crop(screenshot: np.ndarray, level_bbox: tuple, guess: str = None) -> str:
    """RULE #2 collection: bank a bar-level crop to QUEUE_DIR for Duncan to label
    with review_level_queue.py. Deduped against crops already queued and capped at
    QUEUE_MAX. Returns the saved path, or None if skipped (dup / queue full / too
    small). Never writes into the permanent library -- only the review tool does."""
    crop = _crop(screenshot, level_bbox)
    if crop.size == 0 or min(crop.shape[:2]) < QUEUE_MIN_BADGE_PX:
        return None
    os.makedirs(QUEUE_DIR, exist_ok=True)
    existing = [f for f in os.listdir(QUEUE_DIR) if f.endswith(".png")]
    if len(existing) >= QUEUE_MAX:
        return None
    small = _level_to_gray(crop)
    for filename in existing:
        prev = cv2.imread(os.path.join(QUEUE_DIR, filename), cv2.IMREAD_GRAYSCALE)
        if prev is None:
            continue
        result = cv2.matchTemplate(small, cv2.resize(prev, LEVEL_MATCH_SIZE), cv2.TM_CCOEFF_NORMED)
        if float(result[0][0]) > QUEUE_DEDUP_THRESHOLD:
            return None
    tag = f"{guess}_" if guess else ""
    out_path = os.path.join(QUEUE_DIR, f"{tag}{int(time.time() * 1000)}.png")
    cv2.imwrite(out_path, crop)  # save COLOR so review shows the real badge
    return out_path
