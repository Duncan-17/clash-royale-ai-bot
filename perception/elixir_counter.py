"""
Reads current elixir from the live match UI: the pink droplet + number in
the bottom-left of the hand row, and the 10-segment bar next to it.

Two independent signals, combined:

- The NUMBER is the primary, trusted signal for the integer part. It's one
  of only 11 possible values (0-10), rendered in a consistent bold white
  font, so template matching against real examples (same approach as
  card_detector.py) gets high-confidence, unambiguous reads.
- The BAR is measured by pixel fill (algorithmic, no templates needed) to
  get the FRACTIONAL progress toward the next elixir point. The bar
  renders completed points as solid pink and the in-progress segment as a
  lighter partial fill whose width grows continuously as elixir
  regenerates (confirmed by comparing two live captures ~15s apart during
  calibration: the partial-fill width shrank from ~73% of a segment down
  to ~15% of the next segment, consistent with continuous regen rather
  than a fixed "next up" indicator).

Bar geometry below (BAR_LEFT_X, BAR_RIGHT_X, sample rows) was calibrated
against a real 900x1600 capture by locating the segment tick-divider
lines pixel-by-pixel; positions were identical across two independent
captures. Re-calibrate (same way hand_region.py notes) if LDPlayer's
resolution setting ever changes.

Usage:
    python -m perception.elixir_counter
Reads test_screenshot.png, prints the number-match result and the
bar-based continuous estimate.
"""

import os
import time

import cv2
import numpy as np

# --- Number (digit) region + template matching -----------------------

NUMBER_REGION = (246, 1522, 312, 1572)  # (x1, y1, x2, y2) — wide enough for "10"
TEMPLATE_DIR = "assets/elixir_templates"
MATCH_SIZE = (64, 64)
CONFIDENCE_THRESHOLD = 0.85
DUPLICATE_THRESHOLD = 0.995
MAX_EXAMPLES_PER_VALUE = 20

# --- Bar geometry -------------------------------------------------------

BAR_LEFT_X = 244.5    # pixel x of the 0-elixir mark (left edge of segment 1)
BAR_RIGHT_X = 864.0   # pixel x of the 10-elixir mark (right edge of segment 10)
NUM_SEGMENTS = 10
# Rows sampled through the flat body of the bar (avoids the rounded top/
# bottom bevel, which has its own highlight/shadow colors).
BAR_SAMPLE_Y_ROWS = [1550, 1554, 1558, 1562, 1566, 1570, 1574]
FILLED_SUM_THRESHOLD = 210  # sum(R,G,B): solid pink ~621, partial-fill glow ~270-300, unfilled navy ~176-186
FILLED_MAX_GREEN = 200  # excludes the white (255,255,255) elixir-number glyph, which overlaps the
# bar's leftmost ~90px and would otherwise register as "filled" regardless of actual elixir level —
# both real fill colors (pink G=137, glow G=88) have low green, white digit text doesn't.


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, MATCH_SIZE)


def load_templates() -> dict:
    """Returns {value(str): [gray_example, ...]}."""
    templates = {}
    if not os.path.isdir(TEMPLATE_DIR):
        return templates
    for name in os.listdir(TEMPLATE_DIR):
        value_dir = os.path.join(TEMPLATE_DIR, name)
        if not os.path.isdir(value_dir):
            continue
        examples = []
        for filename in os.listdir(value_dir):
            if not filename.endswith(".png"):
                continue
            img = cv2.imread(os.path.join(value_dir, filename), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            examples.append(cv2.resize(img, MATCH_SIZE))
        if examples:
            templates[name] = examples
    return templates


def best_match(crop: np.ndarray, templates: dict):
    """Returns (value_str, confidence). `crop` is a color (BGR) image."""
    if not templates:
        return None, 0.0
    resized_gray = to_gray(crop)
    best_value, best_score = None, -1.0
    for value, examples in templates.items():
        for example in examples:
            result = cv2.matchTemplate(resized_gray, example, cv2.TM_CCOEFF_NORMED)
            score = float(result[0][0])
            if score > best_score:
                best_value, best_score = value, score
    return best_value, best_score


def add_example(value: str, crop: np.ndarray):
    """Saves `crop` (color or grayscale) as a new example for elixir value
    `value` ("0".."10"), deduped against near-identical examples already on
    file. Returns the saved path, or None if skipped."""
    value_dir = os.path.join(TEMPLATE_DIR, value)
    os.makedirs(value_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(value_dir) if f.endswith(".png")]
    if len(existing_files) >= MAX_EXAMPLES_PER_VALUE:
        return None

    gray_full = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray_small = to_gray(crop)

    for filename in existing_files:
        existing = cv2.imread(os.path.join(value_dir, filename), cv2.IMREAD_GRAYSCALE)
        if existing is None:
            continue
        result = cv2.matchTemplate(gray_small, cv2.resize(existing, MATCH_SIZE), cv2.TM_CCOEFF_NORMED)
        if float(result[0][0]) > DUPLICATE_THRESHOLD:
            return None

    out_path = os.path.join(value_dir, f"{int(time.time() * 1000)}.png")
    cv2.imwrite(out_path, gray_full)
    return out_path


def read_number(screenshot: np.ndarray, templates: dict = None):
    """Returns (value_str_or_None, confidence) from template matching.
    Pass `templates` (from load_templates()) when calling this every frame
    in a loop — loading from disk on every call is wasteful."""
    x1, y1, x2, y2 = NUMBER_REGION
    crop = screenshot[y1:y2, x1:x2]
    if templates is None:
        templates = load_templates()
    value, score = best_match(crop, templates)
    if score < CONFIDENCE_THRESHOLD:
        return None, score
    return value, score


CONFIRM_RUN = 4  # consecutive filled px (scanning right-to-left) required to confirm we've hit the real fill region


def read_bar_fill(screenshot: np.ndarray) -> float:
    """Returns a continuous elixir estimate (0.0-10.0) by measuring how far
    across the bar the fill (solid pink OR the lighter in-progress glow)
    extends.

    Scans right-to-left and combines rows by majority vote (not by
    averaging raw pixel colors first) — the elixir NUMBER glyph is drawn
    directly on top of the bar's leftmost ~90px, and its black outline +
    white fill, blended across multiple sample rows, produces intermediate
    colors that can spuriously look like real fill. Scanning from the
    right avoids the digit entirely for anything but very low elixir, and
    per-row majority voting (rather than blending colors together) avoids
    manufacturing colors no single pixel actually has."""
    left, right = int(BAR_LEFT_X), int(BAR_RIGHT_X)
    width = right - left

    votes = np.zeros(width, dtype=int)
    for y in BAR_SAMPLE_Y_ROWS:
        row = screenshot[y, left:right].astype(int)
        sums = row.sum(axis=1)
        green = row[:, 1]
        filled = (sums > FILLED_SUM_THRESHOLD) & (green < FILLED_MAX_GREEN)
        votes += filled

    filled = votes >= (len(BAR_SAMPLE_Y_ROWS) // 2 + 1)  # majority of rows agree

    boundary_offset = 0  # stays 0 (elixir 0) if no confirmed fill run is ever found
    consecutive_filled = 0
    for i in range(width - 1, -1, -1):
        if filled[i]:
            consecutive_filled += 1
            if consecutive_filled >= CONFIRM_RUN:
                boundary_offset = i + CONFIRM_RUN
                break
        else:
            consecutive_filled = 0

    fraction = boundary_offset / width
    return round(fraction * NUM_SEGMENTS, 2)


def read_elixir(screenshot: np.ndarray, templates: dict = None) -> dict:
    """Combines both signals. Returns a dict:
        value: int or None       — trusted integer read (from the number)
        value_confidence: float  — number-match confidence
        bar_estimate: float      — continuous 0-10 estimate from the bar
        fraction: float          — progress (0-1) toward the next point,
                                    taken from the bar, cross-checked
                                    against the number when both agree
    Falls back to floor(bar_estimate) for `value` if the number isn't
    confidently recognized (e.g. no templates yet for that digit). Pass
    `templates` (from load_templates()) when calling this every frame in a
    loop — loading from disk on every call is wasteful.
    """
    number_str, number_conf = read_number(screenshot, templates)
    bar_estimate = read_bar_fill(screenshot)

    if number_str is not None:
        value = int(number_str)
    else:
        value = int(bar_estimate)  # fallback, floor

    fraction = bar_estimate - int(bar_estimate)
    # Sanity check: if the bar's floor disagrees with the trusted number by
    # more than 1, the bar measurement is probably noisy (e.g. mid-transition
    # animation frame) — don't report a fraction we can't stand behind.
    if abs(int(bar_estimate) - value) > 1:
        fraction = 0.0

    return {
        "value": value,
        "value_confidence": number_conf,
        "bar_estimate": bar_estimate,
        "fraction": fraction,
    }


if __name__ == "__main__":
    screenshot = cv2.imread("test_screenshot.png")
    if screenshot is None:
        raise FileNotFoundError("test_screenshot.png not found — capture one first.")

    result = read_elixir(screenshot)
    print(f"Number read: {result['value']} (confidence={result['value_confidence']:.3f})")
    print(f"Bar estimate: {result['bar_estimate']:.2f} / 10")
    print(f"Fraction toward next point: {result['fraction']:.2f}")
