"""
Semi-automated data collection: runs continuously during a live match,
capturing the hand every INTERVAL_SECONDS. For each of the 4 slots:

- If it's already confidently recognized (>= CONFIDENCE_THRESHOLD), the
  crop is auto-filed as a new example for that card (deduped against
  near-identical examples already on file) — no typing needed. This is
  what actually builds up robustness: many real examples of each known
  card, captured across whatever animation moments, elixir states, and
  lighting happen to occur during normal play.
- If it's not recognized, the crop is saved to a review queue instead of
  interrupting play — batch-label everything in the queue afterward with
  review_queue.py, in one pass, instead of stopping mid-match every time.

If config/deck.txt exists, matching is restricted to just those 8 cards
instead of the whole template library — more accurate (no risk of a false
match against an unrelated card from some other deck) and it means any
recognition failure is known for certain to be an under-represented
example of one of THOSE 8, not a genuinely new card. Run
seed_deck_from_catalog.py first so every deck card has at least a rough
starting reference (from its Supercell catalog icon) before your first
collection session — otherwise a deck card with zero examples can't be
recognized at all on its first appearance.

Also auto-collects elixir-number examples in the same loop (same capture,
no extra ADB round-trips). Unlike cards, this can't bootstrap off its own
template confidence — plenty of digit values may have zero templates on
day one, so there's nothing to match against yet. Instead it self-labels
off elixir_counter.py's BAR reading, which needs no templates at all: a
frame only gets auto-filed when the bar's continuous estimate is within
ELIXIR_STABLE_MARGIN of a whole number (i.e. we just ticked over, not
mid-regen), so the inferred integer label is trustworthy even before any
digit template exists for it. This is a much shorter tail than cards —
only 11 possible values, most of which show up within a game or two — so
there's no separate review-queue step for it; anything ambiguous is just
skipped rather than queued.

Also auto-collects level-badge digit examples (added 2026-08-17, per
Duncan's request to grow perception/side_reader.py's digit-template
library beyond its single bootstrap "3" example) — same capture, but this
one needs the full unit-detection pass (perception/unit_detector.py) to
find each on-screen unit's bbox first, since a badge's position moves
with whatever unit it belongs to (unlike the elixir number's fixed screen
region). This is closer to the CARD case than the elixir case: there's no
independent signal (like elixir's bar) to self-label a brand-new digit
value from nothing, so a badge crop is only auto-filed when it ALREADY
matches an existing template with high confidence (AUTO_FILE_LEVEL_THRESHOLD,
stricter than side_reader.py's own looser LEVEL_MATCH_THRESHOLD used just
to accept a candidate as "probably a real badge" — a bad auto-filed
example would corrupt the template set more permanently than a one-off
loose classification, so this bar is set higher). A candidate that passed
side_reader.py's own color/shape search but doesn't match anything
confidently gets queued to LEVEL_QUEUE_DIR instead — batch-label with
review_level_queue.py, same two-step pattern as cards, since (unlike
elixir's 11-value tail) new digit values could plausibly keep showing up
for a while as different cards' levels are seen.

Also auto-collects spawn-icon detections (added 2026-08-17, per Duncan's
request to grow perception/spawn_icon_reader.py's template library beyond
its 2 bootstrap screenshots, and to get real usage data on how often it
fires). Piggybacks on the same per-frame unit-detection pass level-badge
collection already added above. UNLIKE every other collected asset type
here, this is ALWAYS queued for Duncan's own review, NEVER auto-filed
directly into assets/spawn_icon_templates/ — a first version did auto-file
with no human check at all (side comes from color, which felt "confident
enough" to skip review), and Duncan caught a genuinely ambiguous real
example that had already gone into the permanent library unreviewed. This
detector is new/unvalidated; a wrong permanent example is a worse outcome
than a few seconds of review time. See SPAWN_ICON_QUEUE_DIR's own comment
and CLAUDE.md's "never write to a permanent asset library without review"
rule. Run `python -m perception.review_spawn_icon_queue` afterward.

Usage:
    python -m perception.auto_collect
    python -m perception.auto_collect --seconds 120   # stop after a fixed duration instead of Ctrl+C
Stop anytime with Ctrl+C — everything saved so far is kept.
"""

import argparse
import os
import time

import cv2

from config.hand_region import hand_slot_regions
from control.actions import EmulatorController
from perception.capture import ScreenCapture
from perception.card_detector import (
    CONFIDENCE_THRESHOLD,
    DUPLICATE_THRESHOLD,
    add_example,
    best_match,
    load_templates,
    to_gray,
)
from perception.elixir_counter import NUMBER_REGION as ELIXIR_NUMBER_REGION
from perception.elixir_counter import add_example as add_elixir_example
from perception.elixir_counter import read_bar_fill
from perception.side_reader import (
    LEVEL_MATCH_SIZE,
    add_level_example,
    best_level_match,
    candidate_badge_crop,
    load_level_templates,
)
from perception.spawn_icon_reader import MATCH_SIZE as SPAWN_ICON_MATCH_SIZE
from perception.spawn_icon_reader import find_spawn_icon
from perception.unit_detector import UnitDetector

INTERVAL_SECONDS = 1.0
QUEUE_DIR = "assets/unlabeled_queue"
LEVEL_QUEUE_DIR = "assets/unlabeled_level_queue"  # kept separate from QUEUE_DIR
# (cards) since these are a different kind of crop (a badge digit, not a
# hand-card icon) -- mixing them would make review_queue.py's dedup/display
# logic ambiguous about which add_example() to file a labeled item under.
SPAWN_ICON_QUEUE_DIR = "assets/unlabeled_spawn_icon_queue"  # every detected
# icon lands here for Duncan's own review, NEVER auto-filed directly into
# assets/spawn_icon_templates/ -- see the collection loop's own comment for
# why (a real 2026-08-17 incident: an early version auto-filed with zero
# human check, and Duncan caught a genuinely ambiguous example that had
# already gone in unreviewed). Filenames are tagged "_claimed_<side>" with
# what find_spawn_icon() itself believed, for review_spawn_icon_queue.py to
# show as a starting suggestion -- NOT as a trusted label; the reviewer
# (Duncan) makes the actual call, this detector is too new/unvalidated to
# trust unsupervised for something written into a permanent asset library.
DECK_FILE = "config/deck.txt"
ELIXIR_STABLE_MARGIN = 0.12  # bar estimate must be within this of a whole number to trust it as a label
AUTO_FILE_LEVEL_THRESHOLD = 0.85  # stricter than side_reader.py's own
# LEVEL_MATCH_THRESHOLD=0.65 (which only needs to gate "is this probably a
# real badge, not a boot") -- see module docstring's own reasoning for why
# auto-filing a permanent template needs a higher bar than a one-off
# per-frame classification.


def load_deck():
    """Returns the deck card list from config/deck.txt, or None if that
    file doesn't exist (falls back to matching the whole library)."""
    if not os.path.isfile(DECK_FILE):
        return None
    names = []
    with open(DECK_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    return names


def already_queued(crop) -> bool:
    """Dedupes against near-identical crops already sitting in the queue,
    so a card you can't identify yet doesn't fill the queue with hundreds
    of near-copies of the same unrecognized card."""
    if not os.path.isdir(QUEUE_DIR):
        return False
    gray = to_gray(crop)
    for filename in os.listdir(QUEUE_DIR):
        if not filename.endswith(".png"):
            continue
        existing = cv2.imread(os.path.join(QUEUE_DIR, filename))
        if existing is None:
            continue
        result = cv2.matchTemplate(gray, to_gray(existing), cv2.TM_CCOEFF_NORMED)
        if float(result[0][0]) > DUPLICATE_THRESHOLD:
            return True
    return False


def already_queued_level(crop) -> bool:
    """Same as already_queued() but for LEVEL_QUEUE_DIR, resized/compared
    at LEVEL_MATCH_SIZE (badge crops are much smaller than hand-card
    slots, so reusing card_detector.py's MATCH_SIZE here would be wrong)."""
    if not os.path.isdir(LEVEL_QUEUE_DIR):
        return False
    gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop, LEVEL_MATCH_SIZE)
    for filename in os.listdir(LEVEL_QUEUE_DIR):
        if not filename.endswith(".png"):
            continue
        existing = cv2.imread(os.path.join(LEVEL_QUEUE_DIR, filename), cv2.IMREAD_GRAYSCALE)
        if existing is None:
            continue
        result = cv2.matchTemplate(gray, cv2.resize(existing, LEVEL_MATCH_SIZE), cv2.TM_CCOEFF_NORMED)
        if float(result[0][0]) > DUPLICATE_THRESHOLD:
            return True
    return False


def already_queued_spawn_icon(crop, side) -> bool:
    """Same as already_queued_level() but for SPAWN_ICON_QUEUE_DIR, only
    comparing against already-queued items tagged with the same claimed
    `side` (an icon that later turns out misclassified shouldn't dedupe
    away a genuinely different-side one that happens to look similar)."""
    if not os.path.isdir(SPAWN_ICON_QUEUE_DIR):
        return False
    gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop, SPAWN_ICON_MATCH_SIZE)
    for filename in os.listdir(SPAWN_ICON_QUEUE_DIR):
        if not filename.endswith(f"_claimed_{side}.png"):
            continue
        existing = cv2.imread(os.path.join(SPAWN_ICON_QUEUE_DIR, filename), cv2.IMREAD_GRAYSCALE)
        if existing is None:
            continue
        result = cv2.matchTemplate(gray, cv2.resize(existing, SPAWN_ICON_MATCH_SIZE), cv2.TM_CCOEFF_NORMED)
        if float(result[0][0]) > DUPLICATE_THRESHOLD:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=None,
                         help="Run for this many seconds instead of until Ctrl+C")
    args = parser.parse_args()

    os.makedirs(QUEUE_DIR, exist_ok=True)
    os.makedirs(LEVEL_QUEUE_DIR, exist_ok=True)
    os.makedirs(SPAWN_ICON_QUEUE_DIR, exist_ok=True)

    print("Connecting to LDPlayer...")
    controller = EmulatorController()
    capture = ScreenCapture(controller)
    templates = load_templates()
    unit_detector = UnitDetector()
    level_templates = load_level_templates()

    deck = load_deck()
    if deck:
        missing = [name for name in deck if name not in templates]
        templates = {name: examples for name, examples in templates.items() if name in deck}
        print(f"Deck restriction active ({DECK_FILE}): matching only against {sorted(templates.keys())}")
        if missing:
            print(
                f"WARNING: no examples at all for {missing} — these can't be recognized "
                f"yet. Run seed_deck_from_catalog.py, or label_hand.py once for each."
            )
    else:
        print(f"No {DECK_FILE} found — matching against the full library (all cards on file).")

    total_examples = sum(len(v) for v in templates.values())
    print(f"Loaded {total_examples} example(s) across {len(templates)} card(s).")
    print("Collecting during play — press Ctrl+C to stop.\n")

    print("Also auto-collecting elixir-number examples (self-labeled off the bar reading).")
    total_level_examples = sum(len(v) for v in level_templates.values())
    print(f"Also auto-collecting level-badge examples ({total_level_examples} existing example(s) "
          f"across {len(level_templates)} value(s)) — matched confidently -> filed, unmatched -> queued.\n")

    added_by_card = {}  # card_name -> [file paths added this run]
    added_by_elixir = {}  # value_str -> [file paths added this run]
    added_by_level = {}  # value_str -> [file paths added this run]
    queued_paths = []
    queued_level_paths = []
    queued_spawn_icon_paths = []
    seen = 0
    start = time.time()
    try:
        while args.seconds is None or (time.time() - start) < args.seconds:
            screenshot = capture.screenshot()
            for (x1, y1, x2, y2) in hand_slot_regions():
                crop = screenshot[y1:y2, x1:x2]
                seen += 1
                name, score = best_match(crop, templates)

                if score >= CONFIDENCE_THRESHOLD:
                    path = add_example(name, crop)
                    if path:
                        added_by_card.setdefault(name, []).append(path)
                else:
                    if not already_queued(crop):
                        ts = int(time.time() * 1000)
                        out_path = os.path.join(QUEUE_DIR, f"{ts}.png")
                        cv2.imwrite(out_path, crop)
                        queued_paths.append(out_path)

            bar_estimate = read_bar_fill(screenshot)
            nearest = round(bar_estimate)
            if abs(bar_estimate - nearest) <= ELIXIR_STABLE_MARGIN and 0 <= nearest <= 10:
                ex1, ey1, ex2, ey2 = ELIXIR_NUMBER_REGION
                elixir_crop = screenshot[ey1:ey2, ex1:ex2]
                path = add_elixir_example(str(nearest), elixir_crop)
                if path:
                    added_by_elixir.setdefault(str(nearest), []).append(path)

            detected_units = unit_detector.detect(screenshot)

            for unit in detected_units:
                badge_crop = candidate_badge_crop(screenshot, unit["bbox"])
                if badge_crop is None or badge_crop.size == 0:
                    continue
                value, score = best_level_match(badge_crop, level_templates)
                if score >= AUTO_FILE_LEVEL_THRESHOLD:
                    path = add_level_example(value, badge_crop)
                    if path:
                        added_by_level.setdefault(value, []).append(path)
                        level_templates = load_level_templates()  # pick up the new example for the next unit this same frame
                elif not already_queued_level(badge_crop):
                    ts = int(time.time() * 1000)
                    out_path = os.path.join(LEVEL_QUEUE_DIR, f"{ts}.png")
                    cv2.imwrite(out_path, badge_crop)
                    queued_level_paths.append(out_path)

            for unit in detected_units:
                side, _area, icon_bbox = find_spawn_icon(screenshot, unit["bbox"])
                if side is None or icon_bbox is None:
                    continue
                ix1, iy1, ix2, iy2 = icon_bbox
                icon_crop = screenshot[iy1:iy2, ix1:ix2]
                if icon_crop.size == 0:
                    continue
                # Deliberately ALWAYS queued for Duncan's own review, never
                # auto-filed straight into assets/spawn_icon_templates/ --
                # corrected 2026-08-17 after an early version auto-filed
                # directly (no human check at all, unlike every other
                # collected asset type in this project) and Duncan caught a
                # genuinely ambiguous real example that had gone in
                # unreviewed. This detector is new and unvalidated; a wrong
                # permanent example is worse than a few seconds of his time
                # reviewing a small queue. See CLAUDE.md's "never write to a
                # permanent asset library without review" rule.
                if not already_queued_spawn_icon(icon_crop, side):
                    ts = int(time.time() * 1000)
                    out_path = os.path.join(SPAWN_ICON_QUEUE_DIR, f"{ts}_claimed_{side}.png")
                    cv2.imwrite(out_path, icon_crop)
                    queued_spawn_icon_paths.append(out_path)

            time.sleep(INTERVAL_SECONDS)
    except KeyboardInterrupt:
        pass

    total_added = sum(len(v) for v in added_by_card.values())
    total_elixir_added = sum(len(v) for v in added_by_elixir.values())
    print(f"\nStopped. Saw {seen} slot-crops this run.\n")

    if added_by_card:
        print(f"Auto-filed {total_added} new card example(s) — go spot-check these and delete any that are wrong:\n")
        for card_name, paths in sorted(added_by_card.items()):
            print(f"  {card_name} (+{len(paths)}):")
            for p in paths:
                print(f"    {p}")
        print()
    else:
        print("No new card examples auto-filed this run.\n")

    if added_by_elixir:
        print(f"Auto-filed {total_elixir_added} new elixir example(s):\n")
        for value, paths in sorted(added_by_elixir.items(), key=lambda kv: int(kv[0])):
            print(f"  elixir={value} (+{len(paths)}):")
            for p in paths:
                print(f"    {p}")
        missing = sorted(set(str(v) for v in range(11)) - set(added_by_elixir.keys()), key=int)
        if missing:
            print(f"\nStill no example this run for: {missing} (may already have some from a prior run).")
        print()
    else:
        print("No new elixir examples auto-filed this run (values were mid-regen every time this ran — normal for a short session).\n")

    if queued_paths:
        print(f"{len(queued_paths)} unrecognized card(s) saved for labeling:")
        for p in queued_paths:
            print(f"    {p}")
        print("Run `python -m perception.review_queue` to label these.")

    total_level_added = sum(len(v) for v in added_by_level.values())
    if added_by_level:
        print(f"\nAuto-filed {total_level_added} new level-badge example(s) — go spot-check these and delete any that are wrong:\n")
        for value, paths in sorted(added_by_level.items(), key=lambda kv: int(kv[0])):
            print(f"  level={value} (+{len(paths)}):")
            for p in paths:
                print(f"    {p}")
        print()
    else:
        print("\nNo new level-badge examples auto-filed this run.\n")

    if queued_level_paths:
        print(f"{len(queued_level_paths)} unrecognized level badge(s) saved for labeling:")
        for p in queued_level_paths:
            print(f"    {p}")
        print("Run `python -m perception.review_level_queue` to label these.")

    if queued_spawn_icon_paths:
        print(f"\n{len(queued_spawn_icon_paths)} spawn-icon detection(s) saved for your review "
              f"(NOT auto-filed — filenames end \"_claimed_<side>\" showing what the detector guessed, "
              f"not a trusted label):")
        for p in queued_spawn_icon_paths:
            print(f"    {p}")
        print("Run `python -m perception.review_spawn_icon_queue` to review these.")
    else:
        print("\nNo spawn-icon detections this run.")


if __name__ == "__main__":
    main()
