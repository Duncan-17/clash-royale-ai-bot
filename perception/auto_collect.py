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

INTERVAL_SECONDS = 1.0
QUEUE_DIR = "assets/unlabeled_queue"
DECK_FILE = "config/deck.txt"
ELIXIR_STABLE_MARGIN = 0.12  # bar estimate must be within this of a whole number to trust it as a label


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=None,
                         help="Run for this many seconds instead of until Ctrl+C")
    args = parser.parse_args()

    os.makedirs(QUEUE_DIR, exist_ok=True)

    print("Connecting to LDPlayer...")
    controller = EmulatorController()
    capture = ScreenCapture(controller)
    templates = load_templates()

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

    print("Also auto-collecting elixir-number examples (self-labeled off the bar reading).\n")

    added_by_card = {}  # card_name -> [file paths added this run]
    added_by_elixir = {}  # value_str -> [file paths added this run]
    queued_paths = []
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


if __name__ == "__main__":
    main()
