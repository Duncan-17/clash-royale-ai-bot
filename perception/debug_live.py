"""
Live spot-check overlay: shows what the detector is seeing WHILE you play,
so you can catch a bad recognition the moment it happens instead of
discovering it later by reading a wall of terminal text. This is purely a
viewer — it never saves anything (that's auto_collect.py's job) — so it's
safe to run at the same time as auto_collect.py in a second terminal if
you want live feedback and data collection running together.

Draws a box around each of the 4 hand slots on top of a live screenshot,
labeled with the best-guess card name and confidence, color-coded:
  green  = confident match (>= CONFIDENCE_THRESHOLD)
  orange = a guess, but a weak one (within 0.15 of the threshold) — worth
           watching; if this keeps happening for the same card it likely
           needs more/better examples
  red    = UNKNOWN (below threshold entirely)
Also draws a box around the elixir number readout and the elixir bar,
same color coding, labeled with the read value / confidence / continuous
bar estimate (see perception/elixir_counter.py).

A resizable window opens ("Live hand detection") — drag/resize it next to
the LDPlayer window. Press 'q' with that window focused, or Ctrl+C in the
terminal, to stop.

Performance note: on this setup, ADB screencap (capture.screenshot()) is
the dominant cost per frame (~0.5-0.6s, measured via profiling — template
matching itself is ~0.1s), so the realistic ceiling here is roughly 1.5-2
fps, not a smooth live feed. A per-frame timing readout is drawn in the
corner so this is visible rather than mysterious. --interval below no
longer ADDS its own delay on top of that processing time (the previous
version did, roughly doubling per-frame latency) — it now only pads the
frame up to the requested period if processing finished early. A real
fix (multiple fps) means swapping capture.py's ADB-screencap backend for
scrcpy or adbnativeblitz, as flagged in the roadmap — not done here.

Usage:
    python -m perception.debug_live
    python -m perception.debug_live --interval 0.3   # target refresh period (only pads if processing was faster)
"""

import argparse
import time

import cv2

from config.hand_region import hand_slot_regions
from perception.auto_collect import load_deck
from perception.capture import ScreenCapture
from perception.card_detector import CONFIDENCE_THRESHOLD as CARD_CONFIDENCE_THRESHOLD
from perception.card_detector import best_match as card_best_match
from perception.card_detector import load_templates as load_card_templates
from control.actions import EmulatorController
from perception.elixir_counter import BAR_LEFT_X, BAR_RIGHT_X, NUMBER_REGION
from perception.elixir_counter import CONFIDENCE_THRESHOLD as ELIXIR_CONFIDENCE_THRESHOLD
from perception.elixir_counter import load_templates as load_elixir_templates
from perception.elixir_counter import read_elixir

WEAK_MARGIN = 0.15  # scores within this far below threshold are drawn orange instead of red
WINDOW_NAME = "Live hand detection (press q to quit)"
BAR_Y_TOP, BAR_Y_BOTTOM = 1524, 1584  # for drawing only — see elixir_counter.py for the calibrated fill-measurement rows


def color_for(name, score, threshold):
    if name not in (None, "UNKNOWN") and score >= threshold:
        return (0, 200, 0)  # green (BGR)
    if score >= threshold - WEAK_MARGIN:
        return (0, 165, 255)  # orange
    return (0, 0, 255)  # red


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=0.5,
                         help="Target seconds between refreshes — only pads the frame if "
                              "processing finished early, never adds delay on top of it")
    args = parser.parse_args()

    print("Connecting to LDPlayer...")
    controller = EmulatorController()
    capture = ScreenCapture(controller)
    card_templates = load_card_templates()
    elixir_templates = load_elixir_templates()

    deck = load_deck()
    if deck:
        card_templates = {name: examples for name, examples in card_templates.items() if name in deck}
        print(f"Deck restriction active: matching only against {sorted(card_templates.keys())}")
    else:
        print("No config/deck.txt found — matching against the full library.")

    total_examples = sum(len(v) for v in card_templates.values())
    print(f"Loaded {total_examples} card example(s) across {len(card_templates)} card(s).")
    print(f"Loaded {sum(len(v) for v in elixir_templates.values())} elixir-digit example(s) "
          f"across {len(elixir_templates)} value(s).")
    print("Opening live view — press 'q' in that window (or Ctrl+C here) to stop.\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame_start = time.time()

            t0 = time.time()
            screenshot = capture.screenshot()
            capture_ms = (time.time() - t0) * 1000
            display = screenshot.copy()

            t0 = time.time()
            for (x1, y1, x2, y2) in hand_slot_regions():
                crop = screenshot[y1:y2, x1:x2]
                name, score = card_best_match(crop, card_templates)
                if name is None or score < CARD_CONFIDENCE_THRESHOLD:
                    label = f"UNKNOWN ({score:.2f})" if name is not None else "no templates"
                    box_color = color_for("UNKNOWN", score if score is not None else 0.0, CARD_CONFIDENCE_THRESHOLD)
                else:
                    label = f"{name} ({score:.2f})"
                    box_color = color_for(name, score, CARD_CONFIDENCE_THRESHOLD)

                cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 3)
                text_y = y1 - 10 if y1 - 10 > 10 else y2 + 25
                cv2.putText(display, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, box_color, 2, cv2.LINE_AA)

            elixir = read_elixir(screenshot, elixir_templates)
            match_ms = (time.time() - t0) * 1000

            ex1, ey1, ex2, ey2 = NUMBER_REGION
            number_color = color_for(str(elixir["value"]), elixir["value_confidence"], ELIXIR_CONFIDENCE_THRESHOLD)
            cv2.rectangle(display, (ex1, ey1), (ex2, ey2), number_color, 2)
            cv2.putText(display, f"elixir={elixir['value']} ({elixir['value_confidence']:.2f})",
                        (ex1, ey1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, number_color, 2, cv2.LINE_AA)

            cv2.rectangle(display, (int(BAR_LEFT_X), BAR_Y_TOP), (int(BAR_RIGHT_X), BAR_Y_BOTTOM), (255, 200, 0), 2)
            cv2.putText(display, f"bar={elixir['bar_estimate']:.2f} (+{elixir['fraction']:.2f})",
                        (int(BAR_LEFT_X), BAR_Y_BOTTOM + 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 200, 0), 2, cv2.LINE_AA)

            total_ms = (time.time() - frame_start) * 1000
            fps = 1000 / total_ms if total_ms > 0 else 0.0
            timing_text = f"FPS {fps:.1f} | capture {capture_ms:.0f}ms | match {match_ms:.0f}ms"
            cv2.putText(display, timing_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, display)

            # Only pad up to the requested interval if processing finished
            # early — capture.screenshot() is normally the dominant cost
            # (~0.5-0.6s) so this is usually a no-op; waitKey still pumps
            # window events (drag/resize) and checks for 'q' either way.
            elapsed = time.time() - frame_start
            remaining_ms = max(1, int((args.interval - elapsed) * 1000))
            key = cv2.waitKey(remaining_ms) & 0xFF
            if key == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
