"""
The "macro" — watches TV Royale replays back-to-back, unattended, running
the full auto-labeling pipeline (two game_knowledge.cycle_tracker.CycleTracker
instances, one per TV Royale player, feeding data/auto_labeler.py's
localize_play + YoloDatasetWriter) through each one. See Project_Status /
clash-royale-ai-pipeline-explained.md section 5 for the full design.

NOT runnable yet. Two things block it, both needing a live session with
Duncan — can't be done blind, same reason config/hand_region.py,
config/red_zones.py, and every other pixel calibration in this project was
done against a real captured screenshot, not guessed:

1. config/tv_royale_regions.py doesn't exist yet — needs calibrating both
   players' hand-slot/elixir/"Next" regions and the arena crop against a
   real TV Royale screenshot, the same way config/hand_region.py was
   calibrated for live play (see find_coords.py).
2. The actual menu navigation (how to enter TV Royale mode from the main
   menu, how to detect a replay has ended, how to advance to the next one)
   isn't known yet — Clash Royale's TV Royale UI flow hasn't been explored.
   The functions below are structural placeholders, not guesses — see
   each one's docstring for exactly what's needed to fill them in.

The per-frame pipeline logic itself (dual play detection -> localize_play
-> YoloDatasetWriter.write) is real and already tested — see
data/auto_labeler.py and data/test_auto_labeler.py — just not wired into
the main loop below yet, since that wiring needs the region constants from
point 1.

Usage (once calibrated):
    python -m data.tv_royale_watcher --minutes 120
    python -m data.tv_royale_watcher --max-replays 20
"""

import argparse
import time

from control.actions import EmulatorController
from data.auto_labeler import YoloDatasetWriter
from game_knowledge.cycle_tracker import CycleTracker
from perception.capture import ScreenCapture
from perception.unit_detector import UnitDetector
from perception.unit_tracker import UnitTracker

# from config.tv_royale_regions import (...)  # uncomment once calibrated


def enter_tv_royale(controller: EmulatorController):
    """Navigate from wherever the app currently is into TV Royale spectator
    mode. Needs real tap coordinates from a live calibration session (see
    module docstring) — deliberately not guessed."""
    raise NotImplementedError(
        "TV Royale navigation hasn't been calibrated yet. Needs a live session: "
        "open TV Royale by hand once, capture each screen along the way and use "
        "find_coords.py to get real tap coordinates, then fill this in."
    )


def replay_has_ended(screenshot) -> bool:
    """True if the current screenshot shows a replay-ended state (results
    screen, "next replay" prompt, etc.) rather than active gameplay. Needs
    a real captured example of that screen to calibrate against — same
    reasoning as enter_tv_royale."""
    raise NotImplementedError("Needs a real captured 'replay ended' screenshot to calibrate against.")


def advance_to_next_replay(controller: EmulatorController):
    """Tap whatever advances from a finished replay to the next one."""
    raise NotImplementedError("Needs real tap coordinates from a live calibration session.")


def run(minutes: float = None, max_replays: int = None):
    print("Connecting to LDPlayer...")
    controller = EmulatorController()
    capture = ScreenCapture(controller)
    unit_detector = UnitDetector()

    writer = YoloDatasetWriter()
    unit_tracker_top = UnitTracker()
    unit_tracker_bottom = UnitTracker()
    cycle_top = CycleTracker(deck=[])
    cycle_bottom = CycleTracker(deck=[])

    enter_tv_royale(controller)

    start = time.time()
    replays_watched = 0
    try:
        while True:
            if minutes is not None and (time.time() - start) / 60 >= minutes:
                break
            if max_replays is not None and replays_watched >= max_replays:
                break

            screenshot = capture.screenshot()
            if replay_has_ended(screenshot):
                advance_to_next_replay(controller)
                replays_watched += 1
                continue

            # Per-frame pipeline — structure only; region-dependent pieces
            # wait on config/tv_royale_regions.py (see module docstring):
            #   top_hand = [screenshot crop per top-player hand slot]
            #   bottom_hand = [same for the bottom player]
            #   top_elixir = read_elixir(screenshot, ..., region=TOP_ELIXIR_REGION)
            #   bottom_elixir = read_elixir(screenshot, ..., region=BOTTOM_ELIXIR_REGION)
            #   cycle_top.update(top_hand, top_elixir, now=time.time())
            #   cycle_bottom.update(bottom_hand, bottom_elixir, now=time.time())
            #   -- on a detected last_play from either side:
            #      new_tracks = unit_tracker_*.update(unit_detector.detect(screenshot))
            #      bbox = data.auto_labeler.localize_play(new_tracks, ...)
            #      if bbox: writer.write(arena_crop, [(card, bbox)])  -- class only,
            #      no side field (see data/auto_labeler.py's module docstring for why)
            pass
    except KeyboardInterrupt:
        pass
    finally:
        print(f"Stopped after watching {replays_watched} replay(s).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=None,
                         help="Stop after this many minutes (default: run until Ctrl+C)")
    parser.add_argument("--max-replays", type=int, default=None,
                         help="Stop after watching this many replays")
    args = parser.parse_args()
    run(minutes=args.minutes, max_replays=args.max_replays)


if __name__ == "__main__":
    main()
