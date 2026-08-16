"""
Live spot-check overlay: shows what the detector is seeing WHILE you play,
so you can catch a bad recognition the moment it happens instead of
discovering it later by reading a wall of terminal text. This is purely a
viewer for the gameplay window — it never saves screenshots (that's
auto_collect.py's job) — so it's safe to run at the same time as
auto_collect.py in a second terminal if you want live feedback and data
collection running together. It DOES write a data log by default, see
"Logging" below.

Two windows, on purpose:
- "Live hand detection" — the gameplay screenshot with color-coded box
  outlines around each hand slot, the elixir number, the elixir bar, and
  every detected unit on the board. Has two display modes, toggled live
  with the 'd' key (starts in simple mode):
    simple   — boxes only, no text.
    detailed — boxes plus a compact label each: card name + confidence
               for hand slots, value/estimate for elixir, and for units a
               short color-coded tag combining the card's initials with
               its tracker ID (e.g. "K3" = Knight, track #3).
- "Debug panel" — a separate, plain dark panel with HAND/ELIXIR text
  (small and fixed-size, so no reflow problem there) plus two STABLE
  GRIDS for units — see "Why a grid" below. Opens positioned to the right
  of the gameplay window by default; drag either window if that doesn't
  match your monitor layout.
Box color coding:
  green  = confident card match (>= CONFIDENCE_THRESHOLD) / ally unit
  orange = a card guess, but a weak one (within 0.15 of the threshold) —
           worth watching; if this keeps happening for the same card it
           likely needs more/better examples
  red    = UNKNOWN card (below threshold) / enemy unit

Why a grid (Duncan's feedback, 2026-08-15): the original panel rendered
units as a plain text list, one line per currently-known unit. Every time
a unit appeared or disappeared, the line count changed and everything
below it reflowed up/down — with units flickering in and out (see
unit_tracker.py's docstring for why that happens even for real, unmoving
units), the whole panel was constantly jumping, making it hard to follow.
Fix: SlotGrid assigns each track_id a FIXED position (0..49) that it
keeps for as long as it exists, freed for reuse once it's gone — so a
given unit always renders in the same grid cell while tracked, and only
that cell's content changes, never the layout around it. Both grids
("LIVE" and "TRACKED", one row of headers each) share the SAME slot
assignment (driven by TRACKED's track_id set, which is always a superset
of LIVE's — see unit_tracker.py), so a given unit sits in the same cell
in both grids — e.g. seeing a cell occupied in TRACKED but blank in LIVE
means "the tracker still believes this exists, just not detected this
exact frame," directly visible by comparing the two grids at a glance
instead of reading text.

Logging (added 2026-08-15, per Duncan): writes one JSON line per frame to
logs/debug_live_<timestamp>.jsonl by default (see --log / --no-log) —
hand/elixir/live-units/tracked-units, everything needed to reconstruct
exactly when a given track flickered, how long occlusions lasted, etc.
without needing to watch it happen live. Also captures hand+elixir per
frame even though this session's ask was specifically about troop
tracking, since it's free to include now and Duncan flagged this log
will also be useful later for cross-checking against placement/action
detection once that exists (Phase 7-ish) — better to capture the full
per-frame snapshot now than redesign the log format later.

Press 'q' with either window focused, or Ctrl+C in the terminal, to stop.
Press 'd' to toggle the gameplay window between simple and detailed.

Performance note (updated 2026-08-15): capture.py now uses adbnativeblitz
(background H.264 stream) instead of per-call ADB screencap, so
capture.screenshot() itself is ~0.5ms — no longer the bottleneck. Card
template matching is ~35ms for 4 slots at the current MATCH_SIZE=64x64
(see card_detector.py's comment on that constant); unit detection
(perception/unit_detector.py's YOLOv10 ONNX model) adds another ~45-50ms
on top. Combined realistic ceiling is around 12fps. --interval below only
PADS a frame up to the target period if processing finished early, never
adds delay on top of it.

Usage:
    python -m perception.debug_live
    python -m perception.debug_live --interval 0.3   # slow it down deliberately (e.g. less CPU)
    python -m perception.debug_live --no-log          # skip writing the JSONL log
    python -m perception.debug_live --log path.jsonl  # explicit log path instead of the timestamped default
"""

import argparse
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np

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
from perception.unit_detector import UnitDetector
from perception.unit_tracker import UnitTracker

WEAK_MARGIN = 0.15  # scores within this far below threshold are drawn orange instead of red
GAMEPLAY_WINDOW = "Live hand detection (press q to quit, d to toggle detail)"
PANEL_WINDOW = "Debug panel"
PANEL_WIDTH = 420
BAR_Y_TOP, BAR_Y_BOTTOM = 1524, 1584  # for drawing only — see elixir_counter.py for the calibrated fill-measurement rows

WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
DARK_GRAY = (70, 70, 70)
PANEL_BG = (30, 30, 30)
ALLY_COLOR = (0, 200, 0)
ENEMY_COLOR = (0, 0, 255)

GRID_SLOTS = 50
GRID_COLS = 10
GRID_ROWS = 5
CELL_W = 38
CELL_H = 22


def color_for(name, score, threshold):
    if name not in (None, "UNKNOWN") and score >= threshold:
        return (0, 200, 0)  # green (BGR)
    if score >= threshold - WEAK_MARGIN:
        return (0, 165, 255)  # orange
    return (0, 0, 255)  # red


def compact_tag(name: str, track_id: int) -> str:
    """e.g. "knight" + 3 -> "K3", "mini_pekka" + 7 -> "MP7" — a short,
    at-a-glance ID rather than a full name+confidence label."""
    letters = "".join(part[0] for part in name.split("_")).upper()
    return f"{letters}{track_id}"


def draw_box(display, box, color, label=None):
    x1, y1, x2, y2 = box
    cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
    if label:
        text_y = y1 - 8 if y1 - 8 > 10 else y2 + 18
        cv2.putText(display, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)


class SlotGrid:
    """Assigns each track_id a stable slot index (0..num_slots-1) that it
    keeps for as long as it's present, freed for reuse once it's gone.
    See module docstring's "Why a grid" for the problem this solves."""

    def __init__(self, num_slots=GRID_SLOTS):
        self.num_slots = num_slots
        self._slot_of = {}  # track_id -> slot index
        self._free = list(range(num_slots))

    def assign(self, track_ids_present) -> dict:
        present = set(track_ids_present)
        gone = [tid for tid in self._slot_of if tid not in present]
        for tid in gone:
            self._free.append(self._slot_of.pop(tid))
        self._free.sort()
        for tid in present:
            if tid not in self._slot_of and self._free:
                self._slot_of[tid] = self._free.pop(0)
        return self._slot_of


def draw_grid(canvas, x0, y0, slot_of, entries_by_id, dim_ids=frozenset()):
    """entries_by_id: {track_id: (label, color)}. Draws each present
    track_id's label at its assigned fixed slot; empty slots get a faint
    dot so the grid's shape stays visible even when mostly empty."""
    slot_to_id = {s: tid for tid, s in slot_of.items()}
    for slot in range(GRID_COLS * GRID_ROWS):
        row, col = divmod(slot, GRID_COLS)
        cx = x0 + col * CELL_W
        cy = y0 + row * CELL_H
        tid = slot_to_id.get(slot)
        if tid is None or tid not in entries_by_id:
            cv2.circle(canvas, (cx + 3, cy - 4), 1, DARK_GRAY, -1)
            continue
        label, color = entries_by_id[tid]
        if tid in dim_ids:
            color = tuple(c // 2 for c in color)
        cv2.putText(canvas, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)


class PanelWriter:
    """Small helper so the panel-building code below reads as a flat list
    of lines instead of manually tracking a running y-coordinate."""

    def __init__(self, canvas, x=14, y=28, line_height=26):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.line_height = line_height

    def line(self, text, color=WHITE, scale=0.55, thickness=1):
        cv2.putText(self.canvas, text, (self.x, self.y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)
        self.y += self.line_height

    def gap(self, px=10):
        self.y += px

    def grid(self, slot_of, entries_by_id, dim_ids=frozenset()):
        draw_grid(self.canvas, self.x + 4, self.y, slot_of, entries_by_id, dim_ids)
        self.y += GRID_ROWS * CELL_H


def build_panel(slot_results, elixir, live_units, tracked_units, slot_of, timing_text, show_details):
    height = 760
    canvas = np.full((height, PANEL_WIDTH, 3), PANEL_BG, dtype=np.uint8)
    w = PanelWriter(canvas)

    w.line("DEBUG PANEL", WHITE, scale=0.7, thickness=2)
    mode = "detailed" if show_details else "simple"
    w.line(timing_text, GRAY, scale=0.42)
    w.line(f"gameplay window: {mode}  (press 'd' to toggle)", GRAY, scale=0.42)
    w.gap(14)

    w.line("HAND", GRAY, scale=0.45)
    for i, (name, score, color) in enumerate(slot_results, start=1):
        shown = name if name is not None else "no templates"
        w.line(f"  Slot {i}: {shown:14s} {score:.2f}", color)
    w.gap(14)

    w.line("ELIXIR", GRAY, scale=0.45)
    number_color = color_for(str(elixir["value"]), elixir["value_confidence"], ELIXIR_CONFIDENCE_THRESHOLD)
    w.line(f"  Number:    {elixir['value']}  (conf {elixir['value_confidence']:.2f})", number_color)
    w.line(f"  Bar est:   {elixir['bar_estimate']:.2f} / 10", (255, 200, 0))
    w.line(f"  Fraction:  +{elixir['fraction']:.2f} to next", (255, 200, 0))
    w.gap(14)

    live_entries = {u["track_id"]: (compact_tag(u["name"], u["track_id"]),
                                     ALLY_COLOR if u["side"] == "ally" else ENEMY_COLOR)
                     for u in live_units}
    tracked_entries = {u["track_id"]: (compact_tag(u["name"], u["track_id"]),
                                        ALLY_COLOR if u["side"] == "ally" else ENEMY_COLOR)
                        for u in tracked_units}
    dim_ids = {u["track_id"] for u in tracked_units if not u["visible"]}

    w.line(f"UNITS - LIVE READ ({len(live_units)})  -- raw, will flicker", GRAY, scale=0.42)
    w.grid(slot_of, live_entries)
    w.gap(10)

    w.line(f"UNITS - TRACKED ({len(tracked_units)})  -- dim = presumed, not confirmed", GRAY, scale=0.42)
    w.grid(slot_of, tracked_entries, dim_ids)

    return canvas


def open_log(log_arg, no_log):
    if no_log:
        return None
    path = log_arg or os.path.join("logs", f"debug_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to {path} — hand this file to Claude to analyze tracking patterns.\n")
    # buffering=1: line-buffered, flushes after every frame's line rather than
    # sitting in memory until close() — otherwise a crash or hard kill (not a
    # clean 'q'/Ctrl+C) loses the entire log, which defeats the point of a
    # log that's supposed to help diagnose exactly those unexpected moments.
    return open(path, "w", encoding="utf-8", buffering=1)


def write_log(log_file, frame_number, session_start, slot_results, elixir, live_units, tracked_units):
    if log_file is None:
        return
    record = {
        "frame": frame_number,
        "t": round(time.time() - session_start, 3),
        "hand": [{"slot": i, "name": n, "score": round(s, 3)}
                 for i, (n, s, _) in enumerate(slot_results, start=1)],
        "elixir": {"value": elixir["value"], "conf": round(elixir["value_confidence"], 3),
                   "bar": round(elixir["bar_estimate"], 3)},
        "live": [{"id": u["track_id"], "name": u["name"], "side": u["side"], "conf": round(u["confidence"], 3),
                  "bbox": u["bbox"]}
                 for u in live_units],
        "tracked": [{"id": u["track_id"], "name": u["name"], "side": u["side"],
                     "visible": u["visible"], "missed": u["missed_frames"], "age": u["age"], "bbox": u["bbox"]}
                    for u in tracked_units],
    }
    log_file.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=0.03,
                         help="Target seconds between refreshes — only pads the frame if "
                              "processing finished early, never adds delay on top of it.")
    parser.add_argument("--detailed", action="store_true",
                         help="Start the gameplay window in detailed mode instead of simple "
                              "(can also toggle live with the 'd' key)")
    parser.add_argument("--log", type=str, default=None,
                         help="Path to write a JSONL debug log to (one line per frame). "
                              "Default: logs/debug_live_<timestamp>.jsonl")
    parser.add_argument("--no-log", action="store_true", help="Disable logging entirely")
    args = parser.parse_args()
    show_details = args.detailed

    print("Connecting to LDPlayer...")
    controller = EmulatorController()
    capture = ScreenCapture(controller)
    card_templates = load_card_templates()
    elixir_templates = load_elixir_templates()
    unit_detector = UnitDetector()
    unit_tracker = UnitTracker()
    slot_grid = SlotGrid()

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
    print("Opening live view — press 'q' in either window (or Ctrl+C here) to stop, "
          "'d' to toggle simple/detailed.\n")

    log_file = open_log(args.log, args.no_log)
    frame_number = 0
    session_start = time.time()

    cv2.namedWindow(GAMEPLAY_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(PANEL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PANEL_WINDOW, PANEL_WIDTH, 760)
    # Best-effort default layout: gameplay window at the OS default position,
    # panel placed just to its right. Drag either window afterward if this
    # doesn't match your monitor setup — it's just a starting point.
    cv2.moveWindow(PANEL_WINDOW, 950, 0)

    try:
        while True:
            frame_start = time.time()
            frame_number += 1

            t0 = time.time()
            screenshot = capture.screenshot()
            capture_ms = (time.time() - t0) * 1000
            display = screenshot.copy()

            t0 = time.time()
            slot_results = []  # (name, score, color) per hand slot, for the panel
            for (x1, y1, x2, y2) in hand_slot_regions():
                crop = screenshot[y1:y2, x1:x2]
                name, score = card_best_match(crop, card_templates)
                if name is None or score < CARD_CONFIDENCE_THRESHOLD:
                    box_color = color_for("UNKNOWN", score if score is not None else 0.0, CARD_CONFIDENCE_THRESHOLD)
                    shown_name = name if name is not None else None
                else:
                    box_color = color_for(name, score, CARD_CONFIDENCE_THRESHOLD)
                    shown_name = name
                slot_results.append((shown_name if shown_name else "UNKNOWN", score or 0.0, box_color))
                label = f"{shown_name or 'UNKNOWN'} {score:.2f}" if show_details else None
                draw_box(display, (x1, y1, x2, y2), box_color, label)

            elixir = read_elixir(screenshot, elixir_templates)
            match_ms = (time.time() - t0) * 1000

            ex1, ey1, ex2, ey2 = NUMBER_REGION
            number_color = color_for(str(elixir["value"]), elixir["value_confidence"], ELIXIR_CONFIDENCE_THRESHOLD)
            number_label = f"{elixir['value']} {elixir['value_confidence']:.2f}" if show_details else None
            draw_box(display, (ex1, ey1, ex2, ey2), number_color, number_label)
            bar_label = f"{elixir['bar_estimate']:.2f}" if show_details else None
            draw_box(display, (int(BAR_LEFT_X), BAR_Y_TOP, int(BAR_RIGHT_X), BAR_Y_BOTTOM), (255, 200, 0), bar_label)

            t0 = time.time()
            live_units = unit_tracker.update(unit_detector.detect(screenshot))
            tracked_units = unit_tracker.current_state()
            slot_of = slot_grid.assign(u["track_id"] for u in tracked_units)
            units_ms = (time.time() - t0) * 1000
            # Gameplay window draws the raw live read (what's actually visible
            # in this exact frame) — the panel is where the smoothed/tracked
            # view lives, since drawing stale/presumed boxes on top of
            # gameplay risks implying false confidence about where an
            # occluded unit currently is.
            for u in live_units:
                color = ALLY_COLOR if u["side"] == "ally" else ENEMY_COLOR
                unit_label = compact_tag(u["name"], u["track_id"]) if show_details else None
                draw_box(display, u["bbox"], color, unit_label)

            total_ms = (time.time() - frame_start) * 1000
            fps = 1000 / total_ms if total_ms > 0 else 0.0
            timing_text = f"FPS {fps:.1f} | cap {capture_ms:.1f}ms | cards {match_ms:.1f}ms | units {units_ms:.1f}ms"

            panel = build_panel(slot_results, elixir, live_units, tracked_units, slot_of, timing_text, show_details)
            write_log(log_file, frame_number, session_start, slot_results, elixir, live_units, tracked_units)

            cv2.imshow(GAMEPLAY_WINDOW, display)
            cv2.imshow(PANEL_WINDOW, panel)

            # Only pad up to the requested interval if processing finished
            # early — waitKey still pumps window events (drag/resize) and
            # checks for 'q'/'d' either way.
            elapsed = time.time() - frame_start
            remaining_ms = max(1, int((args.interval - elapsed) * 1000))
            key = cv2.waitKey(remaining_ms) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_details = not show_details
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if log_file:
            log_file.close()
            print(f"Log closed ({frame_number} frame(s) written).")
        print("Stopped.")


if __name__ == "__main__":
    main()
