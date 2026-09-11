"""The live/eval view for OUR detector -- the honest picture of what the detector
sees, plus the THIN stack we actually keep (no old compensation heuristics).

Default: raw detections only (units + bars/levels/clocks/towers), coloured by side
(ally=blue / enemy=red from bar colour + a river-position seed, unknown=amber),
components in cyan -- the clean way to judge a model.

--track layers on the RELIABLE, cheap thin stack: perception/thin_tracker.py
(stable IDs, class debounce, sticky side/level, HP) + tower alive/HP from the
detector. The game panel shows units by side (+level+HP) and tower health.

Deliberately OFF by default (opt-in), because on noisy detections they either lie
or cost too much (Duncan 2026-09-10):
    --plays        opponent-play detection (game_knowledge/play_tracker.py). It
                   manufactures plays that didn't happen when detections flicker.
                   The own-play ground-truth framework is unaffected (needs the bot).
    --read-levels  level-DIGIT reading (numbers vs "Lv?"): template-matches every
                   badge every frame (~15ms busy), and levels are on standby.

Usage:
    python -m perception.detector_view                 # LIVE, raw detector output
    python -m perception.detector_view --track         # + thin tracker + tower/game panel
    python -m perception.detector_view --track --profile   # per-stage ms every 30 frames
    python -m perception.detector_view --static        # one frame from test_screenshot.png
    python -m perception.detector_view --model assets/models/cr_yolo_v3/best.pt  # a specific model
"""

import argparse
import time

import cv2
import numpy as np

from perception.overlay import draw_box, _alpha_rect

ALLY = (255, 150, 60)       # blue
ENEMY = (60, 60, 255)       # red
UNKNOWN = (96, 186, 250)    # amber
COMPONENT = (210, 210, 80)  # cyan-ish for bars/levels/clocks/tower UI
LEVEL_TAG = (60, 215, 255)  # gold (BGR) -- a deliberately distinct colour so the
# level reads at a glance and isn't mistaken for a side/component colour


def _draw_level_tag(disp, bbox, level, has_badge=False):
    """A separate high-contrast plate pinned to the box BOTTOM-left, so it never
    collides with the name plate (drawn above the box) or the in-game health
    bar/level badge (rendered above the unit) -- Duncan's live note was that the
    level was unreadable jammed into the name label up top. Shows 'LvN' when the
    digit is read, 'Lv?' when a badge is detected but that digit isn't collected
    yet, and nothing when no badge is present."""
    if isinstance(level, (int, float)):
        text = f"Lv{int(level)}"
    elif has_badge:
        text = "Lv?"
    else:
        return
    x1, y1, x2, y2 = bbox
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), base = cv2.getTextSize(text, font, 0.5, 1)
    pad, accent = 5, 3
    px1 = max(0, x1)
    py1 = y2 + 4
    px2 = px1 + tw + pad * 2 + accent
    py2 = py1 + th + base + pad * 2
    _alpha_rect(disp, (px1, py1), (px2, py2), (20, 20, 20), 0.72)
    _alpha_rect(disp, (px1, py1), (px1 + accent, py2), LEVEL_TAG, 0.95)
    cv2.putText(disp, text, (px1 + accent + pad, py2 - pad - base // 2),
                font, 0.5, LEVEL_TAG, 1, cv2.LINE_AA)

COMPONENT_RAW = {
    "bar", "bar-level", "clock", "tower-bar", "king-tower-bar", "elixir",
    "king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
    "dagger-duchess-tower-bar", "skeleton-king-bar", "emote", "text", "selected",
}

# The static towers + their bars -- hidden with --no-ui so you can judge TROOPS
# without the tower/bar clutter (we track towers via tower_reader anyway).
TOWER_CLASSES = {
    "king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
    "tower-bar", "king-tower-bar", "dagger-duchess-tower-bar",
}


def filter_ui(dets):
    return [d for d in dets if d.get("raw_name", d["name"]) not in TOWER_CLASSES]


def track(dets, tracker):
    """Stabilize UNIT detections through the ThinTracker (class debounce, sticky
    side, IDs); components pass through per-frame. tracker=None -> passthrough."""
    if tracker is None:
        return dets
    units = [d for d in dets if not _is_component(d)]
    comps = [d for d in dets if _is_component(d)]
    return tracker.update(units) + comps


def make_detector(conf, model_path=None, read_levels=False):
    # yolo-only since Task 2 (2026-09-10): the katacr / units.onnx comparison
    # detectors were retired once v4 was settled as the detector.
    from perception.yolo_detector import YoloDetector, DEFAULT_MODEL
    return YoloDetector(model_path=model_path or DEFAULT_MODEL,
                        include_non_units=True, conf=conf, read_levels=read_levels)


PANEL_W = 360
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _is_component(d):
    return d.get("raw_name", d["name"]) in COMPONENT_RAW


def _side_color(d):
    return {"ally": ALLY, "enemy": ENEMY}.get(d.get("side", "unknown"), UNKNOWN)


def annotate(frame, dets):
    disp = frame.copy()
    counts = {"ally": 0, "enemy": 0, "unknown": 0, "component": 0}
    for d in dets:
        if _is_component(d):
            color = COMPONENT
            counts["component"] += 1
            draw_box(disp, d["bbox"], color, label=f"{d['name']} {d['confidence']:.2f}")
            continue
        color = _side_color(d)
        counts[d.get("side") if d.get("side") in ("ally", "enemy") else "unknown"] += 1
        tag = f"#{d['id']} " if "id" in d else ""
        draw_box(disp, d["bbox"], color, label=f"{tag}{d['name']} {d['confidence']:.2f}")
        _draw_level_tag(disp, d["bbox"], d.get("level"), d.get("has_badge"))  # gold plate below box
    header = (f"{len(dets)} dets  ally:{counts['ally']} enemy:{counts['enemy']} "
             f"unk:{counts['unknown']} comp:{counts['component']}")
    _alpha_rect(disp, (0, 0), (disp.shape[1], 30), (20, 20, 20), 0.6)
    cv2.putText(disp, header, (10, 21), cv2.FONT_HERSHEY_DUPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return disp


def info_panel(dets, height):
    """A text list of what the detector saw this frame -- units (name, side,
    conf) then components -- so you can read exactly what's detected/missed."""
    panel = np.full((height, PANEL_W, 3), 25, np.uint8)
    units = sorted([d for d in dets if not _is_component(d)], key=lambda d: -d["confidence"])
    comps = sorted([d for d in dets if _is_component(d)], key=lambda d: -d["confidence"])

    y = 22
    cv2.putText(panel, f"UNITS ({len(units)})", (8, y), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    y += 20
    for d in units[:34]:
        txt = (f"{d['name'][:16]:16} {str(d.get('side','?'))[:4]:4} "
               f"{_lvl(d.get('level'), d.get('has_badge'))} {d['confidence']:.2f}")
        cv2.putText(panel, txt, (8, y), FONT, 0.42, _side_color(d), 1, cv2.LINE_AA)
        y += 16
    y += 8
    cv2.putText(panel, f"COMPONENTS ({len(comps)})", (8, y), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    y += 20
    tally = {}
    for d in comps:
        tally[d["name"]] = tally.get(d["name"], 0) + 1
    for name, n in sorted(tally.items(), key=lambda kv: -kv[1]):
        cv2.putText(panel, f"{name[:22]:22} x{n}", (8, y), FONT, 0.42, COMPONENT, 1, cv2.LINE_AA)
        y += 16
    return panel


def _hp(v):
    return f"{int(round(v*100)):3d}%" if isinstance(v, (int, float)) else "  -"


def _lvl(level, has_badge=False):
    # fixed 3-char column so it never jitters (RULE 1):
    #   "L11" = read;  "L ?" = badge detected but that digit not collected yet;
    #   "L -" = no badge at all (an undamaged ally shows none; enemies always do).
    if isinstance(level, (int, float)):
        return f"L{int(level):>2}"
    return "L ?" if has_badge else "L -"


# FIXED-LAYOUT panel (DEBUG_CONVENTIONS.md RULE 1): each section has a reserved y
# and a fixed max-row count. Adding/removing an item NEVER shifts another section
# -- empty rows stay blank. Row pitch 16px, header ~18px.
_SECTIONS = {  # label -> (header_y, max_rows)
    "ENEMY":    (18, 11),
    "FRIENDLY": (218, 11),
    "UNKNOWN":  (418, 8),
    "TOWERS":   (568, 6),
    "PLAYS":    (690, 9),
}
_TOWER_ORDER = ("enemy_left", "enemy_king", "enemy_right", "ally_left", "ally_king", "ally_right")


def _draw_section(panel, key, rows):
    y, max_rows = _SECTIONS[key]
    cv2.putText(panel, f"{key}", (8, y), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for i in range(max_rows):  # fixed slots -- unused ones left blank
        if i < len(rows):
            txt, col = rows[i]
            cv2.putText(panel, txt, (12, y + 20 + i * 16), FONT, 0.42, col, 1, cv2.LINE_AA)


def _urow(u):
    # fixed columns so values don't jitter horizontally (RULE 1): name | level | hp
    return f"{u['name'][:13]:13} {_lvl(u.get('level'), u.get('has_badge'))} {_hp(u.get('hp'))}"


def game_panel(units, tower_info, plays, height):
    """Fixed-layout game state: units by side (+HP), tower health, enemy plays."""
    import time
    panel = np.full((height, PANEL_W, 3), 25, np.uint8)

    for side, col, key in (("enemy", ENEMY, "ENEMY"), ("ally", ALLY, "FRIENDLY"),
                           ("unknown", UNKNOWN, "UNKNOWN")):
        us = sorted([u for u in units if u.get("side") == side], key=lambda u: u["name"])
        _draw_section(panel, key, [(_urow(u), col) for u in us])
        # header count -- fixed position, drawn over the header
        hy = _SECTIONS[key][0]
        cv2.putText(panel, f"({len(us)})", (PANEL_W - 60, hy), FONT, 0.45, col, 1, cv2.LINE_AA)

    trows = []
    for n in _TOWER_ORDER:
        info = tower_info.get(n, {})
        if not info.get("alive", True):
            txt, col = f"{n:11} gone", (90, 90, 90)
        else:
            col = COMPONENT if n.startswith("enemy") else ALLY
            txt = f"{n:11} {_hp(info.get('hp'))}"
        trows.append((txt, col))
    _draw_section(panel, "TOWERS", trows)

    now = time.time()
    _draw_section(panel, "PLAYS", [(f"{int(now - t):3d}s ago  {card[:16]}", (230, 230, 230))
                                   for t, card in list(plays)[::-1]])
    return panel


# --- tower state from the DETECTOR, not the flaky pixel reader (RULE 2) ---
from perception.tower_reader import TOWER_BAR_REGIONS
_TOWER_CENTERS = {n: ((x1 + x2) // 2, (y1 + y2) // 2) for n, (x1, y1, x2, y2) in TOWER_BAR_REGIONS.items()}
_TOWER_DET_CLASSES = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
                      "tower-bar", "king-tower-bar", "dagger-duchess-tower-bar"}
_TOWER_MATCH_R = 140      # px: a tower detection within this of a slot counts as that slot
_TOWER_GONE_FRAMES = 45   # slot absent this many frames -> destroyed (debounced, ~1.5s)


class TowerPanel:
    """alive/destroyed from whether the DETECTOR currently sees a tower near each
    slot (the pixel tower_reader false-flagged destroyed after a hit); HP from the
    pixel reader, shown only while alive."""
    def __init__(self):
        self.missed = {n: 0 for n in _TOWER_CENTERS}

    def update(self, components, tower_hp):
        seen = set()
        for d in components:
            if d.get("raw_name") in _TOWER_DET_CLASSES:
                cx = (d["bbox"][0] + d["bbox"][2]) / 2
                cy = (d["bbox"][1] + d["bbox"][3]) / 2
                n = min(_TOWER_CENTERS, key=lambda k: (cx - _TOWER_CENTERS[k][0]) ** 2
                        + (cy - _TOWER_CENTERS[k][1]) ** 2)
                if (cx - _TOWER_CENTERS[n][0]) ** 2 + (cy - _TOWER_CENTERS[n][1]) ** 2 < _TOWER_MATCH_R ** 2:
                    seen.add(n)
        out = {}
        for n in _TOWER_CENTERS:
            self.missed[n] = 0 if n in seen else self.missed[n] + 1
            alive = self.missed[n] < _TOWER_GONE_FRAMES
            out[n] = {"alive": alive, "hp": (tower_hp.get(n, {}).get("hp_fraction") if alive else None)}
        return out


# --- RULE #2 level-badge collection: queue crops for review, never auto-file ---
_COLLECT_INTERVAL = 0.4          # throttle disk work; a badge doesn't change fast
_collect_state = {"t": 0.0, "n": 0}


def _collect_levels(frame, comps):
    """Bank each detected `bar-level` crop to the review queue (deduped, capped).
    Duncan labels them with `python -m perception.review_level_queue`; nothing is
    auto-filed into the permanent library (CLAUDE.md RULE #2). Returns running
    total queued this session."""
    import time as _t
    now = _t.time()
    if now - _collect_state["t"] < _COLLECT_INTERVAL:
        return _collect_state["n"]
    _collect_state["t"] = now
    from perception.level_reader import queue_level_crop
    for d in comps:
        if d.get("raw_name") == "bar-level" and queue_level_crop(frame, d["bbox"]):
            _collect_state["n"] += 1
    return _collect_state["n"]


_prof = {"n": 0, "capture": 0.0, "detect": 0.0, "track": 0.0, "plays": 0.0, "towers": 0.0, "draw": 0.0}
_PROF_KEYS = ("capture", "detect", "track", "plays", "towers", "draw")


def _lap(on, key, t0):
    if on:
        _prof[key] += time.perf_counter() - t0
    return time.perf_counter()


def render_frame(frame, det, ui_filter, tracker, tower_tracker, tower_panel=None,
                 collect_levels=False, play_tracker=None, profile=False):
    t = time.perf_counter()
    raw = ui_filter(det.detect(frame))
    t = _lap(profile, "detect", t)
    comps = [d for d in raw if _is_component(d)]
    units = [d for d in raw if not _is_component(d)]
    banked = _collect_levels(frame, comps) if collect_levels else None
    if tracker is not None:
        tracked = tracker.update(units)
        t = _lap(profile, "track", t)
        plays = []
        if play_tracker is not None:
            play_tracker.update(tracked)  # overlays own-play side + logs opponent plays
            plays = [(e.t, e.card) for e in play_tracker.opponent_plays]
        t = _lap(profile, "plays", t)
        tower_hp = tower_tracker.update(frame) if tower_tracker is not None else {}
        tower_info = tower_panel.update(comps, tower_hp) if tower_panel is not None else {}
        t = _lap(profile, "towers", t)
        annotated = annotate(frame, tracked + comps)
        panel = game_panel(tracked, tower_info, plays, annotated.shape[0])
    else:
        annotated = annotate(frame, raw)
        panel = info_panel(raw, annotated.shape[0])
    t = _lap(profile, "draw", t)
    if banked is not None:
        cv2.putText(annotated, f"lvl-collect queued:{banked}", (10, annotated.shape[0] - 12),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (80, 220, 80), 1, cv2.LINE_AA)
    if profile:
        _prof["n"] += 1
        if _prof["n"] >= 30:
            n = _prof["n"]
            print("profile ms/frame:  " + "   ".join(
                f"{k}={_prof[k] / n * 1000:5.1f}" for k in _PROF_KEYS))
            for k in _prof:
                _prof[k] = 0.0
    return np.hstack([panel, annotated])


def _fit(display, max_h=940):
    h = display.shape[0]
    if h <= max_h:
        return display
    s = max_h / h
    return cv2.resize(display, (int(display.shape[1] * s), max_h), interpolation=cv2.INTER_AREA)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--static", action="store_true", help="Read test_screenshot.png once.")
    p.add_argument("--no-ui", action="store_true", help="Hide the static towers + tower bars.")
    p.add_argument("--model", default=None,
                   help="Path to a specific best.pt (e.g. assets/models/cr_yolo_v3/best.pt). "
                        "Default = yolo_detector.DEFAULT_MODEL (v4).")
    p.add_argument("--track", action="store_true",
                   help="Stabilize units through the ThinTracker (class debounce, sticky side, IDs) "
                        "-- shows the flip-flop fixed.")
    p.add_argument("--plays", action="store_true",
                   help="OPT-IN opponent-play detection (game_knowledge.play_tracker). OFF by "
                        "default: on noisy detections it manufactures plays that didn't happen "
                        "(Duncan 2026-09-10). The own-play framework is unaffected (needs the bot).")
    p.add_argument("--read-levels", action="store_true",
                   help="OPT-IN level-DIGIT reading (numbers instead of 'Lv?'). OFF by default: "
                        "it template-matches every badge every frame (~15ms on a busy frame) and "
                        "levels are on standby anyway. Badge detection ('Lv?') is always on.")
    p.add_argument("--collect-levels", action="store_true",
                   help="Bank each detected level badge (bar-level) crop to "
                        "assets/unlabeled_level_queue/ for review (RULE #2). Grows the "
                        "digit library so more levels read; label with review_level_queue.")
    p.add_argument("--profile", action="store_true",
                   help="Print per-stage timings (detect/track/plays/towers/draw) every 30 frames "
                        "to find where time goes.")
    args = p.parse_args()

    det = make_detector(args.conf, args.model, args.read_levels)
    ui_filter = filter_ui if args.no_ui else (lambda d: d)
    tracker = tower_tracker = tower_panel = play_tracker = None
    if args.track:
        from perception.thin_tracker import ThinTracker
        from perception.tower_reader import TowerStateTracker
        tracker = ThinTracker()
        tower_tracker = TowerStateTracker()
        tower_panel = TowerPanel()
        if args.plays:  # opt-in; off by default so noisy detections don't fake plays
            from game_knowledge.play_tracker import PlayTracker
            play_tracker = PlayTracker()
    window = f"Detector view: yolo{' +track' if args.track else ''} (q=quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    if args.static:
        frame = cv2.imread("test_screenshot.png")
        if frame is None:
            raise FileNotFoundError("test_screenshot.png not found.")
        cv2.imshow(window, _fit(render_frame(frame, det, ui_filter, tracker, tower_tracker, tower_panel, args.collect_levels, play_tracker, args.profile)))
        print("Static frame — press any key in the window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    from perception.capture import ScreenCapture
    from control.actions import EmulatorController
    print("Connecting to LDPlayer... (q in the window to quit)")
    capture = ScreenCapture(EmulatorController())
    try:
        while True:
            _t = time.perf_counter()
            frame = capture.screenshot()
            if args.profile:
                _prof["capture"] += time.perf_counter() - _t
            cv2.imshow(window, _fit(render_frame(frame, det, ui_filter, tracker, tower_tracker, tower_panel, args.collect_levels, play_tracker, args.profile)))
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        capture.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
