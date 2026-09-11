"""A THIN tracker for the good v3 detector -- the "reduce" plan's replacement
for the heavy perception/unit_tracker.py.

unit_tracker.py grew large (class accumulator, velocity reacquisition, zone
classifier, side debounce, spawn-icon seeding) to survive the OLD weak detector.
The v3 YOLO is strong and reads side from bars natively, so almost all of that is
unnecessary. This keeps only what a good detector still benefits from:

  1. STABLE IDS -- greedy IoU match frame-to-frame.
  2. CLASS DEBOUNCE -- a track reports a locked class that only changes after a
     new class persists CLASS_LOCK_HITS frames. This is the fix for the observed
     goblin<->spear-goblin (and archer<->musketeer) frame-to-frame FLIP-FLOP:
     the detector wobbling on near-identical classes no longer flickers the label.
  3. STICKY SIDE -- ally/enemy comes from the detected bar's colour, but only
     appears on frames where a bar is visible (a damaged unit). Once a track has
     a side it KEEPS it (an undamaged unit shows no bar, so absence isn't a
     change), flipping only if the opposite side persists CLASS_LOCK_HITS frames.

Deliberately NOT here (unlike unit_tracker): velocity/Kalman reacquisition,
zone-based side inference, spawn-icon seeding, class hallucination accumulators.
Add back only if a real need shows up against the new detector.
"""

import time
from collections import deque

IOU_MATCH = 0.2        # min IoU to bind a detection to an existing track
MAX_MISSED = 8         # drop a track after this many frames unseen (~0.3s at 30fps)
CLASS_LOCK_HITS = 3    # consecutive frames a NEW class/side must persist to take over


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


class _Track:
    __slots__ = ("id", "bbox", "conf", "cls", "side", "hp", "level", "has_badge",
                 "age", "missed", "logged_play", "_pend_cls", "_pend_cls_n",
                 "_pend_side", "_pend_side_n", "_pend_lvl", "_pend_lvl_n")

    def __init__(self, tid, det):
        self.id = tid
        self.bbox = det["bbox"]
        self.conf = det["confidence"]
        self.cls = det["name"]                 # locked (reported) class
        self.side = det.get("side", "unknown")
        self.hp = det.get("hp_fraction")       # latest bar-fill read (None if no bar)
        self.level = det.get("level")          # sticky: a unit's level never changes
        self.has_badge = det.get("has_badge", False)  # per-frame: is a badge visible now
        self.age = 1
        self.missed = 0
        self.logged_play = False               # enemy-play logged once per track
        self._pend_cls, self._pend_cls_n = det["name"], 1
        self._pend_side, self._pend_side_n = self.side, 1
        self._pend_lvl, self._pend_lvl_n = self.level, 1

    def update(self, det):
        self.bbox = det["bbox"]
        self.conf = det["confidence"]
        if det.get("hp_fraction") is not None:
            self.hp = det["hp_fraction"]       # keep last-known when no bar this frame
        self.has_badge = det.get("has_badge", False)
        self.age += 1
        self.missed = 0
        self._debounce_class(det["name"])
        self._debounce_side(det.get("side", "unknown"), det.get("side_source"))
        self._debounce_level(det.get("level"))

    def _debounce_class(self, name):
        if name == self.cls:
            self._pend_cls, self._pend_cls_n = name, 0
            return
        if name == self._pend_cls:
            self._pend_cls_n += 1
        else:
            self._pend_cls, self._pend_cls_n = name, 1
        if self._pend_cls_n >= CLASS_LOCK_HITS:
            self.cls = name
            self._pend_cls_n = 0

    def _debounce_side(self, side, source=None):
        # 'unknown' never overrides a known side (no bar this frame is normal).
        if side == "unknown" or side == self.side:
            return
        if self.side == "unknown":  # first real reading SEEDS it (position OK here)
            self.side = side
            self._pend_side, self._pend_side_n = side, 1
            return
        # An established track only FLIPS on an authoritative BAR-COLOUR reading.
        # A crude river-POSITION guess must not flip it -- units cross the map, and
        # position feeds a sustained wrong side every frame (Duncan 2026-09-10: an
        # ally Giant read enemy after crossing the bridge). Position seeds, never
        # overrides.
        if source == "position":
            return
        if side == self._pend_side:
            self._pend_side_n += 1
        else:
            self._pend_side, self._pend_side_n = side, 1
        if self._pend_side_n >= CLASS_LOCK_HITS:  # a sustained opposite bar reading flips it
            self.side = side
            self._pend_side_n = 0

    def _debounce_level(self, level):
        # A unit's level is constant, so this is mostly "first real reading wins
        # and sticks" -- absence (None: no badge this frame, normal for an
        # undamaged ally) never clears it. A DIFFERENT non-None value only takes
        # over if it persists CLASS_LOCK_HITS frames, so a one-frame misread of a
        # near-lookalike digit can't flip a settled level.
        if level is None or level == self.level:
            return
        if self.level is None:  # first real reading takes immediately
            self.level = level
            self._pend_lvl, self._pend_lvl_n = level, 1
            return
        if level == self._pend_lvl:
            self._pend_lvl_n += 1
        else:
            self._pend_lvl, self._pend_lvl_n = level, 1
        if self._pend_lvl_n >= CLASS_LOCK_HITS:
            self.level = level
            self._pend_lvl_n = 0

    def as_dict(self):
        return {"id": self.id, "name": self.cls, "side": self.side, "hp": self.hp,
                "level": self.level, "has_badge": self.has_badge, "bbox": self.bbox,
                "confidence": self.conf, "age": self.age}


class ThinTracker:
    def __init__(self, iou_match=IOU_MATCH, max_missed=MAX_MISSED):
        self.iou_match = iou_match
        self.max_missed = max_missed
        self._tracks = []
        self._next_id = 0
        # SIMPLIFIED enemy plays: a (time, card) each time a track first becomes
        # enemy -- a lightweight stand-in for the real cycle-based play detector
        # (a new enemy unit ~ a play). Once per track, so it doesn't re-log.
        self.plays = deque(maxlen=12)

    def update(self, detections):
        """detections: unit dicts from yolo_detector (name, side, bbox, confidence).
        Returns the currently-visible tracks as dicts (stable id/name/side)."""
        # Greedy IoU match, best pairs first, one-to-one.
        pairs = []
        for di, d in enumerate(detections):
            for ti, t in enumerate(self._tracks):
                iou = _iou(d["bbox"], t.bbox)
                if iou >= self.iou_match:
                    pairs.append((iou, di, ti))
        pairs.sort(reverse=True)
        used_d, used_t = set(), set()
        for _iouval, di, ti in pairs:
            if di in used_d or ti in used_t:
                continue
            self._tracks[ti].update(detections[di])
            used_d.add(di)
            used_t.add(ti)

        for ti, t in enumerate(self._tracks):
            if ti not in used_t:
                t.missed += 1
        for di, d in enumerate(detections):
            if di not in used_d:
                self._tracks.append(_Track(self._next_id, d))
                self._next_id += 1

        self._tracks = [t for t in self._tracks if t.missed <= self.max_missed]

        # Log a simplified enemy play when a track is first confirmed enemy.
        now = time.time()
        for t in self._tracks:
            if t.side == "enemy" and not t.logged_play:
                t.logged_play = True
                self.plays.append((now, t.cls))

        return [t.as_dict() for t in self._tracks if t.missed == 0]
