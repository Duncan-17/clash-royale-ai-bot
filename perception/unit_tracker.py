"""
Assigns persistent track IDs to unit_detector.py's raw per-frame
detections, so the same physical unit is recognized as the same unit
across frames instead of looking like it keeps dying and respawning.

Why this is needed: unit_detector.py is completely stateless — every
frame it reports "here's what I currently see," with no memory of
previous frames. Detection confidence naturally dips for a frame or two
during animation, rotation, or brief occlusion behind another unit, which
makes a real, unmoving unit vanish from one frame's detection list and
reappear in the next. Confirmed live (2026-08-15, Duncan): without
tracking, this looks exactly like units repeatedly being placed and dying
— it's a raw-detection flicker problem, not a placement/death problem, so
the fix belongs here, not in unit_detector.py or in the (not yet built)
cycle-tracking logic that will eventually consume this.

Approach: simple greedy nearest-neighbor matching between frames,
constrained to the same card name and side (an ally Knight can never
become an enemy Giant), with a grace period (MAX_MISSED_FRAMES) before a
track is dropped — a track survives a few consecutive missed frames
rather than being deleted the instant one frame doesn't detect it. This
is deliberately simpler than a full tracker like SORT/DeepSORT (no
Kalman filtering, no learned re-identification features): units move
slowly relative to this project's frame rate (~12fps with unit detection
on — see debug_live.py's performance note), so plain distance-based
matching between consecutive frames is enough. Known limitation: several
identical units of the same card standing close together (e.g. a
Skeleton Army) can have their track IDs swapped between frames, since
nothing here distinguishes between two visually-identical units of the
same type — accepted for now, since nothing downstream needs per-
individual continuity yet, only "how many of X are on the board" and
"is this a new appearance or the same one as before," both of which
still work correctly even if two interchangeable units swap IDs.

Confirmation delay (MIN_HITS_TO_CONFIRM, added 2026-08-15): a brand-new
track isn't reported anywhere (not in update()'s return, not in
current_state()) until it's been matched MIN_HITS_TO_CONFIRM times.
Motivated by log analysis of a full match (see Project_Status): 44% of
all track IDs ever created (158 of 356) were matched exactly ONCE, ever
— pure single-frame noise blips, never a real unit. Requiring 2 hits
before reporting eliminates all of those outright, and — importantly —
doesn't touch how an ESTABLISHED track behaves at all (missed_frames/
grace-period logic is unchanged); it only delays a brand-new track's
first appearance by one extra matched frame. Real units are still
reported quickly (2 frames at ~12fps is under 200ms); one-off noise
never surfaces at all. This is a real, measured, but partial fix — see
Project_Status for what's known to remain and why.

Usage:
    tracker = UnitTracker()
    tracked = tracker.update(unit_detector.detect(screenshot))
    # tracked is the same list of dicts unit_detector.detect() returns,
    # each with an added "track_id" (stable across calls) and "age"
    # (frames since first seen) field. This is the raw, instant-by-instant
    # read — it will still flicker frame to frame, correctly, since it's
    # reporting exactly what was detected THIS frame, nothing smoothed.
    #
    # For the smoothed view (bridges brief occlusion instead of reporting
    # it as gone), call current_state() instead/as well — see its
    # docstring for what "visible" does and doesn't mean.
"""

import math

MAX_MISSED_FRAMES = 8  # grace period before a track is dropped — a few hundred ms at ~12fps,
# long enough to bridge animation/rotation flicker, short enough that a genuinely-gone unit
# doesn't linger falsely for long
MATCH_DISTANCE_PX = 90  # max center-to-center distance to consider two boxes the same unit
# across consecutive frames — generous enough for fast units' per-frame movement plus normal
# detection-box jitter, tight enough to avoid matching two distinct nearby units
MIN_HITS_TO_CONFIRM = 2  # a track must be matched this many times before it's reported anywhere —
# see module docstring, kills single-frame noise blips (44% of all tracks in a real match) at
# the cost of a one-extra-frame delay (~<200ms at ~12fps) before a genuine new unit is reported


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class UnitTracker:
    def __init__(self, max_missed_frames=MAX_MISSED_FRAMES, match_distance_px=MATCH_DISTANCE_PX,
                 min_hits_to_confirm=MIN_HITS_TO_CONFIRM):
        self.max_missed_frames = max_missed_frames
        self.match_distance_px = match_distance_px
        self.min_hits_to_confirm = min_hits_to_confirm
        self._tracks = {}  # track_id -> {name, side, bbox, center, missed_frames, age, hits}
        self._next_id = 1

    def _is_confirmed(self, track_id) -> bool:
        return self._tracks[track_id]["hits"] >= self.min_hits_to_confirm

    def update(self, detections: list) -> list:
        """detections: unit_detector.detect()'s output for one frame.
        Returns the SUBSET of that list belonging to CONFIRMED tracks
        (matched at least min_hits_to_confirm times — see module
        docstring), each with "track_id" and "age" added. A brand-new
        detection is tracked internally starting this frame but does NOT
        appear in the returned list until it's confirmed."""
        candidates = []  # (distance, detection_index, track_id)
        for i, det in enumerate(detections):
            det_center = _center(det["bbox"])
            for track_id, track in self._tracks.items():
                if track["name"] != det["name"] or track["side"] != det["side"]:
                    continue
                dist = _distance(det_center, track["center"])
                if dist <= self.match_distance_px:
                    candidates.append((dist, i, track_id))

        candidates.sort(key=lambda c: c[0])
        matched_detections = set()
        assignment = {}  # detection_index -> track_id
        active_this_frame = set()  # every track_id matched OR newly created this frame
        for dist, i, track_id in candidates:
            if i in matched_detections or track_id in active_this_frame:
                continue
            assignment[i] = track_id
            matched_detections.add(i)
            active_this_frame.add(track_id)

        confirmed_detections = []
        for i, det in enumerate(detections):
            det_center = _center(det["bbox"])
            if i in assignment:
                track_id = assignment[i]
                track = self._tracks[track_id]
                track["bbox"] = det["bbox"]
                track["center"] = det_center
                track["missed_frames"] = 0
                track["age"] += 1
                track["hits"] += 1
            else:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = {
                    "name": det["name"], "side": det["side"],
                    "bbox": det["bbox"], "center": det_center,
                    "missed_frames": 0, "age": 1, "hits": 1,
                }
                active_this_frame.add(track_id)
            det["track_id"] = track_id
            det["age"] = self._tracks[track_id]["age"]
            if self._is_confirmed(track_id):
                confirmed_detections.append(det)

        for track_id in list(self._tracks.keys()):
            if track_id not in active_this_frame:
                self._tracks[track_id]["missed_frames"] += 1
                # Only confirmed tracks get a grace period — an unconfirmed
                # (single-hit) track that misses even one frame is dropped
                # immediately rather than lingering unconfirmed indefinitely.
                limit = self.max_missed_frames if self._is_confirmed(track_id) else 0
                if self._tracks[track_id]["missed_frames"] > limit:
                    del self._tracks[track_id]

        return confirmed_detections

    def current_state(self) -> list:
        """Returns every currently-tracked unit, including ones inside
        their grace period (not detected in the most recent update() call,
        but not yet dropped) — the smoothed/de-flickered view.

        Each entry has "visible": True if matched in the most recent
        update() call, False if currently coasting on the grace period.
        Important: False does NOT mean "confirmed still alive but hidden"
        — it means "no detection this frame, and we haven't given up yet."
        There is no signal here (or anywhere in this project yet) that can
        actually distinguish real occlusion from the unit having died from
        the model simply missing an obvious frame — all three look
        identical from a single frame's detection output. Telling those
        apart for real would need something this project doesn't have
        yet (recognizing a death animation specifically, or cross-
        referencing HP dropping to zero via the planned-but-unbuilt HP
        reader). Report "visible=False" to the user as "presumed / last
        confirmed N frames ago," never as a claim about the unit's actual
        fate.
        """
        result = []
        for track_id, track in self._tracks.items():
            if not self._is_confirmed(track_id):
                continue
            result.append({
                "track_id": track_id,
                "name": track["name"],
                "side": track["side"],
                "bbox": track["bbox"],  # last known position — may be stale if the unit moved while unconfirmed
                "age": track["age"],
                "visible": track["missed_frames"] == 0,
                "missed_frames": track["missed_frames"],
            })
        return result
