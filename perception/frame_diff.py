"""
Fallback board localizer for the TV Royale auto-labeling pipeline — see
data/auto_labeler.py for how this fits into the full pipeline, and
clash-royale-ai-pipeline-explained.md section 5 for the plain-English
version. Only used when the PRIMARY localization path (reusing
perception/unit_tracker.py's own new-track detection) finds nothing — i.e.
specifically for detection misses, not the everyday mechanism. Deliberately
narrow, not a general motion tracker: pure `cv2.absdiff`-based frame
differencing between a "before" and "after" arena crop.

Important: this module has NO way to tell "a new unit spawned" apart from
"an existing unit moved" on its own — that distinction has to come from
WHICH frames the caller passes in. Call this with frames that bracket a
single known play event (a crop from just before the play, a crop from
shortly after), not consecutive video frames, or it will just find normal
unit movement and report it as a "new" blob.

No live TV Royale access needed to build or test this — verified here with
synthetic images (see perception/test_frame_diff.py), not real footage yet
(no TV Royale capture pipeline exists to source real footage from at the
time this was written — see config/tv_royale_regions.py's absence, still
pending a live calibration session).
"""

import cv2
import numpy as np

DIFF_THRESHOLD = 30  # per-pixel grayscale intensity difference (0-255)
# treated as "changed". First-pass value, not yet tuned against real
# footage — revisit once this actually runs against TV Royale captures.

MIN_BLOB_AREA = 200  # pixels — filters sub-pixel noise/compression
# artifacts out of being reported as a "new unit" candidate. Also a
# first-pass guess, same caveat as DIFF_THRESHOLD.

MORPH_KERNEL = np.ones((5, 5), np.uint8)  # closes small gaps in a real
# unit's diff mask (antialiasing, partial-transparency edges) so it reads
# as one blob instead of several fragments.


def find_new_blobs(before_bgr: np.ndarray, after_bgr: np.ndarray) -> list:
    """Returns a list of (x1, y1, x2, y2) bounding boxes for regions
    present in after_bgr but not in before_bgr. Both images must be the
    same size — true by construction for this project's fixed-camera
    capture (no alignment/registration step needed, unlike a handheld or
    panning camera source)."""
    before_gray = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(before_gray, after_gray)
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        if cv2.contourArea(c) < MIN_BLOB_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, x + w, y + h))
    return boxes


def closest_blob_to(boxes: list, point) -> tuple:
    """Returns whichever box's center is nearest to point=(x, y), or None
    if boxes is empty. For picking the single most plausible "new unit"
    candidate when an expected position is available (e.g. near a played
    side's own bridge), rather than blindly taking the largest blob."""
    if not boxes:
        return None

    def center(b):
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    return min(boxes, key=lambda b: (center(b)[0] - point[0]) ** 2 + (center(b)[1] - point[1]) ** 2)
