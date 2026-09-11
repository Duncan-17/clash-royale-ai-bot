"""Shared overlay-drawing primitives for the live debug views — the rounded
object-ID boxes, the translucent label plates, and the alpha-blended rect
they're built on. Extracted from perception/debug_live.py (2026-08-30) so the
TV Royale hand debug view (perception/tv_royale_debug.py) reuses the exact
same look Duncan settled on there rather than a drifting copy.

Kept dependency-light on purpose (cv2 + numpy only) so importing it doesn't
drag in a detector/model the way importing debug_live would.
"""

import cv2
import numpy as np

BOX_CORNER_RADIUS = 9  # rounded-corner radius; clamped to the box's own size below
BOX_THICKNESS = 2


def _alpha_rect(display, pt1, pt2, color, alpha):
    """Blends a filled rect into `display` at `alpha` opacity, clipped to the
    frame. Translucency (rather than solid fills) is most of what separates a
    readable overlay from a "low poly" look -- a label can sit over gameplay
    without punching an opaque hole in it."""
    h, w = display.shape[:2]
    x1, y1 = max(0, pt1[0]), max(0, pt1[1])
    x2, y2 = min(w, pt2[0]), min(h, pt2[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi = display[y1:y2, x1:x2]
    overlay = np.full(roi.shape, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def draw_label(display, text, x1, y1, y2, color, scale=0.5, thickness=1):
    """Draws `text` on a translucent dark plate with a thin accent bar in
    `color`, positioned just above (x1, y1) with enough clearance that neither
    the text nor its plate overlaps the box it's labeling -- falls back to just
    below (x1, y2) when there's no room above. A thin border in `color` keeps
    the status color visible at a glance even though the text itself sits on
    black rather than on the (often busy, similarly-colored) footage."""
    font = cv2.FONT_HERSHEY_DUPLEX  # softer/rounder strokes than HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 7, 5
    accent_w = 3  # the colored bar down the plate's left edge
    plate_h = text_h + baseline + pad_y * 2
    if y1 - plate_h - 4 > 0:
        plate_y2 = y1 - 4
        plate_y1 = plate_y2 - plate_h
    else:
        plate_y1 = y2 + 4
        plate_y2 = plate_y1 + plate_h
    plate_w = text_w + pad_x * 2 + accent_w
    # Clamp horizontally so a label near the frame edge stays fully readable
    # instead of running off-screen.
    frame_w = display.shape[1]
    plate_x1 = max(0, min(x1, frame_w - plate_w))
    plate_x2 = plate_x1 + plate_w

    _alpha_rect(display, (plate_x1, plate_y1), (plate_x2, plate_y2), (20, 20, 20), 0.72)
    _alpha_rect(display, (plate_x1, plate_y1), (plate_x1 + accent_w, plate_y2), color, 0.95)
    cv2.putText(display, text, (plate_x1 + accent_w + pad_x, plate_y2 - pad_y - baseline // 2),
                font, scale, (245, 245, 245), thickness, cv2.LINE_AA)


def draw_box(display, box, color, label=None, radius=BOX_CORNER_RADIUS,
             thickness=BOX_THICKNESS):
    """A fully-closed, anti-aliased rounded rectangle. cv2 has no rounded-rect
    primitive, so this is 4 straight edges plus 4 quarter-circle ellipse arcs.
    Radius is clamped to half the shorter side so small boxes degrade to a
    normal rounded shape instead of the arcs overlapping and rendering inside
    out."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return
    r = max(0, min(radius, w // 2, h // 2))
    t, aa = thickness, cv2.LINE_AA
    cv2.line(display, (x1 + r, y1), (x2 - r, y1), color, t, aa)
    cv2.line(display, (x1 + r, y2), (x2 - r, y2), color, t, aa)
    cv2.line(display, (x1, y1 + r), (x1, y2 - r), color, t, aa)
    cv2.line(display, (x2, y1 + r), (x2, y2 - r), color, t, aa)
    if r > 0:
        for (cx, cy, start) in ((x1 + r, y1 + r, 180), (x2 - r, y1 + r, 270),
                                (x2 - r, y2 - r, 0), (x1 + r, y2 - r, 90)):
            cv2.ellipse(display, (cx, cy), (r, r), 0, start, start + 90, color, t, aa)
    if label:
        draw_label(display, label, x1, y1, y2, color)
