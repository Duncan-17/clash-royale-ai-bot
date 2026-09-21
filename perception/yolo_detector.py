"""Inference wrapper for OUR fine-tuned standard YOLOv8 detector
(assets/models/cr_yolo_vN/best.pt, default v4 -> see DEFAULT_MODEL), exposing the
same detect() interface as unit_detector.UnitDetector / katacr_detector, so it
drops into debug_live.py and detector_view.py as `--detector yolo`. Trained in
training/ (see training/README.md): v1 synthetic, v2 +bar/level/clock components,
v3 synthetic+real.

Plain Ultralytics YOLOv8 (CLASS + box only, no belonging channel). Three things
this wrapper does on top of raw predict() (all added 2026-09-08 from live tests):
  - SIDE FROM BAR COLOUR: v2+ detects the HP `bar`; we read its colour
    (blue=ally/red=enemy, perception/bar_reader.side_from_bar_color) and
    associate bars->units (apply_bar_sides). Units with no visible bar stay
    'unknown' (an undamaged troop shows no bar) -- the pipeline fills those in.
    This is what replaces side_reader.py.
  - ASPECT PAD: the model trained at KataCR's ~0.634 aspect (568x896) but the
    LDPlayer frame is 900x1600 (~0.5625); we pad the frame horizontally to the
    train aspect (TRAIN_ASPECT) so units aren't shrunk when letterboxed, then map
    boxes back.
  - ARENA-BAND FILTER: drop detections centred outside ARENA_Y_MIN..ARENA_Y_MAX
    -- the model trained on full frames and otherwise reads hand-tray UI (a card's
    elixir badge) as a `clock`.
"""

import cv2
import numpy as np

from config.red_zones import should_suppress_detection, RIVER_Y_TOP, RIVER_Y_BOTTOM

# The model trained on KataCR's 568x896 frames (aspect ~0.634). Our LDPlayer
# frame is 900x1600 (~0.5625) -- taller, so when ultralytics letterboxes it to
# imgsz the arena content ends up NARROWER/smaller than in training (~0.89x),
# which pushes small units (archers, spirits) below detection. Padding the frame
# horizontally to the training aspect restores the trained scale without
# distorting units (added 2026-09-08 to chase the archer misses).
TRAIN_ASPECT = 568 / 896

DEFAULT_MODEL = "assets/models/cr_yolo_v4/best.pt"  # v4 = v3 + LONE level-badge
# training (undamaged-enemy case); live-confirmed by Duncan reading levels well
# (2026-09-10). v3 kept at assets/models/cr_yolo_v3/best.pt as the prior default.
DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 896  # matches the training imgsz

# Arena band (fractions of full height). The model trained on full frames that
# INCLUDE the hand tray + top banner, and it mistakes hand-card elixir badges for
# clocks and other UI for units (Duncan, 2026-09-08). The playable arena is this
# band -- anything centred outside it is UI, so we drop it. Bottom kept a touch
# below unit_detector's 0.80 so the bottom king tower survives but the hand
# tray (cards start ~0.83) is excluded.
ARENA_Y_MIN = 0.05
ARENA_Y_MAX = 0.85

# Raw detector classes that are NOT troops/buildings -- the towers, HP/level bars,
# deploy clock, and other UI. The AUTHORITATIVE list (this is what detect() splits on
# with include_non_units); consumers that want "board troops only" should filter with
# THIS, not a hand-kept subset or a "'tower' in name" substring test (which also drops
# the real buildings bomb-tower / inferno-tower).
NON_UNIT_CLASSES = {
    "king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
    "dagger-duchess-tower-bar", "tower-bar", "king-tower-bar", "bar",
    "bar-level", "clock", "emote", "text", "elixir", "selected",
    "skeleton-king-bar",
}


def _to_our_name(name: str) -> str:
    """KataCR-style class name (hyphens) -> this project's slug convention."""
    return name.replace("-", "_")


class YoloDetector:
    def __init__(self, model_path=DEFAULT_MODEL, conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ,
                 include_non_units=False, read_levels=False):
        from ultralytics import YOLO
        import torch
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.include_non_units = include_non_units
        # read_levels: run the level-DIGIT template match (level_reader.read_level)
        # per badge. OFF by default -- it's ~0.9ms/badge x every badge x every
        # frame (~15ms on a busy frame) re-reading a level that never changes, and
        # was a big source of the live-view lag (Duncan 2026-09-10). Badge
        # DETECTION is always on (cheap) so units still show "Lv?"; turn this on
        # (detector_view --read-levels) only when you actually want the numbers.
        self.read_levels = read_levels
        self.device = 0 if torch.cuda.is_available() else "cpu"
        # Non-unit UI classes (mirror katacr_detector's split) -- filtered out
        # unless include_non_units, so unit_tracker/opponent_play_tracker only
        # ever see troops/buildings/spells. Shared as module-level NON_UNIT_CLASSES
        # so board consumers filter on the same authoritative set.
        self._non_unit = NON_UNIT_CLASSES

    def _pad_to_train_aspect(self, img):
        """Pad the frame horizontally to TRAIN_ASPECT. Returns (padded, x_offset)
        so detections can be mapped back to the original frame."""
        h, w = img.shape[:2]
        target_w = int(round(h * TRAIN_ASPECT))
        if target_w <= w:
            return img, 0
        pad = target_w - w
        left = pad // 2
        padded = cv2.copyMakeBorder(img, 0, 0, left, pad - left,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded, left

    def detect(self, screenshot: np.ndarray) -> list:
        from perception.bar_reader import apply_bar_sides, associate_level_badges, side_from_bar_color
        from perception.level_reader import read_level
        full_h, full_w = screenshot.shape[:2]
        infer_img, x_off = self._pad_to_train_aspect(screenshot)
        res = self.model.predict(infer_img, imgsz=self.imgsz, conf=self.conf,
                                 device=self.device, verbose=False)[0]
        # Collect ALL detections (bars/clocks/UI included) -- bars are needed to
        # assign side before non-units are filtered.
        dets = []
        for b in res.boxes:
            raw_name = self.model.names[int(b.cls)]
            is_non_unit = raw_name in self._non_unit
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            x1, x2 = x1 - x_off, x2 - x_off  # map padded coords back to the frame
            bbox = (max(0, int(round(x1))), max(0, int(round(y1))),
                    min(full_w, int(round(x2))), min(full_h, int(round(y2))))
            # Drop anything centred outside the arena band -- UI (hand tray,
            # banner) is not the board. Kills the hand-elixir-badge -> "clock".
            cy_frac = (bbox[1] + bbox[3]) / 2 / full_h
            if not (ARENA_Y_MIN <= cy_frac <= ARENA_Y_MAX):
                continue
            our_name = raw_name if is_non_unit else _to_our_name(raw_name)
            if not is_non_unit and should_suppress_detection(our_name, bbox):
                continue
            dets.append({
                "name": our_name,
                "confidence": float(b.conf),
                "side": "unknown",
                "bbox": bbox,
                "raw_name": raw_name,
            })

        # v2's payoff: our model detects the HP bar but not its team, so read the
        # bar's COLOUR for side, then associate bars->units (bar_reader, the same
        # logic KataCR's belonging path used). This is what replaces side_reader.
        for d in dets:
            if d["raw_name"] in ("bar", "bar-level"):
                s = side_from_bar_color(screenshot, d["bbox"])
                if s:
                    d["side"] = s
        units = apply_bar_sides(dets, screenshot)  # units w/ side+hp; bars/UI dropped

        # Position fallback: a unit with no visible bar has side='unknown', which
        # left the friend/foe picture poor (most units). Units are PLACED on the
        # owner's half, so split by the river -- above = enemy, below = ally.
        # Bar colour stays authoritative (it wins for damaged/crossed units);
        # this only fills the 'unknown's. Ambiguous river band stays unknown.
        for u in units:
            if u.get("side") == "unknown":
                cy = (u["bbox"][1] + u["bbox"][3]) / 2
                if cy < RIVER_Y_TOP:
                    u["side"], u["side_source"] = "enemy", "position"
                elif cy > RIVER_Y_BOTTOM:
                    u["side"], u["side_source"] = "ally", "position"

        # LEVEL: the model detects the team-coloured level badge as `bar-level`.
        # Bind each to its unit and read the digit (perception/level_reader). Note
        # the game's asymmetry (side_reader docstring): an ENEMY troop always shows
        # its badge; an ALLY troop shows one only once damaged -- so an ally's level
        # often reads None until it takes damage. read_level returns None rather
        # than guess a value it has no template for; the thin tracker keeps a
        # known level sticky across the frames where no badge shows.
        # has_badge = a bar-level was detected+associated (badge is present, so a
        # level EXISTS to know); level = the digit READ from it, or None if we have
        # no template for that value yet. The two differ deliberately: the overlay
        # shows "Lv?" (badge seen, number not yet collected) vs nothing (no badge).
        for u in units:
            u["level"], u["level_bbox"], u["has_badge"] = None, None, False
        level_dets = [d for d in dets if d.get("raw_name") == "bar-level"]
        if level_dets:
            for ui, ld in associate_level_badges(units, level_dets).items():
                units[ui]["level_bbox"] = ld["bbox"]
                units[ui]["has_badge"] = True          # cheap: badge detected -> "Lv?"
                if self.read_levels:                   # the expensive part, opt-in
                    value, _conf = read_level(screenshot, ld["bbox"])
                    units[ui]["level"] = int(value) if value is not None else None

        if self.include_non_units:
            # keep the raw component/UI detections too (clocks, bars, towers)
            return units + [d for d in dets if d["raw_name"] in self._non_unit]
        return units


if __name__ == "__main__":
    import cv2
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test_screenshot.png"
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"{path} not found.")
    det = YoloDetector(include_non_units=True)
    out = det.detect(img)
    print(f"{len(out)} detections on {path}:")
    for d in sorted(out, key=lambda d: -d["confidence"])[:25]:
        print(f"  {d['name']:22} {d['confidence']:.2f}  bbox={d['bbox']}")
