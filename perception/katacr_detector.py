"""
EXPERIMENTAL drop-in alternative to perception/unit_detector.py's UnitDetector,
backed by KataCR's YOLOv8 detectors instead of build-a-bot's units.onnx.

Why this exists (2026-08-20): every remaining detection bug this project has
been patching downstream for weeks -- class hallucinations (Valkyrie,
spear_goblin<->archer), low recall on small units (goblins), no spell coverage
at all, cards missing entirely -- traces back to the base model rather than the
logic on top of it. build-a-bot's units.onnx is a YOLOv10m whose own embedded
metadata dates it 2024-08-31, ~2 years stale, and it has NO classes for spells,
health bars, or the deploy-delay clock.

KataCR (https://github.com/wty-yy/KataCR, MIT, 459*) publishes two YOLOv8l
detectors covering 155 classes -- including, as NATIVE DETECTABLE CLASSES,
several things this project hand-built entire pixel-calibrated modules for:
  'tower-bar'/'king-tower-bar'  <- perception/tower_reader.py
  'bar'/'bar-level'             <- perception/side_reader.py
  'clock'                       <- perception/spawn_icon_reader.py
  'selected'                    <- the hand-selection signal we'd found nothing for
  19 spell classes              <- the entire unbuilt spell-detection module
Reported metrics from the checkpoints' own train_metrics (on THEIR test set,
not ours -- treat as their claim, not a verified result here):
  detector1: P 0.905 R 0.803 mAP50 0.842 mAP50-95 0.623
  detector2: P 0.911 R 0.853 mAP50 0.879 mAP50-95 0.748

IMPORTANT CAVEAT, do not lose this: these checkpoints are dated 2024-04/2024-05,
which is actually ~4 months OLDER than the build-a-bot model they'd replace. So
this does NOT fix the "cards released after the training cutoff are invisible"
problem (Berserker/Ronin/Spirit Empress/Suspicious Bush/Void, plus the 7
unclassifiable cards found while building game_knowledge/card_reference.json) --
if anything that gap is slightly worse. The case for switching is structural
coverage (spells/bars/clock) and resolution (896 vs 480x352, which is the most
plausible explanation for our small-unit recall problem), NOT recency.

Why this reimplements inference instead of importing KataCR's own ComboDetector:
their combo_detect module imports their TRAINING path (train -> custom_validator
-> custom_dataset -> generator), which imports jax -- a heavy, Linux-oriented
dependency this project has no other use for (their README documents the
Linux/JAX constraint). Rather than install jax to run inference, the ~40 lines
their inference actually needs are reimplemented here, matching their
custom_utils.non_max_suppression exactly on the one non-obvious detail: the
model emits 4 box channels + (nc-1) real class scores + 1 BELONGING channel
(ally/enemy), so a naive ultralytics predict() call misreads that last channel
as if it were a class (it surfaces as a bogus 'padding_belong' detection).

Also note these are AGPL-3.0 (same as the build-a-bot model already in use) --
consistent with this project's existing private/non-distributed-use decision,
see Project_Status. Revisit if this repo is ever distributed.

Status: EXPERIMENTAL, on branch experiment/detector-model-swap. Not wired in by
default -- run `python -m perception.debug_live --detector katacr` to compare
live against the current detector.
"""

import os
import sys

import cv2
import numpy as np

from config.red_zones import should_suppress_detection

# The checkpoints pickle a custom class (katacr.yolov8.custom_model.CRDetectionModel),
# so `katacr` must be importable to unpickle them at all. Only the narrow
# custom_model chain is needed and it's jax-free -- it's their combo_detect ->
# train -> custom_validator -> custom_dataset -> generator chain that pulls jax
# (see module docstring), which is exactly why this module avoids that entry point.
VENDOR_KATACR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "vendor", "katacr")
if os.path.isdir(VENDOR_KATACR) and VENDOR_KATACR not in sys.path:
    sys.path.insert(0, VENDOR_KATACR)

MODEL_DIR = "assets/models/katacr"
DEFAULT_MODELS = [os.path.join(MODEL_DIR, "detector1.pt"),
                  os.path.join(MODEL_DIR, "detector2.pt")]

# Their training input is the ARENA CROP resized to (568, 896) -- see KataCR's
# build_dataset/constant.py `image_size_part2 = (568, 896)` and its per-screen-ratio
# 'part2_*' crop fractions. Two real corrections were needed to get this working,
# both found empirically (a first version fed the whole 900x1600 screen letterboxed
# into a square 896 and produced garbage -- it misdetected the HAND-CARD ROW as
# board units, e.g. a Balloon card in hand read as a Balloon on the board, and
# missed two Giants our current detector caught at 0.92):
#   1. Feed the arena crop only, not the full screen.
#   2. Width must be stride-32 aligned; 568 is not (568/32 = 17.75) and makes the
#      FPN concat fail outright, so 576 is used.
# With both applied, confidences jumped to 0.97-0.98 and the belonging channel
# resolved cleanly binary (~0.97 vs ~0.00). KataCR defines no crop for this
# project's 1.78 screen ratio (they list 2.13/2.16/2.22), so this reuses
# unit_detector.py's own already-verified arena band instead.
NET_W, NET_H = 576, 896
DEFAULT_CONF = 0.4  # matches unit_detector.py's own MIN_CONF for comparability
DEFAULT_IOU = 0.6   # KataCR's own combo_detect default

# KataCR class names are hyphenated ('spear-goblin'); this project uses
# underscores ('spear_goblin') everywhere (unit_detector.CLASS_NAMES,
# unit_metadata, spawn_data). Normalizing here keeps every downstream consumer
# (unit_tracker, opponent_play_tracker, red-zone filtering) working unchanged.
# Names with no counterpart in our taxonomy (towers, bars, clock, elixir,
# emote, text, selected) are deliberately NOT renamed -- they're passed through
# as-is so they're visible in the debug view, and callers that only care about
# troops can filter on NON_UNIT_CLASSES below.
NON_UNIT_CLASSES = {
    "king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
    "dagger-duchess-tower-bar", "tower-bar", "king-tower-bar", "bar",
    "bar-level", "clock", "emote", "text", "elixir", "selected",
    "skeleton-king-bar", "evolution-symbol", "ice-spirit-evolution-symbol",
    "padding_belong",  # not a real class -- the belonging channel, see module docstring
}


# KataCR names that don't simply hyphen-swap into this project's taxonomy. Without
# this, 'mini-pekka' became 'mini_pekka' while everything downstream
# (unit_metadata.TARGET_TYPE, spawn_data, red_zones' is_air lookup) keys on
# 'minipekka' -- so a real Mini P.E.K.K.A. silently missed every metadata lookup.
# Found 2026-08-20 by diffing every KataCR class name against unit_detector.CLASS_NAMES.
NAME_OVERRIDES = {"mini-pekka": "minipekka"}

# Classes suppressed as noise for unit-tracking purposes. Two groups:
#  - padding_*: not real classes at all, just taxonomy filler in their label list
#    (padding_belong is the belonging channel; padding_0/1/2 are pure slots). These
#    surfaced as bogus detections until filtered.
#  - projectile/VFX classes: real KataCR classes, but they're transient effects rather
#    than units, and Duncan reported 'dirt' firing constantly in a match where neither
#    player had Miner or Goblin Drill (its only legitimate sources). Feeding these to
#    unit_tracker/opponent_play_tracker would manufacture phantom "units".
# Kept separate from NON_UNIT_CLASSES because those (bars, clock, elixir) are WANTED
# signals we deliberately surface; these are simply noise.
NOISE_CLASSES = {
    "padding_0", "padding_1", "padding_2", "padding_belong",
    "axe", "bomb", "dirt", "goblin-ball", "skeleton-king-skill",
}


def _to_our_name(katacr_name: str) -> str:
    if katacr_name in NAME_OVERRIDES:
        return NAME_OVERRIDES[katacr_name]
    return katacr_name.replace("-", "_")




class KataCRDetector:
    """Same interface as perception.unit_detector.UnitDetector: detect(screenshot)
    returns a list of {name, confidence, side, bbox}. `side` is "ally"/"enemy"
    read straight from the model's own belonging channel -- unlike units.onnx,
    which has no side output at all (this project replaced its companion
    side.onnx with the algorithmic perception/side_reader.py)."""

    def __init__(self, model_paths=None, conf=DEFAULT_CONF, iou=DEFAULT_IOU,
                 include_non_units=True, device=None, half=None):
        import torch  # imported lazily -- torch is only needed for this experimental path
        from ultralytics import YOLO
        self._torch = torch
        self.conf = conf
        self.iou = iou
        self.include_non_units = include_non_units
        # These are YOLOv8l at 576x896 -- roughly 12x the compute of the YOLOv10m@480x352
        # this replaces, so CPU inference measured ~457ms/frame (~2fps) and was reported
        # unwatchable in the live debug view. GPU is effectively required for live use.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # fp16 roughly halves GPU inference time at no meaningful accuracy cost for
        # detection; never used on CPU, where fp16 is typically SLOWER (no native support).
        self.half = (self.device != "cpu") if half is None else half
        paths = model_paths or DEFAULT_MODELS
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                f"KataCR weights not found: {missing}. Download them from the links in "
                "KataCR's README (see this module's docstring) into assets/models/katacr/."
            )
        self._models = []
        for p in paths:
            y = YOLO(p)
            m = y.model.to(self.device).eval()
            m = m.half() if self.half else m.float()
            self._models.append((m, y.names))

    def _raw_boxes(self, model, names, tensor):
        """Runs one detector and returns (xyxy_in_net_space, conf, class_idx, bel)
        after that model's own NMS. Splitting channels as 4 + (nc-1) classes + 1
        belonging matches KataCR's custom_utils.non_max_suppression exactly."""
        torch = self._torch
        with torch.no_grad():
            pred = model(tensor)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        pred = pred[0].transpose(0, 1)  # (num_boxes, channels)

        nc = len(names)  # includes the belonging channel as its last entry
        box, cls, bel = pred[:, :4], pred[:, 4:4 + nc - 1], pred[:, 4 + nc - 1]
        conf, j = cls.max(1)
        keep = conf > self.conf
        if int(keep.sum()) == 0:
            return None
        box, conf, j, bel = box[keep], conf[keep], j[keep], bel[keep]

        xy, wh = box[:, :2], box[:, 2:4]
        xyxy = torch.cat([xy - wh / 2, xy + wh / 2], dim=1)

        import torchvision
        keep_idx = torchvision.ops.nms(xyxy, conf, self.iou)
        return xyxy[keep_idx], conf[keep_idx], j[keep_idx], bel[keep_idx]

    def detect(self, screenshot: np.ndarray) -> list:
        torch = self._torch
        import torchvision
        full_h, full_w = screenshot.shape[:2]

        # Arena crop first -- see NET_W/NET_H's comment for why this is required and
        # what broke without it. Reuses unit_detector.py's own verified arena band so
        # both detectors are looking at exactly the same region of the board.
        from perception.unit_detector import UNIT_Y_END, UNIT_Y_START
        ay1, ay2 = int(UNIT_Y_START * full_h), int(UNIT_Y_END * full_h)
        arena = screenshot[ay1:ay2, :]
        arena_h, arena_w = arena.shape[:2]

        rgb = cv2.cvtColor(arena, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (NET_W, NET_H), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        tensor = (tensor.half() if self.half else tensor.float()).div(255)

        parts = []
        for model, names in self._models:
            got = self._raw_boxes(model, names, tensor)
            if got is None:
                continue
            xyxy, conf, j, bel = got
            parts.append((xyxy, conf, [names[int(idx)] for idx in j], bel))
        if not parts:
            return []

        all_xyxy = torch.cat([p[0] for p in parts])
        all_conf = torch.cat([p[1] for p in parts])
        all_names = [n for p in parts for n in p[2]]
        all_bel = torch.cat([p[3] for p in parts])

        # Cross-model NMS: the two detectors are trained on overlapping class sets
        # split by object size, so the same physical object is frequently found by
        # BOTH. Without this the debug view showed obvious duplicates (two
        # 'tower-bar' boxes on one bar, two 'clock' boxes on one spawn icon).
        # Class-agnostic on purpose -- the duplicates often disagree on class,
        # which per-class NMS would keep both of.
        final = torchvision.ops.nms(all_xyxy, all_conf, self.iou)

        sx, sy = arena_w / NET_W, arena_h / NET_H
        results = []
        for k in final.tolist():
            name = all_names[k]
            if name in NOISE_CLASSES:
                continue
            is_non_unit = name in NON_UNIT_CLASSES
            if is_non_unit and not self.include_non_units:
                continue
            x1, y1, x2, y2 = all_xyxy[k].tolist()
            bbox = (max(0, int(round(x1 * sx))),
                    max(0, int(round(y1 * sy)) + ay1),   # + ay1: back to full-frame
                    min(full_w, int(round(x2 * sx))),
                    min(full_h, int(round(y2 * sy)) + ay1))
            our_name = _to_our_name(name)
            # Red-zone filtering, added 2026-08-20 per Duncan after live testing:
            # this path originally had NONE, and produced exactly the false positives
            # config/red_zones.py exists to kill -- a recurring 'the_log' dead-centre
            # in the river band at (438, 672), an 'archer_queen' in the trees at
            # cx~30, an 'ice_spirit' on a rock at cx~861. Uses the same shared rules
            # as unit_detector.py so both detectors behave identically here.
            #
            # Deliberately NOT applied to the non-unit UI classes: tower bars, the
            # deploy clock and the elixir readout legitimately live inside tower
            # exclusion zones and outside the arena bounds, so suppressing them
            # would throw away the very detections that make this model interesting.
            if not is_non_unit and should_suppress_detection(our_name, bbox):
                continue
            results.append({
                "name": name if is_non_unit else our_name,
                "confidence": float(all_conf[k]),
                # bel>0.5 is the model's own belonging split. Polarity is confirmed
                # empirically against a known frame -- see this module's __main__.
                "side": "enemy" if float(all_bel[k]) > 0.5 else "ally",
                "bbox": bbox,
                "raw_name": name,
            })
        return results


if __name__ == "__main__":
    # Empirical check of the belonging channel's polarity against a frame with
    # KNOWN ground truth: temp_img/opponent_plays/20260817_141624_961306_played_giant.png
    # contains a confirmed ALLY giant around (130,342) and a confirmed ENEMY giant
    # around (612,509) -- established when that snapshot was reviewed with Duncan
    # (see Project_Status's spawn-icon section). If the sides below come out
    # swapped, flip the comparison in _infer_one rather than guessing.
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "temp_img/opponent_plays/20260817_141624_961306_played_giant.png"
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    det = KataCRDetector()
    results = det.detect(img)
    print(f"{len(results)} detection(s) in {path}:\n")
    for d in sorted(results, key=lambda r: -r["confidence"]):
        marker = ""
        cx = (d["bbox"][0] + d["bbox"][2]) / 2
        cy = (d["bbox"][1] + d["bbox"][3]) / 2
        if d["raw_name"] == "giant":
            if abs(cx - 200) < 90 and abs(cy - 405) < 90:
                marker = "   <-- should be ALLY (known ground truth)"
            elif abs(cx - 685) < 90 and abs(cy - 580) < 90:
                marker = "   <-- should be ENEMY (known ground truth)"
        print(f"  {d['raw_name']:24s} {d['side']:6s} conf={d['confidence']:.2f} "
              f"bbox={d['bbox']}{marker}")
