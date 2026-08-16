"""
Detects troops/buildings on the board and classifies each as ally or
enemy — the first vision module in this project that's a real neural
network rather than template matching or pixel math, since units can
appear anywhere on the board, at any scale/rotation, over a changing
background (see clash-royale-ai-pipeline-explained.md for why that rules
out template matching here).

Reuses a pre-trained YOLOv10 model (ONNX format) from
Pbatch/ClashRoyaleBuildABot (vendor/build-a-bot/) rather than training our
own — see Project_Status for the full reasoning (MIT-licensed repo code;
the model FILE's own embedded metadata separately claims AGPL-3.0 from its
export tooling, decided acceptable for this project's private/
non-distributed use). Recognizes 97 of our 122 cards — see Project_Status
for the precise gap analysis (most of the "missing" 25 are spells, which
were never in scope for a troop/building detector; the real gap is 5
troops from 2025 releases, deliberately not solved yet).

Two models, both copied into assets/models/:
- units.onnx — the actual troop/building detector. Input: one video
  frame. Output: up to 300 candidate detections, each
  [left, top, right, bottom, confidence, class_id] in the model's own
  480x352 letterboxed coordinate space. YOLOv10 is NMS-free (its export
  metadata confirms this), so no separate de-duplication pass is needed
  on top of the confidence filter.
- side.onnx — a tiny separate classifier (16x16 input) that labels a
  single detected unit's crop as "ally" or "enemy". Run per-detection
  here rather than build-a-bot's shortcut of skipping it for units that
  couldn't possibly be ours based on deck contents — we don't have a
  deck->unit mapping yet, and the model is cheap enough that running it
  unconditionally isn't worth the complexity of avoiding.

Preprocessing ported from build-a-bot's onnx_detector.py/unit_detector.py
(resize preserving aspect ratio, then pad to the model's fixed input size
with constant 114 — standard YOLO "letterbox" preprocessing) rather than
imported directly, for two reasons: (1) their version is PIL-based, this
project is cv2/numpy throughout; (2) porting explicitly rather than
depending on vendor/ at runtime means this still works if the vendor
clone is ever deleted — only the two .onnx files, now copied into
assets/models/, are an actual dependency.

Important bug class avoided on purpose: the crop fed to the model is
converted BGR->RGB before inference. cv2 loads images as BGR (this
project's convention throughout), but the model was trained via PIL-based
preprocessing, which loads RGB — feeding it BGR directly would silently
swap red and blue, the same category of bug this project already hit and
fixed once for card templates (see capture.py's git history). Verify
visually (see this file's __main__ block) rather than trusting this by
assumption alone.

Arena crop: only the middle vertical band of the screen (5%-80% of
height) is fed to the model, matching what it was actually trained on —
cuts off the top opponent-info bar and the bottom hand/elixir row.
Verified visually against a real 900x1600 capture before relying on it
(see conversation history) — the crop cleanly isolates the full arena
including both king towers, with no unit-relevant content clipped.

Usage:
    python -m perception.unit_detector
Reads test_screenshot.png, prints each detection (name, side, confidence,
bbox), and saves debug_unit_detections.png with boxes drawn — open it to
confirm detections visually before trusting this.
"""

import cv2
import numpy as np
import onnxruntime as ort

from config.red_zones import in_tower_detection_exclusion, in_water_zone
from perception.unit_metadata import is_air

MODEL_PATH = "assets/models/units.onnx"
SIDE_MODEL_PATH = "assets/models/side.onnx"

UNIT_Y_START = 0.05
UNIT_Y_END = 0.80
MIN_CONF = 0.4  # raised from build-a-bot's default 0.3 (2026-08-15) based on log analysis of a
# real match: noise (single/few-frame) tracks averaged confidence 0.502 vs 0.744 for real ones,
# and at 0.4 only 6.7% of REAL tracks' own frames would dip below it (vs 13.3% at 0.5) — the
# tracker's grace period (unit_tracker.py) absorbs sporadic single-frame dips like that without
# losing identity, so this trades a small, mostly-invisible cost for filtering out a real chunk
# of low-confidence noise. See Project_Status for the full numbers this was based on.
PAD_VALUE = 114  # standard YOLO letterbox padding color (mid-gray)

# Class order verified directly against units.onnx's own embedded metadata
# (Ultralytics export 'names' field), not copied blind from build-a-bot's
# Python source — see Project_Status for how this was cross-checked.
# Index 46 relabeled from the model's own "hungry_dragon" to "baby_dragon"
# (current in-game name, matching this project's naming convention) —
# everything else matched the model's metadata exactly, order included.
CLASS_NAMES = [
    "archer", "archer_queen", "balloon", "bandit", "barbarian", "barbarian_hut",
    "bat", "battle_healer", "battle_ram", "bomb_tower", "bomber", "bowler",
    "brawler", "cannon", "cannon_cart", "dark_prince", "dart_goblin",
    "electro_dragon", "electro_giant", "electro_spirit", "electro_wizard",
    "elite_barbarian", "elixir_collector", "elixir_golem_large",
    "elixir_golem_medium", "elixir_golem_small", "executioner", "fire_spirit",
    "firecracker", "fisherman", "flying_machine", "furnace", "giant",
    "giant_skeleton", "giant_snowball", "goblin", "goblin_cage", "goblin_drill",
    "goblin_hut", "golden_knight", "golem", "golemite", "guard", "heal_spirit",
    "hog", "hog_rider", "baby_dragon", "hunter", "ice_golem", "ice_spirit",
    "ice_wizard", "inferno_dragon", "inferno_tower", "knight", "lava_hound",
    "lava_pup", "little_prince", "lumberjack", "magic_archer", "mega_knight",
    "mega_minion", "mighty_miner", "miner", "minion", "minipekka", "monk",
    "mortar", "mother_witch", "musketeer", "night_witch", "pekka",
    "phoenix_egg", "phoenix_large", "phoenix_small", "prince", "princess",
    "ram_rider", "rascal_boy", "rascal_girl", "royal_ghost", "royal_giant",
    "royal_guardian", "royal_hog", "royal_recruit", "skeleton",
    "skeleton_dragon", "skeleton_king", "sparky", "spear_goblin", "tesla",
    "tombstone", "valkyrie", "wall_breaker", "witch", "wizard", "x_bow", "zappy",
]


def _should_suppress(name: str, bbox: tuple) -> bool:
    """Rejects a detection whose CENTER falls in a known false-positive
    zone. Two different kinds of zone here on purpose — see config/
    red_zones.py's module docstring for why they're not the same thing:
    tower detection-exclusion zones are deliberately LARGER than where a
    troop can actually stand (they also cover the defending figure/
    apparatus whose sprite was getting misclassified as a troop, not just
    the tower's true footprint), while water zones ARE a genuine
    impossible-position check (ground units really cannot be in the
    river) and only apply to non-flying units."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    if in_tower_detection_exclusion(cx, cy):
        return True
    if not is_air(name) and in_water_zone(cx, cy):
        return True
    return False


class OnnxDetector:
    """Generic ONNX wrapper: load a model, expose its expected input size,
    handle the resize+pad("letterbox")+normalize preprocessing shared by
    both models here. Ported from build-a-bot's onnx_detector.py (cv2/numpy
    instead of PIL — see module docstring)."""

    def __init__(self, model_path):
        providers = list(
            set(ort.get_available_providers()) & {"CUDAExecutionProvider", "CPUExecutionProvider"}
        )
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.output_name = self.session.get_outputs()[0].name
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.model_height, self.model_width = input_shape[2], input_shape[3]

    def _resize(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        ratio = height / width
        if ratio > self.model_height / self.model_width:
            new_height = self.model_height
            new_width = int(self.model_height / ratio)
        else:
            new_width = self.model_width
            new_height = int(self.model_width * ratio)
        return cv2.resize(image, (new_width, new_height))

    def _pad(self, image: np.ndarray):
        height, width = image.shape[:2]
        dx = self.model_width - width
        dy = self.model_height - height
        pad_right, pad_left = dx // 2, dx - dx // 2
        pad_bottom, pad_top = dy // 2, dy - dy // 2
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        padded = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=PAD_VALUE,
        )
        return padded, padding

    def preprocess(self, bgr_image: np.ndarray):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # model expects RGB — see module docstring
        resized = self._resize(rgb_image)
        padded, padding = self._pad(resized.astype(np.float16))
        chw = padded.transpose(2, 0, 1) / 255.0
        return np.expand_dims(chw.astype(np.float16), axis=0), padding

    def fix_bboxes(self, boxes: np.ndarray, width: int, height: int, padding) -> np.ndarray:
        pad_left, pad_right, pad_top, pad_bottom = padding
        boxes[:, [0, 2]] -= pad_left
        boxes[:, [1, 3]] -= pad_top
        boxes[..., [0, 2]] *= width / (self.model_width - pad_left - pad_right)
        boxes[..., [1, 3]] *= height / (self.model_height - pad_top - pad_bottom)
        return boxes

    def infer(self, model_input: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: model_input})[0]


class SideDetector(OnnxDetector):
    """Classifies a single detected unit's crop as 'ally' or 'enemy'."""

    def run(self, bgr_crop: np.ndarray) -> str:
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (16, 16), interpolation=cv2.INTER_CUBIC)
        model_input = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
        pred = self.session.run([self.output_name], {self.input_name: model_input})[0]
        return ("ally", "enemy")[int(np.argmax(pred[0]))]


class UnitDetector(OnnxDetector):
    def __init__(self, model_path=MODEL_PATH, side_model_path=SIDE_MODEL_PATH):
        super().__init__(model_path)
        self.side_detector = SideDetector(side_model_path)

    def detect(self, screenshot: np.ndarray) -> list:
        """Returns a list of dicts: name, confidence, side ('ally'/'enemy'),
        bbox (x1, y1, x2, y2) in the ORIGINAL screenshot's pixel coordinates."""
        full_height, full_width = screenshot.shape[:2]
        y1 = int(UNIT_Y_START * full_height)
        y2 = int(UNIT_Y_END * full_height)
        arena_crop = screenshot[y1:y2, :]

        model_input, padding = self.preprocess(arena_crop)
        raw = self.infer(model_input)[0]
        raw = raw[raw[:, 4] > MIN_CONF]
        if len(raw) == 0:
            return []

        boxes = self.fix_bboxes(raw.copy(), full_width, full_height, padding)
        # boxes' y-values are currently scaled as if they spanned the full
        # frame height — rescale down to the actual cropped band, then
        # shift back into full-frame coordinates (mirrors build-a-bot's
        # unit_detector.py _post_process, verified against a real capture
        # in this module's __main__ block rather than trusted blindly).
        boxes[:, [1, 3]] *= (UNIT_Y_END - UNIT_Y_START)
        boxes[:, [1, 3]] += UNIT_Y_START * full_height

        detections = []
        for left, top, right, bottom, conf, cls in boxes:
            bbox = (int(round(left)), int(round(top)), int(round(right)), int(round(bottom)))
            bbox = (
                max(0, bbox[0]), max(0, bbox[1]),
                min(full_width, bbox[2]), min(full_height, bbox[3]),
            )
            name = CLASS_NAMES[int(cls)]
            if _should_suppress(name, bbox):
                continue
            crop = screenshot[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            side = self.side_detector.run(crop) if crop.size > 0 else "unknown"
            detections.append({
                "name": name,
                "confidence": float(conf),
                "side": side,
                "bbox": bbox,
            })
        return detections


if __name__ == "__main__":
    screenshot = cv2.imread("test_screenshot.png")
    if screenshot is None:
        raise FileNotFoundError("test_screenshot.png not found — capture one first.")

    detector = UnitDetector()
    detections = detector.detect(screenshot)

    print(f"{len(detections)} unit(s) detected:\n")
    display = screenshot.copy()
    for d in detections:
        print(f"  {d['name']:16s} {d['side']:6s} conf={d['confidence']:.2f} bbox={d['bbox']}")
        color = (0, 200, 0) if d["side"] == "ally" else (0, 0, 255)
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display, f"{d['name']} {d['confidence']:.2f}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.imwrite("debug_unit_detections.png", display)
    print("\nSaved debug_unit_detections.png — open it to confirm detections visually.")
