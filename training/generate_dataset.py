"""Jax-free synthetic dataset generator for the Clash Royale detector.

Composites KataCR's pre-cut unit sprites (RGBA) onto their annotated arena
backgrounds and writes STANDARD YOLO labels -- the fast, unlimited, perfectly-
labeled data route (see training/README.md). KataCR's own generator.py imports
jax through a viz helper; this reimplements just the compositing we need,
jax-free, so it runs on Windows.

What it does per image:
  - pick a background (its .txt already labels the static towers -> kept as base
    labels, first 5 YOLO fields; KataCR's extra state columns are dropped),
  - paste N random unit sprites (troops/buildings/spells) at random arena
    positions, alpha-composited, recording each one's YOLO bbox,
  - write image + `class cx cy w h` label file, split train/val.

Sprites keep native size (they were cut at the background's own resolution, so
scale already matches); ultralytics resizes to imgsz at train time. Belonging
(ally/enemy) is intentionally dropped -- both variants map to the same class,
which only makes the detector more robust; our pipeline reads side separately.

This is a v1: random placement, no lane/position priors and no attached
bars/clocks/UI (those classes are excluded from pasting and left to real
TV Royale data later). Good enough to train a detector that finds units well.

Usage:
    python training/generate_dataset.py --n 4000 --out training/datasets/cr_synth
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np
import yaml

DATASET_ROOT = "vendor/cr-detection-dataset"
BG_DIR = os.path.join(DATASET_ROOT, "images/part2/background")

# KataCR's canonical generation resolution (build_dataset/generation_config.py
# background_size). The pre-cut sprites were segmented at THIS scale, but most
# backgrounds ship at ~1030x1622 (~1.8x bigger), which made native-size sprites
# paste ~1.8x too small (Duncan spotted it 2026-09-07). Resizing every bg to
# this canonical size restores correct sprite-to-board scale; tower labels are
# normalized so they're unaffected. Framing is consistent across bg resolutions
# (king-tower is ~0.16 of frame height at every size), so the resize is clean.
CANONICAL_W, CANONICAL_H = 568, 896
SEG_DIR = os.path.join(DATASET_ROOT, "images/segment")
NAMES_YAML = os.path.join(DATASET_ROOT, "images/part2/ClashRoyale_detection.yaml")

# Pure-UI / tower / background classes we do NOT paste as free-floating units
# (towers already come labeled in the backgrounds; bars/clocks/etc. attach to
# units in real play and are left to real data). Everything else is fair game.
EXCLUDE = {
    "king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
    "dagger-duchess-tower-bar", "tower-bar", "king-tower-bar", "bar", "bar-level",
    "clock", "emote", "text", "elixir", "selected", "skeleton-king-bar",
    "backgrounds", "background-items",
}


def load_names():
    with open(NAMES_YAML, encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    # names: {idx: name}; build name->idx
    return {name: idx for idx, name in names.items()}, [names[i] for i in sorted(names)]


def build_sprite_pool(name2idx):
    """{class_idx: [sprite_path, ...]} for the paste-able unit classes."""
    pool = {}
    for cls in os.listdir(SEG_DIR):
        if cls in EXCLUDE or cls not in name2idx:
            continue
        d = os.path.join(SEG_DIR, cls)
        if not os.path.isdir(d):
            continue
        sprites = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png")]
        if sprites:
            pool[name2idx[cls]] = sprites
    return pool


# Components attached to units (KataCR generation_config): the HP bar sits at a
# unit's top-center, the level badge at the bar's left end, the deploy clock on
# the unit. Detecting these NATIVELY is how we retire side_reader/spawn_icon_
# reader (bar COLOR -> side, clock -> spawn), so we train them in. They come in
# ally(blue)/enemy(red) color variants (filename '<name>_<variant>_...').
COMPONENT_CLASSES = ["bar", "bar-level", "clock"]
BAR_PROB = 0.5     # of a unit getting an HP bar (+ level badge) -- a DAMAGED unit
LONE_LEVEL_PROB = 0.25  # of a unit showing ONLY its level badge, no HP bar. This
# is the real UNDAMAGED-ENEMY case: an enemy troop ALWAYS shows its (red) level
# badge, but its health bar only appears once damaged. Before this case existed,
# `bar-level` was pasted ONLY alongside `bar` (nested under it), so the model
# never saw a lone badge and under-detected it live -- Duncan, 2026-09-09: level
# "picked up when the troop had a healthbar but not when there's no healthbar".
# Needs a RETRAIN (v4) to take effect; v3 still misses lone badges.
CLOCK_PROB = 0.15  # else: chance of a deploy clock instead


def load_component_pool(name2idx):
    """{comp: {'idx': class_idx, 'variants': {variant: [paths]}}}."""
    pool = {}
    for c in COMPONENT_CLASSES:
        d = os.path.join(SEG_DIR, c)
        if c not in name2idx or not os.path.isdir(d):
            continue
        byv = {}
        for f in os.listdir(d):
            if f.endswith(".png"):
                parts = f.split("_")
                byv.setdefault(parts[1] if len(parts) > 1 else "0", []).append(os.path.join(d, f))
        if byv:
            pool[c] = {"idx": name2idx[c], "variants": byv}
    return pool


def load_no_bar_names():
    """Spell/object class names that should NOT get an HP bar or clock."""
    sys.path.insert(0, os.path.abspath("vendor/katacr"))
    try:
        from katacr.constants.label_list import spell_unit_list, object_unit_list
        return set(spell_unit_list) | set(object_unit_list)
    except Exception:
        return set()


def _place_component(canvas, sprite, x, y, cls_idx, labels):
    box = paste_sprite(canvas, sprite, int(x), int(y))
    if box:
        H, W = canvas.shape[:2]
        x1, y1, x2, y2 = box
        labels.append((cls_idx, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))


def attach_components(canvas, unit_bbox, unit_name, comp_pool, no_bar):
    """Maybe paste a bar+level (above the unit) or a clock (on it), matching a
    single side color. Appends their YOLO labels. Returns the new labels."""
    labels = []
    if unit_name in no_bar or not comp_pool:
        return labels
    ux1, uy1, ux2, uy2 = unit_bbox
    ucx, ucy = (ux1 + ux2) // 2, (uy1 + uy2) // 2
    side = random.choice(["0", "1"])

    def pick(comp):
        v = comp_pool[comp]["variants"]
        paths = v.get(side) or sum(v.values(), [])
        return cv2.imread(random.choice(paths), cv2.IMREAD_UNCHANGED)

    r = random.random()
    if r < BAR_PROB + LONE_LEVEL_PROB and "bar" in comp_pool and "bar-level" in comp_pool:
        # A level badge either WITH a health bar (damaged unit, r < BAR_PROB) or
        # ALONE (undamaged enemy, always badged). The badge is placed at the bar's
        # left end EITHER WAY -- a bar sprite is picked for geometry even when it's
        # not drawn -- so the model sees the badge in the SAME spot with and
        # without a bar, which is the whole point of the lone case.
        draw_bar = r < BAR_PROB
        bar = pick("bar")
        bh, bw = bar.shape[:2]
        bx, by = ucx - bw // 2, uy1 - bh + 2          # bar just above the unit's top
        if draw_bar:
            _place_component(canvas, bar, bx, by, comp_pool["bar"]["idx"], labels)
        lvl = pick("bar-level")
        lh, lw = lvl.shape[:2]
        _place_component(canvas, lvl, bx - lw + 3, (by + bh // 2) - lh // 2,
                         comp_pool["bar-level"]["idx"], labels)  # left end of the bar
    elif r < BAR_PROB + LONE_LEVEL_PROB + CLOCK_PROB and "clock" in comp_pool:
        clk = pick("clock")
        ch, cw = clk.shape[:2]
        _place_component(canvas, clk, ucx - cw // 2, ucy - ch // 2,
                         comp_pool["clock"]["idx"], labels)  # centered on the unit
    return labels


def load_backgrounds():
    """[(bg_bgr, [ (cls,cx,cy,w,h), ... ]), ...] -- image + its base tower labels."""
    bgs = []
    for f in os.listdir(BG_DIR):
        if not f.endswith(".jpg"):
            continue
        img = cv2.imread(os.path.join(BG_DIR, f))
        if img is None:
            continue
        img = cv2.resize(img, (CANONICAL_W, CANONICAL_H))  # match the sprite scale
        labels = []
        txt = os.path.join(BG_DIR, f[:-4] + ".txt")
        if os.path.isfile(txt):
            for line in open(txt, encoding="utf-8"):
                p = line.split()
                if len(p) >= 5:
                    labels.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        bgs.append((img, labels))
    return bgs


def paste_sprite(canvas, sprite_rgba, x, y):
    """Alpha-composite sprite's top-left at (x,y). Returns None if it doesn't
    fit, else the (x1,y1,x2,y2) it occupied."""
    sh, sw = sprite_rgba.shape[:2]
    H, W = canvas.shape[:2]
    if x < 0 or y < 0 or x + sw > W or y + sh > H:
        return None
    rgb = sprite_rgba[:, :, :3].astype(float)
    a = (sprite_rgba[:, :, 3:4].astype(float)) / 255.0
    roi = canvas[y:y + sh, x:x + sw].astype(float)
    canvas[y:y + sh, x:x + sw] = (rgb * a + roi * (1 - a)).astype(np.uint8)
    return x, y, x + sw, y + sh


def gen_one(bgs, pool, names, comp_pool, no_bar, min_units, max_units):
    bg, base = random.choice(bgs)
    canvas = bg.copy()
    H, W = canvas.shape[:2]
    labels = list(base)  # start with the towers already labeled in the bg
    n = random.randint(min_units, max_units)
    for _ in range(n):
        cls = random.choice(list(pool))
        sprite = cv2.imread(random.choice(pool[cls]), cv2.IMREAD_UNCHANGED)
        if sprite is None or sprite.ndim != 3 or sprite.shape[2] != 4:
            continue
        sh, sw = sprite.shape[:2]
        # Place within the ARENA PLAYFIELD only -- the backgrounds are full game
        # frames, so the top band (~0-13% = trophy/timer banner) and bottom band
        # (~84-100% = hand tray + elixir) are UI, NOT playfield. Pasting a unit
        # there would teach the detector a false "unit here" on UI. Keep units
        # between those bands (the towers, already labeled in the bg, sit inside).
        x_lo, x_hi = int(W * 0.05), W - sw - int(W * 0.05)
        y_lo, y_hi = int(H * 0.13), int(H * 0.84) - sh
        if x_hi <= x_lo or y_hi <= y_lo:
            continue
        x = random.randint(x_lo, x_hi)
        y = random.randint(y_lo, y_hi)
        box = paste_sprite(canvas, sprite, x, y)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        labels.append((cls, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
        # Attach a bar+level or a clock to this unit (native detection of the
        # things side_reader/spawn_icon_reader do by pixel search).
        labels.extend(attach_components(canvas, (x1, y1, x2, y2), names[cls], comp_pool, no_bar))
    return canvas, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4000, help="Total images to generate.")
    p.add_argument("--out", default="training/datasets/cr_synth", help="Output dataset dir.")
    p.add_argument("--min-units", type=int, default=4)
    p.add_argument("--max-units", type=int, default=14)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    random.seed(args.seed)

    name2idx, names = load_names()
    pool = build_sprite_pool(name2idx)
    comp_pool = load_component_pool(name2idx)
    no_bar = load_no_bar_names()
    bgs = load_backgrounds()
    print(f"{len(names)} classes | {len(pool)} paste-able unit classes | "
          f"{sum(len(v) for v in pool.values())} sprites | {len(bgs)} backgrounds")
    print(f"components: {sorted(comp_pool)} | {len(no_bar)} spell/object classes get no bar")

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

    n_val = int(args.n * args.val_frac)
    for i in range(args.n):
        split = "val" if i < n_val else "train"
        canvas, labels = gen_one(bgs, pool, names, comp_pool, no_bar, args.min_units, args.max_units)
        stem = f"{i:06d}"
        cv2.imwrite(os.path.join(args.out, "images", split, stem + ".jpg"), canvas)
        with open(os.path.join(args.out, "labels", split, stem + ".txt"), "w", encoding="utf-8") as f:
            for cls, cx, cy, w, h in labels:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{args.n}")

    data_yaml = {
        "path": os.path.abspath(args.out),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(names)},
    }
    with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    print(f"Done. Dataset at {args.out} (data.yaml written). "
          f"Train: python training/finetune.py --data {os.path.join(args.out, 'data.yaml')}")


if __name__ == "__main__":
    main()
