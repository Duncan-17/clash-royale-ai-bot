"""Convert KataCR's real annotated gameplay frames (vendor/cr-detection-dataset/
images/part2, ~6966 frames with labels) into standard YOLO, and write a combined
data.yaml that trains on REAL + our SYNTHETIC together.

Why (2026-09-08): the synthetic-only model detects archers fine on synthetic
(val recall 0.78, ~same as goblins) but poorly on REAL frames -- a domain gap,
worst for visually-subtle units like archers. Real labeled frames close that gap
directly. KataCR's part2 uses the SAME 201-class taxonomy as our synthetic set
(config from ClashRoyale_detection.yaml), and its .txt is YOLO + extra state
columns we drop -- so it combines cleanly. Real frames also bring real motion/
effects/backgrounds (helps the animation losses + false positives too).

Output: training/datasets/cr_real/{images,labels}/{train,val} and
training/datasets/cr_combined.yaml (train/val = synthetic + real).

Usage:
    python training/prepare_real_data.py
"""

import glob
import os
import random
import shutil

import yaml

DATASET_ROOT = "vendor/cr-detection-dataset"
PART2 = os.path.join(DATASET_ROOT, "images/part2")
NAMES_YAML = os.path.join(PART2, "ClashRoyale_detection.yaml")
SYNTH = "training/datasets/cr_synth_v2"
OUT = "training/datasets/cr_real"
COMBINED_YAML = "training/datasets/cr_combined.yaml"
VAL_FRAC = 0.1


def main():
    frames = [j for j in glob.glob(os.path.join(PART2, "**", "*.jpg"), recursive=True)
              if os.path.isfile(j[:-4] + ".txt") and os.sep + "background" not in j]
    random.seed(0)
    random.shuffle(frames)
    n_val = int(len(frames) * VAL_FRAC)
    print(f"{len(frames)} real labeled frames -> {n_val} val / {len(frames)-n_val} train")

    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)

    for i, jpg in enumerate(frames):
        split = "val" if i < n_val else "train"
        stem = f"real_{i:06d}"
        shutil.copy(jpg, os.path.join(OUT, "images", split, stem + ".jpg"))
        with open(jpg[:-4] + ".txt", encoding="utf-8") as f, \
             open(os.path.join(OUT, "labels", split, stem + ".txt"), "w", encoding="utf-8") as o:
            for line in f:
                p = line.split()
                if len(p) >= 5:  # keep only class cx cy w h, drop KataCR's state columns
                    o.write(" ".join(p[:5]) + "\n")
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(frames)}")

    names = yaml.safe_load(open(NAMES_YAML, encoding="utf-8"))["names"]
    combined = {
        "train": [os.path.abspath(os.path.join(SYNTH, "images/train")),
                  os.path.abspath(os.path.join(OUT, "images/train"))],
        "val": [os.path.abspath(os.path.join(SYNTH, "images/val")),
                os.path.abspath(os.path.join(OUT, "images/val"))],
        "names": names,
    }
    with open(COMBINED_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(combined, f, sort_keys=False, allow_unicode=True)
    print(f"Done. Combined dataset yaml: {COMBINED_YAML}\n"
          f"Retrain: python training/finetune.py --data {COMBINED_YAML} --name cr_finetune_v3")


if __name__ == "__main__":
    main()
