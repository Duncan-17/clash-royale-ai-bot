"""Turn the reviewed board crops (assets/board_labeled/<card>/, filed by
perception/review_board_queue.py) into a standard YOLO dataset of REAL frames
from Duncan's emulator, and write a combined data.yaml that trains the detector
on synthetic + KataCR-real + THESE.

Each labelled crop is one single-object training image: the padded crop is the
image, and its sidecar's crop_box (the unit's box within the crop) is the label
box. The class INDEX comes from the model's existing taxonomy (read from
training/datasets/cr_combined_v4.yaml, the authoritative name<->index map for the
current weights), via the card's detector class in game_knowledge/card_reference.json:
  card slug -> detected_class (e.g. goblins -> 'goblin', dart_goblin -> 'dart_goblin')
           -> hyphen taxonomy name ('spear_goblin' -> 'spear-goblin') -> class index.

HELD ASIDE (not an error -- reported): a card with no detector class (the gap cards
berserker/ronin/goblin_gang/... whose detected_class is null), a class not in the
taxonomy, or an ambiguous multi-class card (rascals). Those need a taxonomy extension
before they can train; their labelled crops stay in assets/board_labeled/ for then.

Output: training/datasets/cr_real_duncan/{images,labels}/{train,val} and
training/datasets/cr_combined_v5.yaml (v4's train/val + this).

Usage:
    python training/prepare_board_data.py
    python training/prepare_board_data.py --val-frac 0.15
"""

import argparse
import glob
import json
import os
import random
import shutil

import yaml

from game_knowledge.play_localizer import detected_classes

LABELED_DIR = "assets/board_labeled"
V4_YAML = "training/datasets/cr_combined_v4.yaml"
OUT = "training/datasets/cr_real_duncan"
OUT_YAML = "training/datasets/cr_combined_v5.yaml"
DEFAULT_VAL_FRAC = 0.1


def _name_to_idx(yaml_path):
    """The taxonomy's name -> index map, from the current model's combined yaml."""
    names = yaml.safe_load(open(yaml_path, encoding="utf-8"))["names"]
    if isinstance(names, dict):
        return {v: int(k) for k, v in names.items()}
    return {n: i for i, n in enumerate(names)}


def _class_idx_for_card(card, name_to_idx):
    """The single taxonomy class index this card's units train as, or None if the
    card can't map cleanly (gap / not-in-taxonomy / ambiguous multi-class)."""
    classes = detected_classes(card)
    if not classes:
        return None                              # gap card (no detector class)
    hits = [name_to_idx[c.replace("_", "-")] for c in classes if c.replace("_", "-") in name_to_idx]
    hits = sorted(set(hits))
    return hits[0] if len(hits) == 1 else None   # ambiguous multi-class -> hold aside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    args = ap.parse_args()

    if not os.path.isfile(V4_YAML):
        raise SystemExit(f"{V4_YAML} not found -- run prepare_real_data.py first (it writes the "
                         "taxonomy the current weights use).")
    if not os.path.isdir(LABELED_DIR):
        raise SystemExit(f"No labelled crops at {LABELED_DIR} -- review some first "
                         "(python -m perception.review_board_queue).")
    name_to_idx = _name_to_idx(V4_YAML)

    # gather (image, class_idx, box) or hold aside
    samples, held = [], {}
    for card in sorted(os.listdir(LABELED_DIR)):
        cdir = os.path.join(LABELED_DIR, card)
        if not os.path.isdir(cdir):
            continue
        idx = _class_idx_for_card(card, name_to_idx)
        pngs = [p for p in glob.glob(os.path.join(cdir, "*.png"))]
        if idx is None:
            held[card] = len(pngs)
            continue
        for png in pngs:
            js = png[:-4] + ".json"
            box = None
            if os.path.isfile(js):
                box = json.load(open(js, encoding="utf-8")).get("crop_box")
            if not box:
                box = [0.5, 0.5, 0.9, 0.9]       # fallback: unit ~fills the crop
            samples.append((png, idx, box))

    if not samples:
        print("No trainable labelled crops yet.")
        if held:
            print("Held aside (need a taxonomy extension):",
                  ", ".join(f"{c} x{n}" for c, n in sorted(held.items())))
        return

    random.seed(0)
    random.shuffle(samples)
    n_val = int(len(samples) * args.val_frac)
    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)

    for i, (png, idx, box) in enumerate(samples):
        split = "val" if i < n_val else "train"
        stem = f"duncan_{i:06d}"
        shutil.copy(png, os.path.join(OUT, "images", split, stem + ".jpg"))
        cx, cy, bw, bh = box
        with open(os.path.join(OUT, "labels", split, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write(f"{idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    v4 = yaml.safe_load(open(V4_YAML, encoding="utf-8"))
    combined = {
        "train": list(v4["train"]) + [os.path.abspath(os.path.join(OUT, "images/train"))],
        "val": list(v4["val"]) + [os.path.abspath(os.path.join(OUT, "images/val"))],
        "names": v4["names"],
    }
    with open(OUT_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(combined, f, sort_keys=False, allow_unicode=True)

    by_class = {}
    for _, idx, _ in samples:
        by_class[idx] = by_class.get(idx, 0) + 1
    print(f"{len(samples)} trainable crops ({len(samples)-n_val} train / {n_val} val) across "
          f"{len(by_class)} classes -> {OUT}")
    if held:
        print("Held aside (no detector class yet -- taxonomy extension needed):",
              ", ".join(f"{c} x{n}" for c, n in sorted(held.items())))
    print(f"Combined dataset yaml: {OUT_YAML}\n"
          f"Retrain: python training/finetune.py --data {OUT_YAML} --name cr_finetune_v5")


if __name__ == "__main__":
    main()
