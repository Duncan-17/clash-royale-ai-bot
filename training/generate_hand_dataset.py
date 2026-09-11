"""Synthetic dataset generator for the HAND-CARD classifier (YOLOv8-cls).

In-hand a card looks like its CURRENT card art plus an elixir gem bottom-middle,
just dimmer/more muted than the vivid store icon and cropped tighter (Duncan,
2026-09-11). So we can synthesize training crops exactly from the official card
art (assets/card_icons/, fetched from the CR API), and let AUGMENTATION cover the
tray dimness, the TV-Royale cross-view tilt, zoom/framing, and compression.

The elixir badge is deliberately NOT synthesized: it sits bottom-middle, is the
same for every card of a cost, and is non-discriminative. Instead we FOLD IN the
REAL in-hand crops we have (assets/hand_templates/, and KataCR's card_classification
set) -- those carry real badges -- so the classifier learns the art is the signal
and the badge region is noise. Real crops also correct any reworked-art card whose
API icon is stale (e.g. musketeer).

Output = ImageNet-style folders (what ultralytics classify wants):
    <out>/train/<card_slug>/*.jpg
    <out>/val/<card_slug>/*.jpg

    python training/generate_hand_dataset.py --n 90 --out training/datasets/hand_cls
    python training/generate_hand_dataset.py --sample   # montage for a few cards, no full gen
"""

import argparse
import glob
import os
import random

import cv2
import numpy as np

ICON_DIR = "assets/card_icons"
HAND_TEMPLATES = "assets/hand_templates"
# The two card_icons files with the awkward API spelling -> the project's readable
# slug (mirrors perception.card_detector.ICON_NAME_ALIASES). Inlined so this
# training script has no perception dependency.
ICON_NAME_ALIASES = {"p_e_k_k_a": "pekka", "mini_p_e_k_k_a": "mini_pekka"}
KATACR_CLS = "vendor/cr-detection-dataset/images/card_classification"
IMGSZ = 128            # classifier input; slot crops are ~104-155px
SLOT_ASPECT = 0.80     # w/h of a hand slot (TV Royale ~0.78-0.80, live ~0.82)
BG_GRAY = 90           # tray is darker than the store icon's white frame
VAL_FRAC = 0.15
LOADING_PROB = 0.4     # fraction of crops rendered in the "loading"/not-affordable
# state -- a pure-greyscale card under a lighter translucent film that retreats
# clockwise as elixir fills, ending on the plain grey card (Duncan 2026-09-11: the
# v1 classifier nailed fully-loaded cards but misread these loading ones).


def _slug_from_katacr(name):
    return name.replace("-", "_")


def load_art(path):
    """RGBA/again icon -> BGR art composited over a dark tray grey, cropped to the
    art's bbox (drops the transparent framed border)."""
    icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if icon is None:
        return None
    if icon.ndim == 3 and icon.shape[2] == 4:
        a = icon[:, :, 3:4] / 255.0
        rgb = icon[:, :, :3].astype(np.float32)
        art = (rgb * a + BG_GRAY * (1 - a)).astype(np.uint8)
        ys, xs = np.where(icon[:, :, 3] > 10)
        if len(ys):
            art = art[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return art
    return icon[:, :, :3] if icon.ndim == 3 else cv2.cvtColor(icon, cv2.COLOR_GRAY2BGR)


def _aspect_crop(img, aspect, zoom):
    """Center-crop `img` to width/height == aspect, then zoom in by `zoom`
    (>1 = tighter), staying inside the image."""
    h, w = img.shape[:2]
    cw, ch = w, h
    if w / h > aspect:      # too wide -> limit width
        cw = int(h * aspect)
    else:                    # too tall -> limit height
        ch = int(w / aspect)
    cw, ch = int(cw / zoom), int(ch / zoom)
    cw, ch = max(8, min(cw, w)), max(8, min(ch, h))
    x0 = (w - cw) // 2 + random.randint(-w // 20, w // 20)
    y0 = (h - ch) // 2 + random.randint(-h // 20, h // 20)
    x0, y0 = max(0, min(x0, w - cw)), max(0, min(y0, h - ch))
    return img[y0:y0 + ch, x0:x0 + cw]


def _perspective(img, amt):
    """Mild trapezoid warp to mimic TV Royale's cross-view tilt."""
    h, w = img.shape[:2]
    dx = int(w * amt)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[dx, 0], [w - dx, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _greyed(img):
    """Fully black-and-white version of the crop -- the in-game 'card not ready'
    card is pure greyscale (not a desaturated colour). Slight brightness jitter."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = np.clip(g * random.uniform(0.95, 1.15), 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _loading_state(img, fill=None):
    """Greyed card + the clockwise loading screen: a LIGHTER overlay sits over the
    not-yet-loaded portion and retreats CLOCKWISE as elixir fills, leaving the plain
    greyed card behind (Duncan 2026-09-11: lighter screen, removes clockwise, ends
    on the base greyed card). fill 0->1 = how much has loaded (None = random)."""
    base = _greyed(img)
    h, w = base.shape[:2]
    fill = random.uniform(0.0, 1.0) if fill is None else fill
    ys, xs = np.mgrid[0:h, 0:w]
    ang = (np.degrees(np.arctan2(xs - w / 2.0, -(ys - h / 2.0))) + 360) % 360  # 0=top, clockwise
    not_loaded = ang > fill * 360                         # still under the lighter screen
    # A translucent light-grey FILM (blend toward light), not a pure brightness
    # boost -- reads as a thicker/denser screen over the card (Duncan 2026-09-11).
    film = np.full_like(base, 235, dtype=np.float32)
    alpha = 0.6                                            # brighter + harder to see through
    lighter = np.clip(base.astype(np.float32) * (1 - alpha) + film * alpha, 0, 255).astype(np.uint8)
    return np.where(not_loaded[..., None], lighter, base)


def augment(art):
    """One randomly-augmented slot-sized BGR crop from a card's art."""
    img = _aspect_crop(art, SLOT_ASPECT * random.uniform(0.95, 1.05),
                       zoom=random.uniform(1.0, 1.25))
    img = cv2.resize(img, (IMGSZ, IMGSZ), interpolation=cv2.INTER_AREA)
    # dim + desaturate (tray is muted vs the vivid icon)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= random.uniform(0.6, 0.95)                 # desaturate
    hsv[:, :, 2] *= random.uniform(0.7, 1.05)                 # dim/brighten
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if random.random() < 0.6:
        img = _perspective(img, random.uniform(0.0, 0.10))
    if random.random() < 0.4:
        M = cv2.getRotationMatrix2D((IMGSZ / 2, IMGSZ / 2), random.uniform(-6, 6), 1.0)
        img = cv2.warpAffine(img, M, (IMGSZ, IMGSZ), borderMode=cv2.BORDER_REPLICATE)
    if random.random() < LOADING_PROB:   # greyed loading state (added last, axis-aligned)
        img = _loading_state(img)
    if random.random() < 0.3:
        k = random.choice([3, 3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    q = random.randint(55, 92)                                 # jpeg artifacts
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


def real_crops_for(slug):
    """Real in-hand crops we already have for `slug` (ours + KataCR), resized to
    the classifier size. These carry real badges + real tray rendering."""
    out = []
    for p in glob.glob(os.path.join(HAND_TEMPLATES, slug, "*.png")):
        if "catalog_seed" in p:
            continue
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is not None:
            out.append(cv2.resize(im, (IMGSZ, IMGSZ), interpolation=cv2.INTER_AREA))
    kc = os.path.join(KATACR_CLS, slug.replace("_", "-"))
    if os.path.isdir(kc):
        for p in glob.glob(os.path.join(kc, "*")):
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is not None:
                out.append(cv2.resize(im, (IMGSZ, IMGSZ), interpolation=cv2.INTER_AREA))
    return out


def _save(img, out, split, slug, i):
    d = os.path.join(out, split, slug)
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, f"{slug}_{split}_{i:04d}.jpg"), img)


def generate(out, n_per_card):
    icons = sorted(glob.glob(os.path.join(ICON_DIR, "*.png")))
    print(f"{len(icons)} card icons -> {out}  ({n_per_card} synthetic/card + real crops)")
    total_r = 0
    for path in icons:
        # canonicalize the icon filename to the project's slug (p_e_k_k_a -> pekka,
        # mini_p_e_k_k_a -> mini_pekka) so the CLASS name matches everything else AND
        # real_crops_for() finds the real crops in assets/hand_templates/<slug>/.
        raw = os.path.splitext(os.path.basename(path))[0]
        slug = ICON_NAME_ALIASES.get(raw, raw)
        art = load_art(path)
        if art is None:
            print(f"  skip {slug} (unreadable icon)")
            continue
        # synthetic
        for i in range(n_per_card):
            split = "val" if i < int(n_per_card * VAL_FRAC) else "train"
            _save(augment(art), out, split, slug, i)
        # real crops (both classes get a val slice)
        reals = real_crops_for(slug)
        total_r += len(reals)
        for j, im in enumerate(reals):
            split = "val" if j % 6 == 0 else "train"
            _save(im, out, split, slug, 1000 + j)
    print(f"done. folded in {total_r} real crops across cards.")
    print(f"train: python training/finetune_hand_cls.py --data {out}")


def sample(out_png="temp_img/hand_synth_sample.png"):
    """Montage: for a few cards, a row of synthetic variants + a REAL crop, to eyeball
    zoom / dimness / framing before committing to a full gen + train."""
    import subprocess
    cards = ["knight", "musketeer", "giant", "goblins"]
    rows = []
    for slug in cards:
        art = load_art(os.path.join(ICON_DIR, f"{slug}.png"))
        if art is None:
            continue
        base = cv2.resize(_aspect_crop(art, SLOT_ASPECT, 1.1), (IMGSZ, IMGSZ), interpolation=cv2.INTER_AREA)
        # 2 colour + loading progression (fill 15/40/65/90% -- watch the lighter screen retreat clockwise)
        variants = [augment(art) for _ in range(2)] + \
            [_loading_state(base.copy(), fill=f) for f in (0.0, 0.33, 0.66, 1.0)]
        reals = real_crops_for(slug)
        real = reals[0] if reals else np.full((IMGSZ, IMGSZ, 3), 60, np.uint8)
        tiles = variants + [np.full((IMGSZ, 6, 3), 0, np.uint8), real]
        # label the real one
        strip = np.full((18, IMGSZ, 3), 25, np.uint8)
        cv2.putText(strip, "REAL", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 220, 90), 1)
        tiles[-1] = np.vstack([strip, real]) if real.shape[0] == IMGSZ else real
        h = max(t.shape[0] for t in tiles)
        tiles = [t if t.shape[0] == h else np.vstack([t, np.full((h - t.shape[0], t.shape[1], 3), 25, np.uint8)]) for t in tiles]
        lab = np.full((h, 90, 3), 25, np.uint8)
        cv2.putText(lab, slug[:11], (3, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
        rows.append(np.hstack([lab] + tiles))
    w = max(r.shape[1] for r in rows)
    rows = [r if r.shape[1] == w else np.hstack([r, np.full((r.shape[0], w - r.shape[1], 3), 25, np.uint8)]) for r in rows]
    montage = np.vstack([np.vstack([r, np.full((6, w, 3), 0, np.uint8)]) for r in rows])
    os.makedirs("temp_img", exist_ok=True)
    cv2.imwrite(out_png, montage)
    print(f"wrote {out_png}  (left label | 5 synthetic variants | REAL crop)")
    subprocess.Popen(["mspaint.exe", out_png.replace("/", "\\")])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="training/datasets/hand_cls")
    p.add_argument("--n", type=int, default=90, help="synthetic crops per card")
    p.add_argument("--sample", action="store_true", help="montage a few cards, no full gen")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    random.seed(args.seed)
    if args.sample:
        sample()
    else:
        generate(args.out, args.n)


if __name__ == "__main__":
    main()
