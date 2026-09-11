"""
Associates the detector's `bar` / `bar-level` detections with the units they
belong to, and uses the BAR's own side as the authoritative team reading.

Why this exists (2026-08-20, Duncan's chosen next step: "set up the yolo to start
seeing and processing lvl indicators and healthbars since they should be
extremely consistent once learned and easy to train"):

A troop sprite carries NO team signal — Clash Royale doesn't skin-tint units by
side. That's the fact that made `side.onnx` unfixable and is documented at length
in Project_Status. What DOES carry team colour is the level badge and health bar
rendered above a unit. `perception/side_reader.py` already exploits this, but it
has to FIND those elements itself: a per-unit colour-blob search with hand-tuned
thresholds, shape filters, and a digit-template crosscheck, all of which took
several rounds of live debugging (a Giant's orange hair and a Knight's boot both
had to be explicitly excluded).

perception/katacr_detector.py's model detects `bar` and `bar-level` as first-class
objects, so that entire search collapses into "which detected bar belongs to which
detected unit". This module answers only that question. It deliberately does NOT
re-derive side from pixels — the model already emits a belonging channel per
detection, so an associated bar's OWN side is what gets propagated.

Why the bar's side beats the unit's side, which is the whole point:
the model predicts belonging for a `giant` box from a sprite with no team colour
in it (the same impossible task side.onnx failed at), but predicts it for a `bar`
box from something that is literally colour-coded by team. Same model, same
channel, categorically different evidence quality.

Association geometry, measured from real detections rather than assumed (see the
table this module's __main__ prints): a bar's horizontal centre sits within a few
px of its unit's, and its BOTTOM edge lands within roughly ±15px of the unit's TOP
edge. Width is not a usable signal — `bar-level` (a small badge) and `bar` (a wide
health bar) differ by 5x relative to the same unit.

The hard case, confirmed with Duncan against a real frame: an ally and an enemy
Knight overlapping, with the enemy's bar rendered above BOTH. A naive
"nearest unit below the bar" rule assigns that bar to the ally Knight and flips a
correct side to a wrong one. Association is therefore ONE-TO-ONE and greedy by
best score — one bar can claim only one unit — and anything outside the measured
geometry is left unassociated rather than forced onto a guess.

HP FILL reading (added 2026-08-20, once Duncan captured a frame with both an ally
and an enemy Giant at visibly-not-full health — "mostly full" is exactly the right
capture for this, since a full bar has no empty section to measure and is precisely
how tower_reader.py's EMPTY colours were mis-derived three separate times).

Colours were measured from that real capture's own fill→empty transitions, and
produced a genuinely useful cross-check: they are essentially IDENTICAL to the
independently-derived tower-bar colours (ally filled 251,204,112 vs tower
252,207,114; enemy empty 73,48,93 vs tower 73,49,92). The game uses one UI palette
for both, so this independently validates the tower calibration that was corrected
separately — two different measurement sessions on two different UI elements
converging on the same values is much stronger evidence than either alone.

This completes the localisation + measurement halves of the long-planned "Unit HP
reading" feature (Project_Status, "planned 2026-08-15, not started").
"""

# Measured from real detections, not assumed -- see module docstring. Kept
# deliberately tight: a loose match is what produces the overlapping-Knights
# failure, and an unassociated unit (side falls back to the model's own, weaker
# per-unit reading) is much cheaper than a confidently WRONG side.
MAX_CENTER_DX_PX = 30      # bar centre vs unit centre, horizontally
MIN_BOTTOM_OFFSET_PX = -20  # bar bottom relative to unit top: negative = bar sits above
MAX_BOTTOM_OFFSET_PX = 25   # positive = bar overlaps slightly into the unit box

BAR_CLASSES = ("bar", "bar-level")

# Non-unit detector classes (UI/components/towers) -- everything that is NOT a
# troop/building the tracker should see. Inlined here 2026-09-10 when the legacy
# katacr_detector.py was retired (Task 2); this used to import that module's
# NON_UNIT_CLASSES. bar/bar-level are handled separately (BAR_CLASSES) since we
# READ them for side/hp, so they're not in this "ignore" set's job here.
NON_UNIT_CLASSES = {
    "king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
    "dagger-duchess-tower-bar", "tower-bar", "king-tower-bar", "bar",
    "bar-level", "clock", "emote", "text", "elixir", "selected",
    "skeleton-king-bar", "evolution-symbol", "ice-spirit-evolution-symbol",
    "padding_belong",
}

# Measured 2026-08-20 from assets/tower_reader_calibration/unit_bars_both_giants.png --
# a real capture with an ally Giant at ~92% and an enemy Giant at ~84%. See module
# docstring for why a NOT-full bar is the only capture that can produce a correct EMPTY
# value, and for the cross-check against tower_reader.py's independently-derived palette.
UNIT_BAR_FILLED_BGR = {"ally": (251.0, 204.0, 112.0), "enemy": (90.0, 35.0, 219.0)}
UNIT_BAR_EMPTY_BGR = {"ally": (120.0, 80.0, 62.0), "enemy": (73.0, 48.0, 93.0)}
BAR_CLEAN_MAX_DIST = 60      # same tolerance tower_reader.py uses, same reasoning
BAR_MIN_CLEAN_ROW_FRACTION = 0.5


def read_bar_fill(screenshot, bar_bbox, side):
    """Returns the bar's fill fraction 0.0-1.0, or None if no clean bar-coloured
    row could be found (occlusion, mid-animation, a bad box).

    Same technique as perception/tower_reader.py, and deliberately so: pick the
    row with the highest fraction of pixels close to EITHER the filled or the
    empty reference colour, then measure what share of that row's clean pixels
    are filled. Selecting on "close to one of the two SPECIFIC bar colours"
    rather than "roughly the right hue" is the exact lesson tower_reader.py
    learned the hard way -- a looser hue test happily picks a dark border row
    and reports 0% on a healthy bar."""
    import numpy as np
    x1, y1, x2, y2 = bar_bbox
    region = screenshot[max(0, y1):y2, max(0, x1):x2]
    if region.size == 0 or side not in UNIT_BAR_FILLED_BGR:
        return None
    filled = np.array(UNIT_BAR_FILLED_BGR[side])
    empty = np.array(UNIT_BAR_EMPTY_BGR[side])

    best_row, best_frac = None, -1.0
    for i in range(region.shape[0]):
        row = region[i].astype(np.float32)
        d_f = np.abs(row - filled).sum(axis=1)
        d_e = np.abs(row - empty).sum(axis=1)
        clean = (d_f < BAR_CLEAN_MAX_DIST) | (d_e < BAR_CLEAN_MAX_DIST)
        frac = float(clean.mean())
        if frac > best_frac:
            best_row, best_frac = i, frac
    if best_frac < BAR_MIN_CLEAN_ROW_FRACTION:
        return None

    row = region[best_row].astype(np.float32)
    d_f = np.abs(row - filled).sum(axis=1)
    d_e = np.abs(row - empty).sum(axis=1)
    clean = (d_f < BAR_CLEAN_MAX_DIST) | (d_e < BAR_CLEAN_MAX_DIST)
    if not clean.any():
        return None
    return float((d_f[clean] < BAR_CLEAN_MAX_DIST).mean())


def _center_x(bbox):
    return (bbox[0] + bbox[2]) / 2


def _association_score(bar_bbox, unit_bbox):
    """Lower is better; None if the pair falls outside the measured geometry.
    Scores on horizontal centre alignment plus vertical offset from the unit's
    top edge -- the two signals that actually separated correct from incorrect
    pairs in real data. Bar WIDTH is deliberately ignored: `bar-level` and `bar`
    differ ~5x relative to the same unit, so it carries no usable signal."""
    dx = abs(_center_x(bar_bbox) - _center_x(unit_bbox))
    if dx > MAX_CENTER_DX_PX:
        return None
    dy = bar_bbox[3] - unit_bbox[1]  # bar bottom vs unit top
    if not (MIN_BOTTOM_OFFSET_PX <= dy <= MAX_BOTTOM_OFFSET_PX):
        return None
    return dx + abs(dy)


def associate_bars(detections: list) -> dict:
    """Given one frame's detections from perception.yolo_detector.YoloDetector
    (run with include_non_units=True so bars are present), returns
    {unit_index: bar_detection} for every unit a bar could be confidently matched
    to. Indices are positions in `detections`.

    One-to-one and greedy by best score: each bar claims at most one unit and each
    unit accepts at most one bar. That matters for the real failure case -- two
    overlapping units of the same class where only ONE has a visible bar. Forcing
    the spare unit onto the same bar is exactly how a correct side becomes a wrong
    one (confirmed with Duncan on a real ally/enemy Knight pair)."""
    units = [i for i, d in enumerate(detections) if d.get("raw_name") not in BAR_CLASSES
             and not _is_non_unit(d)]
    bars = [i for i, d in enumerate(detections) if d.get("raw_name") in BAR_CLASSES]

    scored = []
    for bi in bars:
        for ui in units:
            s = _association_score(detections[bi]["bbox"], detections[ui]["bbox"])
            if s is not None:
                scored.append((s, bi, ui))
    scored.sort(key=lambda t: t[0])

    used_bars, used_units, out = set(), set(), {}
    for _s, bi, ui in scored:
        if bi in used_bars or ui in used_units:
            continue
        used_bars.add(bi)
        used_units.add(ui)
        out[ui] = detections[bi]
    return out


def associate_level_badges(units: list, level_dets: list) -> dict:
    """Bind detected `bar-level` badges to units, one-to-one greedy by best score
    (same geometry + overlapping-units safety as associate_bars). Returns
    {unit_index: level_det}, indices into `units`.

    Separate from associate_bars deliberately: a unit can carry BOTH its wide
    health `bar` (for HP + side) AND its small `bar-level` badge (for the level
    digit), but associate_bars lumps the two classes into one pool and one-to-one
    matching would hand a unit only one of them. This associates the level badges
    on their own so the level read is independent of the HP read. `units` are unit
    detections (each with a `bbox`); `level_dets` are the raw `bar-level` dets."""
    scored = []
    for li, ld in enumerate(level_dets):
        for ui, u in enumerate(units):
            s = _association_score(ld["bbox"], u["bbox"])
            if s is not None:
                scored.append((s, li, ui))
    scored.sort(key=lambda t: t[0])

    used_l, used_u, out = set(), set(), {}
    for _s, li, ui in scored:
        if li in used_l or ui in used_u:
            continue
        used_l.add(li)
        used_u.add(ui)
        out[ui] = level_dets[li]
    return out


def _is_non_unit(det: dict) -> bool:
    return det.get("raw_name") in NON_UNIT_CLASSES


def side_from_bar_color(screenshot, bbox):
    """Ally vs enemy from a bar/bar-level box's COLOUR (blue=ally, red=enemy).
    For detectors WITHOUT a belonging channel (our fine-tuned YOLOv8, which
    detects the bar as an object but not its team). Counts pixels closer to each
    team's filled bar colour; returns 'ally', 'enemy', or None if the box has no
    clean bar-coloured pixels. Same reference colours read_bar_fill uses."""
    import numpy as np
    x1, y1, x2, y2 = bbox
    region = screenshot[max(0, y1):y2, max(0, x1):x2].reshape(-1, 3).astype(float)
    if region.size == 0:
        return None
    d_ally = np.linalg.norm(region - np.array(UNIT_BAR_FILLED_BGR["ally"]), axis=1)
    d_enemy = np.linalg.norm(region - np.array(UNIT_BAR_FILLED_BGR["enemy"]), axis=1)
    ally = int(((d_ally < d_enemy) & (d_ally < BAR_CLEAN_MAX_DIST)).sum())
    enemy = int(((d_enemy < d_ally) & (d_enemy < BAR_CLEAN_MAX_DIST)).sum())
    if ally == 0 and enemy == 0:
        return None
    return "ally" if ally >= enemy else "enemy"


def apply_bar_sides(detections: list, screenshot=None) -> list:
    """Returns a NEW list of unit detections (bars/UI classes dropped) with each
    unit's `side` replaced by its associated bar's side where one was found, plus
    two extra fields:
      - `side_source`: "bar" if it came from an associated bar, else "unit"
      - `bar_bbox`: the associated bar's box, or None -- this is what a future
        HP-fill reader will measure inside (see module docstring)

    Units with no associated bar keep the model's own per-unit side. That's the
    correct fallback rather than "unknown": a fresh, undamaged ally troop shows no
    bar at all by design (the asymmetry documented in side_reader.py), so absence
    of a bar is normal and expected, not a failure."""
    assoc = associate_bars(detections)
    out = []
    for i, d in enumerate(detections):
        if d.get("raw_name") in BAR_CLASSES or _is_non_unit(d):
            continue
        bar = assoc.get(i)
        unit = dict(d)
        unit["bar_bbox"] = bar["bbox"] if bar else None
        if bar is not None:
            unit["side"] = bar["side"]
            unit["side_source"] = "bar"
            unit["hp_fraction"] = (read_bar_fill(screenshot, bar["bbox"], bar["side"])
                                    if screenshot is not None and bar["raw_name"] == "bar"
                                    else None)
        else:
            unit["side_source"] = "unit"
            unit["hp_fraction"] = None
        out.append(unit)
    return out


if __name__ == "__main__":
    import sys

    import cv2

    from perception.yolo_detector import YoloDetector

    path = sys.argv[1] if len(sys.argv) > 1 else "test_screenshot.png"
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"{path} not found.")
    # YoloDetector already runs apply_bar_sides internally, so its units carry the
    # associated bar's side + hp_fraction. Print them to eyeball the association.
    units = [d for d in YoloDetector(include_non_units=False).detect(img)]
    print(f"{len(units)} units on {path} (side / side_source / hp / level):")
    for u in sorted(units, key=lambda d: -d["confidence"]):
        hp = "" if u.get("hp_fraction") is None else f"{u['hp_fraction']*100:.0f}%"
        print(f"  {u['name']:16s} {u.get('side','?'):7s} {u.get('side_source','?'):8s} "
              f"{hp:>5s}  lvl={u.get('level')}  bbox={u['bbox']}")
