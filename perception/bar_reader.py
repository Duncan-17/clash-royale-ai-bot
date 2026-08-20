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

NOT built here yet, deliberately: reading HP FILL PERCENTAGE inside the bar. The
model gives the bar's bounding box, not its fill, so that needs the same
pixel-fill technique perception/tower_reader.py uses — and, as that module's
history proves three times over, an EMPTY-colour value guessed from a full-health
capture is silently wrong. That needs real captures of damaged units first.
Unit HP is a long-standing planned feature (Project_Status, "Unit HP reading —
planned 2026-08-15, not started"); this module is the localisation half of it.
"""

# Measured from real detections, not assumed -- see module docstring. Kept
# deliberately tight: a loose match is what produces the overlapping-Knights
# failure, and an unassociated unit (side falls back to the model's own, weaker
# per-unit reading) is much cheaper than a confidently WRONG side.
MAX_CENTER_DX_PX = 30      # bar centre vs unit centre, horizontally
MIN_BOTTOM_OFFSET_PX = -20  # bar bottom relative to unit top: negative = bar sits above
MAX_BOTTOM_OFFSET_PX = 25   # positive = bar overlaps slightly into the unit box

BAR_CLASSES = ("bar", "bar-level")


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
    """Given one frame's detections from perception.katacr_detector.KataCRDetector
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


def _is_non_unit(det: dict) -> bool:
    from perception.katacr_detector import NON_UNIT_CLASSES
    return det.get("raw_name") in NON_UNIT_CLASSES


def apply_bar_sides(detections: list) -> list:
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
        else:
            unit["side_source"] = "unit"
        out.append(unit)
    return out


if __name__ == "__main__":
    import glob

    import cv2

    from perception.katacr_detector import KataCRDetector

    det = KataCRDetector(include_non_units=True)
    frames = sorted(glob.glob("temp_img/side_conflicts/*.png"))[-14:]
    agree = disagree = unassoc = 0
    print("bar-vs-unit side, per associated pair (disagreements are the interesting ones --")
    print("the bar is the more trustworthy of the two, see module docstring):\n")
    for p in frames:
        dets = det.detect(cv2.imread(p))
        assoc = associate_bars(dets)
        for ui, bar in assoc.items():
            u = dets[ui]
            same = u["side"] == bar["side"]
            agree, disagree = (agree + 1, disagree) if same else (agree, disagree + 1)
            if not same:
                print(f"  {p.split(chr(92))[-1][-18:]}  {u['name']:16s} "
                      f"unit={u['side']:6s} bar={bar['side']:6s} ({bar['raw_name']})")
        unassoc += sum(1 for i, d in enumerate(dets)
                       if d.get("raw_name") not in BAR_CLASSES and not _is_non_unit(d)
                       and i not in assoc)
    total = agree + disagree
    print(f"\nassociated pairs: {total}  agree: {agree}  disagree: {disagree}"
          f"  ({agree / total * 100:.0f}% agreement)" if total else "\nno pairs")
    print(f"units with NO bar (expected -- undamaged ally troops show none): {unassoc}")
