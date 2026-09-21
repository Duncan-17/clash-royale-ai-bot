"""BUILD A -- Play<->Unit binding for TV Royale ("loose play localization").

When a play fires (side S, card C, ~time T -- GROUND TRUTH from the dual cycle
tracker) this binds it to the board unit that just APPEARED and STAMPS that unit
with card=C + side=S, so card / side / placement come from the play instead of the
flickering detector. It fixes the two board gaps the 2026-09-12 handoff flagged:

  #1 SHARED-CARD SIDE -- an enemy dart_goblin at the bridge read friendly. BOTH decks
     run dart_goblin, so no per-CARD prior can say whose it is; only linking THIS play
     to THIS new unit ("the one that just left the ENEMY hand") disambiguates.
  #2 CLASS FLICKER -- a friendly berserker read as baby_dragon/bandit. The cycle
     tracker KNOWS berserker was played; the bound unit is stamped 'berserker' and it
     sticks over the detector's wobble.

How it differs from game_knowledge/play_tracker.py (the LIVE binder): there, OWN plays
carry a known placement position and the OPPONENT's plays are *inferred* from new enemy
tracks. Here BOTH players' plays are already ground truth (card+side+time) and we have
NO placement -- localizing it is the whole point -- so binding is APPEARANCE-driven (the
newest unit on the play's side within a time window), not proximity-to-a-known-point.

Reuses perception/thin_tracker.ThinTracker for stable IDs + class/side/level debounce on
the TV Royale board (same 900x1600 arena). This layer adds birth tracking, the binding,
and the authoritative stamp. Card metadata (game_knowledge/card_reference.json): only
TROOP/BUILDING plays spawn a trackable unit, so only those are bound (spells, incl.
goblin_barrel/graveyard, abstain). A card's detected_class(es) are a MATCH BONUS, never
required -- the 7 detector-gap troops (berserker, ronin, goblin_giant, ...) have none,
and #2 is exactly the case where the detector's class is wrong, so binding leans on
side-half + timing + uniqueness and CONFIDENCE-GATES: no clear single new unit -> abstain
(never a guess), matching the rest of the learning pipeline.
"""

import functools
import json
import os

from perception.thin_tracker import ThinTracker

_REF_PATH = os.path.join(os.path.dirname(__file__), "card_reference.json")

BIND_WINDOW = 3.0     # s: how far BACK a unit's birth counts as "just appeared" for a play
BIND_FORWARD = 1.5    # s: keep trying to bind a play this long if no unit had appeared yet
HALF_MARGIN = 0.15    # normalized y slack around the river midline (0.5) -- bridge plays sit
# near the middle, so a top(enemy) play accepts a birth up to 0.5+margin and a
# bottom(friendly) play down to 0.5-margin. Wider than a hard half so bridge units bind;
# still rejects a unit clearly in the opponent's half (miner/drill deep placements abstain).


@functools.lru_cache(maxsize=1)
def _card_ref():
    with open(_REF_PATH, encoding="utf-8") as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def _ref(card):
    """card_reference entry, hyphen-tolerant: keys are underscore (x_bow) but a deck/
    hand slug can be hyphenated (x-bow) -- try the given slug, then its underscore form."""
    r = _card_ref()
    return r.get(card) or r.get((card or "").replace("-", "_"))


def card_category(card):
    """'troop' | 'building' | 'spell' | None, from card_reference.json."""
    e = _ref(card)
    return e.get("category") if e else None


def detected_classes(card):
    """The detector class name(s) this card shows up as -- a set (empty for spells and
    the detector-gap troops, whose detected_class is null)."""
    e = _ref(card)
    dc = e.get("detected_class") if e else None
    return set(dc) if dc else set()


def spawns_trackable_unit(card):
    """True if playing this card puts a trackable board unit down (troop/building).
    Spells (incl. goblin_barrel/graveyard, whose spawns are transient swarms we don't
    try to localize) and unknown cards -> False, so the localizer abstains on them."""
    if not card or str(card).startswith("unknown_"):
        return False
    return card_category(card) in ("troop", "building")


def _class_match(unit_name, expected):
    """Loose match of a detector unit class to a card's expected detected_class set
    (taxonomies differ slightly: goblins->goblin, three_musketeers->musketeer)."""
    if not unit_name or not expected:
        return False
    if unit_name in expected:
        return True
    a = unit_name.replace("_", "")
    for e in expected:
        b = e.replace("_", "")
        if a == b or a.rstrip("s") == b.rstrip("s") or a in b or b in a:
            return True
    return False


def _center_norm(bbox, w, h):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h


def _on_half(cy, det_side, margin):
    """Is a birth at normalized y=cy on det_side's half of the arena (with slack)?
    det_side 'enemy' = top (small y), 'ally' = bottom (large y)."""
    if det_side == "enemy":
        return cy <= 0.5 + margin
    return cy >= 0.5 - margin


class PlayLocalizer:
    """Owns a board ThinTracker + the play->unit bindings for one TV Royale game.

    Per frame: update(dets, now, w, h). On each play: bind(play, now). For the board to
    log/draw: resolved_units() -- the tracker's current units with bound ones stamped.
    Build a fresh one per game (or call reset())."""

    def __init__(self, bind_window=BIND_WINDOW, bind_forward=BIND_FORWARD,
                 half_margin=HALF_MARGIN):
        self.bind_window = bind_window
        self.bind_forward = bind_forward
        self.half_margin = half_margin
        self.reset()

    def reset(self):
        self._tracker = ThinTracker()
        self._births = {}     # track_id -> {"t", "cx", "cy"}
        self._extra = {}      # track_id -> {"struggle", "off_deck", "side_source"} (latest frame)
        self._bindings = {}   # track_id -> {"card", "det_side", "play_t"}
        self._pending = []    # bindable plays awaiting a unit: {"play", "t0"}
        self._tracks = []     # last update()'s raw ThinTracker dicts
        self._wh = (None, None)

    # -- per frame ----------------------------------------------------------
    def update(self, dets, now, w, h):
        """Advance the tracker with this frame's board units (already resolve_sides'd),
        record each track's birth (time + normalized centre), carry the per-frame
        struggle/off_deck/side_source, and retry any deferred plays that were waiting
        for their unit to appear."""
        self._wh = (w, h)
        self._tracks = self._tracker.update(dets)
        # ThinTracker sets track.bbox to its matched detection's bbox verbatim, so an
        # exact-bbox lookup recovers that detection's extra fields for this track.
        det_by_bbox = {tuple(d["bbox"]): d for d in dets}
        for t in self._tracks:
            tid = t["id"]
            d = det_by_bbox.get(tuple(t["bbox"]), {})
            self._extra[tid] = {
                "struggle": bool(d.get("struggle", False)),
                "off_deck": bool(d.get("off_deck", False)),
                "side_source": d.get("side_source"),
            }
            if tid not in self._births:
                cx, cy = _center_norm(t["bbox"], w, h)
                self._births[tid] = {"t": now, "cx": cx, "cy": cy}
        # deferred binding: a play may fire a frame or two before its sprite is detected.
        still = []
        for pend in self._pending:
            if now - pend["t0"] > self.bind_forward:
                continue  # gave up on this one (its unit never clearly appeared)
            if self._try_bind(pend["play"], now) is None:
                still.append(pend)
        self._pending = still

    # -- on a play ----------------------------------------------------------
    def bind(self, play, now):
        """Bind a ground-truth play {side, card, t} to the board unit that just
        appeared. Returns {track_id, card, det_side, placement} on an immediate bind,
        else None -- a bindable play with no unit YET is queued and retried for
        bind_forward seconds by update(). placement is in CANONICAL (self-bottom)
        coords for the acting player, ready for the learning sample."""
        if not spawns_trackable_unit(play.get("card")):
            return None
        r = self._try_bind(play, now)
        if r is None:
            self._pending.append({"play": play, "t0": now})
        return r

    def _try_bind(self, play, now):
        card = play.get("card")
        det_side = "ally" if play.get("side") == "friendly" else "enemy"
        expected = detected_classes(card)
        cands = []
        for t in self._tracks:
            tid = t["id"]
            if tid in self._bindings:
                continue
            b = self._births.get(tid)
            if b is None or now - b["t"] > self.bind_window:
                continue
            if not _on_half(b["cy"], det_side, self.half_margin):
                continue
            cands.append(t)
        if not cands:
            return None
        if expected:
            matched = [t for t in cands if _class_match(t["name"], expected)]
            if len(matched) == 1:
                chosen = matched[0]
            elif len(matched) > 1:
                return None                     # ambiguous class match -> abstain
            elif len(cands) == 1:
                chosen = cands[0]               # class wobbled, but a single new unit fits
            else:
                return None
        else:                                   # detector-gap troop: no class signal
            chosen = cands[0] if len(cands) == 1 else None
            if chosen is None:
                return None
        tid = chosen["id"]
        self._bindings[tid] = {"card": card, "det_side": det_side, "play_t": now}
        b = self._births[tid]
        cy = b["cy"] if play.get("side") == "friendly" else 1.0 - b["cy"]
        return {"track_id": tid, "card": card, "det_side": det_side,
                "placement": {"x": round(b["cx"], 3), "y": round(cy, 3)}}

    # -- output -------------------------------------------------------------
    def resolved_units(self):
        """This frame's visible tracks as board-unit dicts, with bound units STAMPED
        (name=card, side from the play, side_source='play', struggle/off_deck cleared)
        and unbound units carrying the tracker's debounced class/side + the detector's
        struggle/off_deck flags. Shape matches what the learning logger / overlay read."""
        out = []
        for t in self._tracks:
            tid = t["id"]
            ex = self._extra.get(tid, {})
            bd = self._bindings.get(tid)
            if bd:
                name, side, src = bd["card"], bd["det_side"], "play"
                struggle = off_deck = False
            else:
                name, side, src = t["name"], t["side"], ex.get("side_source")
                struggle = ex.get("struggle", False)
                off_deck = ex.get("off_deck", False)
            conf = t.get("confidence")
            out.append({
                "id": tid, "bbox": t["bbox"], "name": name, "side": side,
                "side_source": src, "level": t.get("level"),
                "conf": round(conf, 2) if conf is not None else None,
                "hp_fraction": t.get("hp"), "struggle": struggle,
                "off_deck": off_deck, "bound": bool(bd), "age": t.get("age"),
            })
        return out
