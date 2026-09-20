"""Banks BOARD-UNIT crops while watching TV Royale, to grow the detector's
training data with REAL frames from Duncan's own emulator -- the lever
training/README.md names for the detector's two live weaknesses (confusable
classes like archer<->musketeer / goblin<->spear-goblin, and the KataCR-vs-
Duncan's-device domain gap).

BUILD A made this cheap: game_knowledge/play_localizer binds each ground-truth
play to the board unit that appeared, so a bound unit's crop comes with a
TRUE card label for free. Two kinds of crop are worth banking (Duncan 2026-09-14):
  - BOUND      -- the localizer bound this unit to a play -> proposed label is the
                  played card (ground truth). One key to confirm in review.
  - STRUGGLE   -- the detector's class is in neither deck (u['struggle']/off_deck):
                  exactly the confusable/unknown reads we want to relabel + retrain.

RULE #2 (CLAUDE.md): this only QUEUES to assets/unlabeled_board_queue/ for Duncan to
review + label with perception/review_board_queue.py. It NEVER files into a permanent
library -- not even a bound crop, whose label is ground truth but whose SOURCE (the
new, not-yet-battle-tested binder) is exactly what review guards. The label is a
pre-filled SUGGESTION, not a trusted write.

Each crop is padded (PAD_FRAC) for context and saved with a JSON sidecar carrying the
unit's box WITHIN the crop (so ingest writes an exact YOLO label, not a whole-crop
box), the proposed label, the side, and the 8 candidate cards from that side's deck
(the review tool turns labelling into a 1-of-8 keypress). Banked ONCE per track id.
"""

import json
import os
import time

import cv2

QUEUE_DIR = "assets/unlabeled_board_queue"
PAD_FRAC = 0.25          # pad the bbox by this fraction each side for training context
CAP_PER_LABEL = 40       # stop banking a given proposed label after this many (queue balance)
MIN_BOX_PX = 14          # skip boxes smaller than this (too small to label/train usefully)
STRUGGLE_MIN_AGE = 4     # only bank a STRUGGLE crop after the track has persisted this many
# frames unbound -- gives a play a chance to bind it first, so a soon-to-be-bound unit
# isn't banked as a (wrong-guess) struggle crop.


def _slug(s):
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(s))


class BoardCropCollector:
    def __init__(self, queue_dir=QUEUE_DIR, cap_per_label=CAP_PER_LABEL, pad_frac=PAD_FRAC):
        self.queue_dir = queue_dir
        self.cap_per_label = cap_per_label
        self.pad_frac = pad_frac
        os.makedirs(queue_dir, exist_ok=True)
        self._banked = set()          # (game_id, track_id) already banked
        self._counts = self._seed_counts()
        self._enemy_deck = []         # top player's 8 cards (card slugs)
        self._friendly_deck = []      # bottom player's 8 cards
        self.saved = 0

    def _seed_counts(self):
        counts = {}
        for f in os.listdir(self.queue_dir):
            if f.endswith(".png"):
                counts[f.split("__", 1)[0]] = counts.get(f.split("__", 1)[0], 0) + 1
        return counts

    def set_decks(self, enemy_deck, friendly_deck):
        """The two decks for THIS game (enemy=top, friendly=bottom) -- the candidate
        lists the review tool offers per unit, keyed by the unit's side."""
        self._enemy_deck = list(enemy_deck or [])
        self._friendly_deck = list(friendly_deck or [])

    def _candidates(self, side):
        if side == "ally":
            return self._friendly_deck or (self._friendly_deck + self._enemy_deck)
        if side == "enemy":
            return self._enemy_deck or (self._enemy_deck + self._friendly_deck)
        return self._friendly_deck + self._enemy_deck   # side unknown -> up to 16

    def observe(self, frame, units, game_id="live"):
        """Bank the crop of each BOUND or STRUGGLE unit not already banked this game.
        `units` = game_knowledge/play_localizer.PlayLocalizer.resolved_units() (each
        has id, bbox, name, side, bound, struggle/off_deck, age)."""
        h, w = frame.shape[:2]
        for u in units:
            bound = bool(u.get("bound"))
            struggle = bool(u.get("struggle") or u.get("off_deck"))
            if not (bound or struggle):
                continue
            if struggle and not bound and (u.get("age") or 0) < STRUGGLE_MIN_AGE:
                continue                       # give a play a chance to bind it first
            key = (game_id, u.get("id"))
            if key in self._banked:
                continue
            saved = self._bank(frame, w, h, u, bound, game_id)
            self._banked.add(key)              # mark tried either way (don't recheck each frame)
            if saved:
                self.saved += 1

    def _bank(self, frame, w, h, u, bound, game_id):
        x1, y1, x2, y2 = u["bbox"]
        bw, bh = x2 - x1, y2 - y1
        if bw < MIN_BOX_PX or bh < MIN_BOX_PX:
            return None
        px, py = int(bw * self.pad_frac), int(bh * self.pad_frac)
        cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
        cx2, cy2 = min(w, x2 + px), min(h, y2 + py)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        cw, ch = cx2 - cx1, cy2 - cy1
        box = [round(((x1 + x2) / 2 - cx1) / cw, 4), round(((y1 + y2) / 2 - cy1) / ch, 4),
               round(bw / cw, 4), round(bh / ch, 4)]

        proposed = u.get("name")
        kind = "bound" if bound else "struggle"
        if self._counts.get(proposed, 0) >= self.cap_per_label:
            return None
        stem = f"{_slug(proposed)}__{kind}__{_slug(game_id)}_{u.get('id')}_{int(time.time()*1000)}"
        cv2.imwrite(os.path.join(self.queue_dir, stem + ".png"), crop)
        meta = {
            "proposed": proposed, "kind": kind, "side": u.get("side"),
            "det_name": u.get("name"), "bound": bound,
            "candidates": self._candidates(u.get("side")),
            "crop_box": box, "game": game_id, "track_id": u.get("id"),
        }
        with open(os.path.join(self.queue_dir, stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        self._counts[proposed] = self._counts.get(proposed, 0) + 1
        return stem
