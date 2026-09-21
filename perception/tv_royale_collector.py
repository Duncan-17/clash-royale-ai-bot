"""Banks clean in-hand card crops while spectating TV Royale, to grow
assets/hand_templates/ beyond the weak catalog seeds — the whole point being
that TV Royale can't be "played" (you're watching), so real examples have to
be harvested from the replay's own hands as they cycle.

Queues to assets/unlabeled_tv_hand_queue/ for Duncan to review/label — it does
NOT file into the permanent hand_templates library directly (RULE #2 in
CLAUDE.md; same queue-then-review flow as review_queue.py /
review_spawn_icon_queue.py). The restricted matcher's best-of-16 guess (with
both decks entered in config/tv_royale_decks.py) is encoded in the filename as
a STARTING SUGGESTION for the reviewer, not a trusted label.

What makes a crop worth banking, and why:
- STABLE: the same card must read for STABLE_FRAMES consecutive frames, so we
  skip the card-wipe/transition frames (same reason cycle_tracker debounces).
- NOT SELECTED: a selected card is lifted out of its resting box, so its
  resting crop is junk — we only bank the normal resting art.
- ONE PER APPEARANCE: saved once when a card first goes stable in a slot, not
  every frame it sits there (that would flood the queue with one static frame).
- DEDUP + CAP: near-identical crops for a card are skipped, and each guessed
  card is capped so a common card doesn't crowd out rarer ones.
"""

import os
import time

import cv2

from perception.card_detector import to_gray

QUEUE_DIR = "assets/unlabeled_tv_hand_queue"
STABLE_FRAMES = 3
COLLECT_MIN_SCORE = 0.40  # below this the guess/crop is too shaky to bank; review still catches errors
CAP_PER_CARD = 25         # max queued crops per guessed card before we stop banking it
DEDUP_THRESHOLD = 0.92    # a new crop this similar to the last banked one for its slot is skipped.
# Lowered 0.97 -> 0.92 (2026-09-07): a card's idle ANIMATION makes two banks of
# the same card land ~0.9 similar, which 0.97 let through as near-dupes -- 0.92
# catches those while still keeping genuinely different examples.


class HandCropCollector:
    def __init__(self, queue_dir=QUEUE_DIR, stable_frames=STABLE_FRAMES,
                 min_score=COLLECT_MIN_SCORE, cap_per_card=CAP_PER_CARD):
        self.queue_dir = queue_dir
        self.stable_frames = stable_frames
        self.min_score = min_score
        self.cap_per_card = cap_per_card
        os.makedirs(queue_dir, exist_ok=True)
        self._pending = {}       # slot_key -> [card, streak, saved_this_run]
        self._last_gray = {}     # slot_key -> last banked gray crop (dedup)
        self._counts = self._seed_counts()
        self.saved = 0

    def _seed_counts(self):
        counts = {}
        for f in os.listdir(self.queue_dir):
            if f.endswith(".png"):
                guess = f.split("__", 1)[0]
                counts[guess] = counts.get(guess, 0) + 1
        return counts

    def _is_dup(self, slot_key, crop):
        prev = self._last_gray.get(slot_key)
        if prev is None:
            return False
        cur = to_gray(crop)
        score = float(cv2.matchTemplate(cur, prev, cv2.TM_CCOEFF_NORMED)[0][0])
        return score > DEDUP_THRESHOLD

    def observe(self, slot_key, card, score, is_selected, crop):
        """Feed one slot's read for this frame. Returns a saved path when a new
        example is banked, else None. slot_key: any hashable id for the slot
        (e.g. ('enemy', 2))."""
        if is_selected or card in (None, "UNKNOWN") or score < self.min_score:
            # No usable reading this frame (a selection, a wipe, a misread
            # flicker). Leave the slot's state UNTOUCHED -- resetting it here was
            # a bug: a single flicker between stable frames cleared the
            # "already saved" flag and re-banked the same card, producing the
            # near-identical runs Duncan saw. A genuine card change is detected
            # below by the card name actually differing, not by these gaps.
            return None

        entry = self._pending.get(slot_key)
        if entry and entry[0] == card:
            entry[1] += 1
        else:
            entry = [card, 1, False]  # a genuinely different card = new appearance
            self._pending[slot_key] = entry
        card_name, streak, saved_this_run = entry

        if streak < self.stable_frames or saved_this_run:
            return None
        entry[2] = True  # mark this stable run handled either way

        if self._counts.get(card_name, 0) >= self.cap_per_card:
            return None
        if self._is_dup(slot_key, crop):
            return None

        path = os.path.join(
            self.queue_dir,
            f"{card_name}__{score:.2f}__{slot_key[0]}{slot_key[1]}__{int(time.time()*1000)}.png",
        )
        cv2.imwrite(path, crop)
        self._last_gray[slot_key] = to_gray(crop)
        self._counts[card_name] = self._counts.get(card_name, 0) + 1
        self.saved += 1
        return path


UNCERTAIN_CONF = 0.70  # classifier confidence below this = "doesn't know it well"


class UncertainCardCollector:
    """Banks slot crops the CLASSIFIER is UNSURE about (confidence < conf_below) to
    the review queue -- so watching games self-stocks a queue of exactly the cards
    the classifier doesn't know well (Duncan 2026-09-11: "it can id a card it doesn't
    know well ... self-stock a queue I can sort as it watches games").

    Sort it with review_tv_hand_queue -> labels file into assets/hand_templates/ ->
    the next classifier retrain folds them in (generate_hand_dataset.real_crops_for).
    Deduped per slot + capped per guessed card + time-throttled so one persistently-
    uncertain slot doesn't flood the queue. Same queue/filename format as
    HandCropCollector, so the existing review tool handles it unchanged.
    """

    def __init__(self, queue_dir=QUEUE_DIR, conf_below=UNCERTAIN_CONF, cap_per_card=25,
                 throttle_s=0.5):
        self.queue_dir = queue_dir
        self.conf_below = conf_below
        self.cap_per_card = cap_per_card
        self.throttle_s = throttle_s
        os.makedirs(queue_dir, exist_ok=True)
        self._last_gray = {}   # slot_key -> last banked gray crop (dedup)
        self._last_t = {}      # slot_key -> last bank time (throttle)
        self._counts = {}
        for f in os.listdir(queue_dir):
            if f.endswith(".png"):
                g = f.split("__", 1)[0]
                self._counts[g] = self._counts.get(g, 0) + 1
        self.saved = 0

    def _is_dup(self, slot_key, crop):
        prev = self._last_gray.get(slot_key)
        if prev is None:
            return False
        cur = to_gray(crop)
        return float(cv2.matchTemplate(cur, prev, cv2.TM_CCOEFF_NORMED)[0][0]) > DEDUP_THRESHOLD

    def observe(self, slot_key, card, score, is_selected, crop, off_deck=False):
        """Feed one slot's classifier read. Banks the crop iff the classifier is UNSURE
        (score < conf_below) OR the read is OFF-DECK (off_deck=True: a confident card in
        NEITHER player's known deck -- almost always a misread of a card the classifier
        doesn't know, e.g. a new/evo card outside its vocabulary). Off-deck banks BYPASS
        the per-card cap, so an unknown card isn't blocked by a saturated common class;
        dedup + throttle still bound them. is_selected is accepted for signature-compat
        with HandCropCollector but ignored."""
        if card in (None, "UNKNOWN") or crop.size == 0:
            return None
        if not (off_deck or score < self.conf_below):
            return None
        now = time.time()
        if now - self._last_t.get(slot_key, 0.0) < self.throttle_s:
            return None
        if self._is_dup(slot_key, crop):
            return None
        if not off_deck and self._counts.get(card, 0) >= self.cap_per_card:
            return None
        self._last_t[slot_key] = now
        path = os.path.join(
            self.queue_dir,
            f"{card}__{score:.2f}__{slot_key[0]}{slot_key[1]}__{int(now * 1000)}.png",
        )
        cv2.imwrite(path, crop)
        self._last_gray[slot_key] = to_gray(crop)
        self._counts[card] = self._counts.get(card, 0) + 1
        self.saved += 1
        return path
