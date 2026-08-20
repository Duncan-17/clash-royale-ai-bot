"""
Own-side card cycle tracker — Phase 4, part 1.

The game shows only 4 of our own 8 deck cards at a time; the other 4 sit in
a hidden FIFO queue (play a card -> it goes to the back of the queue -> the
queue's front card fills the now-empty hand slot). This module reconstructs
that queue by elimination rather than reading any on-screen "next card"
hint (there isn't one to read yet — see the bottom of this docstring for
why that's fine): every time a hand slot's confirmed card changes, whichever
new card shows up there IS the queue's current front card, revealed after
the fact instead of ahead of time.

Two-part queue model, chosen to match the actual game rule (unresolved
starting cards must ALL surface once before any newly-requeued card can
resurface, since a requeued card is always appended behind whatever's
already waiting):
- `back_pool`: the set of original deck cards not yet placed in hand and
  not yet revealed via a play — order among them is genuinely unknown
  until each is individually revealed. Shown as "?" placeholders.
- `tail_queue`: a FIFO of cards that have already been played once this
  match, in the exact order they were played — this portion's order is
  fully known in real time, no guessing needed, since we directly observe
  every play.
`back_pool` always empties out after the first four real plays (a "lap"),
after which `tail_queue` alone *is* the queue and everything is fully
deterministic for the rest of the match.

A candidate play (a hand slot's confirmed card changing) is only trusted as
a real play once cross-checked against the elixir signal — elixir should
have dropped by roughly the departing card's known cost, net of regen over
the elapsed interval. This mirrors elixir_counter.py's own two-signal
design: no single frame's read is trusted blind. A mismatch doesn't discard
the play (the hand-slot change genuinely happened) — it's recorded with
confirmed=False instead, since a wrong SILENT trust is worse than a
visible, flagged uncertainty.

Call update() once per frame with the same slot_results/elixir shapes
perception/debug_live.py already computes. Call reset() at the start of
each new match — there's no automatic match-boundary detection anywhere in
this project yet (a known, deliberately deferred gap — see Project_Status),
so this has to be triggered manually (debug_live.py wires it to a keypress).

Two real bugs found and fixed after the first live test (2026-08-15), both
caught by replaying the actual JSONL log rather than trusting the design on
paper:

Bug 1 — desync from blindly trusting internal queue order: once
back_pool is empty, _register_play used to always popleft() tail_queue's
believed front and assume it equalled `revealed`, without checking. A
single high-confidence WRONG read anywhere (still possible even above
CARD_CONFIDENCE_THRESHOLD — the real log had one) breaks that assumption,
and the old code had no way to notice or recover: it silently popped the
wrong card, permanently corrupting tail_queue for the rest of the match
(observed for real: "mini_pekka" got dropped from the model even though it
was still legitimately sitting in hand, while "archers" ended up duplicated
into two queue slots at once). Fixed by always locating `revealed` within
tail_queue directly (deque.remove(), not popleft()) and re-anchoring to
that — self-correcting every single play against the actual observation
instead of trusting the model's own prior belief. A play's `desync` flag is
set True whenever the revealed card wasn't where the model expected it —
this doesn't hide the mismatch, it surfaces it (same "flag it, don't hide
it" spirit as `confirmed`) so a still-elevated desync rate is a visible
signal of a template-matching problem worth chasing, not a silent one.

Bug 2 — below-threshold guesses treated as real reads: slot_results
entries from perception/debug_live.py's per-frame loop carry the card
detector's best GUESS even when its score is below
perception.card_detector.CONFIDENCE_THRESHOLD — debug_live.py only uses the
score to pick a box color (red/orange/green) for a human to read, it never
downgrades the NAME string to "UNKNOWN" itself. update() was blindly
trusting that name, so a few frames of a low-confidence wrong guess (e.g.
"giant" at 0.3 during another card's play animation) could pass the
HAND_STABLE_FRAMES debounce and get treated as a real card in hand. Caught
by inspecting the JSONL log from a real match: a stable, 0.998-confidence
"knight" flickered through ~20 frames of noisy sub-0.5 guesses (including
several of "giant") while the actual play animation ran, before settling on
a different real card at confidence 1.0 — the debounce alone wasn't enough
to reject that stretch, since 3-4 of those noisy frames happened to repeat
the same wrong guess consecutively. Fixed by having update() ignore any
slot_results entry below CARD_CONFIDENCE_THRESHOLD itself (same constant
perception/card_detector.py already defines), rather than assuming the
caller pre-filtered — treated as equivalent to "UNKNOWN" for cycle-tracking
purposes regardless of what debug_live.py chooses to display.

Known limitations, not yet handled (same "flag it, revisit if it actually
causes a problem" spirit as the rest of this project):
- Elixir cross-check only accounts for the "normal" regen rate
  (config.game_constants.ELIXIR_REGEN_SECONDS["normal"]). A play during
  double/triple-elixir phases will use the wrong regen rate for the
  before/after comparison and may get flagged confirmed=False even though
  it was real. Not wired to game-clock/phase detection yet.
- No match-start/end detection — reset() must be called manually.
- The "Next:" on-screen preview card isn't perception-supported yet
  (config/hand_region.py only defines the 4 hand slots) — Project_Status
  previously assumed it was needed for this module; it isn't. The
  elimination approach above gets the same information after the fact
  instead of ahead of time, so no new perception work was required after
  all, just a different algorithm than that doc originally assumed.
"""

import json
import re
from collections import deque

from config.game_constants import ELIXIR_REGEN_SECONDS
from perception.card_detector import CONFIDENCE_THRESHOLD as CARD_CONFIDENCE_THRESHOLD

CARDS_JSON = "config/cards.json"

HAND_STABLE_FRAMES = 3  # consecutive identical reads required before a slot's
# card is "confirmed" — higher than unit_tracker.py's MIN_HITS_TO_CONFIRM=2
# since a wrongly-confirmed play corrupts cycle state, not just one frame's box.

ELIXIR_HISTORY_SECONDS = 3.0  # how far back the rolling elixir buffer reaches —
# long enough to bracket a play's before/after elixir even accounting for the
# ~1s animation window elixir_counter.py's own docs note as unreliable.

ELIXIR_MATCH_TOLERANCE = 1.0  # elixir units of slack allowed between a played
# card's known cost and the observed before/after drop (net of regen) before
# a candidate play is downgraded to confirmed=False. Generous first pass —
# tighten later if false-confirms turn out to be a real problem.


def _slug(name: str) -> str:
    """Matches the naming actually used in deck.txt / assets/hand_templates/
    (e.g. "Mini P.E.K.K.A" -> "mini_pekka"). NOT the same as
    perception/download_card_icons.py's safe_filename(), which was tried
    first here and found to disagree on punctuation-heavy names —
    safe_filename() underscore-substitutes every period individually
    ("mini_p_e_k_k_a"), producing assets/card_icons/mini_p_e_k_k_a.png,
    while the hand-typed hand_templates/deck.txt convention drops periods
    and apostrophes entirely before collapsing whitespace. Confirmed by
    checking both directories directly rather than assuming either was
    right."""
    s = name.lower()
    s = re.sub(r"[.']", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def load_card_costs(cards_json_path: str = CARDS_JSON) -> dict:
    """Returns {slug_name: elixir_cost} for every card in config/cards.json
    that has a fixed cost, keyed the same way deck.txt/hand_templates are,
    so a deck.txt entry can look its cost up directly. Skips cards with no
    fixed cost (Mirror's real cost is "last card played + 1", a special
    rule the official API doesn't encode — not handled here, revisit only
    if Mirror ends up in a tracked deck)."""
    with open(cards_json_path) as f:
        cards = json.load(f)
    return {_slug(c["name"]): c["elixirCost"] for c in cards if "elixirCost" in c}


class CycleTracker:
    def __init__(self, deck: list, confidence_threshold: float = CARD_CONFIDENCE_THRESHOLD):
        if len(deck) != 8:
            print(f"Warning: deck has {len(deck)} card(s), not 8 — cycle tracking "
                  f"will still run but queue-length assumptions may not hold.")
        self.deck = list(deck)
        self.confidence_threshold = confidence_threshold
        self.costs = load_card_costs()
        missing = [c for c in self.deck if c not in self.costs]
        if missing:
            print(f"Warning: no elixir cost found for {missing} — plays of "
                  f"these card(s) will never pass the elixir cross-check.")
        self.reset()

    def reset(self):
        """Clears all state for a new match. Nothing is in hand yet, so
        every deck card starts as an unresolved member of back_pool."""
        self.hand = {slot: None for slot in range(4)}
        self._pending = {slot: (None, 0) for slot in range(4)}  # slot -> (candidate_name, streak)
        self.back_pool = set(self.deck)
        self.tail_queue = deque()
        self._elixir_history = deque()  # (timestamp, value), oldest first
        self.plays = []
        self.plays_this_match = 0

    def _confirm_slot(self, slot: int, name: str) -> bool:
        """Debounced slot update. Returns True the frame a NEW confirmed
        value first takes effect."""
        candidate, streak = self._pending[slot]
        if name == candidate:
            streak += 1
        else:
            candidate, streak = name, 1
        self._pending[slot] = (candidate, streak)

        if streak < HAND_STABLE_FRAMES or self.hand[slot] == candidate:
            return False
        self.hand[slot] = candidate
        return True

    def _record_elixir(self, now: float, value: float):
        self._elixir_history.append((now, value))
        cutoff = now - ELIXIR_HISTORY_SECONDS
        while self._elixir_history and self._elixir_history[0][0] < cutoff:
            self._elixir_history.popleft()

    def _register_play(self, played: str, revealed: str, now: float) -> dict:
        cost = self.costs.get(played)
        confirmed = False
        if cost is not None and self._elixir_history:
            before_t, before_v = self._elixir_history[0]
            after_v = self._elixir_history[-1][1]
            elapsed = now - before_t
            regen_gain = elapsed / ELIXIR_REGEN_SECONDS["normal"]
            expected_after = before_v - cost + regen_gain
            confirmed = abs(after_v - expected_after) <= ELIXIR_MATCH_TOLERANCE

        # Self-correct against the OBSERVED revealed card rather than
        # trusting the model's own belief about queue order — a single
        # missed/misread frame anywhere upstream (still possible even past
        # the confidence threshold; the real log that caught this had a
        # high-confidence wrong read) would otherwise desync tail_queue
        # permanently and silently for the rest of the match, since
        # nothing else re-anchors it to reality afterward.
        desync = False
        if self.back_pool:
            # Unresolved originals must all surface before any requeued
            # card can — so a reveal while back_pool is nonempty is always
            # one of those originals, by the game's own queue rule.
            self.back_pool.discard(revealed)
        elif revealed in self.tail_queue:
            desync = self.tail_queue[0] != revealed
            self.tail_queue.remove(revealed)
        elif self.tail_queue:
            # revealed doesn't match anything currently tracked at all —
            # can't identify which entry to drop, so fall back to the
            # believed front just to keep the length invariant (back_pool
            # + tail_queue always == 4) from drifting, and flag it loudly.
            self.tail_queue.popleft()
            desync = True
        self.tail_queue.append(played)

        event = {
            "card": played, "revealed": revealed, "cost": cost,
            "confirmed": confirmed, "desync": desync, "frame_time": now,
        }
        self.plays.append(event)
        self.plays_this_match += 1
        return event

    def update(self, slot_results, elixir: dict, now: float) -> dict:
        """slot_results: list of (name, score, ...) per hand slot, same
        order perception/card_detector.identify_hand and debug_live.py's
        per-frame loop already produce. elixir: the dict read_elixir()
        returns. now: a monotonically increasing timestamp (e.g.
        time.time()), used only for the elixir before/after window."""
        self._record_elixir(now, elixir["value"] + elixir["fraction"])
        before_hand = dict(self.hand)

        for slot, result in enumerate(slot_results):
            name, score = result[0], result[1]
            if name in (None, "UNKNOWN") or score < self.confidence_threshold:
                continue
            self._confirm_slot(slot, name)

        last_play = None
        for slot, new_name in self.hand.items():
            old_name = before_hand[slot]
            if new_name is None or new_name == old_name:
                continue
            if old_name is None:
                # First time this slot's card is known at all (match start,
                # or bot attached mid-match) — it's already in hand, not a
                # play out of the hidden queue.
                self.back_pool.discard(new_name)
                continue
            last_play = self._register_play(old_name, new_name, now)

        # Full 8-slot picture: 4 in hand + 4 out of hand (the very next card
        # to be drawn = "on deck", then the remaining 3 further back).
        # "?" marks any out-of-hand position not yet resolved this match.
        out_of_hand = (["?"] * len(self.back_pool)) + list(self.tail_queue)
        out_of_hand = out_of_hand[:4]
        return {
            "hand": dict(self.hand),
            "on_deck": out_of_hand[0],
            "rest_of_queue": out_of_hand[1:],
            "queue": out_of_hand,  # kept for the JSONL log / anything already reading this key
            "queue_fully_resolved": len(self.back_pool) == 0,
            "last_play": last_play,
            "plays_this_match": self.plays_this_match,
        }
