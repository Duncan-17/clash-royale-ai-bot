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

DECK_ADMIT_SECONDS = 0.8  # (learn_deck mode) a card must hold a hand slot
# continuously for at least this long to be admitted to the LEARNED deck. Long
# enough to exclude the brief blank-frame misreads that only flash during a play
# transition (e.g. "mirror"), short enough that a genuine resting card qualifies
# almost immediately. Time-based (not a frame count) so it doesn't depend on the
# caller's capture rate. Tunable.

PLAY_CONFIRM_SECONDS = 0.4  # a newly-shown hand card must remain the confirmed
# card at least this long before the DEPARTURE of the previous card is logged as a
# play. Absorbs transients that briefly occupy a slot during a play transition -- a
# blank-frame misread (e.g. "mirror", including when Mirror is genuinely in the
# deck) or a classifier flip that flickers back -- instead of emitting a phantom
# play. The play is timestamped at the transition, so timing isn't shifted. Tradeoff:
# a card played within this window of appearing is missed (rare vs play cadence).

ELIXIR_PLAY_MIN_DROP = 0.5  # GATE: a real play SPENDS elixir. If elixir did not
# drop by at least this much across the transition (peak just before minus trough
# just after), the slot's card didn't really change -- it's a resting-card misread,
# not a play, and is dropped. A real play drops >= ~1; a misread ~0. Fail-open when
# there's no elixir data (don't want to miss real plays). Tunable.

# --- elixir reconciliation: recover a play the classifier never read, from an
# unexplained elixir spend (see _reconcile_elixir). Handles unread cards ANYWHERE in
# the match (mid-game evos, end-of-game), not just slots unread from the start.
SPEND_MIN = 0.9          # min accumulated elixir drop to count as a play's spend
# (a real play spends >= 1; regen only ever ADDS elixir, so a fall this size is a play).
RECON_LAG_SECONDS = 1.8  # wait this long after a detected spend before deciding it's
# unexplained -- long enough for the matching slot-change play to register first (it
# lags by learn-admission + confirm + PLAY_CONFIRM debounce, ~1.5s worst case in pure
# learn). Unclaimed after this -> infer the unread play. Bigger = safer vs races,
# at the cost of later recovery of genuinely-missed plays.
SPEND_COST_TOL = 1.0     # elixir slack when matching a spend's size to a play's cost.
SUSTAIN_SECONDS = 0.4    # a detected dip must STAY down this long (only regen refilling
# it) to count as a real spend; a momentary elixir-read blip that pops back up faster
# than regen is discarded -- cuts spurious inferences from noisy elixir reads.
SUSTAIN_TOL = 0.6        # elixir may recover this much within SUSTAIN_SECONDS and still
# count as sustained (~regen over that window). Larger = looser.

DECK_SIZE = 8            # a Clash Royale deck is exactly 8 cards. Used to (a) treat a
# full known/menu deck as authoritative, and (b) cap the LEARNED deck so persistent
# classifier misreads can't accumulate a 9th+ "card" that then logs phantom plays.

CYCLE_REPEAT_WINDOW = 3  # GATE: a card can't be replayed until it cycles back into
# hand (>=4 other cards played after it, a Clash Royale rule). So a "play" of a card
# that this player played within the last few plays can't be legit -- the card isn't
# in hand yet -- and is dropped. Deck-independent. Set to 3 (not the theoretical 4)
# to tolerate one missed play without falsely rejecting a real repeat.


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


def _cost_key(slug: str) -> str:
    """Cost/reference tables key hyphens as underscores (X-Bow -> 'x_bow', via _slug),
    but the deck/hand slug for X-Bow is 'x-bow' (assets/hand_templates/x-bow). Normalize
    so its cost still resolves -- without this an x-bow deck logs 'no elixir cost' and its
    plays never pass the elixir cross-check."""
    return (slug or "").replace("-", "_")


def load_card_costs(cards_json_path: str = CARDS_JSON) -> dict:
    """Returns {slug_name: elixir_cost} for every card in config/cards.json
    that has a fixed cost, keyed the same way deck.txt/hand_templates are,
    so a deck.txt entry can look its cost up directly. Skips cards with no
    fixed cost (Mirror's real cost is "last card played + 1", a special
    rule the official API doesn't encode — so Mirror has no entry here and
    its cost is resolved dynamically in _register_play when it's actually
    played: the player's previous play's cost + 1)."""
    with open(cards_json_path) as f:
        cards = json.load(f)
    return {_slug(c["name"]): c["elixirCost"] for c in cards if "elixirCost" in c}


class CycleTracker:
    def __init__(self, deck: list, confidence_threshold: float = CARD_CONFIDENCE_THRESHOLD,
                 learn_deck: bool = False):
        self.learn_deck = learn_deck
        if len(deck) != 8 and not learn_deck:
            print(f"Warning: deck has {len(deck)} card(s), not 8 — cycle tracking "
                  f"will still run but queue-length assumptions may not hold.")
        self.deck = list(deck)
        # The play filter (update()) rejects any read of a card that can't legally
        # be in this player's hand -- a misread of an out-of-hand card can never be a
        # real play. Fixes the constant phantom "mirror" plays (2026-09-11): the
        # unrestricted hand classifier lands on out-of-roster cards like Mirror on
        # the brief blank slot right after a real play, on BOTH hands. Two sources of
        # "legal cards":
        #  - _deck_set: the KNOWN deck (own deck.txt, or the TV Royale menu auto-read),
        #    authoritative and available from frame 1. Empty + not learn_deck -> filter
        #    disabled (legacy behavior).
        #  - _observed: the deck LEARNED from the hands (learn_deck=True). A card that
        #    holds a slot continuously past DECK_ADMIT_SECONDS is really in the deck; a
        #    transition blip never lasts that long. Removes the need for a pre-entered
        #    list AND cross-checks/repairs a mis-read menu deck (their union is allowed).
        self._deck_set = set(self.deck)
        self._has_seed = bool(self.deck)   # a known deck was provided (vs pure learn)
        self.confidence_threshold = confidence_threshold
        self.costs = load_card_costs()
        missing = [c for c in self.deck if _cost_key(c) not in self.costs]
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
        self._observed = set()          # deck LEARNED from the hands (learn_deck mode)
        self._run = {slot: (None, 0.0) for slot in range(4)}  # (card, run_start_time) per slot
        self._last_emitted = {slot: None for slot in range(4)}  # last card ACCEPTED as resting
        self._hand_since = {slot: 0.0 for slot in range(4)}     # when hand[slot] last changed
        # elixir reconciliation (recover unread plays from unexplained spends):
        self._elx_prev = None           # previous frame's elixir level
        self._drop_accum = 0.0          # elixir fallen so far in the current dip
        self._spend_candidate = None    # a dip awaiting the SUSTAIN check: {t, mag, floor}
        self._pending_spends = []       # sustained spends awaiting reconciliation: {t, mag}
        self._claimed = set()           # indices of normal plays already matched to a spend

    def _allowed(self, name: str) -> bool:
        """Whether a READ may drive slot confirmation: the card must be one that can
        legally be in this hand, so a misread of a card that can't be here (the
        blank-frame 'mirror', an out-of-deck classifier flip) is rejected before it
        can confirm a phantom play. This read-block is the PRIMARY filter; the
        play-confirmation debounce (PLAY_CONFIRM_SECONDS) is a second guard that
        additionally absorbs brief transients among ALLOWED cards (e.g. a real
        in-deck Mirror's blank-frame flash)."""
        # A full known deck (8 cards from the menu/list) is AUTHORITATIVE: a card
        # outside it can't be in this hand, so persistent classifier misreads (e.g.
        # "arrows"/"goblin_machine" that aren't in the match) are blocked outright --
        # they can't get "learned" in and logged as phantom plays. This is why a
        # filled list gives the cleanest reads.
        if len(self._deck_set) >= DECK_SIZE:
            return name in self._deck_set
        if self.learn_deck:
            # no full list -> the deck learned from the hands (+ any partial list). The
            # learned set is capped at DECK_SIZE (see _learn), so it can't accumulate
            # misreads beyond a real 8-card deck.
            return name in self._deck_set or name in self._observed
        return (not self._deck_set) or (name in self._deck_set)

    def is_allowed(self, name: str) -> bool:
        """Public view of _allowed."""
        return self._allowed(name)

    def in_deck(self, name: str) -> bool:
        """Whether `name` is believed to be in this player's deck -- for diagnostics
        (e.g. the OFF-DECK panel). Mirrors _allowed so OFF-DECK shows exactly what the
        tracker filters: a full known deck is authoritative; else known + learned."""
        if len(self._deck_set) >= DECK_SIZE:
            return name in self._deck_set
        return name in self._deck_set or name in self._observed

    @property
    def observed_deck(self) -> set:
        """The cards learned to be in this player's deck so far (learn_deck mode)."""
        return set(self._observed)

    def known_cards(self) -> set:
        """This player's cards we're confident about: the given deck + the learned one.
        Used as a side/class prior for the board detector (a troop belongs to whoever's
        deck holds its card)."""
        return set(self._deck_set) | set(self._observed)

    def _learn(self, slot_results, now: float):
        """Grow the learned deck: a card that holds a slot continuously for
        >= DECK_ADMIT_SECONDS is admitted as really in the deck. A blank-frame
        misread (e.g. 'mirror') only flashes, so its run resets before it qualifies."""
        if not self.learn_deck:
            return
        for slot, result in enumerate(slot_results):
            name, score = result[0], result[1]
            valid = name not in (None, "UNKNOWN") and score >= self.confidence_threshold
            run_card, run_start = self._run[slot]
            if valid and name == run_card:
                # Cap the learned deck at DECK_SIZE: a real deck is 8 cards, so once 8
                # are learned, a further "new" card is a persistent misread -> don't
                # admit it (that's what let arrows/goblin_machine become phantom plays).
                if now - run_start >= DECK_ADMIT_SECONDS and (
                        name in self._observed or len(self._observed) < DECK_SIZE):
                    self._observed.add(name)
            else:
                self._run[slot] = (name if valid else None, now)

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

    def _elixir_drop(self, transition_t: float, now: float):
        """How much elixir was spent around a play: peak just before the transition
        minus trough just after. A real play shows a clear positive drop; a misread
        of a resting card shows ~0 (elixir only rises with regen). Robust to single-
        frame glitches via max/min. Returns None if there's no data to compare."""
        before = [v for (t, v) in self._elixir_history if transition_t - 1.5 <= t <= transition_t + 0.15]
        after = [v for (t, v) in self._elixir_history if transition_t <= t <= now]
        if not before or not after:
            return None
        return max(before) - min(after)

    def _register_play(self, played: str, revealed: str, now: float,
                       transition_t: float = None):
        """Record a play, or return None if it fails a validity gate (it was a
        misread of a resting card, not a real play)."""
        cost = self.costs.get(_cost_key(played))
        # Mirror has no FIXED cost (load_card_costs skips it): it copies this
        # player's LAST played card at +1 elixir. So when an in-deck Mirror is
        # played, resolve its cost from this player's previous play (self.plays[-1],
        # not yet including this event) + 1 -- so the elixir check, the log, and
        # reward see the real cost instead of None. Only reachable if Mirror is
        # actually in the deck (an out-of-deck "mirror" misread was already filtered
        # in update()). Double-Mirror is impossible (one Mirror per deck), so the
        # previous play is a normal card with a numeric cost.
        if cost is None and played == "mirror" and self.plays:
            prev_cost = self.plays[-1].get("cost")
            cost = prev_cost + 1 if prev_cost is not None else None

        # GATE 1 -- cycle validity: a card can't be replayed until it has cycled back
        # into hand. If this player played `played` within the last few plays it can't
        # be in hand now, so this "play" is a misread (e.g. the second graveyard that
        # was never played). Deck-independent. See CYCLE_REPEAT_WINDOW.
        if played in [p["card"] for p in self.plays[-CYCLE_REPEAT_WINDOW:]]:
            return None

        # GATE 2 -- elixir: a real play spends elixir. If elixir clearly did not drop
        # around the transition, the slot's card didn't really change (resting-card
        # misread), so it's not a play. Fail-open when there's no elixir data.
        tt = transition_t if transition_t is not None else now
        drop = self._elixir_drop(tt, now)
        if drop is not None and drop < ELIXIR_PLAY_MIN_DROP:
            return None
        # confirmed = the observed drop matches this card's cost (a strong "real play"
        # signal); shown as [OK] vs [?elixir] in the debugger.
        confirmed = (drop is not None and cost is not None
                     and abs(drop - cost) <= ELIXIR_MATCH_TOLERANCE)

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

        # Without a seed deck (pure learn) there's no known queue to reconstruct, so
        # back_pool is empty and the branch above fires on almost every play -- the
        # desync flag would be constant noise. It's only meaningful with a seed deck.
        if not self._has_seed:
            desync = False

        event = {
            "card": played, "revealed": revealed, "cost": cost,
            "confirmed": confirmed, "desync": desync, "frame_time": now,
            "elixir_drop": round(drop, 2) if drop is not None else None,
        }
        self.plays.append(event)
        self.plays_this_match += 1
        return event

    def _reconcile_elixir(self, now: float):
        """Recover a play the classifier never read, from an UNEXPLAINED elixir spend.

        A real play instantly drops the player's elixir by its cost; regen only ever
        adds elixir. So we watch for a completed dip (elixir fell >= SPEND_MIN then
        stopped falling) and buffer it as a spend. After RECON_LAG (long enough for a
        slot-change play to register through the debounce), any spend NOT matched to a
        detected play means a card was played that we couldn't read -> infer it (named
        by elimination from the learned deck, else unknown_<cost>e). Handles unread
        cards ANYWHERE in the match (mid-game evos, end-of-game), not just slots that
        were unread from the start. Returns the inferred event, if any."""
        # 1. detect a completed dip (elixir fell then stopped falling)
        cur = self._elixir_history[-1][1] if self._elixir_history else None
        if cur is not None and self._elx_prev is not None:
            delta = cur - self._elx_prev
            if delta < -0.15:                       # still falling -> part of a dip
                self._drop_accum += -delta
            else:                                   # flat/rising -> dip (if any) is done
                if self._drop_accum >= SPEND_MIN:   # a candidate; confirm only if it STAYS down
                    self._spend_candidate = {"t": now, "mag": self._drop_accum, "floor": cur}
                self._drop_accum = 0.0
        self._elx_prev = cur

        # 1b. SUSTAIN check: a real spend stays near its floor (only regen refills it);
        # a noise blip pops back up fast -> discard. Confirm after SUSTAIN_SECONDS.
        c = self._spend_candidate
        if c is not None and cur is not None:
            if cur > c["floor"] + SUSTAIN_TOL:      # recovered too fast -> was noise
                self._spend_candidate = None
            elif now - c["t"] >= SUSTAIN_SECONDS:   # stayed down -> a real spend
                self._pending_spends.append({"t": c["t"], "mag": c["mag"]})
                self._spend_candidate = None

        # 2. reconcile spends old enough that their slot-change play would have landed
        inferred = None
        for s in [s for s in self._pending_spends if now - s["t"] >= RECON_LAG_SECONDS]:
            self._pending_spends.remove(s)
            if self._claim_spend(s):                # a detected play explains it
                continue
            ev = self._infer_from_spend(s["mag"], s["t"])
            if ev is not None:
                inferred = ev
        return inferred

    def _claim_spend(self, spend) -> bool:
        """True if an as-yet-unclaimed NORMAL play accounts for this spend (matching
        cost, registered around the spend). Each normal play claims one spend."""
        for i, p in enumerate(self.plays):
            if p.get("inferred") or i in self._claimed:
                continue
            if (abs((p.get("cost") or 0) - spend["mag"]) <= SPEND_COST_TOL
                    and spend["t"] - 0.6 <= p.get("frame_time", 0) <= spend["t"] + RECON_LAG_SECONDS + 0.3):
                self._claimed.add(i)
                return True
        return False

    def _infer_from_spend(self, mag: float, t: float):
        """Log a play recovered from an unexplained spend of size `mag`. NAME it by
        elimination from the learned deck (unique card of that cost, in hand, not a
        too-recent play); else unknown_<cost>e -- the play (side/cost/time) isn't lost
        and the crop is banked for review by the uncertain-card collector."""
        cost = round(mag)
        # Only infer a play whose cost EXISTS in this deck. A spend matching no deck
        # card's cost is elixir-read noise, not a play (e.g. "unknown_2e" when the deck
        # has no 2-cost card). Uses known + learned costs; fail-open before anything's
        # learned.
        deck_costs = {self.costs.get(c) for c in (self._deck_set | self._observed)}
        deck_costs.discard(None)
        if deck_costs and cost not in deck_costs:
            return None
        recent = {p["card"] for p in self.plays[-CYCLE_REPEAT_WINDOW:]}
        in_hand = {self.hand[s] for s in range(4) if self.hand[s]}
        candidates = [c for c in self._observed
                      if self.costs.get(c) == cost and c not in in_hand and c not in recent]
        named = candidates[0] if len(candidates) == 1 else None
        card = named if named is not None else f"unknown_{cost}e"
        if card in recent:                 # cycle gate for the inferred card too
            return None
        event = {
            "card": card, "revealed": None, "cost": cost,
            "confirmed": True, "desync": False, "frame_time": t,
            "elixir_drop": round(mag, 2), "inferred": True, "named": named is not None,
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
        self._learn(slot_results, now)   # grow the learned deck (learn_deck mode)
        before_hand = dict(self.hand)

        for slot, result in enumerate(slot_results):
            name, score = result[0], result[1]
            # Skip unknown / low-confidence / not-legal reads: all are "no
            # trustworthy card this frame," so leave the slot's pending streak
            # undisturbed rather than confirming a misread (a card that can't be
            # in this hand -- e.g. the blank-frame "mirror" -- is never a real play).
            if (name in (None, "UNKNOWN") or score < self.confidence_threshold
                    or not self._allowed(name)):
                continue
            self._confirm_slot(slot, name)

        # Note when each slot's CONFIRMED card last changed (drives the debounce).
        for slot in range(4):
            if self.hand[slot] != before_hand[slot]:
                self._hand_since[slot] = now

        # Emit a play only once the newly-shown card has PERSISTED past the debounce
        # (PLAY_CONFIRM_SECONDS). A transient that briefly occupies a slot during a
        # play transition -- a blank-frame misread like "mirror", or a classifier
        # flip that flickers back -- never lasts that long, so it's absorbed instead
        # of logged as a phantom play. `_last_emitted` is the card we've accepted as
        # really resting in the slot; when the confirmed card settles on something
        # new, the card it REPLACED is the play.
        last_play = None
        for slot in range(4):
            cur = self.hand[slot]
            if cur is None or cur == self._last_emitted[slot]:
                continue
            if now - self._hand_since[slot] < PLAY_CONFIRM_SECONDS:
                continue  # not settled yet -- might still be a transient
            old = self._last_emitted[slot]
            self._last_emitted[slot] = cur
            if old is None:
                # First card settled in this slot -- a fill (match start, or a card
                # the classifier couldn't read until now), not a play out of the queue.
                # An unread card that gets PLAYED is recovered separately by the elixir
                # reconciliation below, from its unexplained elixir spend.
                self.back_pool.discard(cur)
            else:
                # _register_play returns None if the "play" fails a validity gate
                # (no elixir spent, or the card couldn't have cycled back) -- a
                # misread, not a real play, so it's not recorded.
                play = self._register_play(old, cur, now, self._hand_since[slot])
                if play is not None:
                    last_play = play

        # Recover any play the classifier never read: an elixir spend not explained
        # by a detected play -> an unread card was played. Runs AFTER slot detection
        # so this frame's normal plays are available to claim their spends.
        inferred = self._reconcile_elixir(now)
        if inferred is not None:
            last_play = inferred

        # Full 8-slot picture: 4 in hand + 4 out of hand (the very next card
        # to be drawn = "on deck", then the remaining 3 further back).
        # "?" marks any out-of-hand position not yet resolved this match. Padded to
        # 4 so on_deck/rest are always defined even before the deck is known (an
        # empty seed deck in learn_deck mode starts with nothing out of hand).
        out_of_hand = (["?"] * len(self.back_pool)) + list(self.tail_queue)
        out_of_hand = (out_of_hand + ["?"] * 4)[:4]
        return {
            "hand": dict(self.hand),
            "on_deck": out_of_hand[0],
            "rest_of_queue": out_of_hand[1:],
            "queue": out_of_hand,  # kept for the JSONL log / anything already reading this key
            "queue_fully_resolved": len(self.deck) > 0 and len(self.back_pool) == 0,
            "last_play": last_play,
            "plays_this_match": self.plays_this_match,
        }
