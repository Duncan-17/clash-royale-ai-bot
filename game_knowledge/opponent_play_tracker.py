"""
Opponent troop/building play detector — Phase 4, part 2.

Per the philosophy doc (§2) and Project_Status, the single most important
piece of opponent-side modeling is accurately detecting WHEN the opponent
plays a card and WHICH card it is — everything else about their hand/deck/
elixir is downstream inference that depends on getting this right first.
Own-side cycle tracking (game_knowledge/cycle_tracker.py) is done and
verified live; this is the next piece, built per Duncan's explicit request
to solidify play detection before deck/elixir inference. Troops/buildings
only — spells are a separate, differently-shaped detection problem
(Project_Status's Tier 1/2/3 plan), not started, not this module's job.

The base signal already exists: perception/unit_detector.py +
perception/unit_tracker.py already detect and track enemy units frame to
frame. What this module adds is turning "a track_id is new" into an
explicit, correctly-attributed play event — correctly, because a new
enemy track does NOT always mean a fresh hand play. Two distinct reasons
it might not, both handled here (see game_knowledge/spawn_data.py for the
underlying data):

1. It's a byproduct of an existing card's ability (Golem splitting on
   death, Witch/Tombstone/Goblin Hut/etc. producing units for free) —
   Project_Status's long-deferred "unit-creation/spawn attribution"
   problem, finally built here.
2. It's ONE of several units a single card produces simultaneously
   (Minions=3, Goblins=4, Skeleton Army=~15, etc.) — missed in this
   module's first design pass, caught by Duncan before any code was
   written: without grouping simultaneous same-class arrivals together,
   one real play would get miscounted as several.

Both are handled by the same mechanism: new same-side, same-class tracks
that appear within GROUPING_WINDOW_SECONDS of each other are buffered into
one candidate group, then the group as a whole is explained — as N copies
of a known multi-unit card, as spawn output from a nearby/recently-vanished
spawner, or (if neither fits) flagged as "uncertain" rather than guessed.
That "uncertain" outcome is the concrete, buildable-now piece of Duncan's
broader ask that the eventual opponent-elixir tracker be able to flag
readings that drastically violate game rules — this module can't check
elixir yet (see below), but it CAN and does flag a vision-level rule
violation (a burst size matching no known card and no known spawner) the
moment it sees one, rather than silently forcing it into "played".

A third reason a "new enemy track" can appear that ISN'T a fresh play,
found from a real live-match log (2026-08-16) rather than anticipated in
advance: it can be one of OUR OWN units, briefly misclassified as enemy by
perception/unit_detector.py's SideDetector (side.onnx). Confirmed by
tracing two real cases in that log: an ally Giant misclassified as enemy
for its first ~2 seconds on the board (an "enemy" giant track died and an
"ally" giant track was born one frame later at a near-identical position),
and a well-established ally Musketeer (alive 296 frames) that flipped to
"enemy" for a few frames after an occlusion. side.onnx returns a bare
argmax label with no confidence exposed anywhere — the exact same class of
bug already fixed once in cycle_tracker.py (trusting a per-frame classifier
with no plausibility gate), showing up here in a new module. Mitigated
(not fixed — side.onnx itself isn't being retrained) by mirroring the
on-death-spawn check: if a same-class ALLY track vanished very recently
near a new enemy group's position, report attribution="misclassified"
instead of "played". Deliberately a 4th category, not folded into
"uncertain" — an "uncertain" event might still be a real (just ambiguous)
enemy play; a "misclassified" one most likely represents zero real enemy
plays and should be excluded entirely from any future opponent-elixir
accounting, which is why the two need to stay distinguishable.

Deliberately NOT built here: opponent elixir estimation. Project_Status
already notes opponent elixir has to be inferred FROM observed play costs,
not the other way around — there's no elixir signal yet to cross-check
against, unlike the own-side cycle tracker which had a real elixir reading
to validate against. Each event still carries elixir_cost when it can be
cheaply determined (see _card_slug_and_cost's known incompleteness note
below), specifically so a future opponent-elixir module can consume this
one's output directly without redoing this attribution work.

Known limitations, flagged rather than hidden (same pattern as
cycle_tracker.py's known-limitations block):
- Can't distinguish a real on-death spawn from a track that merely stopped
  being detected (occlusion, a missed frame past unit_tracker's grace
  period) — the same died-vs-occluded gap Project_Status already documents
  for unit tracking in general. A spawn misattributed to a coincidental
  nearby occlusion is possible, not just theoretical.
- SPAWN_PROXIMITY_PX is a first-pass guess, not yet tuned against real
  footage (no live match available while writing this) — revisit once
  tested live, the same way unit_tracker.py's MATCH_DISTANCE_PX and
  MIN_CONF were tuned only after real match logs existed.
- A single real burst can get split into two separate groups (one
  "uncertain", one incomplete) if its detections straggle across more than
  GROUPING_WINDOW_SECONDS — acceptable for now, revisit if it's a frequent
  problem once tested live.
- goblin_giant isn't a detectable class at all (see spawn_data.py) — its
  spear_goblin death-spawns can never be attributed to a source, since
  nothing is ever tracked to check proximity against.
"""

import math

from config.red_zones import RIVER_Y_TOP
from game_knowledge import spawn_data
from perception.spawn_icon_reader import find_spawn_icon
from game_knowledge.cycle_tracker import load_card_costs

# RIVER_Y_TOP: top edge of the river band, imported directly (not
# duplicated) from config/red_zones.py's already-calibrated real-capture
# coordinates. Anything at or below this y is at-or-past the river toward
# OUR side; anything clearly above it is the opponent's own territory.
# Previously derived here ad hoc as WATER_ZONES[0][1] since red_zones.py
# had no real name for it — now that perception/unit_tracker.py needs the
# same value for its own position-based side signal, red_zones.py exposes
# RIVER_Y_TOP/RIVER_Y_BOTTOM directly; this just imports it.

BURST_MIN_FRACTION = 0.5  # a burst card's group only needs to reach this fraction of its full
# CARD_BURST_COUNT (see _burst_min_count) to resolve "played" instead of "uncertain" — added
# 2026-08-18 per Duncan's direct live report: goblins (expected 4) and Spear Goblins (expected
# 3) almost never get their FULL count confirmed simultaneously in one group, a real detection-
# recall limitation (see Project_Status's goblin-detection-rate findings — small green units,
# ~0.58 avg confidence in an earlier match), not a logic bug worth continuing to flag as
# "possible missed/extra detection" every single time. Requiring the exact full count was
# correct in spirit (avoids guessing) but was making a real, common, correctly-recognized-by-eye
# play resolve "uncertain" almost every time for these specific cards. 0.5 (floored at 2, capped
# at the card's own full count — see _burst_min_count) was picked directly from real log data:
# a genuine Goblins play in a live match log showed 2 of 4 confirmed together in one group at
# least once, never more — chosen to catch that case without inventing detections that never
# happened; still flags anything below the floor (e.g. a lone single goblin) as uncertain rather
# than guessing. Not tuned beyond that one data point — revisit if real logs show it's still
# too strict (or, in the other direction, ever causes an actual false "played").


def _burst_min_count(expected: int) -> int:
    """The minimum group size that still resolves a burst card as "played"
    instead of "uncertain" — see BURST_MIN_FRACTION's own comment for why
    this isn't a strict 100% match anymore. Floored at 2 (never loosens a
    2-unit card like Archers down to accepting a single detection — that
    specific risk wasn't reported/diagnosed, so left at its original
    strictness) and capped at `expected` itself (so a 1-count "burst" like
    Rascals' rascal_boy entry, whose real count is already only 1, can't
    end up needing MORE than its own full count — a real edge case this
    formula would otherwise break)."""
    return min(expected, max(2, math.ceil(expected * BURST_MIN_FRACTION)))


GROUPING_WINDOW_SECONDS = 0.5  # how long to buffer new same-class arrivals
# before explaining them as one group. Generous enough to absorb detection
# staggering across several frames at this project's ~10-27fps range
# (Project_Status), short enough that two genuinely separate real plays of
# the same card back-to-back won't get merged into one (own-side cycle
# tracking data showed real plays are typically several seconds apart at
# minimum, per Project_Status's first live-match log).

RECENT_DEATH_WINDOW_SECONDS = 2.0  # how long a vanished track stays
# eligible as an on-death spawn source. Real on-death spawns appear almost
# immediately; this can't tell true death from long occlusion (see module
# docstring), so it's deliberately short rather than generous.

SIDE_FLIP_WINDOW_SECONDS = 2.0  # how long a vanished ALLY track stays
# eligible as a "this new enemy is probably that same misclassified unit"
# source. Widened from an original 1.0s (2026-08-16, second live-match log):
# the check only runs when a candidate group CLOSES, which is already
# GROUPING_WINDOW_SECONDS (0.5s) after the enemy track first appeared, on
# top of whatever real gap existed between the ally vanishing and the new
# enemy being confirmed — a real case was traced where that combined delay
# was ~1.06s, just barely outside the original 1.0s window, so the ally
# vanish got pruned one frame before the check ran and the case leaked
# through as a false "played". Originally reasoned this didn't need to
# stay tight the way OWN_PLAY_WINDOW_SECONDS does for mirror-deck safety —
# turned out incomplete: a THIRD live-match log (2026-08-16) showed this
# window backfiring on its own, independent of length, when the "ally"
# reading feeding it was itself bogus (a genuinely enemy unit side.onnx
# mislabeled "ally" while it sat in the opponent's own territory) — see
# _could_be_ours, added the same day, which is the real fix for that
# failure mode. This window's length isn't the risk; an untrustworthy
# "ally" reading is, regardless of how long it stays eligible. Now matches
# RECENT_DEATH_WINDOW_SECONDS.

SPAWN_PROXIMITY_PX = 150  # max pixel distance between a new group's center
# and a candidate spawner's (last known) position. First-pass guess, not
# yet verified against real footage — see module docstring.

CONCURRENT_DUPLICATE_PROXIMITY_PX = 100  # tighter than SPAWN_PROXIMITY_PX —
# used for _find_concurrent_ally_duplicate, which is looking for essentially
# the SAME physical unit detected twice in one frame region (one box tagged
# ally, one tagged enemy), not a unit that's moved or a different nearby
# object. Roughly in line with unit_tracker.py's own MATCH_DISTANCE_PX=90
# for "close enough to plausibly be the same object."

CONCURRENT_DUPLICATE_MIN_AGE = 10  # ~1s of continuous confirmed tracking at
# this project's ~10fps floor (Project_Status) — long enough that a track
# that's just noisily oscillating (which tends to keep losing and regaining
# confirmation, resetting age each time) is unlikely to reach it, but short
# enough not to miss a real, quickly-relevant case. This is the trust
# signal for _find_concurrent_ally_duplicate specifically — see that
# method's docstring for why age instead of position.

OWN_PLAY_WINDOW_SECONDS = 2.0  # how long our own confirmed play of a card
# stays eligible to explain a same-card "new enemy" as a misclassification.
# Wider than SIDE_FLIP_WINDOW_SECONDS on purpose — this path also has to
# absorb game_knowledge.cycle_tracker.py's own HAND_STABLE_FRAMES debounce
# on top of unit_tracker's own confirmation delay, so the two systems'
# timestamps for "the same real event" can legitimately be over a second
# apart. The position check (RIVER_Y_TOP) is what keeps this window from
# being too permissive — see _check_own_play_collision.

PLAY_PERSISTENCE_MIN_AGE = 5  # minimum confirmed-frame age a track must
# reach by the time its candidate group closes before a "played" outcome
# is trusted, rather than downgraded to "uncertain". Added 2026-08-16
# against the first EXHAUSTIVE ground-truth match log (every real play
# accounted for, not just an impression-level review) — found several
# short-lived "giant" tracks (2-4 confirmed frames, then gone for good,
# never seen again) with no explaining ally/spawner context at all,
# generating false "played" events. Same category of problem as the
# earlier heal_spirit/inferno_tower/valkyrie/furnace/pekka wrong-class
# hallucinations (a units.onnx-level detection artifact, not a side or
# attribution bug) — just showing up as a common, real class this time
# instead of an obviously-wrong one, so nothing else caught it. A
# genuinely real play's track keeps accumulating age continuously past
# this point (confirmed directly: the one real giant in that match had
# age 7-8 by its own group's close time); a noise blip does not. Set
# above unit_tracker.py's own MIN_HITS_TO_CONFIRM=2 since that bar alone
# was already proven, by this same data, to still let noise through.
# First-pass value from one data point — tighten/loosen if real plays
# start getting wrongly downgraded, the same way every other threshold in
# this file has been tuned only after real data existed to tune it against.

# Buildings genuinely never move once placed — a different, cleaner problem
# than the troop-misclassification cases above. Real bug found 2026-08-16
# (fourth live match): the same physical Goblin Hut got read as "played"
# THREE separate times (three different track_ids, but bboxes within ~5px
# of each other for the whole match) — its track evidently kept getting
# lost (occlusion by passing troops, a confidence dip, etc.) and
# re-acquired as a "new" track by unit_tracker.py, well past what a single
# grace period could bridge. Only 1 of those 3 was a real play; Duncan
# confirmed directly ("he did not actually play 4" [huts]). Restricted to
# genuinely stationary classes only — deliberately excludes goblin_drill
# (surfaces after a delay, arguably still "placed" fresh each time) and
# cannon_cart (transforms/relocates after damage) since neither is as
# unambiguously static as the rest of this list.
BUILDING_CLASSES = frozenset({
    "barbarian_hut", "bomb_tower", "cannon", "elixir_collector", "furnace",
    "goblin_cage", "goblin_hut", "inferno_tower", "mortar", "tesla",
    "tombstone", "x_bow",
})

BUILDING_REACQUIRE_PROXIMITY_PX = 40  # much tighter than SPAWN_PROXIMITY_PX
# on purpose — a real building's bbox should barely move at all between
# reads (a few px of detection jitter, not the tens-of-px drift a moving
# troop has), so a tight tolerance is both safe and more precise here.

BUILDING_REACQUIRE_WINDOW_SECONDS = 45.0  # widened from an original 8.0s
# (2026-08-16, sixth live-match log): a real goblin_hut's track went fully
# untracked for ~8.2s — juuust past the original window — right as it
# spawned a spear_goblin, so the spawn-attribution lookup (see
# _find_spawn_source's use of this same list, added the same day) missed
# it by a hair and the spawn got misread as a fresh play. Buildings like
# Goblin Hut/Tombstone/Barbarian Hut/Furnace have real multi-tens-of-
# seconds lifetimes in the actual game, so an 8s cutoff was never matched
# to what it needed to bridge — widened generously past that rather than
# re-tuned to the one observed gap. A lost building track has no "it must
# still be nearby" time pressure the way an occluded moving troop does (it
# is not going anywhere), so there's little cost to being generous here;
# the tight BUILDING_REACQUIRE_PROXIMITY_PX is what keeps this safe, not
# the window length. A genuine same-tile rebuild within this window being
# (harmlessly) treated as a re-acquisition instead of a fresh play is an
# accepted, rare edge case, not chased further here.


class OpponentPlayTracker:
    def __init__(self, own_deck: list = ()):
        self.costs = load_card_costs()
        self.own_deck = set(own_deck)
        self.reset()

    def reset(self):
        """Clears all state for a new match — no automatic match-boundary
        detection exists anywhere in this project yet (same already-
        documented gap as cycle_tracker.py), so this is called manually."""
        self._prev_ids = set()
        self._prev_by_id = {}
        self._recently_vanished = []  # [{track_id, name, bbox, vanished_at}]
        self._prev_ally_ids = set()
        self._prev_ally_by_id = {}
        self._recently_vanished_ally = []  # [{track_id, name, bbox, vanished_at}]
        self._recently_vanished_buildings = []  # [{track_id, name, bbox, vanished_at}] -- enemy only
        self._own_recent_plays = []  # [(card_slug, timestamp)] -- set fresh each update() call
        self._pending = {}  # class_name -> {"members": [(track_id, bbox)], "first_seen": t}
        self.plays_this_match = 0

    def _prune_vanished(self, now: float):
        cutoff = now - RECENT_DEATH_WINDOW_SECONDS
        self._recently_vanished = [v for v in self._recently_vanished if v["vanished_at"] >= cutoff]

    def _prune_vanished_ally(self, now: float):
        cutoff = now - SIDE_FLIP_WINDOW_SECONDS
        self._recently_vanished_ally = [v for v in self._recently_vanished_ally if v["vanished_at"] >= cutoff]

    def _prune_vanished_buildings(self, now: float):
        cutoff = now - BUILDING_REACQUIRE_WINDOW_SECONDS
        self._recently_vanished_buildings = [v for v in self._recently_vanished_buildings if v["vanished_at"] >= cutoff]

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def _group_center(self, members):
        centers = [self._center(bbox) for _, bbox in members]
        return (sum(c[0] for c in centers) / len(centers),
                sum(c[1] for c in centers) / len(centers))

    def _find_spawn_source(self, spawned_class: str, group_center, alive_by_id: dict):
        """Searches recently-vanished (on_death spawners) and currently-
        alive (periodic spawners) enemy tracks for a plausible source of
        spawned_class near group_center. Returns (source_track_id,
        source_card) or (None, None) if nothing plausible is nearby."""
        for spawner, kind, _count in spawn_data.spawners_of(spawned_class):
            if kind == "on_death":
                for v in self._recently_vanished:
                    if v["name"] == spawner and self._dist(self._center(v["bbox"]), group_center) <= SPAWN_PROXIMITY_PX:
                        return v["track_id"], spawner
            elif kind == "periodic":
                for tid, track in alive_by_id.items():
                    if track["name"] == spawner and self._dist(self._center(track["bbox"]), group_center) <= SPAWN_PROXIMITY_PX:
                        return tid, spawner
                # Fallback added 2026-08-16: every periodic spawner in
                # spawn_data.py that's also a building (goblin_hut,
                # tombstone, barbarian_hut, furnace) can lose live tracking
                # for many seconds at a time without actually being gone —
                # same reasoning _find_building_reacquisition already uses.
                # A real goblin_hut spawning a spear_goblin was missed here
                # because this branch only ever checked alive_by_id, never
                # the building's own recently-vanished record. Troop
                # spawners (witch, night_witch, skeleton_king) are
                # deliberately NOT covered by this fallback — a troop that
                # stops being tracked may really have died, unlike a
                # building, so "currently alive" stays the right bar there.
                if spawner in BUILDING_CLASSES:
                    for v in self._recently_vanished_buildings:
                        if v["name"] == spawner and self._dist(self._center(v["bbox"]), group_center) <= SPAWN_PROXIMITY_PX:
                            return v["track_id"], spawner
        return None, None

    def _find_ally_flip_source(self, class_name: str, group_center, now: float):
        """Searches recently-vanished ALLY tracks of the same class near
        group_center — a plausible sign the "new enemy" is really one of
        our own units the side classifier just flipped. Returns
        (track_id, elapsed_seconds) or (None, None)."""
        for v in self._recently_vanished_ally:
            if v["name"] == class_name and self._dist(self._center(v["bbox"]), group_center) <= SPAWN_PROXIMITY_PX:
                return v["track_id"], now - v["vanished_at"]
        return None, None

    def _find_concurrent_ally_duplicate(self, class_name: str, group_center, alive_ally_by_id: dict):
        """Searches CURRENTLY ALIVE ally tracks (not vanished) of the same
        class near group_center — catches a DIFFERENT failure mode than
        _find_ally_flip_source: two SIMULTANEOUS conflicting detections of
        the same physical unit (one tagged ally, one tagged enemy) rather
        than a temporal flip-then-vanish. Confirmed for real 2026-08-16: an
        ally musketeer stayed continuously visible the whole time a nearby
        "enemy" musketeer track was born and lived a few tens of pixels
        away — it never vanished, so _find_ally_flip_source (which only
        ever looks at VANISHED tracks) could not have caught this no
        matter how the windows were tuned.

        Deliberately uses age, not _could_be_ours's position check, as the
        trust signal — found while testing THIS exact real case: the real
        ally musketeer had already pushed past the river (same "our unit
        legitimately crossed" situation _could_be_ours already accepts as
        an unavoidable blind spot for a VANISHED reading), so the position
        gate would have rejected the real fix here too. The difference is
        this track is CURRENTLY alive — its age is direct, measured
        evidence of how established it is, not an indirect proxy the way
        position is for a track we no longer have live state on. Still
        applies _class_in_own_deck (unambiguous, no reason to drop it).
        Returns (track_id, distance_px) or (None, None)."""
        for tid, track in alive_ally_by_id.items():
            if (track["name"] == class_name and track["age"] >= CONCURRENT_DUPLICATE_MIN_AGE
                    and self._class_in_own_deck(track["name"])):
                d = self._dist(self._center(track["bbox"]), group_center)
                if d <= CONCURRENT_DUPLICATE_PROXIMITY_PX:
                    return tid, d
        return None, None

    def _find_building_reacquisition(self, class_name: str, group_center, now: float, own_track_ids):
        """Searches recently-vanished ENEMY buildings of the same class
        very close to group_center — a real building whose track was lost
        and just got re-acquired, not a fresh placement. See
        BUILDING_CLASSES/BUILDING_REACQUIRE_PROXIMITY_PX's comments for
        why this is a different, cleaner problem than the troop-side
        misclassification checks above (a building never moves). Returns
        (track_id, elapsed_seconds) or (None, None).

        own_track_ids excludes a group's own track(s) from matching against
        THEMSELVES — real bug, caught 2026-08-16: a track can flicker
        visible True->False->True within its own life, before its own
        candidate group even closes (GROUPING_WINDOW_SECONDS gives it time
        to). That self-flicker gets recorded as a "vanish" the same as any
        other, and without this exclusion a brand-new building would match
        against its own flicker record — same position by construction —
        and get wrongly reported "reacquired" instead of "played" the very
        first time it's ever seen."""
        for v in self._recently_vanished_buildings:
            if v["track_id"] in own_track_ids:
                continue
            if v["name"] == class_name and self._dist(self._center(v["bbox"]), group_center) <= BUILDING_REACQUIRE_PROXIMITY_PX:
                return v["track_id"], now - v["vanished_at"]
        return None, None

    def _check_own_play_collision(self, card_slug: str, group_center, now: float):
        """Duncan's own catch (2026-08-16): timing alone isn't safe here —
        if the opponent runs a similar/mirror deck, they could genuinely
        play the same card around the same time we do, and suppressing
        that would be a real false negative, not a fix. Position is what
        makes this safe: a GENUINE opponent play can only first appear on
        their own side of the board, never ours — so this only fires when
        BOTH the card and timing line up with one of our own recent plays
        AND the new group's position isn't clearly on the opponent's own
        side (i.e. it's at/past the river, consistent with being one of
        our own units instead). Returns the matching own-play timestamp,
        or None if no collision (including: real match on card+time, but
        position looks like genuine opponent territory — not suppressed)."""
        if group_center[1] < RIVER_Y_TOP:
            return None  # clearly the opponent's own side — never suppress on timing alone
        for slug, t in self._own_recent_plays:
            if slug == card_slug and abs(now - t) <= OWN_PLAY_WINDOW_SECONDS:
                return t
        return None

    def _class_in_own_deck(self, unit_class: str) -> bool:
        """Exact, unambiguous half of the "could this ally reading be
        real" question — is this even a card we could own at all. An
        "ally" reading of a class not in our deck (a goblin or
        spear_goblin trickling from the opponent's own Goblin Hut, say)
        is automatically wrong, no heuristic needed. If self.own_deck
        wasn't provided, this is permissive (True) rather than rejecting
        everything."""
        if not self.own_deck:
            return True
        slug, _cost = self._card_slug_and_cost(unit_class, 1)
        return slug in self.own_deck

    def _could_be_ours(self, unit_class: str, bbox) -> bool:
        """Gate for trusting a VANISHED "ally"-labeled track as real,
        added 2026-08-16 after a third live-match log showed the
        ally-vanish check backfiring: a genuinely ENEMY giant got
        mislabeled "ally" by side.onnx while sitting deep in the
        opponent's OWN territory (bbox center y well above RIVER_Y_TOP,
        nowhere our own units could legitimately be that early in their
        life) — every subsequent real appearance of that same enemy giant
        then got wrongly suppressed as "misclassified", since the bogus
        "ally" reading was treated as ground truth. Two independent
        checks, both must pass: (1) _class_in_own_deck, and (2) position
        has to be plausible for our side (at/past the river) — a
        heuristic, not a certainty (our own units DO legitimately cross
        the river later in a match), but rules out exactly the failure
        mode that motivated this method. Position is the only trust
        signal available for a track we no longer have live state for —
        see _find_concurrent_ally_duplicate for why a CURRENTLY-alive
        track can and should be judged differently (by age, not
        position)."""
        return self._class_in_own_deck(unit_class) and self._center(bbox)[1] >= RIVER_Y_TOP

    def _card_slug_and_cost(self, unit_class: str, count: int):
        if unit_class == "musketeer":
            slug = "three_musketeers" if count == 3 else "musketeer"
        elif unit_class == "skeleton":
            slug = "skeleton_army" if count >= spawn_data.SKELETON_ARMY_BURST_MIN else "skeletons"
        else:
            slug = spawn_data.CARD_SLUG_FOR_BURST.get(unit_class, unit_class)
        return slug, self.costs.get(slug)

    def _base_event(self, name, count, track_ids, bboxes, now):
        return {"card": name, "count": count, "track_ids": track_ids, "bboxes": bboxes, "frame_time": now}

    def _played_event(self, name, count, track_ids, bboxes, now, slug, cost):
        e = self._base_event(name, count, track_ids, bboxes, now)
        e.update({"attribution": "played", "card_slug": slug, "elixir_cost": cost,
                   "source_track_id": None, "source_card": None, "flag_reason": None})
        self.plays_this_match += 1
        return e

    def _group_persisted(self, track_ids, alive_by_id: dict) -> bool:
        """See PLAY_PERSISTENCE_MIN_AGE's comment for why this check
        exists. True if at least one member of the group has reached the
        minimum confirmed-frame age by now (looked up in alive_by_id,
        this frame's live enemy tracks) — a track fully dropped from
        alive_by_id by close time, or one that never accumulated enough
        confirmed frames before going invisible for good, fails this."""
        return any(alive_by_id.get(tid, {}).get("age", 0) >= PLAY_PERSISTENCE_MIN_AGE for tid in track_ids)

    def _finalize_played(self, name, count, track_ids, bboxes, now, slug, cost, alive_by_id):
        """Every "-> played" outcome funnels through here so
        _group_persisted applies uniformly across all the different
        places a "played" result can be produced (ordinary single-unit,
        a matched multi-unit burst, the skeleton/musketeer special
        cases), rather than needing the same check repeated at each one."""
        if not self._group_persisted(track_ids, alive_by_id):
            return self._uncertain_event(name, count, track_ids, bboxes, now,
                f"{name} was detected but didn't persist past a few confirmed frames before its "
                f"window closed — likely a brief model misdetection, not confirmed as a real play")
        return self._played_event(name, count, track_ids, bboxes, now, slug, cost)

    def _spawned_event(self, name, count, track_ids, bboxes, now, source_id, source_card):
        e = self._base_event(name, count, track_ids, bboxes, now)
        e.update({"attribution": "spawned", "card_slug": None, "elixir_cost": None,
                   "source_track_id": source_id, "source_card": source_card, "flag_reason": None})
        return e

    def _uncertain_event(self, name, count, track_ids, bboxes, now, reason):
        e = self._base_event(name, count, track_ids, bboxes, now)
        e.update({"attribution": "uncertain", "card_slug": None, "elixir_cost": None,
                   "source_track_id": None, "source_card": None, "flag_reason": reason})
        return e

    def _misclassified_event(self, name, count, track_ids, bboxes, now, source_id, reason):
        e = self._base_event(name, count, track_ids, bboxes, now)
        e.update({"attribution": "misclassified", "card_slug": None, "elixir_cost": None,
                   "source_track_id": source_id, "source_card": name, "flag_reason": reason})
        return e

    def _reacquired_event(self, name, count, track_ids, bboxes, now, source_id, elapsed):
        e = self._base_event(name, count, track_ids, bboxes, now)
        e.update({"attribution": "reacquired", "card_slug": None, "elixir_cost": None,
                   "source_track_id": source_id, "source_card": name,
                   "flag_reason": f"same {name} lost tracking {elapsed:.1f}s ago and just got a new "
                                   f"track_id at nearly the same position — not a fresh placement"})
        return e

    def _explain_skeleton(self, count, track_ids, bboxes, now, center, alive_by_id):
        """"skeleton" is three-way ambiguous by count alone: Skeletons(3),
        Skeleton Army(~15), or a Tombstone/Witch/Skeleton King trickle(1-2)
        — handled separately from the generic single-expected-count path
        below since it has no single fixed "expected" burst size."""
        name = "skeleton"
        if count >= spawn_data.SKELETON_ARMY_BURST_MIN:
            slug, cost = self._card_slug_and_cost(name, count)
            return self._finalize_played(name, count, track_ids, bboxes, now, slug, cost, alive_by_id)
        if count == spawn_data.CARD_BURST_COUNT["skeleton"]:
            slug, cost = self._card_slug_and_cost(name, count)
            return self._finalize_played(name, count, track_ids, bboxes, now, slug, cost, alive_by_id)
        source_id, source_card = self._find_spawn_source(name, center, alive_by_id)
        if source_id is not None:
            return self._spawned_event(name, count, track_ids, bboxes, now, source_id, source_card)
        return self._uncertain_event(name, count, track_ids, bboxes, now,
            f"{count} skeleton appeared together — doesn't match Skeletons (3), "
            f"Skeleton Army (>={spawn_data.SKELETON_ARMY_BURST_MIN}), or a known nearby spawner")

    def _explain_musketeer(self, count, track_ids, bboxes, now, alive_by_id):
        """"musketeer" is the one class valid at two different fixed
        counts as two different real cards (1=Musketeer, 3=Three
        Musketeers) — caught during design before this shipped with a
        single-expected-count bug that would have flagged every ordinary
        Musketeer play as "uncertain". See spawn_data.py's note on why
        this isn't in CARD_BURST_COUNT."""
        name = "musketeer"
        if count in (1, 3):
            slug, cost = self._card_slug_and_cost(name, count)
            return self._finalize_played(name, count, track_ids, bboxes, now, slug, cost, alive_by_id)
        return self._uncertain_event(name, count, track_ids, bboxes, now,
            f"{count} musketeer appeared together; expected 1 (Musketeer) or 3 (Three Musketeers)")

    def _explain_group(self, name, members, now, alive_by_id, alive_ally_by_id, screenshot=None):
        count = len(members)
        track_ids = [tid for tid, _ in members]
        bboxes = [bbox for _, bbox in members]
        center = self._group_center(members)

        # Checked before EVERYTHING else, when a screenshot is available: the
        # deploy-delay spawn icon (perception/spawn_icon_reader.py, added
        # 2026-08-17) is a direct visual read of which side just placed a
        # card -- independent of this group's own already-enemy-filtered but
        # potentially WRONG side classification (the persistence-lock and
        # other side-misread bugs this whole feature exists to catch). Per
        # Duncan, the icon sits near ONE representative member's board
        # position for a multi-unit group (a "flag"/lead unit, not
        # necessarily always the same one), so every member's bbox is tried
        # in turn, keeping the first icon actually found. screenshot is
        # optional and None by default (existing tests/callers with no pixel
        # access) -- with no screenshot this check is skipped entirely and
        # behavior is IDENTICAL to before this feature existed, same
        # graceful-degradation spirit as every other optional signal in this
        # project. Only acts on an "ally" read (an "enemy" read just
        # confirms what was already assumed, so no action needed) or no read
        # at all (icon's brief window may have already passed by the time
        # this group's GROUPING_WINDOW_SECONDS closed -- not new information
        # either way, falls through to the existing checks below unchanged).
        if screenshot is not None:
            icon_side = None
            for bbox in bboxes:
                icon_side, _icon_area, _icon_bbox = find_spawn_icon(screenshot, bbox)
                if icon_side is not None:
                    break
            if icon_side == "ally":
                return self._misclassified_event(name, count, track_ids, bboxes, now, None,
                    f"the deploy-delay spawn icon near this {name} read ALLY (blue), not enemy — "
                    f"a direct visual signal this is one of our own units, not a genuine enemy play")

        # Checked first of all: a building never moves, so a "new" one
        # appearing right where one already was is almost certainly the
        # same physical building re-acquiring its track, not a fresh play.
        if name in BUILDING_CLASSES:
            src_id, elapsed = self._find_building_reacquisition(name, center, now, track_ids)
            if src_id is not None:
                return self._reacquired_event(name, count, track_ids, bboxes, now, src_id, elapsed)

        # Checked before anything else: if this looks like it might just be
        # one of our own units the side classifier briefly mislabeled, none
        # of the card-identity logic below is even meaningful to apply.
        flip_id, elapsed = self._find_ally_flip_source(name, center, now)
        if flip_id is not None:
            return self._misclassified_event(name, count, track_ids, bboxes, now, flip_id,
                f"an ally {name} vanished {elapsed:.1f}s ago at a nearby position — "
                f"likely a side-classification flip, not a real enemy play")

        # Same idea, different shape: a currently-ALIVE (never vanished)
        # ally of the same class sitting right on top of this "new enemy"
        # — two simultaneous conflicting reads of one physical unit rather
        # than a flip-then-vanish. See _find_concurrent_ally_duplicate.
        dup_id, dup_dist = self._find_concurrent_ally_duplicate(name, center, alive_ally_by_id)
        if dup_id is not None:
            return self._misclassified_event(name, count, track_ids, bboxes, now, dup_id,
                f"an ally {name} is alive right now only {dup_dist:.0f}px away — likely two "
                f"conflicting reads of the same physical unit, not a real enemy play")

        # Second check, also before any card-identity logic: did WE just
        # play the same card ourselves, at a position not clearly on the
        # opponent's own side of the board? See _check_own_play_collision's
        # docstring for why position is required alongside timing — timing
        # alone would wrongly suppress a genuine mirror-deck coincidence.
        tentative_slug, _tentative_cost = self._card_slug_and_cost(name, count)
        collision = self._check_own_play_collision(tentative_slug, center, now)
        if collision is not None:
            own_t = collision
            return self._misclassified_event(name, count, track_ids, bboxes, now, None,
                f"we played {tentative_slug} ourselves {now - own_t:.1f}s ago and this new enemy "
                f"{name} isn't clearly on the opponent's side of the board — likely the same "
                f"misclassified unit, not a genuine (possibly mirror-deck) opponent play")

        if name == "skeleton":
            return self._explain_skeleton(count, track_ids, bboxes, now, center, alive_by_id)
        if name == "musketeer":
            return self._explain_musketeer(count, track_ids, bboxes, now, alive_by_id)

        expected = spawn_data.CARD_BURST_COUNT.get(name)

        # 1. Match against this card's own known burst size -- loosened
        #    2026-08-18 to accept at least _burst_min_count(expected)
        #    rather than requiring the exact full count every time (see
        #    BURST_MIN_FRACTION's own comment for why). Still capped at
        #    `expected` itself -- MORE than the real card's count is still
        #    anomalous, not accepted here.
        if expected is not None and _burst_min_count(expected) <= count <= expected:
            slug, cost = self._card_slug_and_cost(name, count)
            return self._finalize_played(name, count, track_ids, bboxes, now, slug, cost, alive_by_id)

        # 2. Pure byproduct — never independently playable, always a spawn
        #    even when a specific source can't be pinned down.
        if name in spawn_data.PURE_BYPRODUCT_CLASSES:
            source_id, source_card = self._find_spawn_source(name, center, alive_by_id)
            return self._spawned_event(name, count, track_ids, bboxes, now, source_id, source_card)

        # 3. Known spawn target — try to explain via a nearby/recent
        #    spawner before falling through to "uncertain" below.
        if name in spawn_data.AMBIGUOUS_SPAWNED_CLASSES:
            source_id, source_card = self._find_spawn_source(name, center, alive_by_id)
            if source_id is not None:
                return self._spawned_event(name, count, track_ids, bboxes, now, source_id, source_card)

        # 4. A known multi-unit card, but this count matches neither its
        #    own burst size nor any known spawner — flag rather than guess.
        if expected is not None:
            return self._uncertain_event(name, count, track_ids, bboxes, now,
                f"{count} {name} appeared together; {name.replace('_', ' ')} cards come in groups "
                f"of {expected} (need at least {_burst_min_count(expected)} confirmed together) — "
                f"possible missed/extra detection")

        # 5. Ordinary single-unit card, the common case.
        if count == 1:
            slug, cost = self._card_slug_and_cost(name, count)
            return self._finalize_played(name, count, track_ids, bboxes, now, slug, cost, alive_by_id)

        # 6. Unexpected multiples of what should be a single-unit card.
        return self._uncertain_event(name, count, track_ids, bboxes, now,
            f"{count} {name} appeared together; {name} is normally a single-unit card")

    def update(self, tracked_units: list, now: float, own_recent_plays: list = (), screenshot=None) -> list:
        """tracked_units: perception.unit_tracker.UnitTracker.current_state()'s
        output (confirmed tracks only, both sides). now: a monotonically
        increasing timestamp (e.g. time.time()). own_recent_plays: a list of
        (card_slug, timestamp) for our own recently-confirmed plays (e.g.
        from game_knowledge.cycle_tracker.CycleTracker's last_play events,
        accumulated by the caller — see perception/debug_live.py) — used to
        catch the same-card-as-us misclassification case, see
        _check_own_play_collision. Safe to pass the full match's history
        each call; only entries within OWN_PLAY_WINDOW_SECONDS matter.
        screenshot: the current full-frame BGR image, optional (default
        None) — enables the spawn-icon crosscheck in _explain_group() (see
        its own docstring); omit for existing callers/tests with no pixel
        access, behavior is unchanged from before that feature existed.
        Returns this frame's closed play-group events (usually [])."""
        self._own_recent_plays = own_recent_plays
        enemy = [u for u in tracked_units if u["side"] == "enemy"]
        current_by_id = {u["track_id"]: u for u in enemy}
        current_ids = set(current_by_id)

        ally = [u for u in tracked_units if u["side"] == "ally"]
        current_ally_by_id = {u["track_id"]: u for u in ally}
        current_ally_ids = set(current_ally_by_id)

        new_ids = current_ids - self._prev_ids

        # "Vanished" means EITHER fully dropped from tracked_units OR just
        # flickered invisible while still present — the second case is the
        # dominant real one (confirmed 2026-08-16 from a second live-match
        # log): unit_tracker.py's own grace period keeps a track sitting in
        # tracked_units, visible=False, for up to MAX_MISSED_FRAMES frames
        # after detection actually stops, so a track_id set-membership diff
        # alone misses it entirely — the track_id never leaves the dict
        # during that window. Both a real giant and a real musketeer's
        # misclassification cycle happened completely within that grace
        # window: the ally track stayed present the whole time (same
        # track_id, visible flickering False then True again) while a
        # separate "enemy" track of the same class was born and confirmed
        # in the meantime. Only watching set membership (the original,
        # insufficient version of this check) never saw the ally-side event
        # at all, so it could never explain the enemy-side one either.
        newly_invisible_ids = {tid for tid, track in current_by_id.items()
                                if self._prev_by_id.get(tid, {}).get("visible") and not track["visible"]}
        vanished_ids = (self._prev_ids - current_ids) | newly_invisible_ids
        newly_invisible_ally_ids = {tid for tid, track in current_ally_by_id.items()
                                     if self._prev_ally_by_id.get(tid, {}).get("visible") and not track["visible"]}
        vanished_ally_ids = (self._prev_ally_ids - current_ally_ids) | newly_invisible_ally_ids

        for vid in vanished_ids:
            info = current_by_id.get(vid) or self._prev_by_id.get(vid)
            if info is not None:
                self._recently_vanished.append(
                    {"track_id": vid, "name": info["name"], "bbox": info["bbox"], "vanished_at": now})
                if info["name"] in BUILDING_CLASSES:
                    self._recently_vanished_buildings.append(
                        {"track_id": vid, "name": info["name"], "bbox": info["bbox"], "vanished_at": now})
        self._prune_vanished(now)
        self._prune_vanished_buildings(now)

        for vid in vanished_ally_ids:
            info = current_ally_by_id.get(vid) or self._prev_ally_by_id.get(vid)
            # Gate added 2026-08-16 (see _could_be_ours's docstring): only
            # trust this "ally" reading as a real flip source if it's
            # actually plausible we own it. Without this, a genuinely
            # enemy unit that side.onnx mislabels "ally" (seen for real,
            # sitting deep in the opponent's own territory) gets treated
            # as ground truth, and every later real appearance of that
            # same enemy unit gets wrongly suppressed as "misclassified" —
            # backwards from what this check is supposed to catch.
            if info is not None and self._could_be_ours(info["name"], info["bbox"]):
                self._recently_vanished_ally.append(
                    {"track_id": vid, "name": info["name"], "bbox": info["bbox"], "vanished_at": now})
        self._prune_vanished_ally(now)

        for nid in new_ids:
            track = current_by_id[nid]
            group = self._pending.setdefault(track["name"], {"members": [], "first_seen": now})
            group["members"].append((nid, track["bbox"]))

        events = []
        for name in list(self._pending.keys()):
            group = self._pending[name]
            if now - group["first_seen"] >= GROUPING_WINDOW_SECONDS:
                events.append(self._explain_group(name, group["members"], now, current_by_id, current_ally_by_id, screenshot))
                del self._pending[name]

        self._prev_ids = current_ids
        self._prev_by_id = current_by_id
        self._prev_ally_ids = current_ally_ids
        self._prev_ally_by_id = current_ally_by_id
        return events
