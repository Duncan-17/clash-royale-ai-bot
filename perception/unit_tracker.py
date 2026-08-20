"""
Assigns persistent track IDs to unit_detector.py's raw per-frame
detections, so the same physical unit is recognized as the same unit
across frames instead of looking like it keeps dying and respawning.

Why this is needed: unit_detector.py is completely stateless — every
frame it reports "here's what I currently see," with no memory of
previous frames. Detection confidence naturally dips for a frame or two
during animation, rotation, or brief occlusion behind another unit, which
makes a real, unmoving unit vanish from one frame's detection list and
reappear in the next. Confirmed live (2026-08-15, Duncan): without
tracking, this looks exactly like units repeatedly being placed and dying
— it's a raw-detection flicker problem, not a placement/death problem, so
the fix belongs here, not in unit_detector.py or in the (not yet built)
cycle-tracking logic that will eventually consume this.

Approach: greedy nearest-neighbor matching between frames, gated by a
COMPATIBLE side (see _side_compatible's docstring for why "unknown" is a
wildcard, not a hard mismatch) and a search area around each track's own
CURRENT-FRAME match position — see "Reacquisition" below for what that
position actually is once a track has missed a frame. A track survives a
grace period of consecutive missed frames (MAX_MISSED_FRAMES) rather than
being dropped the instant one frame doesn't detect it. Deliberately
simpler than a full tracker like SORT/DeepSORT (no Kalman filtering, no
learned re-identification features, no Hungarian-optimal assignment —
still a greedy sort) — see Reacquisition below for the one piece of
"real tracker" machinery this DOES borrow (velocity-based prediction) and
why a full Kalman filter wasn't judged worth it for this game specifically.
Known limitation: several identical units of the same card standing close
together (e.g. a Skeleton Army) can have their track IDs swapped between
frames, since nothing here distinguishes between two visually-identical
units of the same type — accepted for now, since nothing downstream needs
per-individual continuity yet, only "how many of X are on the board" and
"is this a new appearance or the same one as before," both of which still
work correctly even if two interchangeable units swap IDs.

CLASS accumulator (added 2026-08-19/20, per the "perception fusion
architecture" design doc): matching no longer requires an EXACT class-
name match between a track and a new detection — only side-compatibility
and being within the track's own search area. A track's REPORTED class
(det["name"]/track["name"]) is instead resolved by _resolve_class's own
debounce (CLASS_LOCK_HITS), the same "accumulate, don't overwrite"
principle already applied to side. Why: under the old hard-equality
matching rule, a single misclassified frame on an otherwise-stable track
(the documented Valkyrie/spear_goblin<->archer hallucination bugs) failed
to match ANYTHING and spun up a separate phantom candidate track instead
of just being absorbed as noise — this fixes that specific failure mode.
It does NOT fix the base model actually hallucinating a wrong class for
several CONSECUTIVE frames in a row (a real, sustained misread will still
eventually relabel the track after CLASS_LOCK_HITS consistent frames,
same as it would visually look "right" to a human watching those frames
too) — that class of bug is a units.onnx accuracy limitation, not
something tracking-layer logic can paper over.

Reacquisition (added 2026-08-19/20, per Duncan's own diagnosis of a real
regression + the fusion-architecture doc): a track that's missed at least
one frame no longer just gets matched against its LAST known position —
see _predicted_center. Each track keeps a simple EMA-smoothed velocity
estimate (_update_velocity), and while missing, its search area is
anchored to that velocity-extrapolated PREDICTED position instead, sized
per how many frames it's been missing (see _search_radius_ok). This is
the direct fix for the already-documented "troop-tracking-gap double-
counting" bug: a unit that drops out of detection mid-walk used to only
be searched for near where it was LAST seen, so if it re-appeared several
tile-widths further along its own path (a real, common case — a fast
unit walking during a ~1s gap), it missed the old MATCH_DISTANCE_PX
radius entirely and got a brand-new track_id instead of reconnecting to
its own history.

Scope decision made WITH Duncan before building this (see conversation):
a full Kalman filter (proper state covariance, IoU + Hungarian-optimal
matching) was considered and explicitly NOT built — simple linear
velocity extrapolation gets the same practical fix at a fraction of the
implementation/tuning cost, and Duncan's own correction is important
context for why: a Kalman filter's constant-velocity process model has
the EXACT SAME blind spot to sudden direction changes that linear
extrapolation does (most CR troops target the nearest enemy and can
retarget/get knocked back at any moment — only a narrow slice of win-
condition cards reliably beeline in a straight line), so the extra
machinery wouldn't actually buy back the thing that matters most for
THIS game. What DOES matter for that specific problem, and what
_search_radius_ok actually does about it: perception/unit_metadata.py's
TARGET_TYPE data (targets_buildings_only()) shapes the search area
itself — TIGHT and DIRECTIONAL (an ellipse along the track's own recent
heading) for a predictable building-targeter with a clear velocity, WIDE
and circular (direction not trusted at all) for everything else. Card
movement SPEED (to size the radius more precisely per unit type, e.g.
Hog Rider vs. a slow tank) is a known, not-yet-filled gap — see
game_knowledge/card_reference.json's movement_speed field — the current
radius constants are one-size-fits-all placeholders pending that data,
not tuned against real per-card speeds yet.

Confirmation delay (MIN_HITS_TO_CONFIRM, added 2026-08-15): a brand-new
track isn't reported anywhere (not in update()'s return, not in
current_state()) until it's been matched MIN_HITS_TO_CONFIRM times.
Motivated by log analysis of a full match (see Project_Status): 44% of
all track IDs ever created (158 of 356) were matched exactly ONCE, ever
— pure single-frame noise blips, never a real unit. Requiring 2 hits
before reporting eliminates all of those outright, and — importantly —
doesn't touch how an ESTABLISHED track behaves at all (missed_frames/
grace-period logic is unchanged); it only delays a brand-new track's
first appearance by one extra matched frame. Real units are still
reported quickly (2 frames at ~12fps is under 200ms); one-off noise
never surfaces at all. This is a real, measured, but partial fix — see
Project_Status for what's known to remain and why.

Usage:
    tracker = UnitTracker()
    tracked = tracker.update(unit_detector.detect(screenshot))
    # tracked is the same list of dicts unit_detector.detect() returns,
    # each with an added "track_id" (stable across calls) and "age"
    # (frames since first seen) field. This is the raw, instant-by-instant
    # read — it will still flicker frame to frame, correctly, since it's
    # reporting exactly what was detected THIS frame, nothing smoothed.
    #
    # For the smoothed view (bridges brief occlusion instead of reporting
    # it as gone), call current_state() instead/as well — see its
    # docstring for what "visible" does and doesn't mean.

Confidence is also persisted per track (added 2026-08-16, alongside the
side-persistence change above) so current_state() can report a
detection-confidence view too, not just side — see
perception/debug_live.py's 'u' key toggle, which needs this to color the
TRACKED grid the same way as the LIVE one in "confidence" mode. Unlike
side, confidence is simply overwritten every matched frame (there's no
"unknown" state for it to persist through — a detection always carries a
real confidence score once it exists at all).
"""

import math

from config.red_zones import RIVER_Y_BOTTOM, RIVER_Y_TOP
from game_knowledge.zone_classifier import classify_zone
from perception.spawn_icon_reader import find_spawn_icon
from perception.unit_metadata import targets_buildings_only

MAX_MISSED_FRAMES = 8  # grace period before a track is dropped — a few hundred ms at ~12fps,
# long enough to bridge animation/rotation flicker, short enough that a genuinely-gone unit
# doesn't linger falsely for long
MATCH_DISTANCE_PX = 90  # max center-to-center distance to consider two boxes the same unit
# across consecutive frames — generous enough for fast units' per-frame movement plus normal
# detection-box jitter, tight enough to avoid matching two distinct nearby units
MIN_HITS_TO_CONFIRM = 2  # a track must be matched this many times before it's reported anywhere —
# see module docstring, kills single-frame noise blips (44% of all tracks in a real match) at
# the cost of a one-extra-frame delay (~<200ms at ~12fps) before a genuine new unit is reported

SIDE_LOCK_HITS = 2  # a badge-based side reading must repeat this many times in a row before
# it's actually written into a track's side — see _resolve_side's docstring. Added 2026-08-18
# after a 2026-08-17 review of real opponent-play-tracker output found a track locking its side
# on a SINGLE bad frame (22 correct "unknown" reads, then one stray "enemy" read, stuck for the
# rest of the track's life — see Project_Status "Opponent-play-tracker debug-image review").

# Priority TIERS for _resolve_side's three side signals, added 2026-08-18 (same day, later
# session) after a real regression report: an earlier version of this fix let POSITION
# permanently LOCK a track's side the moment it resolved, which meant a wrong position guess
# (a legitimate ally push that already crossed the river, or a placement into a neutral zone --
# see _side_from_position's own known gap) could never be corrected by the BADGE even once a
# clean, repeated, genuinely reliable badge reading became available. Duncan's own explicit
# priority order, restated here as the real design: position is only ever a FIRST GUESS,
# refined by the spawn icon, and BOTH of those can always be overridden by the badge -- "the
# #1 most consistent marker our system had... almost 100% accurate" historically, on its own,
# before position/icon existed. A higher tier can always overwrite a lower one; a tier can only
# be matched/exceeded to change a track's side, never dropped below once set for that frame.
TIER_NONE = 0      # nothing has ever set this track's side yet
TIER_POSITION = 1  # set by _side_from_position -- the weakest, a provisional first guess only
TIER_ICON = 2      # set by perception/spawn_icon_reader.py -- a real one-time visual read,
# stronger than a position guess but still just a single moment-of-placement snapshot
TIER_BADGE = 3     # set by perception/side_reader.py's color badge, debounced (SIDE_LOCK_HITS)
# -- the strongest. Unlike the other two, NOT limited to a young track (a badge often only
# becomes visible once a unit is damaged, which can happen long after placement), and CAN
# override an already-set lower tier -- or even re-confirm/correct itself -- at any point in a
# track's life, not just once.

SPAWN_ICON_SEED_MAX_AGE = 5  # only worth searching for perception/spawn_icon_reader.py's
# deploy-delay icon while a track is this young — the icon only shows briefly right at
# placement (see that module's own docstring), so searching for it on an established track
# would just be wasted work. Deliberately tighter than perception/debug_live.py's own
# SPAWN_ICON_PREVIEW_MAX_AGE=15 (a display-only search with no cost to searching a little
# longer); this one runs on every young track every frame, so it's kept to the window that
# actually matters for seeding.

POSITION_SEED_MAX_AGE = 5  # only trust a track's PLACEMENT position (see _side_from_position)
# while it's this young -- once a unit starts walking, its current position stops indicating
# which side placed it (a Giant beelines for the enemy tower and ends up standing deep in enemy
# territory despite being ours). Same window as SPAWN_ICON_SEED_MAX_AGE (both are "only valid at
# the moment of placement" signals) but named separately since the two are conceptually
# independent and may need retuning separately once more real data exists.

RECENT_VANISH_MAX_FRAMES = 30  # how many frames to remember a DROPPED track's last known side
# + position for -- see _find_recently_vanished_match's docstring. Frame-based (not wall-clock)
# to match every other threshold in this module; at this project's realistic ~12fps-with-unit-
# detection ceiling, ~30 frames is roughly 2.5s -- meant to comfortably bridge the "~1s" real
# tracking gaps already documented for a moving unit crossing the board (Project_Status). A
# first-pass guess, not yet tuned against real data.
RECENT_VANISH_MAX_DISTANCE_PX = 250  # how far a reappearing unit could plausibly have walked
# during the gap before being re-detected as a NEW track_id. Deliberately looser than
# MATCH_DISTANCE_PX=90 (that bounds a SINGLE missed frame; this bridges potentially dozens of
# them) -- a rough first-pass guess, not derived from any specific unit's real walk speed.

SIDE_MISMATCH_PENALTY_PX = 55  # added to a candidate's match distance when the detection's
# side disagrees with the track's committed side -- a PREFERENCE, not a veto.
#
# Replaces a hard exclusion (2026-08-20 review). The old behaviour let a single bad side
# reading destroy a track's IDENTITY: a detection whose side conflicted was refused
# outright, so the real unit failed to match anything and spawned a brand-new track_id.
# Duncan hit exactly this -- an ally Giant tracked cleanly for 134 consecutive frames,
# then on the very next frame (zero missed frames, 22px of movement, nowhere near far
# enough to implicate reacquisition) split into a new track. Downstream that reads as a
# fresh unit appearing, which is precisely what produces phantom "played" events in
# game_knowledge/opponent_play_tracker.py.
#
# It also contradicted this project's own fusion-architecture principle that IDENTITY is
# resolved before, and independently of, ATTRIBUTES: side is an attribute, so letting it
# gate matching inverted the intended order. Class was already corrected this way (see the
# CLASS accumulator work); side was the remaining offender.
#
# A penalty keeps the useful part -- when two candidate tracks are otherwise comparable,
# the side-agreeing one still wins -- while guaranteeing a real unit always matches
# SOMETHING rather than fragmenting. Sized larger than typical per-frame jitter but well
# under MATCH_DISTANCE_PX, so it reorders candidates without excluding a lone one.
CLASS_LOCK_HITS = 2  # mirrors SIDE_LOCK_HITS -- see _resolve_class's docstring. A track's
# reported class only changes to a NEW value after this many consecutive matched frames
# agree on it; a single misclassified frame doesn't relabel the track (and, paired with
# matching no longer hard-requiring exact class equality, doesn't fail to MATCH it either).

VELOCITY_EMA_ALPHA = 0.4  # smoothing factor for each track's rolling per-frame velocity
# estimate (see _update_velocity) -- higher weights recent movement more, lower smooths out
# detection-box jitter more. A reasonable middle ground, not tuned against real data yet.
MIN_SPEED_FOR_DIRECTIONAL_PX = 3  # a track's own velocity magnitude (px/frame) must reach
# at least this before its heading is trusted enough to use the tight/directional search
# area below -- freshly spawned, stopped-to-attack, or genuinely stationary units don't have
# a discernible direction yet, and should fall back to the wide circular search regardless
# of card type rather than trusting a near-zero, noise-dominated velocity vector.
MAX_MISS_RADIUS_GROWTH_FRAMES = 20  # missed_frames is capped at this for search-radius
# growth purposes (a track missing far longer than this is already deep into/past its own
# MAX_MISSED_FRAMES grace period anyway) -- keeps the search area from growing unbounded.
WIDE_SEARCH_RADIUS_PER_MISSED_FRAME_PX = 35  # circular search-radius growth rate, per
# missed frame, for a track whose direction can't be trusted (not a building-targeter, or
# not moving clearly yet -- see MIN_SPEED_FOR_DIRECTIONAL_PX). Deliberately generous since
# these units can retarget onto the nearest enemy (or get knocked back by cards like The
# Log/Giant Snowball/Barbarian Barrel) at any moment -- see module docstring's
# "Reacquisition" section for why this is the COMMON case, not the exception. First-pass
# guess, not tuned against real per-card speed data yet (game_knowledge/card_reference.json's
# still-empty movement_speed field would refine this per unit type once filled in).
DIRECTIONAL_LONGITUDINAL_RADIUS_PER_MISSED_FRAME_PX = 45  # along-heading search-radius
# growth rate for a building-targeter with a clear velocity -- generous, since distance
# traveled along a predictable straight-line path is the most trustworthy part of its
# motion. First-pass guess, same caveat as above.
DIRECTIONAL_LATERAL_RADIUS_PX = 60  # FIXED sideways tolerance for the same case -- doesn't
# grow with missed_frames (a beelining unit's heading doesn't drift further off-axis the
# longer it's missing, unlike how far along that heading it could have traveled). Kept
# non-trivial (not near-zero) to absorb real path curvature around obstacles/other units
# and ordinary detection-box jitter, not just perfectly straight lines.


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _side_compatible(track: dict, det_side: str) -> bool:
    """True if a track's currently-known side doesn't outright conflict
    with a new detection's side. "unknown" (perception.side_reader found
    no color signal this frame -- see unit_detector.py's detect() for why
    that's common and expected, e.g. a fresh undamaged ally troop) is
    treated as a wildcard on EITHER side, not a hard mismatch -- added
    2026-08-16 when side_reader.py replaced side.onnx. Without this, a
    real unit's track would be dropped and a spurious new one started the
    moment its side reading flips between "unknown" and a real color
    (e.g. an ally troop the instant it takes its first damage and its
    badge/bar first appears) -- the exact kind of flicker this tracker
    exists to prevent for the DETECTION side of things already (see
    module docstring); this extends the same principle to side.

    A genuine known-vs-known CONFLICT ("ally" vs "enemy") is handled
    differently depending on how strongly the track's current side is
    already established (added 2026-08-18 alongside _resolve_side's
    priority-tier redesign -- a real gap found while fixing the position-
    lock regression): if the track is only at TIER_POSITION or TIER_ICON
    (a provisional guess, not yet confirmed), the match is STILL allowed
    through -- otherwise a wrong position/icon guess could NEVER be
    corrected at all, since the very detection that would correct it
    (carrying the real, conflicting badge reading) would already be
    rejected right here, before _resolve_side (which is what actually
    applies the badge-beats-position priority) ever runs. Once a track has
    reached TIER_BADGE (the strongest signal, already confirmed via its
    own debounce), a conflicting reading is rejected as before -- a track
    that's already properly confirmed shouldn't be casually reopened by
    one disagreeing frame; it can only fail to match and start a separate
    track, same as the original behavior."""
    track_side = track["side"]
    return track_side == "unknown" or det_side == "unknown" or track_side == det_side


def _side_from_position(bbox, tower_state=None) -> str:
    """Returns "ally"/"enemy"/"unknown" from WHERE a unit's bbox sits on
    the board. Added 2026-08-18 per Duncan's direct correction: a troop
    can only ever be PLACED on the placing player's own half of the board
    (never the opponent's, and never mid-river).

    `tower_state` (perception.tower_reader.TowerStateTracker.update()'s
    own return dict) is what determines whether a princess tower's
    destruction has opened a NEUTRAL pocket on that flank -- when
    provided, this defers entirely to game_knowledge.zone_classifier's
    real per-lane deterministic zone logic (built 2026-08-20, closing the
    gap this function used to explicitly document as unclosed: "treats
    the ENTIRE far half as exclusively the opponent's forever," a known
    false-positive source once a tower actually falls). When `tower_state`
    is None (an older caller, or a test that hasn't wired it up), falls
    back to the ORIGINAL pure-river approximation below -- still correct
    by default whenever no princess tower has fallen, which is most of a
    match, but without the neutral-zone correction.

    Uses the bbox's BOTTOM edge (y2), not its center, as the ground-
    position proxy — same reasoning perception/spawn_icon_reader.py's own
    docstring already establishes for tall troops (a sprite's visual
    center sits well above its actual feet/board position for a tall
    unit like a Giant; the bottom edge is much closer to true ground
    position)."""
    ground_x = (bbox[0] + bbox[2]) / 2
    ground_y = bbox[3]

    if tower_state is not None:
        zone = classify_zone(ground_x, ground_y, tower_state)
        return zone if zone in ("ally", "enemy") else "unknown"  # "neutral" and the
        # river band itself both mean "no side information," same "unknown" semantics

    if ground_y > RIVER_Y_BOTTOM:
        return "ally"
    if ground_y < RIVER_Y_TOP:
        return "enemy"
    return "unknown"


def _resolve_side(track: dict, det: dict, screenshot, tower_state=None):
    """Updates `track`'s side-related fields in place based on this frame's
    detection, and sets det["side"] to the track's resulting side (which
    may still be "unknown"). Replaces the original single-line "lock onto
    whatever det['side'] first says" logic, AND a first tiered redesign
    (2026-08-18, same day) that still had a real bug: that version let
    POSITION or the ICON permanently "seed" a track (never revisited
    again), which Duncan caught live -- his own troops kept reading
    "enemy" once they'd crossed onto the opponent's side or into a
    neutral zone, even with a clean, clearly-visible blue level badge +
    health bar, because the earlier (wrong) position guess had already
    permanently locked the track before the badge ever got a chance to
    weigh in. Redesigned around explicit PRIORITY TIERS (TIER_POSITION <
    TIER_ICON < TIER_BADGE, see their own comments) instead: a higher tier
    can always overwrite whatever a lower tier already decided, so the
    badge -- Duncan's own "#1 most consistent marker," proven "almost
    100% accurate" on its own before position/icon existed -- can still
    correct an earlier wrong guess at any point in a track's life, not
    just once near placement. Order below is "what's checked," not "what
    wins" -- position/icon only ever WRITE a tier weaker than or equal to
    their own rank (so a lower tier can never clobber a higher one already
    set), while the badge is intentionally unconditional and re-checked
    every single frame:

    1. Placement POSITION (_side_from_position) — the FIRST GUESS, per
       Duncan's explicit framing: cheap (pure geometry), always-available
       near placement, but only a provisional default UNLESS `tower_state`
       is provided, in which case a polarized-zone read is deterministic
       proof instead (see game_knowledge.zone_classifier, wired in
       2026-08-20) — still only WRITTEN at TIER_POSITION here, though,
       since a track's tier still just reflects "which signal set this,"
       not "how certain was it" (a deterministic-but-TIER_POSITION read
       still can't be overridden by a weaker tier, only by badge, which
       already can't disagree with a real game rule for long in practice).
       Only ever applies once, from TIER_NONE up to TIER_POSITION (the
       `< TIER_POSITION` guard is only ever true while nothing has set a
       side yet) — it cannot re-fire or override anything once ANY tier
       is set, since a unit's later position no longer indicates who
       placed it once it's walking.
    2. The deploy-delay SPAWN ICON (perception/spawn_icon_reader.py) — a
       single direct color read at the moment of placement, refining a
       position guess (or filling in when position had no answer, e.g. a
       river-band placement). Only runs while `track["side_tier"] <
       TIER_ICON` (skips the real image-processing cost once already at
       icon-or-better) and while young (SPAWN_ICON_SEED_MAX_AGE).
    3. The per-frame color BADGE (perception/side_reader.py) — the
       STRONGEST, checked every frame regardless of current tier or age,
       gated by a same-value debounce (SIDE_LOCK_HITS) rather than a
       single reading (see that constant's own comment for the original
       persistence-lock bug this prevents). Whenever its debounce is
       satisfied THIS frame, it writes track["side"]/tier=TIER_BADGE,
       unconditionally overwriting a lower tier's guess (the whole point
       of this redesign) or re-confirming/correcting an existing badge-tier
       value -- EXCEPT one asymmetric guard: a debounced "enemy" candidate
       is rejected outright if the unit's CURRENT position is still
       unambiguously on the ally's own half (see the position-plausibility
       check inline below, added 2026-08-18 after a real false-positive
       badge read flipped a stationary Giant sitting behind the ally king
       tower to "enemy" for the rest of its life). No such guard exists
       for a debounced "ally" candidate, since a real ally unit can
       legitimately walk anywhere, including deep into enemy territory --
       gating that direction too would undo the exact correction this
       redesign exists to allow.

    This redesign also required loosening _side_compatible (see its own
    docstring): a track still at TIER_POSITION or TIER_ICON now allows a
    conflicting detection to match through anyway, specifically so a
    genuinely correcting badge reading gets a chance to reach this
    function at all -- previously a conflicting detection was rejected at
    the MATCHING step, before _resolve_side ever ran, which would have
    silently defeated this entire redesign. A track already at TIER_BADGE
    keeps the original strict behavior (a conflicting reading can't match
    at all, only spin up a separate track) -- an already-confirmed track
    shouldn't be casually reopened by one disagreeing frame.

    Known, accepted limitation NOT solved by this redesign (a genuinely
    new tracking design, per Duncan's own point 5 in Project_Status "Next
    up", is the intended real fix): per-individual track identity is still
    thin (see the module docstring's own note on swapped IDs for
    interchangeable nearby units), and the recently-vanished bridge below
    is still a cheap proximity heuristic, not real re-identification.
    """
    if track["side_tier"] < TIER_POSITION and track["age"] <= POSITION_SEED_MAX_AGE:
        position_side = _side_from_position(det["bbox"], tower_state)
        if position_side != "unknown":
            track["side"] = position_side
            track["side_tier"] = TIER_POSITION

    if (track["side_tier"] < TIER_ICON and screenshot is not None
            and track["age"] <= SPAWN_ICON_SEED_MAX_AGE):
        icon_side, _icon_area, _icon_bbox = find_spawn_icon(screenshot, det["bbox"])
        if icon_side is not None:
            track["side"] = icon_side
            track["side_tier"] = TIER_ICON

    det_side = det["side"]
    if det_side != "unknown":
        if det_side == track["side_candidate"]:
            track["side_candidate_hits"] += 1
        else:
            track["side_candidate"] = det_side
            track["side_candidate_hits"] = 1
        if track["side_candidate_hits"] >= SIDE_LOCK_HITS:
            candidate = track["side_candidate"]
            # A badge claiming "enemy" is only trustworthy if the unit's
            # CURRENT position doesn't make that literally impossible by
            # the game's own placement rules -- an enemy troop can never
            # legitimately stand on the ally's own exclusive half (same
            # invariant _side_from_position already relies on). Added
            # 2026-08-18 after a real, confirmed case in a live match log:
            # a Giant sat essentially stationary (~10px of drift) deep
            # behind the ally's own king tower for 6 straight frames
            # (correctly "ally" the whole time), then a badge misread
            # flipped it to "enemy" for the rest of its life, even though
            # it never moved anywhere near the river. Deliberately NOT
            # symmetric: a badge claiming "ally" is never positionally
            # impossible, since a genuine ally unit can legitimately walk
            # anywhere, including deep into enemy territory -- that's
            # exactly the case this whole tier redesign exists to let the
            # badge correct, so gating THAT direction too would undo it.
            if not (candidate == "enemy" and _side_from_position(det["bbox"], tower_state) == "ally"):
                track["side"] = candidate
                track["side_tier"] = TIER_BADGE

    det["side"] = track["side"]
    det["side_tier"] = track["side_tier"]  # exposed so a caller (e.g. debug_live.py's display)
    # can distinguish a still-provisional TIER_POSITION guess from an icon/badge-confirmed one --
    # see _resolve_side's own docstring for what each tier means.


def _find_recently_vanished_match(vanished: list, name: str, center: tuple):
    """Returns the entry in `vanished` (see UnitTracker._recently_vanished's
    own comment) that's the best candidate for "this new detection is
    actually the SAME physical unit as a track that recently got dropped
    and lost its old track_id" -- same name, closest center within
    RECENT_VANISH_MAX_DISTANCE_PX -- or None.

    Added 2026-08-18 per Duncan's direct diagnosis of a real regression:
    the position-priority fix above (_side_from_position) is very accurate
    for a track's TRUE first placement, but this project already has a
    documented, unfixed bug (Project_Status "troop-tracking-gap double-
    counting") where a moving unit's track can be LOST mid-walk (a missed-
    frames streak exceeding MAX_MISSED_FRAMES, or simply moving further
    than MATCH_DISTANCE_PX between sightings) and reacquired under a BRAND
    NEW track_id. Before this fix, that brand-new track started completely
    fresh -- with no memory that it used to be a known "ally" -- and if the
    reacquisition happened to land past the river (a real ally push
    crossing onto the opponent's side, which is exactly when a fast-moving
    unit is likely to also be losing/regaining detection), the position
    check would immediately and confidently (re-)lock it "enemy", directly
    contradicting an already-reliable badge read (a damaged unit with a
    visible level number + health bar, which Duncan reports has been
    "almost 100% accurate" on its own). Duncan's own diagnosis, and the
    one implemented here: this belongs in unit_tracker.py specifically,
    since the system is "already very strong at gauging... a new SPAWNED
    unit" and the problem is specifically "losing and refinding it over
    and over" on the far side.

    Deliberately NOT a full re-identification/velocity-aware reacquisition
    system (that's the real fix, planned as its own future tracking-system
    redesign per Project_Status "Next up" point 5, and Duncan's own framing
    of this as something "flushed out more from a tracking system" later).
    This is a narrow, cheap bridge for SIDE specifically: it doesn't try to
    merge the old and new track_id's identity (per-individual continuity
    still isn't relied on anywhere downstream, per this module's own
    long-standing known limitation on swapped IDs for identical nearby
    units), it only lets a freshly-recreated track inherit the side a very
    recently-lost same-name track already had, instead of re-deciding it
    from scratch via position/icon/badge as if it were a genuinely new
    unit. Proximity+recency are a rough proxy for "probably the same
    unit," not a guarantee -- untuned against real data yet."""
    best, best_dist = None, None
    for entry in vanished:
        if entry["name"] != name:
            continue
        dist = _distance(center, entry["center"])
        if dist <= RECENT_VANISH_MAX_DISTANCE_PX and (best is None or dist < best_dist):
            best, best_dist = entry, dist
    return best


def _predicted_center(track: dict) -> tuple:
    """Where a track's next detection is expected, given its own recent
    velocity (_update_velocity) and how long it's been missing. For a
    track matched every frame (missed_frames == 0) this is just its last
    known center -- identical to the tracker's original behavior. Only
    diverges once a track has actually missed a frame, extrapolating
    along its own recent motion instead of freezing at the last known
    point -- see module docstring's "Reacquisition" section."""
    vx, vy = track["velocity"]
    mf = track["missed_frames"]
    cx, cy = track["center"]
    return (cx + vx * mf, cy + vy * mf)


def _search_radius_ok(det_center: tuple, track: dict, match_distance_px: float) -> bool:
    """True if det_center falls within `track`'s current reacquisition
    search area. A track matched last frame (missed_frames == 0) uses the
    plain circular `match_distance_px` check around its own last center --
    the tracker's original, unmodified behavior for the common frame-to-
    frame case (passed in rather than read from the module constant so a
    UnitTracker constructed with a custom match_distance_px is honored
    here too). A track that's actually missing uses its PREDICTED center
    (_predicted_center) and a search area that grows with how long it's
    been missing, shaped by whether it's a predictable building-targeter
    (perception.unit_metadata.targets_buildings_only) with a clear enough
    velocity to trust a heading at all -- see module docstring's
    "Reacquisition" section and MIN_SPEED_FOR_DIRECTIONAL_PX/the
    WIDE_*/DIRECTIONAL_* constants' own comments for the reasoning:
    - Not clearly moving yet, or not a building-targeter: WIDE, circular
      (direction can't be trusted at all for these).
    - A building-targeter with a clear velocity: TIGHT sideways, GENEROUS
      along its own heading -- an ellipse test via the offset's projection
      onto/against the velocity vector, not just a bigger circle."""
    mf = track["missed_frames"]
    if mf == 0:
        return _distance(det_center, track["center"]) <= match_distance_px

    predicted = _predicted_center(track)
    mf_capped = min(mf, MAX_MISS_RADIUS_GROWTH_FRAMES)
    vx, vy = track["velocity"]
    speed = math.hypot(vx, vy)
    dx = det_center[0] - predicted[0]
    dy = det_center[1] - predicted[1]

    if targets_buildings_only(track["name"]) and speed >= MIN_SPEED_FOR_DIRECTIONAL_PX:
        ux, uy = vx / speed, vy / speed  # unit vector along the track's own heading
        longitudinal = dx * ux + dy * uy       # offset projected ONTO the heading
        lateral = dx * (-uy) + dy * ux         # offset projected PERPENDICULAR to it
        long_radius = DIRECTIONAL_LONGITUDINAL_RADIUS_PER_MISSED_FRAME_PX * mf_capped
        lat_radius = DIRECTIONAL_LATERAL_RADIUS_PX
        return (longitudinal / long_radius) ** 2 + (lateral / lat_radius) ** 2 <= 1.0

    radius = match_distance_px + WIDE_SEARCH_RADIUS_PER_MISSED_FRAME_PX * mf_capped
    return math.hypot(dx, dy) <= radius


def _update_velocity(track: dict, new_center: tuple):
    """EMA-updates a track's own velocity estimate (px/frame) in place,
    from its last known center to `new_center`. Divides the raw
    displacement by the number of frames actually elapsed (1 +
    missed_frames) so a multi-frame gap doesn't get misread as one huge
    single-frame jump -- called BEFORE track["center"] is overwritten
    with the new position, using missed_frames as it stood coming INTO
    this match (not yet reset to 0). VELOCITY_EMA_ALPHA smooths per-frame
    detection-box jitter, the same "smooth over a noisy per-frame signal"
    spirit as everything else in this module."""
    elapsed = 1 + track["missed_frames"]
    old_cx, old_cy = track["center"]
    raw_vx = (new_center[0] - old_cx) / elapsed
    raw_vy = (new_center[1] - old_cy) / elapsed
    old_vx, old_vy = track["velocity"]
    track["velocity"] = (
        VELOCITY_EMA_ALPHA * raw_vx + (1 - VELOCITY_EMA_ALPHA) * old_vx,
        VELOCITY_EMA_ALPHA * raw_vy + (1 - VELOCITY_EMA_ALPHA) * old_vy,
    )


def _resolve_class(track: dict, det: dict):
    """Updates track["name"] (the committed/reported class) and det["name"]
    based on this frame's detection, mirroring _resolve_side's same-value
    debounce (SIDE_LOCK_HITS) but for class instead of side -- see module
    docstring's "CLASS accumulator" section for the full reasoning. Unlike
    side, class has no "unknown" wildcard state (YOLO always emits some
    class guess), so every disagreement is a real candidate to weigh, not
    a no-signal frame to skip."""
    det_name = det["name"]
    if det_name == track["class_candidate"]:
        track["class_candidate_hits"] += 1
    else:
        track["class_candidate"] = det_name
        track["class_candidate_hits"] = 1
    if track["class_candidate_hits"] >= CLASS_LOCK_HITS:
        track["name"] = track["class_candidate"]
    det["name"] = track["name"]


class UnitTracker:
    def __init__(self, max_missed_frames=MAX_MISSED_FRAMES, match_distance_px=MATCH_DISTANCE_PX,
                 min_hits_to_confirm=MIN_HITS_TO_CONFIRM):
        self.max_missed_frames = max_missed_frames
        self.match_distance_px = match_distance_px
        self.min_hits_to_confirm = min_hits_to_confirm
        self._tracks = {}  # track_id -> {name, side, bbox, center, velocity, missed_frames,
        # age, hits, side_candidate, side_candidate_hits, side_tier (see _resolve_side),
        # class_candidate, class_candidate_hits (see _resolve_class)}
        self._next_id = 1
        self._recently_vanished = []  # [{name, side, side_tier, center, dropped_at_frame}, ...]
        # -- last known side/tier/position of tracks recently DROPPED (not just missing-with-
        # grace-period -- see _find_recently_vanished_match's docstring), pruned each update()
        # by RECENT_VANISH_MAX_FRAMES
        self._frame_count = 0

    def _is_confirmed(self, track_id) -> bool:
        return self._tracks[track_id]["hits"] >= self.min_hits_to_confirm

    def update(self, detections: list, screenshot=None, tower_state=None) -> list:
        """detections: unit_detector.detect()'s output for one frame.
        Returns the SUBSET of that list belonging to CONFIRMED tracks
        (matched at least min_hits_to_confirm times — see module
        docstring), each with "track_id" and "age" added. A brand-new
        detection is tracked internally starting this frame but does NOT
        appear in the returned list until it's confirmed.

        screenshot: optional full-frame BGR image (same one `detections`
        was produced from). None-safe — omitting it just means young
        tracks fall straight through to the badge-debounce side logic with
        no spawn-icon seed available (see _resolve_side). Passed through so
        this can check perception/spawn_icon_reader.py's icon for young
        tracks without the caller re-deriving detection bboxes itself.

        tower_state: optional perception.tower_reader.TowerStateTracker
        .update()'s own return dict (added 2026-08-20). None-safe —
        omitting it means position-based side resolution falls back to
        the old whole-half approximation (see _side_from_position's
        docstring) instead of real per-lane deterministic zone
        classification. Pass it once a TowerStateTracker is available in
        the caller's own loop (see perception/debug_live.py)."""
        self._frame_count += 1
        self._recently_vanished = [
            e for e in self._recently_vanished
            if self._frame_count - e["dropped_at_frame"] <= RECENT_VANISH_MAX_FRAMES
        ]

        candidates = []  # (distance, detection_index, track_id)
        for i, det in enumerate(detections):
            det_center = _center(det["bbox"])
            for track_id, track in self._tracks.items():
                # NOTE: neither class NOR side is a hard gate here, on purpose -- see the
                # module docstring's "CLASS accumulator" and "Identity before attributes"
                # sections. Matching is decided by the search area alone; _resolve_class
                # and _resolve_side own the reported class/side.
                if not _search_radius_ok(det_center, track, self.match_distance_px):
                    continue
                dist = _distance(det_center, _predicted_center(track))
                # Side disagreement is a soft PENALTY, not an exclusion: given two
                # otherwise-comparable candidate tracks, prefer the one whose side
                # already agrees -- but never refuse the match outright. See
                # SIDE_MISMATCH_PENALTY_PX for why exclusion was actively harmful.
                if not _side_compatible(track, det["side"]):
                    dist += SIDE_MISMATCH_PENALTY_PX
                candidates.append((dist, i, track_id))

        candidates.sort(key=lambda c: c[0])
        matched_detections = set()
        assignment = {}  # detection_index -> track_id
        active_this_frame = set()  # every track_id matched OR newly created this frame
        for dist, i, track_id in candidates:
            if i in matched_detections or track_id in active_this_frame:
                continue
            assignment[i] = track_id
            matched_detections.add(i)
            active_this_frame.add(track_id)

        confirmed_detections = []
        for i, det in enumerate(detections):
            det_center = _center(det["bbox"])
            if i in assignment:
                track_id = assignment[i]
                track = self._tracks[track_id]
                # Velocity update needs the OLD center + the missed_frames
                # elapsed getting here -- must run BEFORE they're overwritten below.
                _update_velocity(track, det_center)
                track["bbox"] = det["bbox"]
                track["center"] = det_center
                track["confidence"] = det["confidence"]
                track["missed_frames"] = 0
                track["age"] += 1
                track["hits"] += 1
                # Resolve the track's reported class (see module docstring's "CLASS
                # accumulator" section) before side -- side's own position check
                # (_side_from_position) reads det["bbox"], not det["name"], so order
                # between these two doesn't matter functionally, but resolving identity
                # (class) before attributes (side) reads more naturally.
                _resolve_class(track, det)
                # Persist/resolve the track's side across frames instead of
                # re-deriving it from scratch every frame -- see
                # _resolve_side's docstring for the priority-tier logic
                # (position < icon < badge, badge always able to correct).
                _resolve_side(track, det, screenshot, tower_state)
            else:
                track_id = self._next_id
                self._next_id += 1
                track = {
                    "name": det["name"], "side": "unknown",
                    "bbox": det["bbox"], "center": det_center, "velocity": (0.0, 0.0),
                    "confidence": det["confidence"],
                    "missed_frames": 0, "age": 1, "hits": 1,
                    "side_candidate": None, "side_candidate_hits": 0,
                    "side_tier": TIER_NONE,
                    "class_candidate": det["name"], "class_candidate_hits": 1,
                }
                self._tracks[track_id] = track
                # Check FIRST whether this "new" detection is more likely a
                # recently-lost track reappearing under a new ID (see
                # _find_recently_vanished_match) -- if so, inherit its
                # already-known side AND tier directly, so a badge-tier
                # inheritance still can't be casually overridden by a fresh
                # (weaker) position/icon read, but a weaker inherited tier
                # can still be upgraded normally by _resolve_side below.
                vanished_match = _find_recently_vanished_match(self._recently_vanished, det["name"], det_center)
                if vanished_match is not None:
                    track["side"] = vanished_match["side"]
                    track["side_tier"] = vanished_match["side_tier"]
                    self._recently_vanished.remove(vanished_match)
                _resolve_side(track, det, screenshot, tower_state)
                active_this_frame.add(track_id)
            det["track_id"] = track_id
            det["age"] = self._tracks[track_id]["age"]
            if self._is_confirmed(track_id):
                confirmed_detections.append(det)

        for track_id in list(self._tracks.keys()):
            if track_id not in active_this_frame:
                self._tracks[track_id]["missed_frames"] += 1
                # Only confirmed tracks get a grace period — an unconfirmed
                # (single-hit) track that misses even one frame is dropped
                # immediately rather than lingering unconfirmed indefinitely.
                limit = self.max_missed_frames if self._is_confirmed(track_id) else 0
                if self._tracks[track_id]["missed_frames"] > limit:
                    dropped = self._tracks[track_id]
                    # Remember a genuinely-known side (AND its tier) for a
                    # little while after the track is gone -- see
                    # _find_recently_vanished_match's docstring for why
                    # (bridges the side across a tracking-gap reacquisition
                    # under a new track_id, not just this exact ID).
                    if self._is_confirmed(track_id) and dropped["side_tier"] > TIER_NONE:
                        self._recently_vanished.append({
                            "name": dropped["name"], "side": dropped["side"],
                            "side_tier": dropped["side_tier"], "center": dropped["center"],
                            "dropped_at_frame": self._frame_count,
                        })
                    del self._tracks[track_id]

        return confirmed_detections

    def current_state(self) -> list:
        """Returns every currently-tracked unit, including ones inside
        their grace period (not detected in the most recent update() call,
        but not yet dropped) — the smoothed/de-flickered view.

        Each entry has "visible": True if matched in the most recent
        update() call, False if currently coasting on the grace period.
        Important: False does NOT mean "confirmed still alive but hidden"
        — it means "no detection this frame, and we haven't given up yet."
        There is no signal here (or anywhere in this project yet) that can
        actually distinguish real occlusion from the unit having died from
        the model simply missing an obvious frame — all three look
        identical from a single frame's detection output. Telling those
        apart for real would need something this project doesn't have
        yet (recognizing a death animation specifically, or cross-
        referencing HP dropping to zero via the planned-but-unbuilt HP
        reader). Report "visible=False" to the user as "presumed / last
        confirmed N frames ago," never as a claim about the unit's actual
        fate.
        """
        result = []
        for track_id, track in self._tracks.items():
            if not self._is_confirmed(track_id):
                continue
            result.append({
                "track_id": track_id,
                "name": track["name"],
                "side": track["side"],
                "side_tier": track["side_tier"],  # see _resolve_side's docstring for what each tier means
                "confidence": track["confidence"],  # last known reading — like bbox, may be stale if unconfirmed
                "bbox": track["bbox"],  # last known position — may be stale if the unit moved while unconfirmed
                "age": track["age"],
                "visible": track["missed_frames"] == 0,
                "missed_frames": track["missed_frames"],
            })
        return result
