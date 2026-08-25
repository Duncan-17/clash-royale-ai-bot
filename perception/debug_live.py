"""
Live spot-check overlay: shows what the detector is seeing WHILE you play,
so you can catch a bad recognition the moment it happens instead of
discovering it later by reading a wall of terminal text. This is purely a
viewer for the gameplay window — it never saves screenshots (that's
auto_collect.py's job) — so it's safe to run at the same time as
auto_collect.py in a second terminal if you want live feedback and data
collection running together. It DOES write a data log by default, see
"Logging" below.

Two windows, on purpose:
- "Live hand detection" — the gameplay screenshot with color-coded box
  outlines around each hand slot, the elixir number, the elixir bar, and
  every detected unit on the board. Has two display modes, toggled live
  with the 'd' key (starts in simple mode):
    simple   — boxes only, no text.
    detailed — boxes plus a compact label each: card name + confidence
               for hand slots, value/estimate for elixir, and for units a
               short color-coded tag combining an optional side letter
               ("A"/"E", see SIDE_LETTER), the card's initials, and its
               tracker ID (e.g. "AK3" = ally Knight, track #3; "EMP7" =
               enemy Mini P.E.K.K.A., track #7; a bare "MP7" with no
               letter means side is still unknown for that track).
- "Debug panel" — a separate, plain dark panel with HAND/ELIXIR/CYCLE/
  OPPONENT PLAYS text (small and fixed-size, so no reflow problem there)
  plus two STABLE GRIDS for units — see "Why a grid" below. Opens
  positioned to the right of the gameplay window by default; drag either
  window if that doesn't match your monitor layout.

Box color coding for CARDS (hand slots, elixir number):
  green  = confident card match (>= CONFIDENCE_THRESHOLD)
  orange = a card guess, but a weak one (within 0.15 of the threshold) —
           worth watching; if this keeps happening for the same card it
           likely needs more/better examples
  red    = UNKNOWN card (below threshold)

Box color coding for UNITS depends on the 'u'-key-toggled view mode (see
UNIT_VIEW_MODES) — "side" mode (default) matches the game's OWN badge
colors directly, not the card scheme above:
  blue   = ally (perception/side_reader.py read a blue badge/health bar)
  red    = enemy (read a red badge/health bar)
  orange = no color signal yet this frame — most commonly a fresh,
           undamaged ally troop, which the game itself shows no badge/
           health bar for at all until it's taken damage (see
           unit_detector.py's detect() docstring)
"confidence" mode instead reuses the card scheme above (green/orange/red
= units.onnx's own detection confidence), unrelated to side entirely —
see UNIT_CONFIDENT_THRESHOLD.

Why a grid (Duncan's feedback, 2026-08-15): the original panel rendered
units as a plain text list, one line per currently-known unit. Every time
a unit appeared or disappeared, the line count changed and everything
below it reflowed up/down — with units flickering in and out (see
unit_tracker.py's docstring for why that happens even for real, unmoving
units), the whole panel was constantly jumping, making it hard to follow.
Fix: SlotGrid assigns each track_id a FIXED position (0..49) that it
keeps for as long as it exists, freed for reuse once it's gone — so a
given unit always renders in the same grid cell while tracked, and only
that cell's content changes, never the layout around it. Both grids
("LIVE" and "TRACKED", one row of headers each) share the SAME slot
assignment (driven by TRACKED's track_id set, which is always a superset
of LIVE's — see unit_tracker.py), so a given unit sits in the same cell
in both grids — e.g. seeing a cell occupied in TRACKED but blank in LIVE
means "the tracker still believes this exists, just not detected this
exact frame," directly visible by comparing the two grids at a glance
instead of reading text.

Logging (added 2026-08-15, per Duncan): writes one JSON line per frame to
logs/debug_live_<timestamp>.jsonl by default (see --log / --no-log) —
hand/elixir/live-units/tracked-units, everything needed to reconstruct
exactly when a given track flickered, how long occlusions lasted, etc.
without needing to watch it happen live. Also captures hand+elixir per
frame even though this session's ask was specifically about troop
tracking, since it's free to include now and Duncan flagged this log
will also be useful later for cross-checking against placement/action
detection once that exists (Phase 7-ish) — better to capture the full
per-frame snapshot now than redesign the log format later.

Side-conflict auto-capture (added 2026-08-16, per Duncan's live report of
an enemy Giant "flickering" between enemy/friend): log analysis proved
this isn't a single track's side flipping (impossible, see
unit_tracker.py's _side_compatible) but two separate track_ids for the
same physical unit a few px apart with genuinely different
perception/side_reader.py reads on consecutive frames — the raw per-frame
color read itself is unstable for some real, still-unexplained cases. A
manually-timed static screenshot couldn't reproduce it (almost certainly
because the instability is motion-related — walking/attack animation —
which a deliberately-paused capture can't catch). See
find_side_conflict(): every frame, checks whether two same-name tracks
with different KNOWN sides are sitting close together, and if so (throttled
to one snapshot per CONFLICT_SNAPSHOT_COOLDOWN_SECONDS), saves the RAW
screenshot plus a small JSON sidecar (track details, frame number, time)
to temp_img/side_conflicts/ automatically — catches the problem live
without needing to time a manual capture around a few-frame event.

CYCLE (added 2026-08-15, Phase 4 part 1 — see game_knowledge/cycle_tracker.py
for the full algorithm): shows the current hand, the best-known order of
the other 4 deck cards ("?" for any not yet resolved this match), and the
most recent detected play (card played -> card revealed, and whether the
elixir signal actually confirmed it or just flagged it as uncertain).
There's no automatic match-boundary detection anywhere in this project yet
(a known, deliberately deferred gap), so cycle state has to be reset by
hand at the start of each new match — press 'r'.

OPPONENT PLAYS (added 2026-08-16, Phase 4 part 2 — see
game_knowledge/opponent_play_tracker.py for the full algorithm): every time
a NEW enemy troop/building was detected, each tagged PLAYED (a genuine
fresh card play, with the card's elixir cost when known), spawned (a
byproduct of an existing card's ability — Golem splitting, Witch/Tombstone/
Goblin Hut/etc. producing units for free — attributed to its source when
one is nearby), UNCERTAIN (a burst size matching neither a known card nor
a known spawner — flagged in magenta, per Duncan's request that misreadings
which drastically violate game rules be surfaced, not hidden), or
MISCLASSIFIED (dark gray — likely one of OUR OWN units the side classifier
briefly mislabeled, not a real enemy play at all; see
opponent_play_tracker.py's module docstring for the two ways this gets
caught), or REACQUIRED (orange — a building, never a fresh placement: the
SAME physical building lost its track and just got a new one at nearly the
same position, a different and cleaner problem than the troop-side
misclassification cases since buildings never move). Troops/buildings
only — spells are a separate, not-yet-built detection problem. The full match's history is kept (not just the last
few) — scroll with the mouse wheel over the Debug panel window to page
through it, per Duncan's request for an ordered scrollable list instead of
the original rolling-last-5 display. Reset alongside cycle tracking on 'r'.

Opponent-play snapshots (added 2026-08-17, per Duncan's request after live
matches kept "struggling a lot for card placements" and the panel's live
scrolling text wasn't enough to review after the fact): every time
OpponentPlayTracker closes out an event this frame (see
save_opponent_play_snapshots()), the RAW full-board screenshot is saved to
temp_img/opponent_plays/ with the event's own bbox(es) boxed and a text
label (card + attribution + elixir cost/flag_reason), plus a JSON sidecar
with the full event dict. Lets Duncan page through exactly what the
tracker saw and concluded for every single play after a match, instead of
only catching it live as the panel scrolls past.

Live spawn-icon markers (added 2026-08-18, per Duncan's request to see
perception/spawn_icon_reader.py's detections live instead of only in the
saved snapshots above): every recently-appeared unit (age <=
SPAWN_ICON_PREVIEW_MAX_AGE, since the icon can only ever show briefly right
after placement) is checked each frame; a found icon draws a CIRCLE
(deliberately not a rectangle, so it's unmistakably a different signal from
the ordinary side/badge boxes) colored blue/red for ally/enemy with a
"SPAWN: ALLY"/"SPAWN: ENEMY" label directly on the gameplay window, plus a
small "SPAWN ICONS (this frame)" line in the Debug panel.

Spawn-icon MISS snapshots (added 2026-08-18, same day, after search-window
tuning alone couldn't close a real reported gap): whenever a young track
(age <= unit_tracker.py's own SPAWN_ICON_SEED_MAX_AGE — the window the
tracker itself actually uses for seeding, not the wider display-only
preview window above) fails to find any icon at all, the raw frame + a
JSON sidecar (track details) is saved to temp_img/spawn_icon_misses/ —
same "catch it live with real data instead of guessing at pixel geometry
again" reasoning as the side-conflict snapshots. Only once per track_id,
throttled by SPAWN_ICON_MISS_COOLDOWN_SECONDS, so a burst of simultaneous
spawns doesn't flood the disk.

Tower boxes (added 2026-08-20, alongside perception/tower_reader.py):
every one of the 6 towers' calibrated bar regions is boxed on the
gameplay window every frame — green/orange/red by HP fraction (>50%/
20-50%/<20%), dark red + "DESTROYED" once TowerStateTracker's debounce
actually confirms it (a single missed frame doesn't flip this — see that
class's own docstring), orange "?" for a king tower that hasn't been
activated yet (no bar to read until it's damaged or a same-side princess
tower falls — NOT the same as destroyed, see TowerStateTracker). Once
destroyed, a tower's region is never read again at all (not just
displayed differently) — see TowerStateTracker.update()'s own comment for
why (a destroyed tower's empty patch of ground is exactly where a troop
is likely to walk next). Mirrored in the Debug panel's own TOWERS section
and in the JSONL log's new "towers" key.

Press 'q' with either window focused, or Ctrl+C in the terminal, to stop.
Press 'd' to toggle the gameplay window between simple and detailed.
Press 'u' to toggle unit box coloring between "side" (blue=ally,
red=enemy, orange=no color signal found yet — perception/side_reader.py)
and "confidence" (green/orange/red = units.onnx's own detection
confidence, same style as card matching). Added 2026-08-16 per Duncan's
feedback that reusing green/red for both card-ID-confidence and unit-side
made it impossible to tell, watching live, whether a red unit box meant
"enemy" or "the system doesn't know what this is" — two unrelated
problems that happened to share a color scheme by coincidence.
Press 'r' to reset cycle tracking, opponent-play tracking, AND tower
tracking (do this at the start of a new match).
Scroll the mouse wheel over the Debug panel to page through OPPONENT PLAYS
history — up reveals older events, down returns toward the most recent.

Performance note (updated 2026-08-15): capture.py now uses adbnativeblitz
(background H.264 stream) instead of per-call ADB screencap, so
capture.screenshot() itself is ~0.5ms — no longer the bottleneck. Card
template matching is ~35ms for 4 slots at the current MATCH_SIZE=64x64
(see card_detector.py's comment on that constant); unit detection
(perception/unit_detector.py's YOLOv10 ONNX model) adds another ~45-50ms
on top. Combined realistic ceiling is around 12fps. --interval below only
PADS a frame up to the target period if processing finished early, never
adds delay on top of it.

Usage:
    python -m perception.debug_live
    python -m perception.debug_live --interval 0.3   # slow it down deliberately (e.g. less CPU)
    python -m perception.debug_live --no-log          # skip writing the JSONL log
    python -m perception.debug_live --log path.jsonl  # explicit log path instead of the timestamped default
"""

import argparse
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np

from config.hand_region import hand_slot_regions
from perception.auto_collect import load_deck
from perception.capture import ScreenCapture
from perception.card_detector import CONFIDENCE_THRESHOLD as CARD_CONFIDENCE_THRESHOLD
from perception.card_detector import best_match as card_best_match
from perception.card_detector import load_templates as load_card_templates
from control.actions import EmulatorController
from perception.elixir_counter import BAR_LEFT_X, BAR_RIGHT_X, NUMBER_REGION
from perception.elixir_counter import CONFIDENCE_THRESHOLD as ELIXIR_CONFIDENCE_THRESHOLD
from perception.elixir_counter import load_templates as load_elixir_templates
from perception.elixir_counter import read_elixir
from perception.spawn_icon_reader import find_spawn_icon
from perception.tower_reader import TOWER_BAR_REGIONS, TowerStateTracker
from perception.unit_detector import UnitDetector
from perception.unit_tracker import SPAWN_ICON_SEED_MAX_AGE as TRACKER_SPAWN_ICON_SEED_MAX_AGE
from perception.unit_tracker import TIER_POSITION, UnitTracker
from game_knowledge.cycle_tracker import CycleTracker
from game_knowledge.opponent_play_tracker import OpponentPlayTracker

WEAK_MARGIN = 0.15  # scores within this far below threshold are drawn orange instead of red
GAMEPLAY_WINDOW = "Live hand detection (q=quit, d=toggle detail, u=toggle side/confidence)"
PANEL_WINDOW = "Debug panel"
PANEL_WIDTH = 420
BAR_Y_TOP, BAR_Y_BOTTOM = 1524, 1584  # for drawing only — see elixir_counter.py for the calibrated fill-measurement rows

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
DARK_GRAY = (70, 70, 70)
PANEL_BG = (30, 30, 30)
# Palette softened 2026-08-20 per Duncan ("why is everything so low poly and neon").
# The HUES still match the in-game ally-blue / enemy-red / uncertain-orange mapping he
# asked for in 2026-08-16 -- what changed is that they're no longer fully-saturated
# channel extremes (255,80,0)/(0,0,255)/(0,165,255), which is what read as "neon"
# against real gameplay footage. These are pulled toward a softer, slightly desaturated
# version of the same hue so they stay instantly distinguishable without vibrating.
ALLY_COLOR = (255, 158, 72)   # BGR -> a calm azure
ENEMY_COLOR = (86, 82, 240)   # BGR -> a soft coral red, not pure (0,0,255)
UNKNOWN_SIDE_COLOR = (96, 186, 250)  # BGR -> warm amber -- perception/side_reader.py found no
# color signal this frame (most commonly a fresh, undamaged ally troop with no
# badge/bar shown yet at all -- see unit_detector.py's detect() docstring).
# Reuses the same orange already used elsewhere in this file (color_for(),
# unit_confidence_color()) for "uncertain," rather than a fourth ad-hoc color.

SIDE_LETTER = {"ally": "A", "enemy": "E"}  # prefixed onto compact_tag()'s label
# per Duncan's request (2026-08-16) -- text redundant with the box color, since
# color alone (especially blue vs. red at a glance, mid-match) isn't always
# enough to be sure which is which. Deliberately no letter for "unknown" -- the
# orange color already signals "no reading yet," and there's no real letter for
# "unsure" the way there is for a known side.


def side_color(side: str, side_tier: int = None):
    """side_tier: optional confidence tier from perception.unit_tracker's
    _resolve_side (TIER_POSITION/TIER_ICON/TIER_BADGE). A still-provisional
    TIER_POSITION-only read renders as the same "uncertain" orange as a
    true unknown, per Duncan's direct request (2026-08-18, after a live
    match where a position-only "enemy" guess displayed with the same full
    confidence as an icon/badge-confirmed one): position alone is only
    ever this system's FIRST GUESS (see unit_tracker.py's own
    _resolve_side docstring for the full priority design), not yet
    confirmed by the spawn icon or the color badge, so it shouldn't look
    as certain on screen as one that has been. None (the default) skips
    this downgrade entirely, for any caller with no tier info available
    (e.g. the spawn-icon marker/panel text, which is always a direct,
    confident color read with no tier concept of its own)."""
    if side_tier is not None and side_tier <= TIER_POSITION:
        return UNKNOWN_SIDE_COLOR
    return {"ally": ALLY_COLOR, "enemy": ENEMY_COLOR}.get(side, UNKNOWN_SIDE_COLOR)


PHANTOM_NOISE_MARKER = "didn't persist past a few confirmed frames"  # substring of the
# flag_reason game_knowledge/opponent_play_tracker.py's _finalize_played() writes for a
# track that never reached PLAY_PERSISTENCE_MIN_AGE -- a units.onnx-level detection
# artifact (see that constant's own comment), not a real event. Filtered out of the
# OPPONENT PLAYS panel/snapshots (added 2026-08-17 per Duncan: this category alone was
# "3/4" of the visible feed, burying genuine plays/uncertain-for-real-reasons events) --
# NOT filtered out of the raw per-frame `opponent_events` passed to write_log(), so the
# JSONL log stays complete for offline analysis even though the live views are decluttered.


def _is_phantom_noise(event: dict) -> bool:
    return event["attribution"] == "uncertain" and PHANTOM_NOISE_MARKER in (event.get("flag_reason") or "")


CONFLICT_SNAPSHOT_DIR = os.path.join("temp_img", "side_conflicts")
CONFLICT_SNAPSHOT_PROXIMITY_PX = 100  # roughly in line with game_knowledge/
# opponent_play_tracker.py's own CONCURRENT_DUPLICATE_PROXIMITY_PX=100 --
# "close enough to plausibly be the same physical unit."
CONFLICT_SNAPSHOT_COOLDOWN_SECONDS = 3.0  # avoid flooding disk if a conflict
# persists across many consecutive frames -- one snapshot per burst is enough.

SPAWN_ICON_MISS_DIR = os.path.join("temp_img", "spawn_icon_misses")
SPAWN_ICON_MISS_COOLDOWN_SECONDS = 2.0  # same "one snapshot per burst" reasoning as
# CONFLICT_SNAPSHOT_COOLDOWN_SECONDS above -- avoids flooding disk if several units spawn
# near-simultaneously and all miss in the same frame.
# Added 2026-08-18 per Duncan's live report: spawn icons still get missed for cards placed
# behind the king tower even after widening SEARCH_BELOW_FACTOR + fixing the crowding-
# selection bug the same session -- rather than guess at the search geometry a THIRD time with
# no real data, this captures the raw frame + exact search inputs the next time it happens, the
# same "verify with real data before touching thresholds again" rigor as every other fix in
# this project (see CLAUDE.md Rule #1). Only saved ONCE per track_id (see captured_miss_ids in
# main()) -- a track's icon window is only a few frames long, and re-saving the same physical
# miss every one of those frames would just be noise, not new information.


def _unit_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _unit_dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def find_side_conflict(tracked_units):
    """Returns (unit_a, unit_b) for the first pair of same-name, differently-
    KNOWN-sided tracks sitting close together this frame, or None. Added
    2026-08-16 after Duncan reported an enemy Giant "flickering" between
    enemy/friend live -- log analysis (logs/debug_live_20260816_171250.jsonl)
    confirmed this ISN'T a single track's side flipping (that can't happen,
    see unit_tracker.py's _side_compatible) but two separate track_ids for
    the same physical unit, a few px apart, with genuinely different
    side_reader.py reads on consecutive frames -- i.e. the raw per-frame
    color read itself is unstable for some real cases, not a downstream
    logic bug. A static test_screenshot.png capture couldn't reproduce it
    (that giant read cleanly) -- almost certainly because the instability
    shows up specifically during motion (walking/attack animation), which a
    single manually-timed still capture can't catch on purpose. This
    detects the exact same pattern live and auto-saves the raw frame so the
    problem is caught automatically instead of needing perfect manual
    timing."""
    known = [u for u in tracked_units if u["side"] != "unknown"]
    for i in range(len(known)):
        for j in range(i + 1, len(known)):
            a, b = known[i], known[j]
            if a["name"] == b["name"] and a["side"] != b["side"]:
                if _unit_dist(_unit_center(a["bbox"]), _unit_center(b["bbox"])) <= CONFLICT_SNAPSHOT_PROXIMITY_PX:
                    return a, b
    return None


OPPONENT_PLAY_SNAPSHOT_DIR = os.path.join("temp_img", "opponent_plays")
ATTRIBUTION_COLOR = {  # matches build_panel()'s own OPPONENT PLAYS color scheme exactly
    "played": (0, 200, 0),
    "spawned": (150, 150, 150),
    "misclassified": (90, 90, 90),
    "reacquired": (200, 140, 0),
    "uncertain": (255, 0, 255),
}

SPAWN_ICON_PREVIEW_MAX_AGE = 15  # only search units this new for a live spawn-icon
# preview -- the icon is only ever visible for a brief window right after placement
# (see perception/spawn_icon_reader.py's own docstring), so re-checking a
# long-established unit every frame would just be wasted work for something that
# can't be there. ~15 confirmed frames is a generous window at this project's
# ~10-12fps-with-units-on range (Project_Status) -- loosened if real play shows
# genuine icons still visible past this.


def save_opponent_play_snapshots(screenshot, events: list):
    """Saves one annotated full-board screenshot per game_knowledge.
    opponent_play_tracker.OpponentPlayTracker event this frame (usually
    zero events; a "play" only closes out of its GROUPING_WINDOW_SECONDS
    buffer occasionally, not every frame) -- added 2026-08-17 per Duncan's
    request for a way to review card-placement detection after a match
    ends, not just watch it fly by live. Draws every bbox in the event's
    own group (there can be several -- e.g. Minions' 3 units) plus a label
    with the card name, attribution, and (for "played") elixir cost or
    (for "uncertain"/"misclassified") the flag_reason, onto a COPY of the
    raw screenshot -- deliberately the raw frame, not debug_live.py's
    `display` (which already has unrelated hand/elixir/unit boxes drawn on
    it that would clutter this specific view). Filename encodes the
    attribution and card so Duncan can sort/filter the folder by eye
    without opening every image -- also writes a JSON sidecar with the
    full event dict for anything not visible in the picture (track_ids,
    exact frame_time, source_track_id for spawned/reacquired/misclassified)."""
    if not events:
        return
    os.makedirs(OPPONENT_PLAY_SNAPSHOT_DIR, exist_ok=True)
    for event in events:
        display = screenshot.copy()
        color = ATTRIBUTION_COLOR.get(event["attribution"], (255, 255, 255))
        for bbox in event["bboxes"]:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)

        countpart = f" x{event['count']}" if event["count"] != 1 else ""
        if event["attribution"] == "played":
            label = event["card_slug"] or event["card"]
            cost = f", {event['elixir_cost']} elixir" if event["elixir_cost"] is not None else ""
            headline = f"PLAYED: {label}{countpart}{cost}"
        elif event["attribution"] == "spawned":
            src = f" (from {event['source_card']})" if event["source_card"] else " (source unknown)"
            headline = f"spawned: {event['card']}{countpart}{src}"
        elif event["attribution"] in ("misclassified", "uncertain"):
            headline = f"{event['attribution']}: {event['card']}{countpart} -- {event['flag_reason']}"
        elif event["attribution"] == "reacquired":
            headline = f"reacquired: {event['card']}{countpart} (same building, track re-lost)"
        else:
            headline = f"{event['attribution']}: {event['card']}{countpart}"

        x1, y1, _, _ = event["bboxes"][0]
        cv2.putText(display, headline, (max(5, x1 - 10), max(20, y1 - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(display, headline, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)  # also top-left, in case the bbox is off-screen-ish

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        card_part = (event["card_slug"] or event["card"]).replace(" ", "_")
        base = f"{stamp}_{event['attribution']}_{card_part}"
        cv2.imwrite(os.path.join(OPPONENT_PLAY_SNAPSHOT_DIR, f"{base}.png"), display)
        with open(os.path.join(OPPONENT_PLAY_SNAPSHOT_DIR, f"{base}.json"), "w", encoding="utf-8") as f:
            json.dump(event, f, indent=2)


UNIT_VIEW_MODES = ["side", "confidence"]  # cycled with the 'u' key -- see
# main()'s key handling. Added 2026-08-16 per Duncan's feedback: green/red
# already mean "card confidently identified / not identified" for hand
# slots, so reusing green=ally/red=enemy for units made it impossible to
# tell, at a glance, whether a red unit box meant "this is an enemy" or
# "the system doesn't know what this is" -- two completely different
# problems (side_reader.py's job vs. units.onnx's own class/bbox
# confidence) that happened to share a color scheme by coincidence.
UNIT_CONFIDENT_THRESHOLD = 0.7  # green at/above this. Real confident
# detections have measured 0.75-0.92 in live testing (Project_Status);
# MIN_CONF=0.4 (unit_detector.py) is the hard floor below which a
# detection is never even returned, so this is a SEPARATE, higher bar for
# "trust this without watching it," same spirit as card_detector.py's
# CONFIDENCE_THRESHOLD.


def unit_confidence_color(conf):
    if conf is None:
        return UNKNOWN_SIDE_COLOR  # tracked-grid entries before confidence was persisted -- shouldn't
        # normally happen once unit_tracker.py's current_state() always includes it, kept as a safe default.
    if conf >= UNIT_CONFIDENT_THRESHOLD:
        return (0, 200, 0)  # green
    if conf >= UNIT_CONFIDENT_THRESHOLD - WEAK_MARGIN:
        return (0, 165, 255)  # orange
    return (0, 0, 255)  # red


def unit_view_color(u: dict, mode: str):
    return side_color(u["side"], u.get("side_tier")) if mode == "side" else unit_confidence_color(u.get("confidence"))

GRID_SLOTS = 50
GRID_COLS = 10
GRID_ROWS = 5
CELL_W = 38
CELL_H = 22

OPPONENT_PLAYS_VISIBLE_ROWS = 8  # fixed-size scrollback window — see
# "OPPONENT PLAYS" in the module docstring for the scroll UX this drives.


def on_panel_scroll(event, x, y, flags, scroll_state):
    """cv2.setMouseCallback handler for PANEL_WINDOW. Verified directly
    against this project's actual cv2 build (5.0.0) rather than assumed:
    cv2.EVENT_MOUSEWHEEL exists, but the commonly-referenced
    cv2.getMouseWheelDelta() helper does NOT exist in this version — the
    signed wheel delta is decoded straight from flags instead (standard
    signed-16-bit-in-the-upper-bits convention). scroll_state is a small
    mutable dict ({"offset": int}) shared with the main loop; offset=0
    always means "showing the newest events," increasing it pages upward
    into older history."""
    if event != cv2.EVENT_MOUSEWHEEL:
        return
    delta = flags >> 16
    if delta > 0:
        scroll_state["offset"] += 1
    elif delta < 0:
        scroll_state["offset"] = max(0, scroll_state["offset"] - 1)


def color_for(name, score, threshold):
    if name not in (None, "UNKNOWN") and score >= threshold:
        return (0, 200, 0)  # green (BGR)
    if score >= threshold - WEAK_MARGIN:
        return (0, 165, 255)  # orange
    return (0, 0, 255)  # red


def compact_tag(name: str, track_id: int, side: str = None, side_tier: int = None) -> str:
    """e.g. "knight" + 3 + "ally" -> "AK3", "mini_pekka" + 7 + "enemy" ->
    "EMP7" — a short, at-a-glance ID rather than a full name+confidence
    label. side is optional (omit or pass "unknown" for no prefix) — see
    SIDE_LETTER's own comment for why "unknown" deliberately gets none.
    side_tier (added 2026-08-18, same reasoning as side_color's own
    side_tier param): a still-provisional TIER_POSITION-only read also
    gets no letter, matching its box now rendering the same "uncertain"
    orange as a true unknown — showing a confident "E"/"A" prefix next to
    an orange (not red/blue) box would read as a contradiction."""
    letters = "".join(part[0] for part in name.split("_")).upper()
    prefix = "" if (side_tier is not None and side_tier <= TIER_POSITION) else SIDE_LETTER.get(side, "")
    return f"{prefix}{letters}{track_id}"


def _alpha_rect(display, pt1, pt2, color, alpha):
    """Blends a filled rect into `display` at `alpha` opacity, clipped to the
    frame. Translucency (rather than the solid fills this file used before) is
    most of what separates a readable overlay from the "low poly" look Duncan
    flagged 2026-08-20 -- a label can sit over gameplay without punching an
    opaque hole in it."""
    h, w = display.shape[:2]
    x1, y1 = max(0, pt1[0]), max(0, pt1[1])
    x2, y2 = min(w, pt2[0]), min(h, pt2[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi = display[y1:y2, x1:x2]
    overlay = np.full(roi.shape, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def draw_label(display, text, x1, y1, y2, color, scale=0.5, thickness=1):
    """Draws `text` on a translucent dark plate with a thin accent bar in
    `color`, positioned just above (x1, y1) with enough clearance that neither
    the text nor its plate overlaps the box it's labeling -- falls back to just below
    (x1, y2) when there's no room above. A thin border in `color` keeps the
    status color visible at a glance even though the text itself now sits
    on black rather than directly on the (often busy, similarly-colored)
    gameplay footage.

    Added 2026-08-20 per Duncan's direct report that the old bare colored-
    text-on-live-gameplay labels were unreadable (no contrast against
    moving footage) and could sit on top of the very box they described."""
    font = cv2.FONT_HERSHEY_DUPLEX  # softer/rounder strokes than HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 7, 5
    accent_w = 3  # the colored bar down the plate's left edge
    plate_h = text_h + baseline + pad_y * 2
    if y1 - plate_h - 4 > 0:
        plate_y2 = y1 - 4
        plate_y1 = plate_y2 - plate_h
    else:
        plate_y1 = y2 + 4
        plate_y2 = plate_y1 + plate_h
    plate_w = text_w + pad_x * 2 + accent_w
    # Clamp horizontally so a label on a unit near the frame edge stays fully
    # readable instead of running off-screen (caught 2026-08-20 while previewing
    # the redesign on a real frame -- detections near the right edge rendered as
    # a truncated "baby"/"ice spirit" with the rest of the plate cut off).
    frame_w = display.shape[1]
    plate_x1 = max(0, min(x1, frame_w - plate_w))
    plate_x2 = plate_x1 + plate_w

    # Dark translucent plate, then a solid accent bar carrying the status color.
    # Colored TEXT on a colored plate was the old look; white text on a neutral
    # plate with a color accent reads far more cleanly over busy footage.
    _alpha_rect(display, (plate_x1, plate_y1), (plate_x2, plate_y2), (20, 20, 20), 0.72)
    _alpha_rect(display, (plate_x1, plate_y1), (plate_x1 + accent_w, plate_y2), color, 0.95)
    cv2.putText(display, text, (plate_x1 + accent_w + pad_x, plate_y2 - pad_y - baseline // 2),
                font, scale, (245, 245, 245), thickness, cv2.LINE_AA)


BOX_CORNER_RADIUS = 9  # rounded-corner radius; clamped to the box's own size below
BOX_THICKNESS = 2


def draw_box(display, box, color, label=None, radius=BOX_CORNER_RADIUS,
             thickness=BOX_THICKNESS):
    """A fully-closed rounded rectangle, anti-aliased (2026-08-20, revised: the
    original 3px aliased cv2.rectangle looked "shitty and retro"; a corner-
    brackets-only version replaced it, and Duncan then asked for full rounded
    squares instead -- he liked the new colors but wanted the box closed).

    cv2 has no rounded-rect primitive, so this is 4 straight edges plus 4
    quarter-circle ellipse arcs. Radius is clamped to half the shorter side so
    small boxes (a Skeleton, a Bat) degrade to a normal rounded shape instead
    of the arcs overlapping and rendering inside out."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return
    r = max(0, min(radius, w // 2, h // 2))
    t, aa = thickness, cv2.LINE_AA
    # straight edges (inset by the corner radius at each end)
    cv2.line(display, (x1 + r, y1), (x2 - r, y1), color, t, aa)
    cv2.line(display, (x1 + r, y2), (x2 - r, y2), color, t, aa)
    cv2.line(display, (x1, y1 + r), (x1, y2 - r), color, t, aa)
    cv2.line(display, (x2, y1 + r), (x2, y2 - r), color, t, aa)
    if r > 0:
        for (cx, cy, start) in ((x1 + r, y1 + r, 180), (x2 - r, y1 + r, 270),
                                (x2 - r, y2 - r, 0), (x1 + r, y2 - r, 90)):
            cv2.ellipse(display, (cx, cy), (r, r), 0, start, start + 90, color, t, aa)
    if label:
        draw_label(display, label, x1, y1, y2, color)


SPAWN_MARKER_MIN_RADIUS = 32  # a FIXED floor, not scaled to the icon's own
# tiny detected size (~20-45px) -- corrected 2026-08-18 after Duncan reported
# seeing the panel's "SPAWN ICONS" text trigger correctly but no visible
# marker on the gameplay window at all ("either microscopic or not getting
# drawn"). A radius scaled to the icon itself (the original approach) looked
# fine in a calm static test but was almost certainly too subtle against a
# busy, moving live game. This marker's job is to grab attention for a rare
# event, not precisely outline the detected blob the way a troop box does.


def draw_spawn_icon_marker(display, icon_bbox, side):
    """Deliberately a DIFFERENT shape from draw_box()'s rectangles (a large
    double-ring circle plus bold "SPAWN"/side text) so a live spawn-icon
    detection is unmistakably a different signal from the ordinary
    side/badge boxes already drawn everywhere else -- added 2026-08-18 per
    Duncan's request to see this live rather than only after the fact in
    temp_img/opponent_plays/. Deliberately oversized/high-contrast (see
    SPAWN_MARKER_MIN_RADIUS's own comment) -- correctness of WHICH unit has
    an icon is already handled by the panel text and the underlying
    detector; this marker's only job is to be impossible to miss."""
    x1, y1, x2, y2 = icon_bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    radius = max(SPAWN_MARKER_MIN_RADIUS, max(x2 - x1, y2 - y1) // 2 + 4)
    color = side_color(side)

    # Redesigned 2026-08-20 per Duncan ("make the spawn icon recognition animation
    # much cleaner"). Was three stacked hard circles (white ring + 5px colored ring
    # + dot), which read as a crude bullseye. Now: a soft translucent halo, a thin
    # anti-aliased ring, and four short cardinal ticks -- a targeting reticle rather
    # than a bullseye. Still unmistakably a different SHAPE from the unit corner-
    # brackets, which was the original reason it's a circle at all.
    overlay_r = radius + 6
    roi_pt1, roi_pt2 = (cx - overlay_r, cy - overlay_r), (cx + overlay_r, cy + overlay_r)
    h, w = display.shape[:2]
    rx1, ry1 = max(0, roi_pt1[0]), max(0, roi_pt1[1])
    rx2, ry2 = min(w, roi_pt2[0]), min(h, roi_pt2[1])
    if rx2 > rx1 and ry2 > ry1:
        roi = display[ry1:ry2, rx1:rx2]
        halo = roi.copy()
        cv2.circle(halo, (cx - rx1, cy - ry1), radius, color, -1, cv2.LINE_AA)
        cv2.addWeighted(halo, 0.18, roi, 0.82, 0, dst=roi)

    cv2.circle(display, (cx, cy), radius, color, 2, cv2.LINE_AA)
    tick_in, tick_out = radius - 5, radius + 5
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(display, (cx + dx * tick_in, cy + dy * tick_in),
                 (cx + dx * tick_out, cy + dy * tick_out), color, 2, cv2.LINE_AA)
    cv2.circle(display, (cx, cy), 2, (245, 245, 245), -1, cv2.LINE_AA)

    draw_label(display, f"SPAWN · {side.upper()}", cx - radius, cy - radius, cy + radius,
               color, scale=0.5, thickness=1)


def draw_tower_hp(display, box, tower_name, info, show_details):
    """Draws a tower's state as an actual miniature PROGRESS BAR sitting just
    above its calibrated bar region, rather than a corner-bracket box plus a
    text percentage. Added 2026-08-20 per Duncan ("change how the health bar
    reading looks") -- an HP fraction is a magnitude, and a filled bar shows a
    magnitude at a glance in a way "68%" in a corner never does. The bar is
    drawn OUTSIDE (above) the real region so it never covers the very pixels
    tower_reader.py is measuring, which matters when eyeballing whether a
    reading is right."""
    x1, y1, x2, y2 = box
    color, label = tower_color_and_label(tower_name, info)
    bar_h = 7
    by2 = y1 - 5
    by1 = by2 - bar_h
    _alpha_rect(display, (x1, by1), (x2, by2), (18, 18, 18), 0.75)

    hp = info["hp_fraction"]
    if info["destroyed"]:
        # Unmistakably different from "0% but alive" -- a hatched/empty track with
        # an X rather than a bar that merely happens to be empty.
        cv2.line(display, (x1, by1), (x2, by2), color, 1, cv2.LINE_AA)
        cv2.line(display, (x1, by2), (x2, by1), color, 1, cv2.LINE_AA)
    elif hp is not None:
        fill_w = int((x2 - x1) * max(0.0, min(1.0, hp)))
        if fill_w > 0:
            _alpha_rect(display, (x1, by1), (x1 + fill_w, by2), color, 0.9)
    cv2.rectangle(display, (x1, by1), (x2, by2), color, 1, cv2.LINE_AA)

    if show_details:
        draw_label(display, f"{tower_name} · {label}", x1, by1, by2, color, scale=0.4)


def tower_color_and_label(tower_name: str, info: dict):
    """(color, label) for one tower's TowerStateTracker entry ({"hp_fraction":
    float_or_None, "destroyed": bool}), used both on the gameplay window's
    tower boxes and (color only) in the panel's TOWERS section. Added
    2026-08-20 alongside wiring perception/tower_reader.py into debug_live.py."""
    if info["destroyed"]:
        return (0, 0, 160), "DESTROYED"  # dark red -- distinct from the brighter
        # red already used for "enemy" elsewhere, so a destroyed tower reads as
        # unambiguously different from an ordinary enemy-side unit box
    hp = info["hp_fraction"]
    if hp is None:
        return (0, 165, 255), "?"  # orange -- same "uncertain/no signal" color used
        # everywhere else in this panel (e.g. a still-provisional TIER_POSITION read)
    if hp > 0.5:
        color = (0, 200, 0)
    elif hp > 0.2:
        color = (0, 165, 255)
    else:
        color = (0, 0, 255)
    return color, f"{hp * 100:.0f}%"


class SlotGrid:
    """Assigns each track_id a stable slot index (0..num_slots-1) that it
    keeps for as long as it's present, freed for reuse once it's gone.
    See module docstring's "Why a grid" for the problem this solves."""

    def __init__(self, num_slots=GRID_SLOTS):
        self.num_slots = num_slots
        self._slot_of = {}  # track_id -> slot index
        self._free = list(range(num_slots))

    def assign(self, track_ids_present) -> dict:
        present = set(track_ids_present)
        gone = [tid for tid in self._slot_of if tid not in present]
        for tid in gone:
            self._free.append(self._slot_of.pop(tid))
        self._free.sort()
        for tid in present:
            if tid not in self._slot_of and self._free:
                self._slot_of[tid] = self._free.pop(0)
        return self._slot_of


def draw_grid(canvas, x0, y0, slot_of, entries_by_id, dim_ids=frozenset()):
    """entries_by_id: {track_id: (label, color)}. Draws each present
    track_id's label at its assigned fixed slot; empty slots get a faint
    dot so the grid's shape stays visible even when mostly empty."""
    slot_to_id = {s: tid for tid, s in slot_of.items()}
    for slot in range(GRID_COLS * GRID_ROWS):
        row, col = divmod(slot, GRID_COLS)
        cx = x0 + col * CELL_W
        cy = y0 + row * CELL_H
        tid = slot_to_id.get(slot)
        if tid is None or tid not in entries_by_id:
            cv2.circle(canvas, (cx + 3, cy - 4), 1, DARK_GRAY, -1)
            continue
        label, color = entries_by_id[tid]
        if tid in dim_ids:
            color = tuple(c // 2 for c in color)
        cv2.putText(canvas, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)


class PanelWriter:
    """Small helper so the panel-building code below reads as a flat list
    of lines instead of manually tracking a running y-coordinate."""

    def __init__(self, canvas, x=14, y=28, line_height=26):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.line_height = line_height

    def line(self, text, color=WHITE, scale=0.55, thickness=1):
        cv2.putText(self.canvas, text, (self.x, self.y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)
        self.y += self.line_height

    def gap(self, px=10):
        self.y += px

    def grid(self, slot_of, entries_by_id, dim_ids=frozenset()):
        draw_grid(self.canvas, self.x + 4, self.y, slot_of, entries_by_id, dim_ids)
        self.y += GRID_ROWS * CELL_H


SPAWN_ICON_PANEL_ROWS = 2  # fixed reserved rows, same "stable layout, no reflow
# jank" reasoning as OPPONENT_PLAYS_VISIBLE_ROWS -- icon detections are rare/
# transient (usually 0-1 per frame), but a fixed row count keeps everything
# below it from jumping around on the rare frame with more than one.


def build_panel(slot_results, elixir, cycle_state, opponent_events_all, scroll_offset, live_units, tracked_units, slot_of, timing_text, show_details, unit_view_mode, spawn_icons_this_frame=(), tower_state=None):
    height = 1260  # bumped from 1140 to fit the new TOWERS section (2026-08-20)
    canvas = np.full((height, PANEL_WIDTH, 3), PANEL_BG, dtype=np.uint8)
    w = PanelWriter(canvas)

    w.line("DEBUG PANEL", WHITE, scale=0.7, thickness=2)
    mode = "detailed" if show_details else "simple"
    w.line(timing_text, GRAY, scale=0.42)
    w.line(f"gameplay window: {mode}  (press 'd' to toggle)", GRAY, scale=0.42)
    w.line(f"unit boxes: {unit_view_mode}  (press 'u' to toggle side/confidence)", GRAY, scale=0.42)
    w.gap(10)

    w.line("SPAWN ICONS (this frame)", GRAY, scale=0.42)
    extra = len(spawn_icons_this_frame) - SPAWN_ICON_PANEL_ROWS
    # Reserve the LAST row for a "+N more" note instead of a full icon line
    # when there's overflow, so the total row count is ALWAYS exactly
    # SPAWN_ICON_PANEL_ROWS regardless of how many icons fired this frame.
    shown_icons = spawn_icons_this_frame[:SPAWN_ICON_PANEL_ROWS - 1] if extra > 0 else spawn_icons_this_frame[:SPAWN_ICON_PANEL_ROWS]
    if not spawn_icons_this_frame:
        w.line("  none", GRAY, scale=0.4)
    for name, side in shown_icons:
        w.line(f"  {name}: {side.upper()}", side_color(side), scale=0.42)
    if extra > 0:
        w.line(f"  (+{extra} more)", GRAY, scale=0.38)
    for _ in range(SPAWN_ICON_PANEL_ROWS - max(len(shown_icons) + (1 if extra > 0 else 0), 1)):
        w.gap(26)
    w.gap(14)

    w.line("HAND", GRAY, scale=0.45)
    for i, (name, score, color) in enumerate(slot_results, start=1):
        shown = name if name is not None else "no templates"
        w.line(f"  Slot {i}: {shown:14s} {score:.2f}", color)
    w.gap(14)

    w.line("ELIXIR", GRAY, scale=0.45)
    number_color = color_for(str(elixir["value"]), elixir["value_confidence"], ELIXIR_CONFIDENCE_THRESHOLD)
    w.line(f"  Number:    {elixir['value']}  (conf {elixir['value_confidence']:.2f})", number_color)
    w.line(f"  Bar est:   {elixir['bar_estimate']:.2f} / 10", (255, 200, 0))
    w.line(f"  Fraction:  +{elixir['fraction']:.2f} to next", (255, 200, 0))
    w.gap(14)

    w.line("TOWERS", GRAY, scale=0.45)
    if tower_state is None:
        w.line("  (tower reader not active)", GRAY, scale=0.4)
    else:
        def _tstr(name):
            color, label = tower_color_and_label(name, tower_state[name])
            return label, color
        for row_label, left_name, right_name in [
            ("Princess ally: ", "ally_left", "ally_right"),
            ("Princess enemy:", "enemy_left", "enemy_right"),
        ]:
            left_label, left_color = _tstr(left_name)
            right_label, right_color = _tstr(right_name)
            w.line(f"  {row_label} L {left_label:10s} R {right_label}", left_color, scale=0.42)
        king_ally_label, king_ally_color = _tstr("ally_king")
        king_enemy_label, king_enemy_color = _tstr("enemy_king")
        w.line(f"  King ally: {king_ally_label:10s} King enemy: {king_enemy_label}", king_ally_color, scale=0.42)
    w.gap(14)

    w.line("CYCLE  (press 'r' to reset for a new match)", GRAY, scale=0.45)
    resolved = "resolved" if cycle_state["queue_fully_resolved"] else "resolving"
    hand_line = "  ".join(f"{i}:{name or '?'}" for i, name in cycle_state["hand"].items())
    w.line(f"  HAND (4):     {hand_line}", (255, 200, 0), scale=0.42)
    w.line(f"  ON DECK (1):  {cycle_state['on_deck']}  <- drawn on the next play", (255, 200, 0), scale=0.42)
    w.line(f"  IN QUEUE (3): {' -> '.join(cycle_state['rest_of_queue'])}  ({resolved})", (255, 200, 0), scale=0.42)
    last_play = cycle_state["last_play"]
    if last_play:
        tag = "confirmed" if last_play["confirmed"] else "UNCONFIRMED"
        if last_play["desync"]:
            tag += ", DESYNC"
        color = (0, 200, 0) if last_play["confirmed"] and not last_play["desync"] else (0, 165, 255)
        if last_play["desync"]:
            color = (255, 0, 255)  # magenta — distinct from the confirmed/unconfirmed colors
        w.line(f"  Last play: {last_play['card']} -> {last_play['revealed']} ({tag})", color, scale=0.42)
    else:
        w.line(f"  Last play: none yet ({cycle_state['plays_this_match']} total this match)", GRAY, scale=0.42)
    w.gap(14)

    total = len(opponent_events_all)
    max_offset = max(0, total - OPPONENT_PLAYS_VISIBLE_ROWS)
    offset = min(scroll_offset, max_offset)
    end = total - offset
    start = max(0, end - OPPONENT_PLAYS_VISIBLE_ROWS)
    visible = opponent_events_all[start:end]
    position = f"{start + 1}-{end} of {total}" if total else "0 of 0"
    w.line(f"OPPONENT PLAYS  (scroll: mouse wheel)  {position}", GRAY, scale=0.45)
    if not visible:
        w.line("  none yet", GRAY, scale=0.42)
    for ev in visible:
        countpart = f" x{ev['count']}" if ev["count"] != 1 else ""
        if ev["attribution"] == "played":
            label = ev["card_slug"] or ev["card"]
            cost = f", {ev['elixir_cost']} elixir" if ev["elixir_cost"] is not None else ""
            w.line(f"  PLAYED: {label}{countpart}{cost}", (0, 200, 0), scale=0.42)
        elif ev["attribution"] == "spawned":
            src = f" (from {ev['source_card']})" if ev["source_card"] else " (source unknown)"
            w.line(f"  spawned: {ev['card']}{countpart}{src}", (150, 150, 150), scale=0.42)
        elif ev["attribution"] == "misclassified":
            w.line(f"  misclassified: {ev['card']}{countpart} -- {ev['flag_reason']}", (90, 90, 90), scale=0.42)
        elif ev["attribution"] == "reacquired":
            w.line(f"  reacquired: {ev['card']}{countpart} (same building, track re-lost)", (200, 140, 0), scale=0.42)
        else:
            w.line(f"  UNCERTAIN: {ev['card']}{countpart} -- {ev['flag_reason']}", (255, 0, 255), scale=0.42)
    # Pad to a fixed row count so the sections below never shift as the
    # visible window's content changes -- same "stable layout" reasoning
    # as the unit grids below (see "Why a grid" in the module docstring).
    for _ in range(OPPONENT_PLAYS_VISIBLE_ROWS - max(len(visible), 1)):
        w.gap(26)
    w.gap(14)

    live_entries = {u["track_id"]: (compact_tag(u["name"], u["track_id"], u["side"], u.get("side_tier")), unit_view_color(u, unit_view_mode))
                     for u in live_units}
    tracked_entries = {u["track_id"]: (compact_tag(u["name"], u["track_id"], u["side"], u.get("side_tier")), unit_view_color(u, unit_view_mode))
                        for u in tracked_units}
    dim_ids = {u["track_id"] for u in tracked_units if not u["visible"]}

    mode_hint = "blue=ally red=enemy orange=no signal (A/E letter prefix)" if unit_view_mode == "side" \
        else "green=confident orange=weak red=low"
    w.line(f"UNITS - LIVE READ ({len(live_units)})  -- raw, will flicker  ({mode_hint})", GRAY, scale=0.42)
    w.grid(slot_of, live_entries)
    w.gap(10)

    w.line(f"UNITS - TRACKED ({len(tracked_units)})  -- dim = presumed, not confirmed", GRAY, scale=0.42)
    w.grid(slot_of, tracked_entries, dim_ids)

    return canvas


def open_log(log_arg, no_log):
    if no_log:
        return None
    path = log_arg or os.path.join("logs", f"debug_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to {path} — hand this file to Claude to analyze tracking patterns.\n")
    # buffering=1: line-buffered, flushes after every frame's line rather than
    # sitting in memory until close() — otherwise a crash or hard kill (not a
    # clean 'q'/Ctrl+C) loses the entire log, which defeats the point of a
    # log that's supposed to help diagnose exactly those unexpected moments.
    return open(path, "w", encoding="utf-8", buffering=1)


def write_log(log_file, frame_number, session_start, slot_results, elixir, cycle_state, opponent_events, live_units, tracked_units, tower_state=None):
    if log_file is None:
        return
    record = {
        "frame": frame_number,
        "t": round(time.time() - session_start, 3),
        "hand": [{"slot": i, "name": n, "score": round(s, 3)}
                 for i, (n, s, _) in enumerate(slot_results, start=1)],
        "elixir": {"value": elixir["value"], "conf": round(elixir["value_confidence"], 3),
                   "bar": round(elixir["bar_estimate"], 3)},
        "cycle": {"hand": cycle_state["hand"], "on_deck": cycle_state["on_deck"],
                  "rest_of_queue": cycle_state["rest_of_queue"],
                  "queue_fully_resolved": cycle_state["queue_fully_resolved"],
                  "last_play": cycle_state["last_play"],
                  "plays_this_match": cycle_state["plays_this_match"]},
        "opponent_plays": opponent_events,
        "live": [{"id": u["track_id"], "name": u["name"], "side": u["side"], "conf": round(u["confidence"], 3),
                  "bbox": u["bbox"]}
                 for u in live_units],
        "tracked": [{"id": u["track_id"], "name": u["name"], "side": u["side"], "conf": round(u["confidence"], 3),
                     "visible": u["visible"], "missed": u["missed_frames"], "age": u["age"], "bbox": u["bbox"]}
                    for u in tracked_units],
        # tower_state added 2026-08-20 alongside perception/tower_reader.py -- lets a
        # future compare_ground_truth.py-style tool correlate opponent-play/side-conflict
        # timing against exactly when a tower actually fell, not just impression from watching.
        "towers": {name: {"hp": None if info["hp_fraction"] is None else round(info["hp_fraction"], 3),
                           "destroyed": info["destroyed"]}
                   for name, info in (tower_state or {}).items()},
    }
    log_file.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=0.03,
                         help="Target seconds between refreshes — only pads the frame if "
                              "processing finished early, never adds delay on top of it.")
    parser.add_argument("--detailed", action="store_true",
                         help="Start the gameplay window in detailed mode instead of simple "
                              "(can also toggle live with the 'd' key)")
    parser.add_argument("--log", type=str, default=None,
                         help="Path to write a JSONL debug log to (one line per frame). "
                              "Default: logs/debug_live_<timestamp>.jsonl")
    parser.add_argument("--no-log", action="store_true", help="Disable logging entirely")
    parser.add_argument("--detector", choices=["onnx", "katacr"], default="onnx",
                         help="Which unit detector to run. 'onnx' = the current "
                              "build-a-bot units.onnx (YOLOv10m). 'katacr' = the "
                              "EXPERIMENTAL KataCR YOLOv8l pair (see "
                              "perception/katacr_detector.py) -- far more accurate in "
                              "offline testing but ~457ms/frame on CPU vs ~36ms, so "
                              "expect ~2fps on CPU. On a CUDA GPU it runs ~20ms/frame (~50fps), "
                              "which is FASTER than the current CPU onnx path.")
    args = parser.parse_args()
    show_details = args.detailed
    unit_view_mode = UNIT_VIEW_MODES[0]  # "side" -- toggled live with 'u'

    print("Connecting to LDPlayer...")
    controller = EmulatorController()
    capture = ScreenCapture(controller)
    card_templates = load_card_templates()
    elixir_templates = load_elixir_templates()
    if args.detector == "katacr":
        from perception.katacr_detector import KataCRDetector
        # include_non_units=False: KataCR also detects towers/bars/clock/elixir, which
        # are genuinely useful (see that module's docstring) but would be nonsense input
        # to unit_tracker/opponent_play_tracker, which expect troops only. Filtered here
        # so this stays a clean A/B of UNIT detection specifically.
        unit_detector = KataCRDetector(include_non_units=False)
        print("DETECTOR: KataCR YOLOv8l x2 (EXPERIMENTAL) — expect ~2fps on CPU.\n"
              "          Its own belonging channel supplies ally/enemy, so sides here do\n"
              "          NOT come from side_reader.py's badge reading.")
    else:
        unit_detector = UnitDetector()
    unit_tracker = UnitTracker()
    tower_tracker = TowerStateTracker()
    slot_grid = SlotGrid()

    deck = load_deck()
    if deck:
        card_templates = {name: examples for name, examples in card_templates.items() if name in deck}
        print(f"Deck restriction active: matching only against {sorted(card_templates.keys())}")
    else:
        print("No config/deck.txt found — matching against the full library.")

    cycle_tracker = CycleTracker(deck or [])
    if not deck:
        print("Cycle tracking needs config/deck.txt (8 cards) to be meaningful — "
              "running anyway, but expect it to stay unresolved.")
    opponent_tracker = OpponentPlayTracker(own_deck=deck or [])
    opponent_events_all = []  # full match history, oldest first — see OPPONENT_PLAYS_VISIBLE_ROWS
    # Card-match throttling state -- see the block in the main loop for why.
    CARD_MATCH_INTERVAL = 0.2  # seconds; ~5Hz, far faster than a hand can actually change
    last_card_match_time = 0.0
    last_slot_results = [("UNKNOWN", 0.0, (0, 165, 255))] * 4
    cycle_state = None
    own_recent_plays = []  # [(card_slug, timestamp)] fed to opponent_tracker.update() each frame
    scroll_state = {"offset": 0}  # shared with on_panel_scroll via the mouse callback below

    total_examples = sum(len(v) for v in card_templates.values())
    print(f"Loaded {total_examples} card example(s) across {len(card_templates)} card(s).")
    print(f"Loaded {sum(len(v) for v in elixir_templates.values())} elixir-digit example(s) "
          f"across {len(elixir_templates)} value(s).")
    print("Opening live view — press 'q' in either window (or Ctrl+C here) to stop, "
          "'d' to toggle simple/detailed, 'u' to toggle unit box side/confidence coloring.\n")

    log_file = open_log(args.log, args.no_log)
    frame_number = 0
    session_start = time.time()
    last_conflict_snapshot_time = 0.0
    last_spawn_icon_miss_time = 0.0
    captured_miss_ids = set()  # track_ids already saved to SPAWN_ICON_MISS_DIR this session

    cv2.namedWindow(GAMEPLAY_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(PANEL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PANEL_WINDOW, PANEL_WIDTH, 1260)
    cv2.setMouseCallback(PANEL_WINDOW, on_panel_scroll, scroll_state)
    # Best-effort default layout: gameplay window at the OS default position,
    # panel placed just to its right. Drag either window afterward if this
    # doesn't match your monitor setup — it's just a starting point.
    cv2.moveWindow(PANEL_WINDOW, 950, 0)

    try:
        while True:
            frame_start = time.time()
            frame_number += 1

            t0 = time.time()
            screenshot = capture.screenshot()
            capture_ms = (time.time() - t0) * 1000
            now_wall = time.time()
            display = screenshot.copy()

            t0 = time.time()
            # Hand-card matching is by far the most expensive thing in this loop
            # (measured 2026-08-20: ~48ms/frame against the full 321-example library,
            # vs ~21ms for the GPU detector), and it is re-identifying cards that only
            # actually change when a card is PLAYED -- a few times per match, not 12
            # times per second. Re-matching at CARD_MATCH_INTERVAL instead of every
            # frame is invisible to a human watching, and is most of the fix for the
            # choppiness Duncan reported.
            #
            # cycle_tracker is deliberately only updated on frames where a FRESH match
            # actually happened (see below). Feeding it the cached tuple repeatedly
            # would let identical values satisfy its HAND_STABLE_FRAMES debounce
            # without any new evidence, which is exactly the kind of false confirmation
            # that debounce exists to prevent.
            cards_fresh = (now_wall - last_card_match_time) >= CARD_MATCH_INTERVAL
            if cards_fresh:
                last_card_match_time = now_wall
                slot_results = []  # (name, score, color) per hand slot, for the panel
                for (x1, y1, x2, y2) in hand_slot_regions():
                    crop = screenshot[y1:y2, x1:x2]
                    name, score = card_best_match(crop, card_templates)
                    if name is None or score < CARD_CONFIDENCE_THRESHOLD:
                        box_color = color_for("UNKNOWN", score if score is not None else 0.0, CARD_CONFIDENCE_THRESHOLD)
                        shown_name = name if name is not None else None
                    else:
                        box_color = color_for(name, score, CARD_CONFIDENCE_THRESHOLD)
                        shown_name = name
                    slot_results.append((shown_name if shown_name else "UNKNOWN", score or 0.0, box_color))
                last_slot_results = slot_results
            else:
                slot_results = last_slot_results

            # Boxes are drawn every frame from whichever results are current, so the
            # overlay never flickers even though matching itself is throttled.
            for (x1, y1, x2, y2), (shown_name, score, box_color) in zip(hand_slot_regions(), slot_results):
                label = f"{shown_name} {score:.2f}" if show_details else None
                draw_box(display, (x1, y1, x2, y2), box_color, label)

            elixir = read_elixir(screenshot, elixir_templates)
            match_ms = (time.time() - t0) * 1000

            cycle_state = (cycle_tracker.update(slot_results, elixir, now=time.time())
                            if cards_fresh else cycle_state)
            if cycle_state["last_play"]:
                own_recent_plays.append((cycle_state["last_play"]["card"], cycle_state["last_play"]["frame_time"]))

            ex1, ey1, ex2, ey2 = NUMBER_REGION
            number_color = color_for(str(elixir["value"]), elixir["value_confidence"], ELIXIR_CONFIDENCE_THRESHOLD)
            number_label = f"{elixir['value']} {elixir['value_confidence']:.2f}" if show_details else None
            draw_box(display, (ex1, ey1, ex2, ey2), number_color, number_label)
            bar_label = f"{elixir['bar_estimate']:.2f}" if show_details else None
            draw_box(display, (int(BAR_LEFT_X), BAR_Y_TOP, int(BAR_RIGHT_X), BAR_Y_BOTTOM), (255, 200, 0), bar_label)

            tower_state = tower_tracker.update(screenshot)
            for name, region in TOWER_BAR_REGIONS.items():
                draw_tower_hp(display, region, name, tower_state[name], show_details)

            t0 = time.time()
            live_units = unit_tracker.update(unit_detector.detect(screenshot), screenshot=screenshot, tower_state=tower_state)
            tracked_units = unit_tracker.current_state()
            slot_of = slot_grid.assign(u["track_id"] for u in tracked_units)
            units_ms = (time.time() - t0) * 1000

            opponent_events = opponent_tracker.update(tracked_units, now=time.time(), own_recent_plays=own_recent_plays, screenshot=screenshot)
            # write_log() below gets the full, unfiltered opponent_events -- only the
            # PANEL/snapshot views (both purely for a human to look at) drop phantom
            # noise, see PHANTOM_NOISE_MARKER's own comment for why.
            opponent_events_all.extend(e for e in opponent_events if not _is_phantom_noise(e))
            save_opponent_play_snapshots(screenshot, [e for e in opponent_events if not _is_phantom_noise(e)])

            conflict = find_side_conflict(tracked_units)
            now = time.time()
            if conflict and (now - last_conflict_snapshot_time) >= CONFLICT_SNAPSHOT_COOLDOWN_SECONDS:
                last_conflict_snapshot_time = now
                a, b = conflict
                os.makedirs(CONFLICT_SNAPSHOT_DIR, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                img_path = os.path.join(CONFLICT_SNAPSHOT_DIR, f"conflict_{stamp}.png")
                info_path = os.path.join(CONFLICT_SNAPSHOT_DIR, f"conflict_{stamp}.json")
                cv2.imwrite(img_path, screenshot)  # RAW frame, not the annotated display -- this is for pixel analysis, not viewing
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump({"track_a": a, "track_b": b, "frame": frame_number, "t": round(now - session_start, 3)}, f, indent=2)
                print(f"Side conflict snapshot saved: {img_path} "
                      f"({a['name']} track{a['track_id']}={a['side']} vs track{b['track_id']}={b['side']})")
            # Gameplay window draws the raw live read (what's actually visible
            # in this exact frame) — the panel is where the smoothed/tracked
            # view lives, since drawing stale/presumed boxes on top of
            # gameplay risks implying false confidence about where an
            # occluded unit currently is.
            for u in live_units:
                color = unit_view_color(u, unit_view_mode)
                # Full readable troop NAME, not just initials (2026-08-20, Duncan:
                # "add a name tag to them so i know what troop its seeing"). The old
                # compact_tag ("EMP7") was built for the panel's dense grid, where
                # space is genuinely tight -- on the gameplay window there's room for
                # the real name, and initials made it impossible to tell at a glance
                # what the model actually thought it saw. Track id is kept (it's how
                # you follow one unit across frames) but de-emphasized after a middot.
                pretty = u["name"].replace("_", " ")
                # A trailing bar glyph marks a side that came from an associated
                # health-bar/level-badge detection rather than the model's own
                # per-unit belonging read -- the far more trustworthy of the two
                # (see perception/bar_reader.py). Lets a wrong side be judged
                # against WHY it was chosen, not just that it was.
                src = " ▮" if u.get("side_source") == "bar" else ""
                hp = u.get("hp_fraction")
                hp_txt = f"  {hp * 100:.0f}%" if hp is not None else ""
                unit_label = f"{pretty} · {u['track_id']}{src}{hp_txt}"
                if show_details:
                    unit_label += f"  {u['confidence']:.2f}"
                draw_box(display, u["bbox"], color, unit_label)

            t0 = time.time()
            spawn_icons_this_frame = []  # [(name, side)] -- for the panel, see build_panel()
            for u in live_units:
                if u["age"] > SPAWN_ICON_PREVIEW_MAX_AGE:
                    continue
                icon_side, _icon_area, icon_bbox = find_spawn_icon(screenshot, u["bbox"])
                if icon_side is not None:
                    draw_spawn_icon_marker(display, icon_bbox, icon_side)
                    spawn_icons_this_frame.append((u["name"], icon_side))
                    print(f"[spawn icon] frame {frame_number}: {u['name']} = {icon_side} "
                          f"at {icon_bbox} (unit bbox {u['bbox']})")  # diagnostic -- remove once
                    # confirmed the on-screen marker is reliably visible; cross-reference this
                    # terminal line against what's actually seen on the gameplay window.
                elif (u["age"] <= TRACKER_SPAWN_ICON_SEED_MAX_AGE and u["track_id"] not in captured_miss_ids
                        and (time.time() - last_spawn_icon_miss_time) >= SPAWN_ICON_MISS_COOLDOWN_SECONDS):
                    # A genuine miss within the window unit_tracker.py itself actually uses for
                    # seeding (not the wider display-only preview window above) -- see
                    # SPAWN_ICON_MISS_DIR's own comment for why this exists instead of guessing
                    # at the search geometry again with no real data.
                    last_spawn_icon_miss_time = time.time()
                    captured_miss_ids.add(u["track_id"])
                    os.makedirs(SPAWN_ICON_MISS_DIR, exist_ok=True)
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    img_path = os.path.join(SPAWN_ICON_MISS_DIR, f"miss_{stamp}.png")
                    info_path = os.path.join(SPAWN_ICON_MISS_DIR, f"miss_{stamp}.json")
                    cv2.imwrite(img_path, screenshot)  # RAW frame -- for pixel analysis, not viewing
                    with open(info_path, "w", encoding="utf-8") as f:
                        json.dump({"track_id": u["track_id"], "name": u["name"], "side": u["side"],
                                    "age": u["age"], "bbox": list(u["bbox"]), "frame": frame_number,
                                    "t": round(time.time() - session_start, 3)}, f, indent=2)
                    print(f"Spawn-icon miss snapshot saved: {img_path} "
                          f"({u['name']} track{u['track_id']} age={u['age']} bbox={u['bbox']})")
            spawn_icon_ms = (time.time() - t0) * 1000

            total_ms = (time.time() - frame_start) * 1000
            fps = 1000 / total_ms if total_ms > 0 else 0.0
            timing_text = f"FPS {fps:.1f} | cap {capture_ms:.1f}ms | cards {match_ms:.1f}ms | units {units_ms:.1f}ms | icons {spawn_icon_ms:.1f}ms"

            panel = build_panel(slot_results, elixir, cycle_state, opponent_events_all, scroll_state["offset"], live_units, tracked_units, slot_of, timing_text, show_details, unit_view_mode, spawn_icons_this_frame, tower_state)
            write_log(log_file, frame_number, session_start, slot_results, elixir, cycle_state, opponent_events, live_units, tracked_units, tower_state)

            cv2.imshow(GAMEPLAY_WINDOW, display)
            cv2.imshow(PANEL_WINDOW, panel)

            # Only pad up to the requested interval if processing finished
            # early — waitKey still pumps window events (drag/resize) and
            # checks for 'q'/'d' either way.
            elapsed = time.time() - frame_start
            remaining_ms = max(1, int((args.interval - elapsed) * 1000))
            key = cv2.waitKey(remaining_ms) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_details = not show_details
            if key == ord("u"):
                idx = (UNIT_VIEW_MODES.index(unit_view_mode) + 1) % len(UNIT_VIEW_MODES)
                unit_view_mode = UNIT_VIEW_MODES[idx]
                print(f"Unit box mode: {unit_view_mode}")
            if key == ord("r"):
                cycle_tracker.reset()
                opponent_tracker.reset()
                tower_tracker = TowerStateTracker()  # a new match means every tower is
                # back to full health/alive -- no reset() method on the class itself since
                # simply re-instantiating is just as clear and matches how opponent_events_all
                # below is reset (reassignment, not a dedicated clear method)
                opponent_events_all = []
                own_recent_plays = []
                # Also clear the throttled card-match cache, so a new match can't
                # briefly show the previous match's last-known hand.
                last_card_match_time = 0.0
                cycle_state = None
                scroll_state["offset"] = 0
                print("Cycle tracker, opponent play tracker, and tower tracker reset — treat this as the start of a new match.")
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if log_file:
            log_file.close()
            print(f"Log closed ({frame_number} frame(s) written).")
        print("Stopped.")


if __name__ == "__main__":
    main()
