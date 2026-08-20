"""
Reads each of the 6 towers' (4 princess + 2 king) health-bar fill
fraction (0.0-1.0) directly from pixel color -- same "read a known UI
element's pixel-bar-fill directly" technique perception/elixir_counter.py
already validates for the elixir bar, not OCR, not a learned model.

A king tower's health bar doesn't render at all until EITHER it's taken
direct damage OR a princess tower on that same side has been destroyed
(Duncan, 2026-08-20, correcting an earlier incomplete version of this
fact) -- there's no bar to read at all for a chunk of a typical match.
See TowerStateTracker's own docstring for the real bug this caused and
how it's handled (an inactive king tower must NOT be confused with a
destroyed one, since both produce the exact same "no clean row" reading).

Princess tower region calibration (2026-08-20), via find_coords.py against a real fresh
capture (test_screenshot.png, all 4 princess towers near/at full health):
Duncan clicked top-left/bottom-right of each bar directly, cross-checked
here (not blindly trusted) by inspecting the ACTUAL pixel data row-by-row
before building anything on top of it, per this project's own established
rigor for every prior calibration:

  ally_left:  (179, 990, 279, 1004)   ally_right:  (650, 992, 750, 1006)
  enemy_left: (177, 232, 279, 248)    enemy_right: (650, 232, 750, 248)

Real finding while inspecting: Duncan warned the tower's own HP NUMBER
overlaps the bar (same category of problem elixir_counter.py already
solved for its own bar). Row-by-row inspection of the real capture found
the overlap is NOT symmetric -- the ally (bottom-of-screen) towers have
the digit sitting in the LOWER portion of the clicked region, while the
enemy (top-of-screen) towers have it in the UPPER portion (presumably so
each digit reads right-side-up toward whichever player is closer to that
tower). This rules out a fixed "skip N rows from the top/bottom" shortcut
-- _best_row() picks the least digit-contaminated row per-frame instead,
the same "best row" technique build-a-bot's own (unusable at this
project's resolution, but still worth borrowing the algorithm from --
see Project_Status) number_detector.py already validated for this exact
problem.

Filled-bar color calibrated directly from the real capture's cleanest
rows (NOT reused from side_reader.py's badge colors by assumption, though
they turned out close -- see below):
  ally  (blue) filled: BGR ~ (252, 207, 114)
  enemy (red)  filled: BGR ~ (91, 38, 221)
A real, unplanned cross-check: rows 18-20 of the enemy_left capture (BGR
~(32, 3, 118)) closely match perception/side_reader.py's ALREADY-
calibrated empty-red badge-bar color (BGR~27-31,0-4,106-107 -- same hue,
much darker, not a neutral gray) -- real evidence the same fill/empty
color pair likely carries over from unit badges to tower bars, not just
an assumption. Still only ONE real data point on the empty side (EMPTY_BGR
is otherwise a first-pass, not independently re-derived from a real
partially-damaged tower yet).

Destroyed-tower detection (2026-08-20, second real capture -- both king
towers damaged/visible for the first time, and Duncan's own ally+enemy
left princess towers both actually destroyed mid-match): confirmed
directly by inspecting the real capture that a destroyed tower shows
literally NO UI element at all (rubble only, no bar, no number) --
NOT a faint or near-zero-fill bar, which was this module's original
(wrong) assumption. This means "no clean bar-colored row found at all"
IS the real destroyed signal, not a separate low-hp-fraction case --
read_tower() returning None already means "no bar detected here," and
TowerStateTracker turns a SUSTAINED streak of that into a confirmed,
permanent "destroyed" (see DESTROYED_CONFIRM_FRAMES's own comment for
why a single frame's None can't be trusted as destroyed outright -- could
just as easily be a transient capture glitch or digit misalignment).

King tower region calibration (2026-08-20), same second capture, via
find_coords.py, Duncan deliberately excluding the decorative gold ring
around the bar itself:
  ally_king:  (401, 1205, 535, 1224)   enemy_king: (401, 44, 536, 67)
Real finding while inspecting: unlike the princess towers' simple digit-
overlap problem, a king tower's bar renders with a highlight/shadow
gradient across its own height -- two visually different bands (a
brighter one, a darker one) that both encode the SAME horizontal fill
transition, not a filled-vs-empty split. _best_row() doesn't need to know
about this explicitly -- it naturally picks whichever band's colors most
closely match the calibrated FILLED_BGR/EMPTY_BGR reference.

This calibration got the strongest real validation of anything in this
module so far: Duncan reported his own king tower at ~69% HP and the
enemy's at ~25% HP at the moment of capture, and the fill fraction
computed from the real column-by-column color transition (independently,
by hand, before any code was written) came out to ~68% and ~24% --
matching almost exactly. FILLED_BGR turned out close to the princess-
tower values (same underlying UI color); EMPTY_BGR did NOT (a lighter,
more desaturated gray-blue/gray-red rather than a dark-but-same-hue
shade) -- confirming king and princess towers are worth calibrating
independently rather than assumed identical (see FILLED_BGR/EMPTY_BGR's
own comments, keyed by (tower_type, side) for exactly this reason).
"""

import os

import cv2
import numpy as np

TOWER_BAR_REGIONS = {
    "ally_left": (179, 990, 279, 1004),
    "ally_right": (650, 992, 750, 1006),
    "enemy_left": (177, 232, 279, 248),
    "enemy_right": (650, 232, 750, 248),
    # King towers added 2026-08-20 from a second real capture (both king towers damaged --
    # per Duncan, a king tower's bar doesn't render at all until it's taken damage, so this
    # was the first capture where one could even be calibrated against). Duncan clicked
    # deliberately excluding the decorative gold ring around the bar itself.
    "ally_king": (401, 1205, 535, 1224),
    "enemy_king": (401, 44, 536, 67),
}
# Vertical padding added around each clicked region before scanning -- the digit was
# found to dip slightly outside Duncan's own clicked box on at least one side (his own
# warning, confirmed by inspection), so the row-scan needs a little headroom beyond the
# exact clicked box to have a chance of finding a clean row at all.
REGION_PAD_Y = 4


def _tower_type(tower_name: str) -> str:
    return "king" if tower_name.endswith("king") else "princess"


def _tower_side(tower_name: str) -> str:
    return "ally" if tower_name.startswith("ally") else "enemy"


# Keyed by (tower_type, side) -- calibrated SEPARATELY per type rather than assumed shared,
# even though the FILLED colors turned out to closely match between princess and king
# towers (a real, interesting cross-check, not an assumption -- see below). The EMPTY
# colors do NOT closely match between the two types, confirming they're worth keeping
# independently calibrated rather than coupled.
FILLED_BGR = {
    ("princess", "ally"): np.array([252.0, 207.0, 114.0]),
    ("princess", "enemy"): np.array([91.0, 38.0, 221.0]),
    # King tower values calibrated 2026-08-20 against a real capture with known ground-
    # truth HP (Duncan: ally king ~69%, enemy king ~25%) -- the computed fill fraction from
    # this exact calibration matched both real values almost exactly (~68%/~24%), the
    # strongest validation any tower_reader.py calibration has had so far. Notably close to
    # the princess-tower values above -- same underlying UI color, not a coincidence, but
    # kept as its own entry rather than reused directly since it was independently measured.
    ("king", "ally"): np.array([254.0, 207.0, 115.0]),
    ("king", "enemy"): np.array([89.0, 35.0, 219.0]),
}
EMPTY_BGR = {
    # Princess-tower ally empty color: REAL measurement (2026-08-20), replacing an earlier
    # unverified guess (BGR~70,40,5, borrowed from side_reader.py's badge colors by
    # analogy) that turned out actively wrong -- a real live-match bug (a healthy, ~90%+
    # tower reading 0% the instant it took any damage) traced directly to this: the old
    # guess happened to closely match a dark BORDER/bezel pixel row that's present at every
    # HP level, not any genuine empty-fill color, which fooled _best_row into preferring
    # that border row over the real bar row every time the real bar stopped being 100%
    # solid FILLED_BGR (i.e. the instant any damage occurred at all -- exactly matching
    # what was reported). Fixed by measuring the REAL transition directly: a real ~90%-
    # damaged capture showed a clean, unambiguous transition from solid FILLED_BGR to
    # BGR~(115,81,66) at the true fill boundary, confirmed consistent across two separate
    # rows (columns ~92/96 out of ~96 usable width, matching Duncan's own "~90+%" estimate
    # almost exactly) -- not a guess this time, a direct real-data measurement.
    ("princess", "ally"): np.array([115.0, 81.0, 66.0]),
    # Enemy side matches a real (if incidental) capture sample already (see module
    # docstring) -- same METHODOLOGY concern as the old ally guess above (not a from-
    # scratch damaged-tower measurement), worth the same scrutiny if a similar bug ever
    # shows up on the enemy side.
    ("princess", "enemy"): np.array([29.0, 2.0, 107.0]),
    # King tower empty colors: REAL data from the same ground-truth-validated capture as
    # the king FILLED_BGR entries above -- notably NOT close to the princess-tower empty
    # colors (more of a muted gray-blue/gray-red than a dark-but-same-hue shade), which is
    # exactly why king and princess towers get independently calibrated color pairs here
    # instead of one shared set.
    ("king", "ally"): np.array([82.0, 91.0, 111.0]),
    ("king", "enemy"): np.array([39.0, 52.0, 74.0]),
}

CLEAN_MAX_DIST = 60  # manhattan BGR distance -- a pixel within this of EITHER FILLED_BGR
# or EMPTY_BGR counts as "clean bar color" (see _clean_mask). Picked from real data
# (clean-row pixels varied by roughly +/-10-15 per channel around the mean), generous
# enough to absorb normal capture noise, not tuned against a large real sample yet.
MIN_CLEAN_ROW_FRACTION = 0.5  # a row is only usable at all if AT LEAST this fraction of
# its pixels are cleanly close to one canonical color or the other.


def _clean_mask(row_bgr: np.ndarray, key: tuple):
    """Per-pixel boolean: True where a pixel in `row_bgr` (a (W, 3) BGR
    array) is closely near EITHER the FILLED or EMPTY reference color for
    `key` (a (tower_type, side) tuple, e.g. ("king", "enemy") -- see
    FILLED_BGR/EMPTY_BGR). Used for both row selection (_best_row) and the
    fill-fraction estimate (_hp_fraction_from_row).

    Real bug found and fixed while calibrating this (2026-08-20): an
    earlier version picked the "best" row using side_reader.py's own
    looser HUE mask (_red_mask/_blue_mask -- reused there for a different
    purpose, tuned against a different UI element) -- broad enough to
    admit anti-aliased digit-edge pixels that are still red/blue-ish but
    not actually a clean bar color, which won the "most matching pixels"
    vote over rows that were genuinely uniform but had slightly fewer
    hue-mask hits. Tightening to "close to one of the two SPECIFIC known
    bar colors" (not just the right hue family) picks the real clean
    strip instead, confirmed against the real capture (see this module's
    own git history for the before/after __main__ output)."""
    diffs_filled = np.abs(row_bgr.astype(np.float32) - FILLED_BGR[key]).sum(axis=1)
    diffs_empty = np.abs(row_bgr.astype(np.float32) - EMPTY_BGR[key]).sum(axis=1)
    return (diffs_filled < CLEAN_MAX_DIST) | (diffs_empty < CLEAN_MAX_DIST)


def _best_row(region_bgr: np.ndarray, key: tuple):
    """Returns the index of the row in `region_bgr` (an (H, W, 3) BGR crop)
    with the HIGHEST fraction of cleanly filled-or-empty-colored pixels
    for `key`, or None if even the best row falls short of
    MIN_CLEAN_ROW_FRACTION -- no usable bar-color signal found at all.
    See module docstring for why a fixed row offset doesn't work here (the
    digit sits on a DIFFERENT side of the bar for ally vs enemy towers).
    For king towers specifically, real inspection also found the bar
    renders with a highlight/shadow gradient across its own height (two
    visually different bands that both encode the SAME horizontal fill
    transition) -- this naturally picks whichever band's colors most
    closely match the calibrated reference, rather than needing to know
    about the gradient explicitly."""
    best_idx, best_frac = None, -1.0
    for i in range(region_bgr.shape[0]):
        frac = float(_clean_mask(region_bgr[i], key).mean())
        if frac > best_frac:
            best_idx, best_frac = i, frac
    if best_frac >= MIN_CLEAN_ROW_FRACTION:
        return best_idx
    return None


def _hp_fraction_from_row(row_bgr: np.ndarray, key: tuple) -> float:
    """Among the pixels in `row_bgr` (a (W, 3) BGR array) that are cleanly
    filled-or-empty-colored (see _clean_mask -- excludes any leftover
    digit/border/edge pixels the row might still contain), the fraction
    that specifically match FILLED rather than EMPTY -- used directly as
    the tower's HP fraction estimate."""
    clean = _clean_mask(row_bgr, key)
    clean_pixels = row_bgr[clean]
    if len(clean_pixels) == 0:
        return 0.0
    diffs_filled = np.abs(clean_pixels.astype(np.float32) - FILLED_BGR[key]).sum(axis=1)
    return float((diffs_filled < CLEAN_MAX_DIST).mean())


def read_tower(screenshot: np.ndarray, tower_name: str):
    """Returns hp_fraction (0.0-1.0) for one tower this frame (princess OR
    king), or None if no clean bar-colored row could be found at all.
    `tower_name` must be a key in TOWER_BAR_REGIONS.

    Deliberately does NOT report "destroyed" here -- a single frame's
    "no clean row found" is genuinely ambiguous on its own (occlusion, a
    capture glitch, digit misalignment -- vs. a real destroyed tower,
    confirmed 2026-08-20 to show literally no UI element at all once
    destroyed, not a faint/near-zero bar). Telling those apart needs the
    SAME kind of debounce this project already uses everywhere else for a
    noisy per-frame signal (see TowerStateTracker below) -- a one-shot
    function can't do that on its own."""
    key = (_tower_type(tower_name), _tower_side(tower_name))
    x1, y1, x2, y2 = TOWER_BAR_REGIONS[tower_name]
    y1p, y2p = max(0, y1 - REGION_PAD_Y), y2 + REGION_PAD_Y
    region = screenshot[y1p:y2p, x1:x2]

    row_idx = _best_row(region, key)
    if row_idx is None:
        return None
    return _hp_fraction_from_row(region[row_idx], key)


DESTROYED_CONFIRM_FRAMES = 60  # a tower must show NO clean bar-color signal for this many
# CONSECUTIVE read_tower() calls before TowerStateTracker actually reports it destroyed --
# same "debounce a noisy per-frame signal before trusting it" pattern used throughout this
# project (unit_tracker.py's SIDE_LOCK_HITS/CLASS_LOCK_HITS, cycle_tracker.py's
# HAND_STABLE_FRAMES). Confirmed 2026-08-20 that a destroyed tower shows literally no UI
# at all (rubble only), which is why "no clean row" is treated as the real destroyed
# signal at all -- but a single miss is still ambiguous with ordinary transient noise.
# Once confirmed, destroyed is treated as PERMANENT (no debounce needed in the other
# direction) -- a real tower never comes back once it's actually destroyed, unlike side/
# class readings elsewhere in this project which genuinely can need later correction.
#
# Raised from an original 5 (2026-08-20, real live-match bug): both enemy princess towers
# were confirmed "destroyed" by frame 5 of an actual match -- well under a second in, long
# before any real damage could plausibly have been dealt -- almost certainly a startup/
# loading transient (the game/UI not fully settled in the very first captured frames), not
# a real destruction. Because destroyed is PERMANENT, a single bad multi-frame streak
# ANYWHERE in a match (a startup moment, a UI overlay, sustained occlusion) could
# irreversibly corrupt tracking for the rest of the match under the old, too-short window.
# The cost of being wrong is deeply asymmetric: a false positive is a permanent, silent
# corruption of every downstream consumer (zone classification, opponent-play attribution);
# a false negative just delays confirming a real destruction by a few extra seconds, which
# costs nothing since nothing here is racing against that. 60 frames is a real handful of
# seconds even at this project's slower loaded framerates (~7-12fps with unit detection on)
# -- long enough that only a genuinely sustained loss of the bar should satisfy it, short
# enough relative to a real multi-minute match that a genuine destruction is still caught
# promptly. Not empirically tuned against a large sample of real transients yet -- if a
# false positive is ever seen again, that's real data to retune this against, not a reason
# to guess again blindly.

HP_CONFIRM_FRAMES = 2  # mirrors SIDE_LOCK_HITS/CLASS_LOCK_HITS -- a NEW hp_fraction reading
# must be confirmed by this many consecutive raw reads landing within HP_AGREEMENT_TOLERANCE
# of each other before it actually updates the reported value. Added 2026-08-20 after a
# real live-match bug, found in the same log as the DESTROYED_CONFIRM_FRAMES fix above: a
# fully-healthy tower holding steady at hp_fraction=1.0 for many consecutive frames would
# occasionally read exactly 0.0 for a single frame, then immediately revert to 1.0 the very
# next frame -- almost certainly a transient VFX/animation artifact (a hit flash, most
# likely, since it recurred on enemy_king specifically while it was under sustained real
# attack), not an actual HP change. There was no debounce on the CONTINUOUS hp_fraction
# value at all before this (only on the discrete "destroyed" signal) -- a single noisy
# frame passed straight through to every consumer (the panel, the log, anything reading
# TowerStateTracker's state).
HP_AGREEMENT_TOLERANCE = 0.08  # how close two consecutive raw reads must be to count as
# "agreeing" -- generous enough to absorb real bar-measurement noise already observed in
# practice (e.g. 0.989 vs 0.99 vs 1.0 readings of the same true 100% state), tight enough
# to still catch a real implausible one-frame spike like the 1.0 -> 0.0 case above.

HP_ZERO_DROP_CONFIRM_FRAMES = 5  # a STRICTER debounce than HP_CONFIRM_FRAMES, specifically
# for a reading that drops ALL THE WAY to near-zero from a previously-confirmed, meaningfully
# healthier value. Added 2026-08-20 after a second real live-match pattern, distinct from the
# single-frame-flash bug HP_CONFIRM_FRAMES/HP_AGREEMENT_TOLERANCE already fix: a damaged
# princess tower would flash to 0.0 for ~2 CONSECUTIVE frames around EVERY new hit, before
# settling on the real new (lower but nonzero) value -- long enough to satisfy the normal
# 2-frame debounce and get "confirmed" just as easily as a real reading, then get corrected
# again a moment later once the real value also confirmed. Almost certainly a hit-flash/
# damage-number animation coinciding with the moment of impact (the tower's OTHER real
# intermediate values in the same match -- 0.734, 0.619, 0.543, etc. -- all read correctly,
# so this isn't a calibration problem). A real, single hit essentially never brings a tower
# from a healthy % straight to empty in one shot -- that only legitimately happens on the
# FINAL blow, which is exactly when the tower is about to be destroyed anyway (TowerState
# Tracker's own separate destroyed-debounce already handles confirming that), so a few extra
# frames of latency specifically on "sudden full drop" readings costs nothing real.
HP_ZERO_DROP_THRESHOLD = 0.05  # a candidate at or below this counts as "dropped to empty" for
# the stricter check above.
HP_ZERO_DROP_MIN_PRIOR = 0.15  # the stricter check only applies when dropping FROM a
# meaningfully healthy prior value -- a tower already near-empty legitimately CAN read another
# near-zero value on its next real hit, and doesn't need (or deserve) the extra scrutiny.


class TowerStateTracker:
    """Stateful wrapper around read_tower() for all 6 towers, debouncing
    the destroyed signal across frames -- see DESTROYED_CONFIRM_FRAMES's
    own comment for why a single frame's "no clean bar row" can't be
    trusted as destroyed outright.

    Real design gap caught and fixed before this shipped (2026-08-20,
    caught by Duncan mid-implementation): a king tower's bar doesn't
    render at all until EITHER it's taken direct damage OR a princess
    tower on that same side has been destroyed (his own correction to an
    earlier, incomplete version of this fact) -- so "no clean bar row" is
    NOT a reliable destroyed signal for a king tower the way it is for a
    princess tower, since a perfectly healthy, not-yet-activated king
    tower produces the EXACT SAME "no clean row" reading as a genuinely
    destroyed one. Fixed with an _ever_read gate, scoped to KING towers
    ONLY (a real regression caught while testing this exact fix: gating
    it universally broke princess-tower detection instead, since a
    princess tower that's ALREADY destroyed by the time tracking starts
    would then never pass _ever_read either and could never be confirmed
    destroyed at all -- a princess tower's bar is always visible from
    frame 1 if it's alive, so there's no equivalent "not yet activated"
    state to protect against the way there is for a king tower). This
    also means: don't expect an inactive king tower to report
    `destroyed: True` OR a meaningful hp_fraction here -- both stay
    None/False until its bar first appears for real.

    hp_fraction is ALSO debounced now (see HP_CONFIRM_FRAMES's own comment)
    -- a single noisy raw reading that disagrees with the previous
    candidate doesn't update the reported value until a second consecutive
    reading confirms it, so a one-frame flash (e.g. a hit-VFX moment)
    can't flicker through to whatever's consuming this state.

    Usage:
        tracker = TowerStateTracker()
        state = tracker.update(screenshot)
        # state: {tower_name: {"hp_fraction": float_or_None, "destroyed": bool}}
        # hp_fraction is the last CONFIRMED value (may be stale while a tower
        # is temporarily unreadable, or while a new candidate value hasn't
        # yet been confirmed -- see HP_CONFIRM_FRAMES).
    """

    def __init__(self):
        self._miss_streak = {name: 0 for name in TOWER_BAR_REGIONS}
        self._destroyed = {name: False for name in TOWER_BAR_REGIONS}
        self._last_hp_fraction = {name: None for name in TOWER_BAR_REGIONS}
        self._ever_read = {name: False for name in TOWER_BAR_REGIONS}
        self._hp_candidate = {name: None for name in TOWER_BAR_REGIONS}
        self._hp_candidate_hits = {name: 0 for name in TOWER_BAR_REGIONS}

    def update(self, screenshot: np.ndarray) -> dict:
        result = {}
        for name in TOWER_BAR_REGIONS:
            # Once destroyed, this region is NEVER read again (per Duncan's explicit
            # request) -- a destroyed tower's now-empty patch of ground is exactly where a
            # troop is likely to walk, and its sprite colors could accidentally satisfy
            # _clean_mask, producing a false "it's alive again with some fill %" reading.
            # destroyed is already permanent (see DESTROYED_CONFIRM_FRAMES's comment), so
            # skipping the read isn't just a performance win -- it's what makes the
            # permanence guarantee actually hold against this specific failure mode.
            if not self._destroyed[name]:
                hp_fraction = read_tower(screenshot, name)
                if hp_fraction is None:
                    # King towers only: a "never yet read" king tower means "not
                    # activated," not destroyed -- see this class's own docstring for why
                    # that gate must NOT apply to princess towers too. (Real bug caught
                    # here while testing: an earlier version used `continue`, which also
                    # skipped populating `result[name]` below for the untouched tower --
                    # fixed by just skipping the streak-increment, not the whole iteration.)
                    if not (_tower_type(name) == "king" and not self._ever_read[name]):
                        self._miss_streak[name] += 1
                        if self._miss_streak[name] >= DESTROYED_CONFIRM_FRAMES:
                            self._destroyed[name] = True
                else:
                    self._miss_streak[name] = 0
                    self._ever_read[name] = True
                    # Debounce the VALUE itself before trusting it (see HP_CONFIRM_FRAMES's
                    # own comment) -- a raw reading only updates _last_hp_fraction once enough
                    # consecutive readings agree with it; a lone noisy frame (e.g. a hit-flash
                    # VFX) resets the candidate but never gets far enough to overwrite the
                    # actually-reported value.
                    candidate = self._hp_candidate[name]
                    if candidate is not None and abs(hp_fraction - candidate) <= HP_AGREEMENT_TOLERANCE:
                        self._hp_candidate_hits[name] += 1
                    else:
                        self._hp_candidate[name] = hp_fraction
                        self._hp_candidate_hits[name] = 1
                    # A sudden drop to near-zero from a meaningfully healthier confirmed value
                    # needs the STRICTER HP_ZERO_DROP_CONFIRM_FRAMES window (see its own
                    # comment) -- an ordinary gradual change still only needs HP_CONFIRM_FRAMES.
                    prior = self._last_hp_fraction[name]
                    is_suspicious_drop = (
                        prior is not None and prior >= HP_ZERO_DROP_MIN_PRIOR
                        and hp_fraction <= HP_ZERO_DROP_THRESHOLD
                    )
                    required_hits = HP_ZERO_DROP_CONFIRM_FRAMES if is_suspicious_drop else HP_CONFIRM_FRAMES
                    if self._hp_candidate_hits[name] >= required_hits:
                        self._last_hp_fraction[name] = self._hp_candidate[name]
            result[name] = {
                "hp_fraction": 0.0 if self._destroyed[name] else self._last_hp_fraction[name],
                "destroyed": self._destroyed[name],
            }
        return result


def read_all_towers(screenshot: np.ndarray) -> dict:
    """One-shot, single-frame read of all 6 towers -- returns
    {tower_name: hp_fraction_or_None}. Convenience for quick/manual
    checks (e.g. this module's own __main__). For anything that needs a
    real destroyed/alive determination, use TowerStateTracker instead --
    a single frame genuinely can't tell "destroyed"/"not yet activated"
    apart from "temporarily unreadable" on its own (see read_tower's and
    TowerStateTracker's own docstrings)."""
    return {name: read_tower(screenshot, name) for name in TOWER_BAR_REGIONS}


if __name__ == "__main__":
    screenshot = cv2.imread("test_screenshot.png")
    if screenshot is None:
        raise FileNotFoundError("test_screenshot.png not found — capture one first.")

    print("=== One-shot read (read_all_towers) ===")
    hp_fractions = read_all_towers(screenshot)
    for name, hp in hp_fractions.items():
        print(f"  {name:12s} hp_fraction={f'{hp:.2f}' if hp is not None else 'UNREADABLE'}")

    print("\n=== TowerStateTracker, fed the same frame repeatedly to demonstrate the")
    print("    destroyed debounce (DESTROYED_CONFIRM_FRAMES consecutive misses) ===")
    tracker = TowerStateTracker()
    state = None
    for _ in range(DESTROYED_CONFIRM_FRAMES + 1):
        state = tracker.update(screenshot)
    display = screenshot.copy()
    for name, (x1, y1, x2, y2) in TOWER_BAR_REGIONS.items():
        r = state[name]
        hp = r["hp_fraction"]
        hp_str = f"{hp:.2f}" if hp is not None else "?"
        label = "DESTROYED" if r["destroyed"] else hp_str
        print(f"  {name:12s} hp_fraction={hp_str}  destroyed={r['destroyed']}")
        color = (0, 0, 255) if r["destroyed"] else (0, 255, 0)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display, f"{name} {label}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    os.makedirs("temp_img", exist_ok=True)
    out_path = os.path.join("temp_img", "debug_tower_reader.png")
    cv2.imwrite(out_path, display)
    print(f"\nSaved {out_path} — open it to confirm visually before trusting this.")
