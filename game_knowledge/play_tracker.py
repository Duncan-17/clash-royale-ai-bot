"""Turns the bot's OWN actions plus the detector's tracks into PLAY events and
authoritative friend/foe knowledge.

The key asymmetry (Duncan, 2026-09-10): the bot KNOWS every card it plays -- the
card, that it is ALLY, where it placed it, and when. So an own play is GROUND
TRUTH, stronger than any detected signal, and "everything I didn't play (and that
isn't spawned by an existing unit) is the ENEMY's." This module is the framework
for that:

  - record_own_play(card, position) -- the bot's action/decision layer calls this
    the instant it taps a card. (No caller yet; the bot doesn't play until Phase 5.
    Until then only opponent-play detection is exercised, which is what you can
    watch live during a manual game.)
  - update(tracks) -- each frame, over thin_tracker's stable tracks:
      * a track already bound to an own play stays authoritatively ALLY (+ its
        real card). This makes the bot's own units rock-solid ours -- they can
        never be misread as enemy (the class of bug that flipped a crossing
        Giant).
      * a NEW ally/unknown track that matches a recent own play (near its placement
        position, within a short window) BINDS to it -- confirmed card + ally.
      * a NEW enemy track that isn't a byproduct of an existing unit is an
        OPPONENT play; its detector UNIT name is mapped to a CARD name.

Deliberately THIN. The old game_knowledge/opponent_play_tracker.py (retired
2026-09-10) grew a stack of heuristics -- persistence gates, burst counts,
ally-vanish/duplicate/collision guards -- to survive a weak detector and a heavy
tracker. This leans on the strong v4 detector + thin_tracker instead and keeps
only what own-play ground truth and a light byproduct guard need. Refinements
(elixir accounting, count-aware card disambiguation) are layered ON later, not
baked in here.
"""

import time
from collections import deque, namedtuple

# --- unit(detector class) -> card(deck/hand slug) -----------------------------
# The detector emits UNIT names (a single "goblin"); a PLAY is a CARD ("goblins").
# Only the MISMATCHES are listed -- the many 1:1 cases (knight, giant, wizard,
# musketeer, mini_pekka, ...) fall through unit_to_card() as identity. Some units
# come from MULTIPLE cards (a "goblin" can be Goblins / Goblin Gang / Goblin Barrel
# / Goblin Hut / Goblin Drill / Goblin Giant); those map to the most common single
# source here and are disambiguated later with count/elixir/spawner context. Card
# slugs match assets/hand_templates/ (see perception/card_detector.known_card_slugs).
UNIT_TO_CARD = {
    "goblin": "goblins",
    "spear_goblin": "spear_goblins",
    "archer": "archers",
    "minion": "minions",
    "skeleton": "skeletons",
    "bat": "bats",
    "barbarian": "barbarians",
    "elite_barbarian": "elite_barbarians",
    "royal_recruit": "royal_recruits",
    "royal_hog": "royal_hogs",
    "guard": "guards",
    "wall_breaker": "wall_breakers",
    "skeleton_dragon": "skeleton_dragons",
    "rascal_boy": "rascals",
    "rascal_girl": "rascals",
    "zappy": "zappies",
}


def unit_to_card(unit_name: str) -> str:
    """Map a detector unit class to the card that plays it (identity if unlisted)."""
    return UNIT_TO_CARD.get(unit_name, unit_name)


# --- byproduct guard: units that a spawner emits, not a direct play -----------
# A witch's skeletons / a hut's goblins should NOT each read as an opponent "play".
# Kept intentionally small and explicit (not the old heavy spawn table): a new
# SPAWNABLE unit that appears next to an existing same-side SPAWNER is treated as
# a byproduct. Extend as needed.
SPAWNER_UNITS = {
    "witch", "night_witch", "goblin_hut", "barbarian_hut", "tombstone",
    "furnace", "goblin_drill",
}
SPAWNABLE_UNITS = {
    "skeleton", "goblin", "spear_goblin", "bat", "barbarian", "fire_spirit",
}

PendingOwnPlay = namedtuple("PendingOwnPlay", "card position t")
PlayEvent = namedtuple("PlayEvent", "kind card track_id t")  # kind: "own" | "opponent"

OWN_MATCH_WINDOW_S = 3.0   # a played card's unit should appear within this (deploy delay + travel)
OWN_MATCH_RADIUS_PX = 120  # ...and near where it was placed
SPAWN_RADIUS_PX = 140      # a byproduct spawns at/near its spawner


def _center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class PlayTracker:
    def __init__(self, own_window=OWN_MATCH_WINDOW_S, own_radius=OWN_MATCH_RADIUS_PX,
                 spawn_radius=SPAWN_RADIUS_PX, log_len=40):
        self.own_window = own_window
        self.own_radius = own_radius
        self.spawn_radius = spawn_radius
        self._pending_own = deque()        # PendingOwnPlay awaiting a matching track
        self.own_plays = deque(maxlen=log_len)       # PlayEvent, bound to a track
        self.opponent_plays = deque(maxlen=log_len)  # PlayEvent
        self._bound = {}                   # track_id -> our played card (authoritative ally)
        self._seen = set()                 # track ids we've processed at least once
        self._opp_logged = set()           # track ids already logged as an opponent play

    # -- called by the bot's action layer the moment it plays a card (Phase 5) --
    def record_own_play(self, card, position, now=None):
        """position = (x, y) in frame pixels where the card was placed."""
        self._pending_own.append(PendingOwnPlay(card, tuple(position), now or time.time()))

    def _expire(self, now):
        while self._pending_own and now - self._pending_own[0].t > self.own_window:
            self._pending_own.popleft()

    def _match_own(self, track, now):
        """Nearest pending own play within window+radius of this track; consumed."""
        tc = _center(track["bbox"])
        best_i, best_d = None, self.own_radius
        for i, p in enumerate(self._pending_own):
            if now - p.t > self.own_window:
                continue
            d = _dist(p.position, tc)
            if d <= best_d:
                best_i, best_d = i, d
        if best_i is None:
            return None
        p = self._pending_own[best_i]
        del self._pending_own[best_i]
        return p

    def _is_byproduct(self, track, tracks):
        if track["name"] not in SPAWNABLE_UNITS:
            return False
        tc = _center(track["bbox"])
        for o in tracks:
            if o["id"] == track["id"] or o.get("side") != "enemy":
                continue
            if o["name"] in SPAWNER_UNITS and _dist(_center(o["bbox"]), tc) <= self.spawn_radius:
                return True
        return False

    def update(self, tracks, now=None):
        """Process one frame of thin_tracker tracks. MUTATES bound tracks in place
        (side='ally', side_source='own_play', card=<played>) so the display shows
        the bot's own units authoritatively, and returns the PlayEvents fired this
        frame (also appended to self.own_plays / self.opponent_plays)."""
        now = now or time.time()
        self._expire(now)
        events = []
        for t in tracks:
            tid = t["id"]

            # 1) already ours (bound to an own play): authoritative ally, forever.
            if tid in self._bound:
                t["side"], t["side_source"], t["card"] = "ally", "own_play", self._bound[tid]
                continue

            # 2) NEW track: try to bind it to one of the bot's own plays.
            if tid not in self._seen:
                self._seen.add(tid)
                if t.get("side") in ("ally", "unknown"):
                    p = self._match_own(t, now)
                    if p is not None:
                        self._bound[tid] = p.card
                        t["side"], t["side_source"], t["card"] = "ally", "own_play", p.card
                        ev = PlayEvent("own", p.card, tid, now)
                        self.own_plays.append(ev)
                        events.append(ev)
                        continue

            # 3) enemy track we haven't logged -> an opponent play (once per track),
            #    unless it's a byproduct of an existing spawner.
            if t.get("side") == "enemy" and tid not in self._opp_logged:
                if not self._is_byproduct(t, tracks):
                    card = unit_to_card(t["name"])
                    self._opp_logged.add(tid)
                    ev = PlayEvent("opponent", card, tid, now)
                    self.opponent_plays.append(ev)
                    events.append(ev)
        return events
