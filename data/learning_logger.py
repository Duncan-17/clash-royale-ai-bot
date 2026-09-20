"""Learning-data logger -- turns each watched TV Royale replay into the imitation
dataset described in clash-royale-learning-schema.md.

Per PLAY it writes one EGOCENTRIC state->action sample; per GAME a header (+ result);
and it ARCHIVES the raw frame at each play (and the end screen) so board/placement/
pixel features can be re-derived later without re-watching.

Three capture layers (Duncan 2026-09-12, "do it all"):
  #3 card data  -- the play (card/cost/timing/inferred/confirmed) + BOTH players'
     elixir and reconstructed hand/cycle. From the validated dual cycle tracker. SOLID.
  #2 YOLO board -- detector units (type/side/pos/level) at the play frame, egocentric-
     canonicalized, + a confidence-gated best-effort PLACEMENT (a newly-appeared self-
     side unit matching the played card). Relies on the detector transferring to the TV
     Royale board (still to validate live); placement stays null unless unambiguous, so
     it never logs a WRONG position -- and the archived frame lets a better localizer
     run offline.
  #1 raw frame  -- archived per play + at match end (crown-reading is a later add).

Egocentric + privileged: every sample is from the ACTING player's view (self_*/opp_*),
board mirrored so self always defends the bottom; opp hand/elixir are logged but tagged
privileged (reward/belief only, never a live-policy input).

Files (under out_dir, default data/learning/):
  games.jsonl              one header per replay (written at game end, with result)
  samples.jsonl            one state->action sample per play
  frames/<game>/NNN.jpg    raw frame at each play; end.jpg = the end screen
"""

import json
import os

import cv2

DEFAULT_OUT = "data/learning"
APPEAR_WINDOW = 2.5      # s: how far back a "newly-appeared" unit counts for placement
APPEAR_DIST = 0.06       # normalized dist below which two detections are the "same" unit
DOUBLE_ELIXIR_T = 120.0  # s into the match: rough single->double boundary (no game clock
# on TV Royale yet, so phase is derived from match-relative time -- approximate).


def _canon_units(units, actor_side, w, h):
    """Detector units -> egocentric list. Detector side: 'ally'=bottom player,
    'enemy'=top. Map to self/opp relative to the actor and mirror y so self is bottom."""
    out = []
    for u in units:
        det = u.get("side")
        if actor_side == "friendly":            # actor is the bottom player
            rel = {"ally": "self", "enemy": "opp"}.get(det, "unknown")
            flip = False
        else:                                   # actor is the top player -> mirror
            rel = {"enemy": "self", "ally": "opp"}.get(det, "unknown")
            flip = True
        x1, y1, x2, y2 = u["bbox"]
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        if flip:
            cy = 1.0 - cy
        # a struggle-case (name impossible for this match) is recorded as unknown, not
        # as a troop that isn't in the game.
        name = "unknown" if u.get("struggle") else u.get("name")
        out.append({
            "type": name, "side": rel,
            "x": round(cx, 3), "y": round(cy, 3),
            "level": u.get("level"),
            "conf": round(u["conf"], 2) if u.get("conf") is not None else None,
            "off_deck": bool(u.get("off_deck", False)),   # class in neither deck = suspect
        })
    return out


def _same_unit(a, b):
    return a["type"] == b["type"] and abs(a["x"] - b["x"]) < APPEAR_DIST and abs(a["y"] - b["y"]) < APPEAR_DIST


_RELIABLE = (None, "bar")   # side_source values that mean a real bar-colour read
# ("position"/"deck"/"board" are weaker inferences, not a bar).


class BoardTeams:
    """Learns each card's OWNING side from reliable board sightings (health-bar colour),
    so a card the hand classifier mis-learned (e.g. zappies read as mortar) is still
    attributable to the right player, and an undamaged troop placed in the WRONG half
    (e.g. after a tower falls) is flipped to its true owner. Also lists cards actually
    seen in the match, so impossible detector reads can be rejected."""

    def __init__(self, min_sightings=3):
        self._owner = {}           # card -> {"enemy": n, "ally": n}  (bar-colour sightings)
        self._played = {}          # card -> {"enemy": n, "ally": n}  (from PLAY events)
        self.min = min_sightings

    def observe(self, units):
        for u in units:
            s = u.get("side")
            if s in ("enemy", "ally") and u.get("side_source") in _RELIABLE:
                d = self._owner.setdefault(u.get("name"), {"enemy": 0, "ally": 0})
                d[s] += 1

    def record_play(self, card, play_side):
        """A PLAY is ground truth for which player owns that card (the card left THAT
        player's hand). One play is authoritative -- no health bar needed. A card BOTH
        players play (e.g. a shared dart_goblin) accrues both sides -> owner() abstains."""
        if not card or str(card).startswith("unknown_"):
            return
        det = "enemy" if play_side == "enemy" else "ally"
        self._played.setdefault(card, {"enemy": 0, "ally": 0})[det] += 1

    def owner(self, name):
        bar = self._owner.get(name, {"enemy": 0, "ally": 0})
        pl = self._played.get(name, {"enemy": 0, "ally": 0})
        e = bar["enemy"] >= self.min or pl["enemy"] >= 1
        a = bar["ally"] >= self.min or pl["ally"] >= 1
        return "enemy" if (e and not a) else ("ally" if (a and not e) else None)

    def seen(self, name):
        return name in self._owner or name in self._played


def resolve_sides(units, enemy_cards, friendly_cards, teams=None):
    """Enemy/friendly attribution for detector units, using every prior available:
      1. a real bar-colour side (damaged troop) is trusted -- always right;
      2. else the card's OWNER learned from reliable board sightings (teams) flips it
         to its true player regardless of position (fixes the zappies-in-wrong-half case);
      3. else the deck prior -- a card in exactly one player's deck is theirs;
      4. else the detector's position guess is left as-is.
    Also flags impossible reads: a name in NEITHER deck AND never reliably seen on the
    board is a struggle-case (u['struggle']=True) -- the logger records it as 'unknown'
    rather than propagating a troop that isn't in the game. Mutates + returns units."""
    for u in units:
        name = u.get("name")
        in_e = any(_card_matches_unit(c, name) for c in enemy_cards)
        in_f = any(_card_matches_unit(c, name) for c in friendly_cards)
        own = teams.owner(name) if teams else None
        seen = teams.seen(name) if teams else False
        if (enemy_cards or friendly_cards) and not (in_e or in_f or seen):
            u["off_deck"] = u["struggle"] = True     # impossible: not in the game
        else:
            u["off_deck"] = not (in_e or in_f) and not seen
        if u.get("side") in ("enemy", "ally") and u.get("side_source") in _RELIABLE:
            continue                                  # bar colour -> trust it
        if own == "enemy" or (in_e and not in_f):
            u["side"], u["side_source"] = "enemy", "board" if own else "deck"
        elif own == "ally" or (in_f and not in_e):
            u["side"], u["side_source"] = "ally", "board" if own else "deck"
    return units


class LearningLogger:
    def __init__(self, out_dir=DEFAULT_OUT):
        self.out_dir = out_dir
        os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)
        self.samples_path = os.path.join(out_dir, "samples.jsonl")
        self.games_path = os.path.join(out_dir, "games.jsonl")
        self._reset_game()

    def _reset_game(self):
        self.game_id = None
        self.decks = None          # {"enemy":[...], "friendly":[...]} (top/bottom)
        self.arena = None
        self.t0 = None
        self.seq = 0
        self.samples = 0
        self._appeared = []        # recent unit appearances: {det_side, type, x, y, t}
        self._prev_canon = None    # previous frame's units in DETECTOR frame (for diff)

    # -- game lifecycle -----------------------------------------------------
    def start_game(self, game_id, decks, arena=None, source="tv_royale"):
        self._reset_game()
        self.game_id = game_id
        self.decks = decks
        self.arena = arena
        self.source = source
        os.makedirs(os.path.join(self.out_dir, "frames", str(game_id)), exist_ok=True)

    def end_game(self, result=None, end_frame=None):
        """Write the game header (with result, if read) and archive the end screen."""
        if self.game_id is None:
            return
        if end_frame is not None:
            cv2.imwrite(os.path.join(self.out_dir, "frames", str(self.game_id), "end.jpg"), end_frame)
        rec = {"game_id": self.game_id, "source": self.source, "arena": self.arena,
               "decks": self.decks, "result": result, "samples": self.samples}
        with open(self.games_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # -- per-frame ----------------------------------------------------------
    def observe_units(self, units, now, w, h):
        """Track unit APPEARANCES (for placement). Called every frame with the raw
        detector units (detector-frame coords). Kept in detector side/coords; canon
        happens at log time (per actor)."""
        cur = []
        for u in units:
            det = u.get("side")
            if det not in ("ally", "enemy"):
                continue
            x1, y1, x2, y2 = u["bbox"]
            cur.append({"det_side": det, "type": u.get("name"),
                        "x": ((x1 + x2) / 2) / w, "y": ((y1 + y2) / 2) / h})
        prev = self._prev_canon or []
        for c in cur:                            # a unit with no match last frame = new
            if not any(_same_unit(c, p) for p in prev):
                self._appeared.append(dict(c, t=now))
        self._appeared = [a for a in self._appeared if now - a["t"] <= APPEAR_WINDOW]
        self._prev_canon = cur

    def _placement(self, play, actor_side, w, h):
        """Best-effort, confidence-gated: a newly-appeared unit on the ACTOR's side
        matching the played card. Returns {x,y} in canonical (self-bottom) coords, or
        None when unclear (never a guess). Spells/unknown cards -> None."""
        card = play.get("card") or ""
        if card.startswith("unknown_"):
            return None
        det_side = "ally" if actor_side == "friendly" else "enemy"
        matches = [a for a in self._appeared
                   if a["det_side"] == det_side and _card_matches_unit(card, a["type"])]
        if len(matches) != 1:                    # 0 or ambiguous -> abstain
            return None
        a = matches[0]
        cy = a["y"] if actor_side == "friendly" else 1.0 - a["y"]
        return {"x": round(a["x"], 3), "y": round(cy, 3)}

    def log_play(self, play, ev, elix, units, frame, now, placement_override=None):
        """Write one egocentric sample for `play`. ev = dual tracker per-frame dict
        {'enemy':..,'friendly':..}; elix = {'enemy':float,'friendly':float}; units =
        board units (or None); frame = the raw BGR frame. placement_override, when
        given, is the EXACT canonical (self-bottom) placement from the play<->unit
        binder (game_knowledge/play_localizer) -- it replaces the appearance heuristic
        below for this sample (used for the acting player, i.e. enemy plays)."""
        if self.game_id is None:
            return None
        if self.t0 is None:
            self.t0 = play["t"]
        t = round(play["t"] - self.t0, 2)
        actor = play["side"]                     # "enemy"(top) or "friendly"(bottom)
        opp = "friendly" if actor == "enemy" else "enemy"
        h, w = frame.shape[:2]

        def state(side):
            e = ev.get(side) or {}
            return {"elixir": round(elix.get(side, 0.0), 2),
                    "leaked": elix.get(side, 0.0) >= 9.95,
                    "hand": list((e.get("hand") or {}).values()),
                    "on_deck": e.get("on_deck"),
                    "rest_of_queue": e.get("rest_of_queue"),
                    "fully_resolved": e.get("queue_fully_resolved")}

        board = {"units": _canon_units(units, actor, w, h) if units else [],
                 "detector_ok": units is not None}
        sample = {
            "game_id": self.game_id, "seq": self.seq, "t": t,
            "actor_side": actor, "mirrored": actor == "enemy", "source": self.source,
            "type": "play",
            "action": {
                "card": play.get("card"), "cost": play.get("cost"),
                "phase": "double" if t >= DOUBLE_ELIXIR_T else "single",
                # Placement perception is only needed for the OPPONENT -- the bot knows
                # its own placement (self-play/live) -- so we only localize ENEMY plays;
                # friendly-side placement comes free from the action, not perception.
                # A binder-supplied placement (exact, from the play<->unit link) wins over
                # the appearance heuristic when present.
                "position": (placement_override if placement_override is not None
                             else (self._placement(play, actor, w, h) if actor == "enemy" else None)),
                "inferred": play.get("inferred", False),
                "named": play.get("named", not play.get("inferred", False)),
                "confirmed": play.get("confirmed", False),
                "revealed": play.get("revealed"),
            },
            "self": state(actor),
            "opp": state(opp),                   # PRIVILEGED (reward/belief only)
            "board": board,
            "reward_raw": {"elixir_spent": play.get("cost")},
        }
        with open(self.samples_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample) + "\n")
        if frame is not None:
            cv2.imwrite(os.path.join(self.out_dir, "frames", str(self.game_id),
                                     f"{self.seq:03d}.jpg"), frame)
        self.seq += 1
        self.samples += 1
        return sample


def _card_matches_unit(card, unit_type):
    """Loose match between a hand-card slug and a detector unit class (the two
    taxonomies differ slightly, e.g. archers/archer, goblins/goblin)."""
    if not card or not unit_type:
        return False
    a, b = card.replace("_", ""), unit_type.replace("_", "")
    return a == b or a.rstrip("s") == b.rstrip("s") or a in b or b in a
