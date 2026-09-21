"""Per-frame TV Royale analysis + self-collection, for the automated watcher.

Wires the hand classifier + dual cycle tracker + the self-stocking queues together
so watching a replay produces:
  - authoritative PLAY events for both players (card + time + side), logged to
    data/tv_royale_plays_log.jsonl -- the learning data,
  - a review queue of hand CARDS the classifier is unsure about
    (perception.tv_royale_collector.UncertainCardCollector), and
  - (optional, --levels) a review queue of troop LEVEL badges the level reader
    can't read yet (perception.level_reader), read off the board via the detector.

start_game(enemy_deck, friendly_deck) once per replay (decks come from the list
auto-read); observe(frame) each frame while the match is live.

TWO ASSUMPTIONS TO VALIDATE LIVE (flagged for the handoff):
  1. DECK->SIDE mapping: the list's left/right decks are mapped left=top=enemy,
     right=bottom=friendly. If cycles look swapped, swap them at the call site.
  2. DETECTOR ON TV ROYALE: the v4 detector's arena geometry was calibrated on the
     live-match view; TV Royale spectator is the same board (same 900x1600), so it
     should transfer, but the --levels path needs a live check before trusting it.
"""

import json
import os
import time

from config import tv_royale_regions as R
from game_knowledge.tv_royale_cycles import DualCycleTracker
from perception.tv_royale_hand_reader import read_both_hands_classifier
from perception.yolo_detector import NON_UNIT_CLASSES

PLAYS_LOG = "data/tv_royale_plays_log.jsonl"
RESTRICT_MIN = 8   # soft-restrict a hand's reads to its deck once this many cards are
# known/learned for that player (i.e. the full 8-card deck) -- see #2 cross-education.


def _deck_prior(ct):
    """A player's known deck as a soft read-restriction prior, or None until the full
    8 are known (before that the classifier's own argmax is trusted)."""
    known = ct.known_cards()
    return known if len(known) >= RESTRICT_MIN else None


class TvrAnalyzer:
    def __init__(self, clf, card_collector=None, detector=None, plays_log=PLAYS_LOG,
                 learn_logger=None, board_collector=None):
        self.clf = clf
        self.card_collector = card_collector   # UncertainCardCollector or None
        self.detector = detector               # YoloDetector(include_non_units=True) or None
        self.plays_log = plays_log
        self.learn_logger = learn_logger       # data.learning_logger.LearningLogger or None
        self.board_collector = board_collector # perception.board_collector.BoardCropCollector or None
        self.cycles = None
        self._logged = 0
        self.game_id = None

    def start_game(self, enemy_deck, friendly_deck, game_id=None):
        from data.learning_logger import BoardTeams
        from game_knowledge.play_localizer import PlayLocalizer
        self.cycles = DualCycleTracker(enemy_deck, friendly_deck)
        self._teams = BoardTeams()          # learns card->owner from board bar-colour
        self._localizer = PlayLocalizer()   # BUILD A: play<->unit binding (side/class/placement)
        self._logged = 0
        self.game_id = game_id
        if self.board_collector is not None:
            self.board_collector.set_decks(enemy_deck, friendly_deck)   # candidates per side
        if self.learn_logger is not None:
            self.learn_logger.start_game(
                game_id, {"enemy": list(enemy_deck), "friendly": list(friendly_deck)})

    def end_game(self, result=None, end_frame=None):
        """Called when a replay ends: finalize the learning game header + end screen."""
        if self.learn_logger is not None:
            self.learn_logger.end_game(result=result, end_frame=end_frame)

    def observe(self, frame, now=None):
        """One live frame. Returns the dual tracker's per-frame dict."""
        now = now or time.time()
        # soft-restrict each hand's reads to its known deck once fully learned (#2)
        enemy, friendly = read_both_hands_classifier(
            frame, self.clf, _deck_prior(self.cycles.enemy), _deck_prior(self.cycles.friendly))
        elix_e, elix_f = R.read_enemy_elixir(frame), R.read_friendly_elixir(frame)
        out = self.cycles.update(enemy, friendly, elix_e, elix_f, now)

        dets = self.detector.detect(frame) if self.detector is not None else None
        # board = troops/buildings only. Filter on the detector's authoritative non-unit
        # set (drops towers/bars/clock/emote/text/elixir/selected) -- NOT a "'tower' in
        # name" test, which would also drop the real bomb_tower / inferno_tower buildings.
        board_units = ([d for d in dets if d.get("raw_name") not in NON_UNIT_CLASSES]
                       if dets else None)
        resolved = None
        if board_units is not None:                         # enemy/friendly attribution (#3)
            from data.learning_logger import resolve_sides
            self._teams.observe(board_units)                # learn card->owner from bar colour
            resolve_sides(board_units, self.cycles.enemy.known_cards(),
                          self.cycles.friendly.known_cards(), teams=self._teams)
            h, w = frame.shape[:2]
            self._localizer.update(board_units, now, w, h)  # BUILD A: stable ids + play stamps
            resolved = self._localizer.resolved_units()
        if self.learn_logger is not None and resolved is not None:
            h, w = frame.shape[:2]
            self.learn_logger.observe_units(resolved, now, w, h)

        # log any new plays (to the plain plays log, and the learning dataset)
        while self._logged < len(self.cycles.plays):
            play = self.cycles.plays[self._logged]
            self._teams.record_play(play["card"], play["side"])   # play = ground-truth owner
            self._log_play(play)
            if self.learn_logger is not None:
                # BUILD A: bind the play to the board unit that just appeared -> stamp its
                # side/class + exact placement; resolve AFTER binding so this sample's board
                # shows the stamp. Binder abstains (placement None) when unclear.
                binding = self._localizer.bind(play, now) if board_units is not None else None
                units_for_sample = self._localizer.resolved_units() if board_units is not None else None
                placement = binding["placement"] if binding else None
                self.learn_logger.log_play(
                    play, out, {"enemy": elix_e, "friendly": elix_f}, units_for_sample,
                    frame, now, placement_override=placement)
            self._logged += 1

        # BUILD B: bank bound/struggle board crops for the detector-retrain review queue
        # (RULE #2: queues only). Uses the post-bind resolved board.
        if self.board_collector is not None and board_units is not None:
            self.board_collector.observe(frame, self._localizer.resolved_units(),
                                         game_id=self.game_id or "live")

        if self.card_collector is not None:
            self._collect_uncertain_cards(frame, enemy, friendly)
        if dets is not None:
            self._collect_unknown_levels(frame, dets)
        return out

    def _collect_uncertain_cards(self, frame, enemy, friendly):
        # off-deck signal: a confident read of a card in NEITHER of this player's known
        # (given + learned) deck cards is likely a card the classifier doesn't know -- bank
        # it for review. Only trusted once the full 8-card deck is known, so we don't flag
        # everything before the deck is learned.
        e_known, f_known = self.cycles.enemy.known_cards(), self.cycles.friendly.known_cards()
        for side, reads, slots, known in (("enemy", enemy, R.enemy_hand_slots(), e_known),
                                          ("friendly", friendly, R.friendly_hand_slots(), f_known)):
            off_ok = len(known) >= RESTRICT_MIN
            for i, (box, rd) in enumerate(zip(slots, reads)):
                x1, y1, x2, y2 = box
                off = off_ok and rd.card not in (None, "UNKNOWN") and rd.card not in known
                self.card_collector.observe((side, i), rd.card, rd.score, False,
                                            frame[y1:y2, x1:x2], off_deck=off)

    def _collect_unknown_levels(self, frame, dets):
        from perception.level_reader import queue_level_crop, read_level
        for d in dets:
            if d.get("raw_name") == "bar-level":
                value, _conf = read_level(frame, d["bbox"])
                if value is None:                 # a level digit we can't read yet -> queue it
                    queue_level_crop(frame, d["bbox"])

    def _log_play(self, play):
        os.makedirs(os.path.dirname(self.plays_log), exist_ok=True)
        rec = {"game": self.game_id, "side": play["side"], "card": play["card"],
               "cost": play.get("cost"), "t": round(play["t"], 2)}
        with open(self.plays_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    @property
    def play_count(self):
        return len(self.cycles.plays) if self.cycles else 0
