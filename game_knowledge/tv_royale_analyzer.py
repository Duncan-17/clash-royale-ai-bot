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

PLAYS_LOG = "data/tv_royale_plays_log.jsonl"


class TvrAnalyzer:
    def __init__(self, clf, card_collector=None, detector=None, plays_log=PLAYS_LOG):
        self.clf = clf
        self.card_collector = card_collector   # UncertainCardCollector or None
        self.detector = detector               # YoloDetector(include_non_units=True) or None (--levels)
        self.plays_log = plays_log
        self.cycles = None
        self._logged = 0
        self.game_id = None

    def start_game(self, enemy_deck, friendly_deck, game_id=None):
        self.cycles = DualCycleTracker(enemy_deck, friendly_deck)
        self._logged = 0
        self.game_id = game_id

    def observe(self, frame, now=None):
        """One live frame. Returns the dual tracker's per-frame dict."""
        now = now or time.time()
        enemy, friendly = read_both_hands_classifier(frame, self.clf)
        out = self.cycles.update(enemy, friendly,
                                 R.read_enemy_elixir(frame), R.read_friendly_elixir(frame), now)
        # log any new plays
        while self._logged < len(self.cycles.plays):
            self._log_play(self.cycles.plays[self._logged])
            self._logged += 1
        if self.card_collector is not None:
            self._collect_uncertain_cards(frame, enemy, friendly)
        if self.detector is not None:
            self._collect_unknown_levels(frame)
        return out

    def _collect_uncertain_cards(self, frame, enemy, friendly):
        for side, reads, slots in (("enemy", enemy, R.enemy_hand_slots()),
                                   ("friendly", friendly, R.friendly_hand_slots())):
            for i, (box, rd) in enumerate(zip(slots, reads)):
                x1, y1, x2, y2 = box
                self.card_collector.observe((side, i), rd.card, rd.score, False, frame[y1:y2, x1:x2])

    def _collect_unknown_levels(self, frame):
        from perception.level_reader import queue_level_crop, read_level
        for d in self.detector.detect(frame):
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
