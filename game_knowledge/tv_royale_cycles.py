"""Dual cycle tracking for TV Royale -- the authoritative play/cycle data engine.

TV Royale shows BOTH players' hands + elixir, so game_knowledge/cycle_tracker.py
(proven reconstructing our own hidden 4-card queue) can be run as TWO instances to
get authoritative "player X played card Y at time T" for the whole match, with no
board-vision inference. That is the clean learning data Duncan wants from TV Royale.

Fed by the hand classifier's reads (perception.tv_royale_hand_reader.
read_both_hands_classifier -> SlotRead per slot, [0]/[1] = card/score, drop-in for
CycleTracker's slot_results) and both elixir bars (config.tv_royale_regions.
read_enemy_elixir / read_friendly_elixir, a float -> the {value, fraction} shape
CycleTracker expects). Each play is cross-checked against its player's elixir drop
inside CycleTracker, exactly as for our own side.

Decks: each CycleTracker needs that player's 8-card deck. In the watcher these come
free from the list auto-read (perception.tv_royale_list reads both decks per game);
for a manual run, config.tv_royale_decks.
"""

from game_knowledge.cycle_tracker import CycleTracker

# The classifier's softmax confidence for a clear read is ~0.9-1.0 (higher/cleaner
# than the template scores CycleTracker's default threshold was tuned for), so a
# read only feeds the cycle when it clears this. Below it the slot is treated as
# unknown that frame (CycleTracker ignores it), which is the safe default.
CLASSIFIER_CONF_THRESHOLD = 0.7


def _elixir_dict(value: float) -> dict:
    """TV Royale's elixir read (a float like 6.55) -> the {value, fraction} shape
    CycleTracker.update expects (it only reads those two keys)."""
    v = max(0.0, float(value))
    return {"value": int(v), "fraction": v - int(v)}


class DualCycleTracker:
    def __init__(self, enemy_deck, friendly_deck,
                 confidence_threshold: float = CLASSIFIER_CONF_THRESHOLD):
        """enemy_deck = top player's 8 card slugs, friendly_deck = bottom player's."""
        # learn_deck=True: each player's deck is also LEARNED from the hands, so a
        # pre-entered list isn't required and a mis-read menu deck gets cross-checked
        # (the allowed set is the union of the given deck + the observed one). See
        # CycleTracker.__init__.
        self.enemy = CycleTracker(enemy_deck, confidence_threshold, learn_deck=True)
        self.friendly = CycleTracker(friendly_deck, confidence_threshold, learn_deck=True)
        self.plays = []   # every play, both players: {side, card, cost, t, ...}

    def update(self, enemy_reads, friendly_reads, elix_enemy, elix_friendly, now):
        """One frame. *_reads = a player's 4 SlotRead; elix_* = that player's elixir
        float; now = a monotonic timestamp. Returns {"enemy": ev, "friendly": ev}
        (each CycleTracker's per-frame dict) and appends any new play(s) to self.plays.

        Pulls new plays straight from each CycleTracker.plays (not ev["last_play"]),
        so if a player has two plays settle on the same frame both are captured."""
        ne0, nf0 = len(self.enemy.plays), len(self.friendly.plays)
        ev_e = self.enemy.update(enemy_reads, _elixir_dict(elix_enemy), now)
        ev_f = self.friendly.update(friendly_reads, _elixir_dict(elix_friendly), now)
        for side, ct, n0 in (("enemy", self.enemy, ne0), ("friendly", self.friendly, nf0)):
            for p in ct.plays[n0:]:
                play = dict(p)
                play["side"] = side
                play["t"] = now
                self.plays.append(play)
        return {"enemy": ev_e, "friendly": ev_f}
