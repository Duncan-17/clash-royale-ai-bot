"""Live visual debug view for TV Royale hand reading — the two-player analogue
of perception/debug_live.py, focused on what the auto-labeling pipeline reads
off the hands: each player's 4 slots, the matched card + score, the elixir, and
crucially the SELECTED (raised) card, which is the "poised to play" signal.

Reuses the same rounded object-ID box + label-plate design as debug_live (via
perception/overlay.py). Boxes are colored by what they mean:
  GOLD   = the slot is SELECTED (card lifted toward the arena, about to play).
           Its true (shifted) position is boxed too, so you can see the raise
           being handled.
  GREEN  = a confident card match (>= --min-confidence).
  ORANGE = a weak match (below it) — expected for catalog seeds until the
           16-card deck restriction + real examples firm things up.
Fill config/tv_royale_decks.py to restrict matching to the two real decks.

Usage:
    python -m perception.tv_royale_debug              # LIVE, template matching
    python -m perception.tv_royale_debug --classifier # LIVE, YOLOv8-cls hand classifier
    python -m perception.tv_royale_debug --static     # one frame from test_screenshot.png
    python -m perception.tv_royale_debug --classifier --collect   # classifier reads + bank crops
    python -m perception.tv_royale_debug --cycles     # LIVE + dual cycle tracker: validate PLAYS
                                                      #   play-by-play (fill config/tv_royale_decks.py
                                                      #   first; press r to reset for a new replay)
"""

import argparse
import importlib
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np

from config import tv_royale_regions as R
from config import tv_royale_decks as TVD
from config.tv_royale_decks import restricted_templates, combined_cards
from game_knowledge.tv_royale_cycles import DualCycleTracker, CLASSIFIER_CONF_THRESHOLD
from perception.card_detector import load_templates
from perception.overlay import draw_box, draw_label, _alpha_rect
from perception.tv_royale_hand_reader import read_hand, read_both_hands_classifier

WINDOW = "TV Royale hand debug (q=quit)"

SELECT = (250, 206, 135)  # selected / poised to play (light blue, BGR)
GREEN = (90, 220, 110)    # confident match
ORANGE = (40, 150, 255)   # weak match
DIM = (170, 150, 90)      # next-card / passive
MAGENTA = (200, 70, 200)  # elixir bar


def _slot_color(read, min_conf):
    if read.is_selected:
        return SELECT
    return GREEN if read.score >= min_conf else ORANGE


def draw_player(display, is_enemy, reads, elixir, min_conf):
    slots = R.enemy_hand_slots() if is_enemy else R.friendly_hand_slots()
    shift = R.selected_shift(is_enemy)
    for i, (box, read) in enumerate(zip(slots, reads), 1):
        color = _slot_color(read, min_conf)
        tag = ("SEL " if read.is_selected else "") + f"{read.card} {read.score:.2f}"
        draw_box(display, box, color, label=tag)
        # For a selected card, also outline where the art actually IS (shifted),
        # so the raise-handling is visible.
        if read.is_selected:
            sx1, sy1, sx2, sy2 = box[0], box[1] + shift, box[2], box[3] + shift
            draw_box(display, (sx1, sy1, sx2, sy2), SELECT, radius=6, thickness=1)

    # Next-card box (passive)
    nxt = R.ENEMY_NEXT if is_enemy else R.FRIENDLY_NEXT
    draw_box(display, nxt, DIM, label="next", radius=6, thickness=1)

    # Elixir bar (value goes in the header to avoid crowding the top edge).
    bar = R.ENEMY_ELIXIR_BAR if is_enemy else R.FRIENDLY_ELIXIR_BAR
    draw_box(display, bar, MAGENTA, radius=4, thickness=2)


# A confident resting match skips the shifted (selection) match -- see
# tv_royale_hand_reader._read_slot. Above this the card can't be selected.
LAZY_RESTING_CONF = 0.6


def cap_templates(templates, n):
    """Subsample each card's examples to at most `n`, evenly spread across
    what's on disk, for MATCH speed. Matching against all ~40 examples/card is
    the live view's bottleneck (~200ms/frame -> 5fps); a spread of ~10 keeps
    accuracy while getting to a smooth framerate (debug_live flagged exactly
    this lever). Collection/accuracy of the stored library is unaffected --
    this only thins what's compared at runtime."""
    out = {}
    for name, exs in templates.items():
        if n and len(exs) > n:
            step = len(exs) / n
            out[name] = [exs[int(i * step)] for i in range(n)]
        else:
            out[name] = exs
    return out


def read_frame(frame, templates, clf=None):
    """All per-frame reads in one place, so the live loop can both draw them and
    feed them to the collector without re-matching. clf!=None -> read the hands
    with the YOLOv8-cls classifier instead of template matching."""
    if clf is not None:
        enemy, friendly = read_both_hands_classifier(frame, clf)
    else:
        enemy = read_hand(frame, templates, is_enemy=True, lazy_conf=LAZY_RESTING_CONF)
        friendly = read_hand(frame, templates, is_enemy=False, lazy_conf=LAZY_RESTING_CONF)
    return {
        "enemy": enemy,
        "friendly": friendly,
        "elix_e": R.read_enemy_elixir(frame),
        "elix_f": R.read_friendly_elixir(frame),
    }


def draw(frame, data, min_conf, note="", backend="templates"):
    display = frame.copy()
    draw_player(display, True, data["enemy"], data["elix_e"], min_conf)
    draw_player(display, False, data["friendly"], data["elix_f"], min_conf)

    # Header: mode + per-player elixir + which slot is selected (+ optional note).
    lib = "decks restricted" if combined_cards() else "FULL library (fill tv_royale_decks.py)"
    sel_e = [i + 1 for i, r in enumerate(data["enemy"]) if r.is_selected] or "-"
    sel_f = [i + 1 for i, r in enumerate(data["friendly"]) if r.is_selected] or "-"
    header = (f"enemy {data['elix_e']:.1f}e sel:{sel_e}   |   "
              f"friendly {data['elix_f']:.1f}e sel:{sel_f}   |   [{backend}|{lib}]{note}")
    _alpha_rect(display, (0, 0), (display.shape[1], 34), (20, 20, 20), 0.6)
    cv2.putText(display, header, (10, 23), cv2.FONT_HERSHEY_DUPLEX, 0.55,
                (240, 240, 240), 1, cv2.LINE_AA)
    return display


def render(frame, templates, min_conf, note="", clf=None, backend="templates"):
    """Convenience: read + draw in one call (used by --static)."""
    return draw(frame, read_frame(frame, templates, clf), min_conf, note, backend=backend)


def _collect_frame(collector, frame, data):
    """Feed every resting slot crop to the collector."""
    for key, reads, slots in (
        ("enemy", data["enemy"], R.enemy_hand_slots()),
        ("friendly", data["friendly"], R.friendly_hand_slots()),
    ):
        for i, (box, rd) in enumerate(zip(slots, reads)):
            x1, y1, x2, y2 = box
            collector.observe((key, i), rd.card, rd.score, rd.is_selected,
                              frame[y1:y2, x1:x2])


def _fit(display, max_h=1000):
    h = display.shape[0]
    if h <= max_h:
        return display
    s = max_h / h
    return cv2.resize(display, (int(display.shape[1] * s), max_h), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# --cycles: live PLAY-BY-PLAY validation of the dual cycle tracker.
#
# Runs the SAME DualCycleTracker the watcher (data/tv_royale_watcher.py) uses,
# on live frames, so each detected play shows the instant it fires -- side,
# card, cost, elixir cross-check (OK / ?e), and the queue-desync flag -- to
# confirm against the actual replay in real time. Writes to a THROWAWAY log
# under logs/, never the real learning log (data/tv_royale_plays_log.jsonl).
# --------------------------------------------------------------------------
FONT = cv2.FONT_HERSHEY_DUPLEX
PANEL_W_CYC = 470
ENE_COL = (70, 70, 235)    # enemy / top player (red, BGR)
FRI_COL = (235, 170, 70)   # friendly / bottom player (blue, BGR)

# Panel text sizing. The whole view is downscaled by _fit before display, so these
# are deliberately large to stay readable. Tune here.
CYC_HEADER_SCALE = 0.78
CYC_ROW_SCALE = 0.66
CYC_ROW_PITCH = 28      # vertical px between rows
CYC_ROW_OFFSET = 36     # first row below its header

# FIXED LAYOUT (DEBUG_CONVENTIONS.md RULE 1): each section has a reserved y and a
# fixed max-row count; adding/removing a play never shifts another section.
_CYC_SECTIONS = {  # label -> (header_y, max_rows)
    "ENEMY (top)":    (40, 5),
    "FRIENDLY (bot)": (228, 5),
    "OFF-DECK":       (416, 4),   # high-conf reads the tracker FILTERS (not in known/learned deck)
    "PLAYS":          (580, 32),  #   (just "mirror" = filter working)
}


def _draw_cyc_section(panel, key, rows):
    y, max_rows = _CYC_SECTIONS[key]
    cv2.putText(panel, key, (10, y), FONT, CYC_HEADER_SCALE, (255, 255, 255), 2, cv2.LINE_AA)
    for i in range(max_rows):  # fixed slots -- unused ones left blank
        if i < len(rows):
            txt, col = rows[i]
            cv2.putText(panel, txt, (14, y + CYC_ROW_OFFSET + i * CYC_ROW_PITCH),
                        FONT, CYC_ROW_SCALE, col, 1, cv2.LINE_AA)


def _queue_rows(ev, elixir, learned_n, col):
    """A player's reconstructed cycle state + learned-deck progress, so the
    tracker's inference is visible next to the arena."""
    if ev is None:
        return [(f"elixir  {elixir:4.1f}", col),
                (f"learned {learned_n}/8", col),
                ("(waiting for match)", col)]
    rest = ", ".join(str(c) for c in ev["rest_of_queue"])
    return [
        (f"elixir  {elixir:4.1f}", col),
        (f"learned {learned_n}/8", col),
        (f"on-deck {ev['on_deck']}", col),
        (f"queue   {rest[:32]}", col),
        (f"resolved {ev['queue_fully_resolved']}", col),
    ]


class CyclesValidator:
    """Drives the dual cycle tracker over live frames + a throwaway play log."""

    def __init__(self):
        os.makedirs("logs", exist_ok=True)
        self.log_path = os.path.join(
            "logs", f"tvr_plays_validation_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
        self.reset()

    def reset(self):
        """(Re)build the tracker from the CURRENT config decks -- reload picks up
        an edit made for a new replay. Bound to 'r' in the live view."""
        importlib.reload(TVD)
        self.enemy_deck = [TVD.normalize(c) for c in TVD.ENEMY_DECK]
        self.friendly_deck = [TVD.normalize(c) for c in TVD.FRIENDLY_DECK]
        self.tracker = DualCycleTracker(self.enemy_deck, self.friendly_deck)
        self.t0 = None
        self.logged = 0
        self.last_ev = {"enemy": None, "friendly": None}
        self.offdeck = {}   # card -> count of high-conf reads not in that hand's deck
        seeded = "(learning from hands only)" if not (self.enemy_deck or self.friendly_deck) \
            else "(pre-seeded from config + learning from hands)"
        print(f"[cycles] reset {seeded}")
        print(f"[cycles]   enemy(top)   ={self.enemy_deck}")
        print(f"[cycles]   friendly(bot)={self.friendly_deck}")
        bad = TVD._validate(self.enemy_deck) + TVD._validate(self.friendly_deck)
        if bad:
            print(f"[cycles] note: {bad} not a known card slug -> not pre-seeded, but the deck "
                  f"is also LEARNED from the hands, so real cards still register once they've "
                  f"held a slot ~1s. Fix the spelling in config/tv_royale_decks.py to pre-seed "
                  f"and get full queue reconstruction.")
        print(f"[cycles] throwaway validation log -> {self.log_path}")

    def observe(self, data, now):
        if self.t0 is None:
            self.t0 = now
        # tally BEFORE update() so the learned deck reflects frames already seen
        self.last_ev = self.tracker.update(
            data["enemy"], data["friendly"], data["elix_e"], data["elix_f"], now)
        self._tally_offdeck(data["enemy"], self.tracker.enemy)
        self._tally_offdeck(data["friendly"], self.tracker.friendly)
        while self.logged < len(self.tracker.plays):
            self._emit(self.tracker.plays[self.logged])
            self.logged += 1

    def _tally_offdeck(self, reads, ct):
        """Count confident reads whose card isn't in the known/learned deck -- e.g.
        the blank-frame 'mirror' or a classifier flip. The debounce is what stops
        these becoming plays; this panel just makes them visible. A card that later
        gets learned is dropped, so OFF-DECK settles to just the genuine misreads."""
        for r in reads:
            card, score = r[0], r[1]
            if not card or card == "UNKNOWN" or score < CLASSIFIER_CONF_THRESHOLD:
                continue
            if ct.in_deck(card):
                self.offdeck.pop(card, None)   # became a known/learned card
            else:
                self.offdeck[card] = self.offdeck.get(card, 0) + 1

    def _emit(self, play):
        rel = round(play["t"] - self.t0, 1)
        rec = {"rel_t": rel, "side": play["side"], "card": play["card"],
               "cost": play.get("cost"), "revealed": play.get("revealed"),
               "confirmed": play.get("confirmed"), "desync": play.get("desync"),
               "inferred": play.get("inferred", False)}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        flag = "OK" if play.get("confirmed") else "?elixir"
        if play.get("inferred"):
            flag += " INFERRED" if play.get("named") else " INFERRED?"
        if play.get("desync"):
            flag += " !DESYNC"
        print(f"PLAY  {play['side'].upper():8} {str(play['card']):16} "
              f"{play.get('cost')}e  t={rel:6.1f}s  [{flag}]")

    def counts(self):
        e = sum(1 for p in self.tracker.plays if p["side"] == "enemy")
        return e, len(self.tracker.plays) - e

    def rel_plays(self):
        t0 = self.t0 or 0
        return [dict(p, rel_t=round(p["t"] - t0, 1)) for p in self.tracker.plays]


def cycles_panel(validator, data, in_match, height):
    """Fixed-layout panel: both players' reconstructed cycles + the running list
    of detected plays (newest first), colored by side."""
    panel = np.full((height, PANEL_W_CYC, 3), 25, np.uint8)
    ev = validator.last_ev
    ne = len(validator.tracker.enemy.observed_deck)
    nf = len(validator.tracker.friendly.observed_deck)
    _draw_cyc_section(panel, "ENEMY (top)", _queue_rows(ev["enemy"], data["elix_e"], ne, ENE_COL))
    _draw_cyc_section(panel, "FRIENDLY (bot)", _queue_rows(ev["friendly"], data["elix_f"], nf, FRI_COL))

    off = sorted(validator.offdeck.items(), key=lambda kv: -kv[1])
    _draw_cyc_section(panel, "OFF-DECK", [(f"{c[:16]:16} x{n}", ORANGE) for c, n in off[:4]])

    rows = []
    for p in validator.rel_plays()[::-1]:
        col = ENE_COL if p["side"] == "enemy" else FRI_COL
        tag = "ENE" if p["side"] == "enemy" else "FRI"
        flag = "" if p.get("confirmed") else "?e"
        if p.get("inferred"):
            flag = ("inf " + flag).strip()   # recovered from elixir, not read directly
        if p.get("desync"):
            flag += "!"
        rows.append((f"{p['rel_t']:6.1f}s {tag} {str(p['card'])[:15]:15} "
                     f"{p.get('cost')}e {flag}", col))
    _draw_cyc_section(panel, "PLAYS", rows)
    e, f = validator.counts()
    hy = _CYC_SECTIONS["PLAYS"][0]
    cv2.putText(panel, f"E{e} F{f}", (PANEL_W_CYC - 140, hy), FONT, CYC_HEADER_SCALE,
                (220, 220, 220), 2, cv2.LINE_AA)

    mtag = "IN MATCH" if in_match else "waiting for match..."
    cv2.putText(panel, mtag, (10, height - 20), FONT, 0.8,
                (GREEN if in_match else ORANGE), 2, cv2.LINE_AA)
    return panel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--static", action="store_true",
                   help="Read one frame from test_screenshot.png instead of the live emulator.")
    p.add_argument("--min-confidence", type=float, default=0.55,
                   help="Score at/above which a match is drawn green (else orange). Default 0.55.")
    p.add_argument("--collect", action="store_true",
                   help="LIVE only: bank clean resting card crops to the review queue "
                        "(assets/unlabeled_tv_hand_queue/) to grow hand_templates. Review "
                        "them with perception.review_tv_hand_queue.")
    p.add_argument("--match-examples", type=int, default=10,
                   help="Max stored examples per card compared at runtime (speed). "
                        "Default 10; 0 = use all (slower). Does not affect the library.")
    p.add_argument("--classifier", action="store_true",
                   help="Read hands with the YOLOv8-cls hand classifier "
                        "(assets/models/hand_cls/best.pt) instead of template matching. "
                        "Unrestricted (all 122 cards) unless --restrict.")
    p.add_argument("--restrict", action="store_true",
                   help="Restrict the classifier to the decks in config/tv_royale_decks.py "
                        "(argmax over 16 instead of 122). Off by default -- a good classifier "
                        "shouldn't need it; use to compare, or once decks are auto-read.")
    p.add_argument("--cycles", action="store_true",
                   help="LIVE only: run the dual cycle tracker (the watcher's exact play "
                        "engine) and show each detected PLAY as it fires -- side / card / cost / "
                        "elixir-check -- to validate play-by-play against the replay. Implies "
                        "--classifier. Decks are LEARNED from the hands, so config/tv_royale_decks.py "
                        "is optional (fill it to pre-seed + get full queue reconstruction). Press r "
                        "to reset for a new replay. Logs to a throwaway file, not the real log.")
    args = p.parse_args()

    templates = cap_templates(restricted_templates(load_templates()), args.match_examples)
    clf, backend = None, "templates"
    if args.classifier or args.cycles:
        from perception.hand_classifier import HandClassifier
        # In --cycles, restrict the classifier to the deck cards when a list is
        # filled -- it can't then output a card that isn't in play (kills confusable-
        # pair flips like zap<->skeleton_dragons at the source). Empty list -> [] ->
        # unrestricted (pure learn). --restrict forces it for plain --classifier too.
        allowed = combined_cards() if (args.restrict or args.cycles) else None
        clf = HandClassifier(allowed=allowed)
        n = len(clf.allowed_idx) if clf.allowed_idx else 122
        backend = f"classifier:{n}"
        print(f"Hand classifier loaded ({'restricted to ' + str(n) if clf.allowed_idx else 'all 122'} cards).")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    if args.static and args.cycles:
        print("--cycles needs a live stream (it tracks plays over time) — ignoring --static.")
        args.static = False

    if args.static:
        frame = cv2.imread("test_screenshot.png")
        if frame is None:
            raise FileNotFoundError("test_screenshot.png not found — capture a TV Royale frame first.")
        cv2.imshow(WINDOW, _fit(render(frame, templates, args.min_confidence, clf=clf, backend=backend)))
        print("Showing static frame. Press any key in the window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Live: pull frames straight from the emulator (never writes test_screenshot.png).
    from perception.capture import ScreenCapture
    from control.actions import EmulatorController
    collector = None
    if args.collect:
        if clf is not None:
            # classifier mode: self-stock a queue of the cards it's UNSURE about
            from perception.tv_royale_collector import UncertainCardCollector, UNCERTAIN_CONF
            collector = UncertainCardCollector()
            print(f"Collecting UNCERTAIN cards (classifier conf < {UNCERTAIN_CONF}) to the "
                  f"review queue -- sort with perception.review_tv_hand_queue.")
        else:
            from perception.tv_royale_collector import HandCropCollector
            collector = HandCropCollector()
            if not combined_cards():
                print("NOTE: --collect works best with both decks filled in "
                      "config/tv_royale_decks.py (guesses are then best-of-16).")
    validator = CyclesValidator() if args.cycles else None
    if validator:
        print("CYCLES validation live. Watch the replay; each PLAY prints here + shows in the "
              "panel (E=enemy/top red, F=friendly/bottom blue). q=quit, r=reset for a new replay.")
    print("Connecting to LDPlayer... (q in the window to quit)")
    capture = ScreenCapture(EmulatorController())
    try:
        while True:
            frame = capture.screenshot()
            data = read_frame(frame, templates, clf)
            note = ""
            if collector:
                _collect_frame(collector, frame, data)
                note = f"   collected:{collector.saved}"
            display = draw(frame, data, args.min_confidence, note, backend)
            if validator:
                in_match = R.is_in_match(frame)
                if in_match:
                    validator.observe(data, time.time())
                display = np.hstack([cycles_panel(validator, data, in_match, display.shape[0]),
                                     display])
            cv2.imshow(WINDOW, _fit(display))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r") and validator:
                validator.reset()
    finally:
        capture.stop()
        cv2.destroyAllWindows()
        if collector:
            print(f"Banked {collector.saved} crop(s) to {collector.queue_dir} — "
                  f"review with: python -m perception.review_tv_hand_queue")


if __name__ == "__main__":
    main()
