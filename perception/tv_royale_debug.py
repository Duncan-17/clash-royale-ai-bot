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
"""

import argparse

import cv2
import numpy as np

from config import tv_royale_regions as R
from config.tv_royale_decks import restricted_templates, combined_cards
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


def _fit(display, max_h=940):
    h = display.shape[0]
    if h <= max_h:
        return display
    s = max_h / h
    return cv2.resize(display, (int(display.shape[1] * s), max_h), interpolation=cv2.INTER_AREA)


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
    args = p.parse_args()

    templates = cap_templates(restricted_templates(load_templates()), args.match_examples)
    clf, backend = None, "templates"
    if args.classifier:
        from perception.hand_classifier import HandClassifier
        clf = HandClassifier(allowed=(combined_cards() if args.restrict else None))
        n = len(clf.allowed_idx) if clf.allowed_idx else 122
        backend = f"classifier:{n}"
        print(f"Hand classifier loaded ({'restricted to ' + str(n) if clf.allowed_idx else 'all 122'} cards).")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

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
            cv2.imshow(WINDOW, _fit(draw(frame, data, args.min_confidence, note, backend)))
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        capture.stop()
        cv2.destroyAllWindows()
        if collector:
            print(f"Banked {collector.saved} crop(s) to {collector.queue_dir} — "
                  f"review with: python -m perception.review_tv_hand_queue")


if __name__ == "__main__":
    main()
