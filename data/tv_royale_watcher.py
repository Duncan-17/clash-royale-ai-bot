"""The macro: watches TV Royale replays back-to-back, unattended, and ANALYSES each
one via game_knowledge/tv_royale_analyzer.TvrAnalyzer -- the hand classifier reads
both hands -> a DUAL cycle tracker reconstructs both players' cycles and logs
authoritative PLAY events (card+time+side) to data/tv_royale_plays_log.jsonl (the
learning data), while self-stocking review queues of the CARDS the classifier is
unsure about and (--levels) the troop LEVEL badges the level reader can't read yet.

Flow (calibrated 2026-09-07 with Duncan; coords/timings in config/tv_royale_nav.py):
  open TV Royale (menu -> TV Royale) -> read the list (Watch buttons + both
  decks per game, perception/tv_royale_list) -> tap a game's Watch -> wait
  WATCH_SETTLE_SECONDS for the entry transition -> analyse while the match is
  live (config.tv_royale_regions.is_in_match gate: both elixir bars present)
  -> when the gate goes False the replay ended -> return to the list -> pick the
  next unwatched game (dedup by deck signature) -> scroll for more -> next arena.

The list's auto-read decks feed each game's two CycleTrackers (and are still logged
to data/tv_royale_decks_log.jsonl). TWO ASSUMPTIONS TO VALIDATE LIVE: the deck->side
mapping (watch_game maps list-left=top=enemy, right=bottom=friendly) and, for
--levels, that the detector transfers to the TV Royale board -- see
tv_royale_analyzer's docstring.

Two behaviors still on provisional defaults (see config/tv_royale_nav.py):
scroll travel, and how to leave a finished replay (defaults to re-opening TV
Royale, which lands on a clean list top whatever the end screen looks like).
Calibrating REPLAY_EXIT_TAP would let it resume mid-list instead of re-scrolling.

Usage:
    python -m data.tv_royale_watcher --minutes 120
    python -m data.tv_royale_watcher --max-replays 20
    python -m data.tv_royale_watcher --arenas 3   # browse this many arenas
"""

import argparse
import json
import os
import time

import cv2

from config import tv_royale_nav as NAV
from config.tv_royale_regions import is_in_match
from control.actions import EmulatorController
from perception.capture import ScreenCapture
from perception.hand_classifier import HandClassifier
from perception.tv_royale_collector import UncertainCardCollector
from game_knowledge.tv_royale_analyzer import TvrAnalyzer
from perception.tv_royale_list import (
    parse_list, deck_signature, load_icon_templates, detect_watch_buttons,
)

DECK_LOG = "data/tv_royale_decks_log.jsonl"
WAIT_FOR_MATCH_TIMEOUT = 18.0   # after tapping Watch, give the replay this long to start
END_CONFIRM_POLLS = 8           # consecutive not-in-match polls (~2s) to declare a replay ended
MAX_SCROLLS_PER_ARENA = 12      # give up scrolling an arena's list after this many


def _sleep(seconds):
    time.sleep(seconds)


def open_tv_royale(controller, capture):
    """Menu -> TV Royale. Lands on the list at the top."""
    controller.tap(*NAV.MENU_ICON)
    _sleep(NAV.MENU_SETTLE_SECONDS)
    controller.tap(*NAV.TV_ROYALE_OPTION)
    _sleep(NAV.MENU_SETTLE_SECONDS)


def next_arena(controller):
    controller.tap(*NAV.ARENA_NEXT)
    _sleep(NAV.MENU_SETTLE_SECONDS)


def scroll_list(controller):
    x1, y1, x2, y2 = NAV.SCROLL_SWIPE
    controller.swipe(x1, y1, x2, y2, NAV.SCROLL_DURATION_MS)
    _sleep(NAV.MENU_SETTLE_SECONDS)


def log_deck(game):
    os.makedirs(os.path.dirname(DECK_LOG), exist_ok=True)
    rec = {
        "time": time.time(),
        "left": [c for c, _ in game["left_deck"]],
        "right": [c for c, _ in game["right_deck"]],
        "left_scores": [round(s, 3) for _, s in game["left_deck"]],
        "right_scores": [round(s, 3) for _, s in game["right_deck"]],
    }
    with open(DECK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def on_list(capture):
    """True if the TV Royale list is currently showing (a Watch button is
    visible) -- used to confirm we've returned from a replay."""
    buttons, _ = detect_watch_buttons(capture.screenshot())
    return len(buttons) > 0


def return_to_list(controller, capture, tries=8):
    """A finished replay ends in transition/results screens; tap OK
    (REPLAY_EXIT_TAP) until the list is back. The list returns at the same
    scroll position with the just-watched game centered, so we do NOT re-open.
    Retries because the transition length is variable -- once the list is
    detected we stop (further taps would land on the list itself)."""
    for _ in range(tries):
        if on_list(capture):
            return True
        controller.tap(*NAV.REPLAY_EXIT_TAP)
        _sleep(NAV.MENU_SETTLE_SECONDS)
    return on_list(capture)


def wait_for_match(capture, timeout=WAIT_FOR_MATCH_TIMEOUT):
    """Poll until the replay is confirmed live, or give up. Returns True if the
    match started."""
    start = time.time()
    while time.time() - start < timeout:
        if is_in_match(capture.screenshot()):
            return True
        _sleep(NAV.POLL_INTERVAL_SECONDS)
    return False


def analyze_match(capture, analyzer):
    """Analyze one replay to its end (match-gate False for END_CONFIRM_POLLS in a
    row). Per frame: classifier hand reads -> dual cycle tracker (logs plays) +
    self-collect uncertain cards / unknown levels. Returns plays detected this game."""
    misses = 0
    plays_start = analyzer.play_count
    end_frame = None
    while misses < END_CONFIRM_POLLS:
        frame = capture.screenshot()
        if is_in_match(frame):
            misses = 0
            analyzer.observe(frame)
        else:
            misses += 1
            end_frame = frame               # last non-match frame ~ the results screen
        _sleep(NAV.POLL_INTERVAL_SECONDS)
    analyzer.end_game(end_frame=end_frame)  # finalize the learning game header + end shot
    return analyzer.play_count - plays_start


def watch_game(controller, capture, game, analyzer):
    """Tap into one game, analyze it, and return to a clean list. Returns plays
    detected (or None if the replay never started)."""
    # DECK->SIDE mapping ASSUMPTION (validate live): list left = top = enemy,
    # right = bottom = friendly. If the cycles come out mirrored, swap these.
    enemy_deck = [c for c, _ in game["left_deck"]]
    friendly_deck = [c for c, _ in game["right_deck"]]
    gid = f"{abs(hash(deck_signature(game))) % 100000:05d}"
    analyzer.start_game(enemy_deck, friendly_deck, game_id=gid)

    controller.tap(*game["watch_xy"])
    _sleep(NAV.WATCH_SETTLE_SECONDS)
    if not wait_for_match(capture):
        print("  replay didn't start (gate never went live) — backing out")
        open_tv_royale(controller, capture)  # recover to a known screen
        return None
    plays = analyze_match(capture, analyzer)
    # Replay ended -> tap OK through the transition screens back to the list
    # (same scroll position, just-watched game centered).
    if not return_to_list(controller, capture):
        print("  couldn't confirm return to list — re-opening TV Royale to recover")
        open_tv_royale(controller, capture)
    return plays


def run(minutes=None, max_replays=None, arenas=1, games_per_arena=None,
        arena_direction="next", levels=False, learn=False):
    print("Connecting to LDPlayer...")
    controller = EmulatorController()
    capture = ScreenCapture(controller)
    icon_templates = load_icon_templates()
    clf = HandClassifier()                      # all 122 cards, unrestricted
    card_collector = UncertainCardCollector()   # self-stocks the cards it's unsure about
    learn_logger = None
    if learn:
        from data.learning_logger import LearningLogger
        learn_logger = LearningLogger()
    detector = None
    if levels or learn:                         # board/placement + level badges both need it
        from perception.yolo_detector import YoloDetector
        detector = YoloDetector(include_non_units=True)
    analyzer = TvrAnalyzer(clf, card_collector=card_collector, detector=detector,
                           learn_logger=learn_logger)
    print(f"Classifier hand reading -> dual cycle tracker (plays -> {analyzer.plays_log}); "
          f"uncertain cards -> review queue{'; troop levels -> level queue' if levels else ''}"
          f"{'; LEARNING dataset -> data/learning/' if learn else ''}.")

    watched = set()
    replays = 0
    start = time.time()

    # Start from wherever the TV Royale list already is if you've opened it
    # yourself (so you can pick the starting arena manually); only open it if
    # we're not already on the list -- re-opening would jump to a default arena.
    if not on_list(capture):
        open_tv_royale(controller, capture)

    arrow = NAV.ARENA_NEXT if arena_direction == "next" else NAV.ARENA_PREV
    try:
        for arena in range(arenas):
            arena_games = 0
            scrolls = 0
            while True:
                if minutes is not None and (time.time() - start) / 60 >= minutes:
                    return
                if max_replays is not None and replays >= max_replays:
                    return
                if games_per_arena is not None and arena_games >= games_per_arena:
                    break  # hit this arena's game cap

                games = parse_list(capture.screenshot(), icon_templates)
                fresh = [g for g in games if deck_signature(g) not in watched]

                if fresh:
                    g = fresh[0]
                    watched.add(deck_signature(g))
                    log_deck(g)
                    left = ",".join(c for c, _ in g["left_deck"][:4]) + ",..."
                    print(f"[arena {arena+1}] game {arena_games+1}"
                          f"{'/'+str(games_per_arena) if games_per_arena else ''} ({left}) "
                          f"— {replays+1} total")
                    plays = watch_game(controller, capture, g, analyzer)
                    if plays is not None:
                        replays += 1
                        arena_games += 1
                        print(f"  {plays} plays detected; {card_collector.saved} uncertain cards queued")
                    scrolls = 0  # progress made; the list stays in place (just-watched
                    # game centered), so we just continue down from here
                    continue

                if scrolls < MAX_SCROLLS_PER_ARENA:
                    scroll_list(controller)
                    scrolls += 1
                    continue
                print(f"  arena {arena+1} ran out of new games after scrolling")
                break  # this arena exhausted

            # Advance to the next arena IN PLACE (tap the arrow; don't re-open,
            # which would reset the arena). The arena banner is a fixed header,
            # so the arrow is tappable regardless of list scroll position.
            if arena < arenas - 1:
                print(f"arena {arena+1} done ({arena_games} games) — advancing arena")
                controller.tap(*arrow)
                _sleep(NAV.MENU_SETTLE_SECONDS)
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        print(f"\nStopped. Watched {replays} replay(s). Plays -> {analyzer.plays_log}; "
              f"{card_collector.saved} uncertain card(s) -> {card_collector.queue_dir} "
              f"(review: python -m perception.review_tv_hand_queue).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=None, help="Stop after this many minutes.")
    p.add_argument("--max-replays", type=int, default=None, help="Stop after this many replays.")
    p.add_argument("--arenas", type=int, default=1, help="How many arenas to browse.")
    p.add_argument("--games-per-arena", type=int, default=None,
                   help="Watch at most this many games per arena, then advance (e.g. 8).")
    p.add_argument("--arena-direction", choices=["next", "prev"], default="next",
                   help="Which arrow to change arena with: 'next'=right, 'prev'=left. Default next.")
    p.add_argument("--levels", action="store_true",
                   help="Also run the detector to self-queue unknown troop level badges. "
                        "Needs a live check that the detector transfers to the TV Royale board.")
    p.add_argument("--learn", action="store_true",
                   help="Write the LEARNING dataset (data/learning/: games.jsonl, samples.jsonl, "
                        "archived frames) -- one egocentric state->action sample per play, with "
                        "detector board + best-effort placement. Runs the detector each frame.")
    args = p.parse_args()
    run(minutes=args.minutes, max_replays=args.max_replays, arenas=args.arenas,
        games_per_arena=args.games_per_arena, arena_direction=args.arena_direction,
        levels=args.levels, learn=args.learn)


if __name__ == "__main__":
    main()
