"""The Phase 5 baseline bot loop: perception -> board state -> policy -> action.

Ties the whole pipeline together:
    capture -> yolo_detector -> thin_tracker -> build_game_state -> RuleBasedPolicy
    -> (live) EmulatorController.play_card + PlayTracker.record_own_play

SAFE BY DEFAULT: runs in DRY-RUN -- it prints the decision each time but does NOT
tap the game. Pass --live to actually play. Start dry, watch the decisions look
sane against the match, THEN go live.

    python -m game_knowledge.bot              # dry-run (log decisions only)
    python -m game_knowledge.bot --live       # actually plays cards
    python -m game_knowledge.bot --once        # one decision from test_screenshot.png (no emulator)
"""

import argparse
import time

ACTION_COOLDOWN_S = 1.2   # min gap between plays -- capture latency means elixir
# hasn't visibly updated for a moment after a play; this stops double-spending.


def _run_once_static():
    """One decision from test_screenshot.png -- no emulator, no tracking history.
    A quick offline sanity check of the state build + policy."""
    import cv2
    from perception.yolo_detector import YoloDetector
    from perception.thin_tracker import ThinTracker
    from game_knowledge.game_state import build_game_state
    from game_knowledge.policy import RuleBasedPolicy

    frame = cv2.imread("test_screenshot.png")
    if frame is None:
        raise FileNotFoundError("test_screenshot.png not found.")
    det = YoloDetector(include_non_units=True)
    raw = det.detect(frame)
    from perception.detector_view import _is_component
    units = [d for d in raw if not _is_component(d)]
    tracked = ThinTracker().update(units)
    state = build_game_state(frame, tracked)
    print(f"elixir={state.elixir} (+{state.elixir_fraction:.2f})  "
          f"hand={[f'{s.card}:{s.cost}' for s in state.hand]}")
    print(f"units: {len(state.units)}  enemy_on_our_half="
          f"{sum(1 for u in state.enemy_units() if (u['bbox'][1]+u['bbox'][3])/2 > 711)}")
    action = RuleBasedPolicy().decide(state)
    print("DECISION:", action if action else "wait")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true",
                   help="Actually tap the game. Default is DRY-RUN (log decisions only).")
    p.add_argument("--once", action="store_true",
                   help="One decision from test_screenshot.png, no emulator (offline sanity check).")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--model", default=None, help="Detector weights (default = DEFAULT_MODEL).")
    args = p.parse_args()

    if args.once:
        _run_once_static()
        return

    from perception.capture import ScreenCapture
    from control.actions import EmulatorController
    from perception.yolo_detector import YoloDetector, DEFAULT_MODEL
    from perception.thin_tracker import ThinTracker
    from perception.tower_reader import TowerStateTracker
    from perception.card_detector import load_templates as load_hand
    from perception.elixir_counter import load_templates as load_elixir
    from perception.detector_view import _is_component
    from game_knowledge.game_state import build_game_state
    from game_knowledge.policy import RuleBasedPolicy
    from game_knowledge.play_tracker import PlayTracker

    controller = EmulatorController()
    capture = ScreenCapture(controller)
    det = YoloDetector(model_path=args.model or DEFAULT_MODEL, include_non_units=True, conf=args.conf)
    tracker, towers = ThinTracker(), TowerStateTracker()
    policy, plays = RuleBasedPolicy(), PlayTracker()
    hand_t, elixir_t = load_hand(), load_elixir()

    mode = "LIVE (tapping the game)" if args.live else "DRY-RUN (logging only, not tapping)"
    print(f"Bot loop started -- {mode}. Ctrl+C to stop.")
    last_action = 0.0
    try:
        while True:
            frame = capture.screenshot()
            raw = det.detect(frame)
            units = [d for d in raw if not _is_component(d)]
            comps = [d for d in raw if _is_component(d)]
            tracked = tracker.update(units)
            plays.update(tracked)  # own-play overlay (no-op until we record) + track ids
            tower_hp = towers.update(frame)
            tower_info = {n: {"alive": True, "hp": d.get("hp_fraction")} for n, d in tower_hp.items()}
            state = build_game_state(frame, tracked, tower_info, hand_t, elixir_t)

            action = policy.decide(state)
            if action and (time.time() - last_action) >= ACTION_COOLDOWN_S:
                last_action = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] e={state.elixir} PLAY {action.card} "
                      f"slot{action.slot} @({action.x},{action.y}) -- {action.reason}"
                      f"{'' if args.live else '   [dry-run]'}")
                if args.live:
                    controller.play_card(action.slot, action.x, action.y)
                    plays.record_own_play(action.card, (action.x, action.y))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        capture.stop()


if __name__ == "__main__":
    main()
