"""
Scores OpponentPlayTracker's real output against a human-verified ground
truth fixture — Duncan's explicit request (2026-08-16) after dictating a
second-by-second account of a real match, so future changes to
opponent_play_tracker.py / spawn_data.py can be checked against real
matches, not just synthetic unit tests (game_knowledge/test_opponent_play_tracker.py)
or one-off ad-hoc log inspection (which is how every fix bundled into this
tool up to now was actually found and verified — this formalizes that
process instead of repeating it by hand each time).

Ground truth fixtures live in game_knowledge/ground_truth/*.json:
    {
        "log_file": "logs/debug_live_<timestamp>.jsonl",
        "own_deck": [...],
        "events": [{"t": 13, "side": "own"|"opponent", "card": "<slug>",
                     "position": "<free text>"}, ...]
    }
Card names in a fixture are SLUGS (config/deck.txt's convention, what a
human would say — "archers", "mini_pekka"), NOT
perception/unit_detector.py's class names ("archer", "minipekka") — see
slug_to_class_count() below for the mapping used to compare the two.

Only "side": "opponent" events are checked against OpponentPlayTracker's
output (that's what this module verifies). "side": "own" events are
recorded for context and future use (e.g. validating cycle_tracker.py
against the same match) but not compared here yet.

Usage:
    python -m game_knowledge.compare_ground_truth game_knowledge/ground_truth/match_20260816_115424.json
"""

import argparse
import json
import sys

from game_knowledge import spawn_data
from game_knowledge.opponent_play_tracker import OpponentPlayTracker

MATCH_TOLERANCE_SECONDS = 6.0  # ground truth timestamps are eyeballed to the
# nearest second while watching replay footage after the fact, not read
# from frame data — generous on purpose, matching against detected events
# that also carry their own real latency (GROUPING_WINDOW_SECONDS and the
# rest of the debounce chain documented in opponent_play_tracker.py).


def slug_to_class_count(slug: str):
    """Reverses the mappings opponent_play_tracker.py already has (rather
    than hand-listing a duplicate table here): spawn_data.CARD_SLUG_FOR_BURST
    for the multi-unit/renamed cases, plus musketeer/skeleton's special
    count-sensitive slugs. Anything not found in either falls back to
    "the slug IS the unit class, count 1" — true for the common case
    (Knight, Giant, Musketeer-as-a-word only appears via the special case
    above, etc.)."""
    if slug == "musketeer":
        return "musketeer", 1
    if slug == "three_musketeers":
        return "musketeer", 3
    if slug == "skeletons":
        return "skeleton", 3
    if slug == "skeleton_army":
        return "skeleton", spawn_data.SKELETON_ARMY_BURST_MIN
    for unit_class, s in spawn_data.CARD_SLUG_FOR_BURST.items():
        if s == slug:
            return unit_class, spawn_data.CARD_BURST_COUNT.get(unit_class, 1)
    return slug, spawn_data.CARD_BURST_COUNT.get(slug, 1)


def replay(log_path: str, own_deck: list):
    """Same remap-and-replay pattern used throughout this session's ad-hoc
    log analysis — formalized here instead of rewritten each time."""
    def remap(tracked):
        return [{"track_id": u["id"], "name": u["name"], "side": u["side"], "bbox": u["bbox"],
                  "age": u["age"], "visible": u["visible"], "missed_frames": u["missed"]} for u in tracked]

    tracker = OpponentPlayTracker(own_deck=own_deck)
    events = []
    own_recent_plays = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cyc = rec.get("cycle") or {}
            last_play = cyc.get("last_play")
            if last_play:
                own_recent_plays.append((last_play["card"], rec["t"]))
            for ev in tracker.update(remap(rec["tracked"]), now=rec["t"], own_recent_plays=own_recent_plays):
                events.append((rec["t"], ev))
    return events


def compare(fixture_path: str, log_path: str = None):
    with open(fixture_path, encoding="utf-8") as f:
        fixture = json.load(f)
    log_path = log_path or fixture["log_file"]
    events = replay(log_path, fixture.get("own_deck", []))

    opponent_gt = [e for e in fixture["events"] if e["side"] == "opponent"]
    results = []
    used = set()
    for gte in opponent_gt:
        unit_class, count = slug_to_class_count(gte["card"])
        best = None
        for i, (t, ev) in enumerate(events):
            if i in used or ev["card"] != unit_class:
                continue
            if abs(t - gte["t"]) > MATCH_TOLERANCE_SECONDS:
                continue
            if best is None or abs(t - gte["t"]) < abs(best[0] - gte["t"]):
                best = (t, ev, i)
        if best is not None:
            t, ev, i = best
            used.add(i)
            status = "MATCH" if ev["attribution"] in ("played", "spawned") else f"WRONG_ATTRIBUTION({ev['attribution']})"
            results.append((gte, status, t, ev))
        else:
            results.append((gte, "MISSED", None, None))

    extras = [(t, ev) for i, (t, ev) in enumerate(events)
              if i not in used and ev["attribution"] in ("played", "spawned")]
    return results, extras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", help="Path to a ground_truth/*.json fixture")
    parser.add_argument("--log", default=None, help="Override the fixture's own log_file path")
    args = parser.parse_args()

    results, extras = compare(args.fixture, args.log)

    matched = sum(1 for _, status, _, _ in results if status == "MATCH")
    print(f"{matched}/{len(results)} ground-truth opponent plays matched (tolerance ±{MATCH_TOLERANCE_SECONDS:.0f}s)\n")
    for gte, status, t, ev in results:
        if status == "MATCH":
            print(f"  MATCH      t={gte['t']:4d} {gte['card']:12s} ({gte['position']}) -> detected t={t:.1f} count={ev['count']}")
        elif status == "MISSED":
            print(f"  MISSED     t={gte['t']:4d} {gte['card']:12s} ({gte['position']}) -> no matching detection nearby")
        else:
            print(f"  {status:22s} t={gte['t']:4d} {gte['card']:12s} ({gte['position']}) -> detected t={t:.1f} but wrong attribution")

    print(f"\n{len(extras)} extra 'played'/'spawned' detection(s) with no matching ground-truth entry (false positives or plays outside the annotated window):")
    for t, ev in extras:
        print(f"  t={t:6.2f} card={ev['card']:12s} count={ev['count']} attrib={ev['attribution']}")

    sys.exit(0 if matched == len(results) else 1)


if __name__ == "__main__":
    main()
