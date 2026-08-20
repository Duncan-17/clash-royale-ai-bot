"""
Synthetic regression suite for OpponentPlayTracker — one small, hand-built
scenario per real bug found during this project's opponent-play-detection
work (2026-08-16), each traced back to an actual live-match log before
being reduced to a minimal reproduction here. Kept separate from
game_knowledge/compare_ground_truth.py, which checks against a REAL
recorded match instead of a synthetic one — the two are complementary:
this file pins down each specific mechanism in isolation and runs in a
fraction of a second, the ground-truth comparison catches anything only
visible against real, messy footage.

Usage:
    python -m game_knowledge.test_opponent_play_tracker
"""

from game_knowledge.opponent_play_tracker import OpponentPlayTracker

next_id = [0]


def unit(name, bbox, side="enemy"):
    next_id[0] += 1
    return {"track_id": next_id[0], "name": name, "side": side, "bbox": bbox, "age": 5, "visible": True, "missed_frames": 0}


def run_frames(tracker, frame_specs):
    """frame_specs: list of (dt, list_of_(label,name,bbox[,side])_tuples_to_add_this_frame, labels_to_remove).
    Returns list of all events seen across all frames. Maintains a rolling
    'alive' dict keyed by a stable label so tests can add/remove units by name."""
    events = []
    now = 0.0
    alive = {}  # label -> unit dict
    for dt, adds, removes in frame_specs:
        now += dt
        for label in removes:
            alive.pop(label, None)
        for spec in adds:
            label, name, bbox = spec[0], spec[1], spec[2]
            side = spec[3] if len(spec) > 3 else "enemy"
            alive[label] = unit(name, bbox, side)
        evs = tracker.update(list(alive.values()), now)
        events.extend(evs)
    return events


def test_1_common_single_card():
    t = OpponentPlayTracker()
    events = run_frames(t, [
        (0.1, [("knight1", "knight", (100, 100, 150, 150))], []),
        (0.6, [], []),  # let grouping window close
    ])
    assert len(events) == 1, events
    e = events[0]
    assert e["attribution"] == "played" and e["card"] == "knight" and e["count"] == 1
    print("test 1 OK:", e["card"], e["attribution"], e["count"])


def test_2_minions_burst():
    t = OpponentPlayTracker()
    events = run_frames(t, [
        (0.1, [("m1", "minion", (100, 100, 120, 120))], []),
        (0.1, [("m2", "minion", (110, 100, 130, 120))], []),
        (0.1, [("m3", "minion", (120, 100, 140, 120))], []),
        (0.6, [], []),
    ])
    assert len(events) == 1, events
    e = events[0]
    assert e["attribution"] == "played" and e["card"] == "minion" and e["count"] == 3
    assert e["card_slug"] == "minions"
    print("test 2 OK:", e["card"], e["attribution"], e["count"], e["card_slug"], "cost=", e["elixir_cost"])


def test_3_golem_split():
    t = OpponentPlayTracker()
    events = run_frames(t, [
        (0.1, [("golem1", "golem", (500, 500, 560, 560))], []),
        (0.5, [], []),  # golem alive a while
        (0.1, [], ["golem1"]),  # golem dies (vanishes)
        (0.2, [("g1", "golemite", (505, 505, 525, 525)), ("g2", "golemite", (530, 505, 550, 525))], []),
        (0.6, [], []),
    ])
    spawned = [e for e in events if e["card"] == "golemite"]
    assert len(spawned) == 1, events
    e = spawned[0]
    assert e["attribution"] == "spawned" and e["count"] == 2 and e["source_card"] == "golem", e
    print("test 3 OK:", e["card"], e["attribution"], e["count"], "source=", e["source_card"])


def test_4_tombstone_trickle():
    t = OpponentPlayTracker()
    events = run_frames(t, [
        (0.1, [("ts", "tombstone", (700, 700, 740, 740))], []),
        (1.0, [], []),  # tombstone sits alive
        (0.1, [("sk1", "skeleton", (705, 705, 715, 715))], []),
        (0.6, [], []),
    ])
    sk_events = [e for e in events if e["card"] == "skeleton"]
    assert len(sk_events) == 1, events
    e = sk_events[0]
    assert e["attribution"] == "spawned" and e["source_card"] == "tombstone", e
    print("test 4 OK:", e["card"], e["attribution"], "source=", e["source_card"])


def test_5_skeletons_vs_army():
    t = OpponentPlayTracker()
    # 3 skeletons far from any spawner -> Skeletons card
    events = run_frames(t, [
        (0.1, [(f"sk{i}", "skeleton", (200 + i * 5, 200, 210 + i * 5, 210)) for i in range(3)], []),
        (0.6, [], []),
    ])
    assert len(events) == 1, events
    e = events[0]
    assert e["attribution"] == "played" and e["count"] == 3 and e["card_slug"] == "skeletons", e
    print("test 5a OK:", e["card"], e["attribution"], e["count"], e["card_slug"])

    # ~15 skeletons -> Skeleton Army
    t2 = OpponentPlayTracker()
    events2 = run_frames(t2, [
        (0.1, [(f"a{i}", "skeleton", (300 + i * 5, 300, 310 + i * 5, 310)) for i in range(15)], []),
        (0.6, [], []),
    ])
    assert len(events2) == 1, events2
    e2 = events2[0]
    assert e2["attribution"] == "played" and e2["count"] == 15 and e2["card_slug"] == "skeleton_army", e2
    print("test 5b OK:", e2["card"], e2["attribution"], e2["count"], e2["card_slug"])


def test_6_two_of_three_minions_now_resolves_played():
    """Updated 2026-08-18: BURST_MIN_FRACTION loosened the burst-match
    threshold (see opponent_play_tracker.py's own comment on why -- real
    live-match detection recall rarely confirms a burst card's FULL count
    simultaneously). A 2-of-3 minion group now correctly resolves "played"
    instead of "uncertain" -- this is the intended new behavior, not a
    regression of the old test's stricter expectation."""
    t = OpponentPlayTracker()
    events = run_frames(t, [
        (0.1, [("m1", "minion", (400, 400, 420, 420)), ("m2", "minion", (410, 400, 430, 420))], []),
        (0.6, [], []),
    ])
    assert len(events) == 1, events
    e = events[0]
    assert e["attribution"] == "played" and e["count"] == 2 and e["card_slug"] == "minions", e
    print("test 6 OK: 2-of-3 minions now resolves played, slug=", e["card_slug"], "cost=", e["elixir_cost"])


def test_6b_a_single_minion_is_still_below_the_loosened_floor():
    """The loosened threshold has a floor (BURST_MIN_FRACTION's own
    max(2, ...)) -- a single stray minion must still resolve "uncertain",
    not "played", since accepting count=1 for a known 3-unit burst card
    would risk a real false positive (an ordinary misdetection treated as
    a full card play)."""
    t = OpponentPlayTracker()
    events = run_frames(t, [
        (0.1, [("m1", "minion", (400, 400, 420, 420))], []),
        (0.6, [], []),
    ])
    assert len(events) == 1, events
    e = events[0]
    assert e["attribution"] == "uncertain" and e["flag_reason"], e
    print("test 6b OK: a single minion (below the loosened floor) still resolves uncertain, reason=", e["flag_reason"])


def test_two_of_four_goblins_resolves_played():
    """Duncan's own reported real-world case (2026-08-18 live match log):
    a genuine Goblins play showed only 2 of 4 confirmed simultaneously --
    must now resolve "played" instead of "uncertain"."""
    t = OpponentPlayTracker()
    events = run_frames(t, [
        (0.1, [("g1", "goblin", (300, 300, 320, 320)), ("g2", "goblin", (330, 300, 350, 320))], []),
        (0.6, [], []),
    ])
    assert len(events) == 1, events
    e = events[0]
    assert e["attribution"] == "played" and e["count"] == 2 and e["card_slug"] == "goblins", e
    print("test goblins-2-of-4 OK:", e["attribution"], "count=", e["count"], "cost=", e["elixir_cost"])


def test_two_spear_goblins_resolves_played_as_the_real_card():
    """Spear Goblins (added to CARD_BURST_COUNT 2026-08-18, previously
    missing entirely -- see spawn_data.py's own comment) with 2 of its
    real 3-unit count confirmed must resolve "played" as the real card,
    not fall into the old "unexpected multiples of a single-unit card"
    branch or stay unresolved via CARD_SLUG_FOR_BURST's new mapping."""
    t = OpponentPlayTracker()
    events = run_frames(t, [
        (0.1, [("sg1", "spear_goblin", (300, 300, 320, 320)), ("sg2", "spear_goblin", (330, 300, 350, 320))], []),
        (0.6, [], []),
    ])
    assert len(events) == 1, events
    e = events[0]
    assert e["attribution"] == "played" and e["card_slug"] == "spear_goblins", e
    print("test spear_goblins-2-of-3 OK:", e["attribution"], "slug=", e["card_slug"], "cost=", e["elixir_cost"])


def test_musketeer_single_and_triple():
    t = OpponentPlayTracker()
    events = run_frames(t, [
        (0.1, [("mk1", "musketeer", (100, 100, 120, 120))], []),
        (0.6, [], []),
    ])
    assert len(events) == 1 and events[0]["attribution"] == "played" and events[0]["count"] == 1
    assert events[0]["card_slug"] == "musketeer"
    print("musketeer x1 OK:", events[0]["card_slug"], "cost=", events[0]["elixir_cost"])

    t2 = OpponentPlayTracker()
    events2 = run_frames(t2, [
        (0.1, [(f"mk{i}", "musketeer", (200 + i * 5, 200, 210 + i * 5, 210)) for i in range(3)], []),
        (0.6, [], []),
    ])
    assert len(events2) == 1 and events2[0]["attribution"] == "played" and events2[0]["count"] == 3
    assert events2[0]["card_slug"] == "three_musketeers"
    print("musketeer x3 OK:", events2[0]["card_slug"], "cost=", events2[0]["elixir_cost"])


def test_7_side_flip_misclassified():
    t = OpponentPlayTracker()
    # An ally giant sits ON OUR SIDE (y >= RIVER_Y_TOP=650), then vanishes
    # (side flip); a "new" enemy giant appears nearby shortly after --
    # should be flagged misclassified, not played.
    events = run_frames(t, [
        (0.1, [("g_ally", "giant", (300, 900, 360, 960), "ally")], []),
        (0.5, [], []),  # ally giant established
        (0.1, [], ["g_ally"]),  # ally giant "vanishes" (really: flips side)
        (0.1, [("g_enemy", "giant", (305, 905, 365, 965), "enemy")], []),
        (0.6, [], []),
    ])
    giant_events = [e for e in events if e["card"] == "giant"]
    assert len(giant_events) == 1, giant_events
    e = giant_events[0]
    assert e["attribution"] == "misclassified", e
    assert e["source_track_id"] is not None and e["flag_reason"]
    print("test 7 OK:", e["card"], e["attribution"], "reason=", e["flag_reason"])


def test_8_real_enemy_play_not_suppressed():
    t = OpponentPlayTracker()
    # A genuine enemy giant with NO recent ally vanish nearby -- must still play.
    events = run_frames(t, [
        (0.1, [("g1", "giant", (800, 800, 860, 860), "enemy")], []),
        (0.6, [], []),
    ])
    assert len(events) == 1 and events[0]["attribution"] == "played", events
    print("test 8 OK:", events[0]["card"], events[0]["attribution"])


def test_9_archers_burst():
    t = OpponentPlayTracker()
    events = run_frames(t, [
        (0.1, [("a1", "archer", (100, 100, 120, 120), "enemy"),
               ("a2", "archer", (110, 100, 130, 120), "enemy")], []),
        (0.6, [], []),
    ])
    assert len(events) == 1, events
    e = events[0]
    assert e["attribution"] == "played" and e["count"] == 2 and e["card_slug"] == "archers", e
    print("test 9 OK:", e["card"], e["attribution"], e["count"], e["card_slug"], "cost=", e["elixir_cost"])


def test_10_own_play_collision_on_our_side():
    t = OpponentPlayTracker()
    # We "played" giant 0.4s before a new enemy giant appears at a position
    # at/past the river (our side) -- should be suppressed as misclassified.
    # Same track_id reused across both calls -- one continuously-tracked
    # unit, not two separate new ones.
    own_plays = [("giant", 20.0)]
    g = unit("giant", (150, 900, 210, 960), "enemy")
    events = t.update([g], now=20.4, own_recent_plays=own_plays)
    events += t.update([g], now=20.95, own_recent_plays=own_plays)  # closes the grouping window
    assert len(events) == 1, events
    e = events[0]
    assert e["attribution"] == "misclassified", e
    print("test 10 OK:", e["card"], e["attribution"], "reason=", e["flag_reason"])


def test_11_mirror_deck_not_suppressed():
    t = OpponentPlayTracker()
    own_plays = [("giant", 20.0)]
    # Same card, same tight timing window, but position is CLEARLY on the
    # opponent's own side (well above the river) -- must NOT be suppressed,
    # this is exactly the mirror-deck scenario Duncan flagged.
    g = unit("giant", (400, 220, 460, 280), "enemy")
    events = t.update([g], now=20.4, own_recent_plays=own_plays)
    events += t.update([g], now=20.95, own_recent_plays=own_plays)
    assert len(events) == 1, events
    e = events[0]
    assert e["attribution"] == "played", e
    print("test 11 OK (mirror-deck case correctly NOT suppressed):", e["card"], e["attribution"])


def test_12_flicker_while_still_present():
    """The real bug from the second live test (2026-08-16): an ally track
    that goes invisible=False but stays IN tracked_units the whole time
    (unit_tracker's own grace period), same track_id throughout -- a naive
    set-membership diff never sees this as a 'vanish' at all, since the
    track_id never leaves the dict. Meanwhile a separate new enemy track of
    the same class gets born during that invisible window."""
    t = OpponentPlayTracker()
    ally_id = 1
    enemy_id = 99

    def frame(ally_visible, include_enemy):
        units = [{"track_id": ally_id, "name": "giant", "side": "ally",
                  "bbox": (300, 1050, 450, 1180), "age": 20, "visible": ally_visible, "missed_frames": 0}]
        if include_enemy:
            units.append({"track_id": enemy_id, "name": "giant", "side": "enemy",
                           "bbox": (295, 1040, 430, 1165), "age": 2, "visible": True, "missed_frames": 0})
        return units

    now = 0.0
    events = []
    events += t.update(frame(True, False), now := now + 0.1)
    events += t.update(frame(True, False), now := now + 0.1)
    events += t.update(frame(False, False), now := now + 0.1)  # ally flickers invisible
    events += t.update(frame(False, True), now := now + 0.1)   # "new" enemy appears during the flicker
    events += t.update(frame(False, True), now := now + 0.1)
    events += t.update(frame(True, True), now := now + 0.1)    # ally becomes visible again, same track_id
    events += t.update(frame(True, True), now := now + 0.6)    # let the grouping window close

    giant_events = [e for e in events if e["card"] == "giant"]
    assert len(giant_events) == 1, giant_events
    e = giant_events[0]
    assert e["attribution"] == "misclassified", e
    print("test 12 OK:", e["card"], e["attribution"], "reason=", e["flag_reason"])


OUR_DECK = ["archers", "arrows", "fireball", "giant", "knight", "minions", "mini_pekka", "musketeer"]


def test_13_ally_deck_gate_rejects_impossible_card():
    """Real bug from the third live-match log: an 'ally' goblin (goblin
    isn't even in the player's deck -- impossible to genuinely be ours) was
    being trusted as a legitimate flip source. With own_deck supplied, an
    'ally' reading of a card we could never own must never suppress a real
    enemy play."""
    t = OpponentPlayTracker(own_deck=OUR_DECK)
    events = run_frames(t, [
        (0.1, [("gob_ally", "goblin", (300, 900, 360, 960), "ally")], []),
        (0.5, [], []),
        (0.1, [], ["gob_ally"]),
        (0.1, [("gob_enemy", "goblin", (305, 905, 365, 965), "enemy")], []),
        (0.6, [], []),
    ])
    goblin_events = [e for e in events if e["card"] == "goblin"]
    assert len(goblin_events) == 1, goblin_events
    assert goblin_events[0]["attribution"] != "misclassified", goblin_events[0]
    print("test 13 OK: goblin NOT misclassified via impossible-card ally reading:", goblin_events[0]["attribution"])


def test_14_ally_position_gate_rejects_enemy_territory():
    """Real bug from the third live-match log: a genuinely ENEMY giant got
    mislabeled 'ally' by side.onnx while sitting deep in the OPPONENT's own
    territory (bbox y well above the river) -- that bogus 'ally' reading
    then wrongly explained away every real subsequent enemy giant
    appearance nearby as 'misclassified'. A fresh 'ally' track positioned
    clearly on the opponent's side must never be trusted as a real vanish
    source, regardless of card identity."""
    t = OpponentPlayTracker(own_deck=OUR_DECK)
    events = run_frames(t, [
        (0.1, [("g_ally_bogus", "giant", (150, 300, 280, 420), "ally")], []),  # y=300-420, opponent's side
        (0.5, [], []),
        (0.1, [], ["g_ally_bogus"]),
        (0.1, [("g_enemy", "giant", (155, 305, 285, 425), "enemy")], []),
        (0.6, [], []),
    ])
    giant_events = [e for e in events if e["card"] == "giant"]
    assert len(giant_events) == 1, giant_events
    assert giant_events[0]["attribution"] == "played", giant_events[0]
    print("test 14 OK: real enemy giant NOT suppressed by a bogus enemy-territory 'ally' reading:",
          giant_events[0]["attribution"])


def test_15_building_reacquisition():
    """Real bug from the fourth live-match log: the same physical Goblin
    Hut got read as 'played' three separate times (three different
    track_ids, bboxes within ~5px of each other) -- its track kept getting
    lost and re-acquired. Only the FIRST appearance should be 'played';
    later ones at nearly the same position should be 'reacquired'."""
    t = OpponentPlayTracker()
    events = run_frames(t, [
        (0.1, [("hut1", "goblin_hut", (155, 519, 276, 628))], []),
        (0.6, [], []),  # first play resolves
        (2.0, [], ["hut1"]),  # track lost (occlusion, confidence dip, etc.)
        (0.1, [("hut2", "goblin_hut", (156, 515, 280, 620))], []),  # re-acquired, nearly same spot
        (0.6, [], []),
    ])
    hut_events = [e for e in events if e["card"] == "goblin_hut"]
    assert len(hut_events) == 2, hut_events
    assert hut_events[0]["attribution"] == "played", hut_events[0]
    assert hut_events[1]["attribution"] == "reacquired", hut_events[1]
    print("test 15 OK: first hut =", hut_events[0]["attribution"], ", second =", hut_events[1]["attribution"])


def test_16_building_far_away_is_a_real_second_play():
    """A second building of the same class far from the first one's
    position IS a genuine second play (e.g. a real second Goblin Hut) --
    must not be swallowed as a reacquisition just because the class matches."""
    t = OpponentPlayTracker()
    events = run_frames(t, [
        (0.1, [("hut1", "goblin_hut", (155, 519, 276, 628))], []),
        (0.6, [], []),
        (2.0, [], ["hut1"]),
        (0.1, [("hut2", "goblin_hut", (600, 900, 700, 1000))], []),  # far away -- real second hut
        (0.6, [], []),
    ])
    hut_events = [e for e in events if e["card"] == "goblin_hut"]
    assert len(hut_events) == 2, hut_events
    assert hut_events[0]["attribution"] == "played" and hut_events[1]["attribution"] == "played", hut_events
    print("test 16 OK: distant second hut correctly treated as a real second play")


def test_17_building_self_flicker_not_reacquisition():
    """Real bug caught while implementing test 15/16: a brand-new building
    can flicker visible True->False->True WITHIN its own life, before its
    own candidate group even closes (GROUPING_WINDOW_SECONDS gives it
    time to) -- that self-flicker gets recorded as a 'vanish' just like
    any other, and without excluding a group's own track_id, a genuine
    first-ever building appearance matched against ITSELF and got wrongly
    reported 'reacquired' instead of 'played'."""
    t = OpponentPlayTracker()
    hut_id = 200
    now = 0.0

    def frame(visible):
        return [{"track_id": hut_id, "name": "goblin_hut", "side": "enemy",
                  "bbox": (162, 381, 278, 489), "age": 5, "visible": visible, "missed_frames": 0}]

    events = []
    events += t.update(frame(True), now := now + 0.1)
    events += t.update(frame(True), now := now + 0.1)
    events += t.update(frame(True), now := now + 0.1)
    events += t.update(frame(False), now := now + 0.1)  # self-flicker, same track_id
    events += t.update(frame(True), now := now + 0.1)
    events += t.update(frame(True), now := now + 0.6)  # let the grouping window close

    hut_events = [e for e in events if e["card"] == "goblin_hut"]
    assert len(hut_events) == 1, hut_events
    assert hut_events[0]["attribution"] == "played", hut_events[0]
    print("test 17 OK: self-flickering brand-new hut correctly 'played', not 'reacquired'")


def test_18_concurrent_ally_duplicate():
    """Real bug from the fifth live-match log: an ally musketeer stayed
    CONTINUOUSLY visible the whole time a nearby 'enemy' musketeer track
    was born and lived -- it never vanished, so the vanish-based ally-flip
    check could never catch this no matter how tuned. Two simultaneous
    conflicting reads of one physical unit, not a flip-then-vanish."""
    t = OpponentPlayTracker(own_deck=OUR_DECK)
    ally = {"track_id": 47, "name": "musketeer", "side": "ally",
            "bbox": (188, 519, 252, 607), "age": 50, "visible": True, "missed_frames": 0}

    now = 0.0
    events = []
    events += t.update([ally], now := now + 0.1)
    events += t.update([ally], now := now + 0.1)
    events += t.update([ally], now := now + 0.1)
    enemy = {"track_id": 57, "name": "musketeer", "side": "enemy",
             "bbox": (185, 576, 248, 657), "age": 2, "visible": True, "missed_frames": 0}
    events += t.update([ally, enemy], now := now + 0.1)  # ally still alive AND enemy appears nearby
    events += t.update([ally, enemy], now := now + 0.1)
    events += t.update([ally, enemy], now := now + 0.6)  # let the grouping window close

    mk_events = [e for e in events if e["card"] == "musketeer"]
    assert len(mk_events) == 1, mk_events
    assert mk_events[0]["attribution"] == "misclassified", mk_events[0]
    print("test 18 OK:", mk_events[0]["card"], mk_events[0]["attribution"], "reason=", mk_events[0]["flag_reason"])


def test_19_concurrent_duplicate_age_gate():
    """A freshly-confirmed 'ally' reading (low age, i.e. plausibly noise
    itself) sitting near a new enemy of the same class must NOT trigger
    the concurrent-duplicate suppression -- only an ally that's been
    stably tracked for a while is trustworthy enough to use as evidence."""
    t = OpponentPlayTracker(own_deck=OUR_DECK)
    young_ally = {"track_id": 47, "name": "musketeer", "side": "ally",
                  "bbox": (188, 519, 252, 607), "age": 2, "visible": True, "missed_frames": 0}
    # Enemy track's own age is deliberately >= PLAY_PERSISTENCE_MIN_AGE so
    # this test isolates the ally age-gate specifically, not conflated
    # with the separate persistence check added later the same day.
    enemy = {"track_id": 57, "name": "musketeer", "side": "enemy",
             "bbox": (185, 576, 248, 657), "age": 6, "visible": True, "missed_frames": 0}
    events = t.update([young_ally, enemy], now=0.1)
    events += t.update([young_ally, enemy], now=0.7)
    mk_events = [e for e in events if e["card"] == "musketeer"]
    assert len(mk_events) == 1 and mk_events[0]["attribution"] == "played", mk_events
    print("test 19 OK: low-age ally correctly did NOT suppress the enemy read:", mk_events[0]["attribution"])


def test_20_spawn_source_survives_long_building_gap():
    """Real bug from the sixth live-match log: a goblin_hut lost live
    tracking for ~8.2s (just past the old 8.0s reacquisition window) right
    as it spawned a spear_goblin -- the spawn-source lookup only checked
    currently-alive tracks, never the building's own recently-vanished
    record, so the spawn got misread as a fresh play. Reproduces the exact
    gap length; must resolve as spawned/goblin_hut now."""
    t = OpponentPlayTracker(own_deck=OUR_DECK)
    events = run_frames(t, [
        (0.1, [("hut", "goblin_hut", (157, 514, 280, 621))], []),
        (0.5, [], []),
        (0.1, [], ["hut"]),  # hut track fully lost
        (8.1, [], []),  # 8.2s gap, past the old 8.0s window
        (0.1, [("sg", "spear_goblin", (160, 520, 200, 560))], []),
        (0.6, [], []),
    ])
    sg_events = [e for e in events if e["card"] == "spear_goblin"]
    assert len(sg_events) == 1, sg_events
    assert sg_events[0]["attribution"] == "spawned" and sg_events[0]["source_card"] == "goblin_hut", sg_events[0]
    print("test 20 OK:", sg_events[0]["card"], sg_events[0]["attribution"], "source=", sg_events[0]["source_card"])


def test_21_phantom_detection_rejected_by_persistence():
    """Real bug from the seventh live-match log, found via an EXHAUSTIVE
    ground-truth comparison (game_knowledge/compare_ground_truth.py) --
    the first time every real play in a match was accounted for, not just
    an impression-level review. Several short-lived 'giant' tracks (2-4
    confirmed frames, then gone for good, never seen again, no explaining
    ally/spawner context at all) generated false 'played' events. A real
    play's track keeps accumulating age; a noise blip does not."""
    t = OpponentPlayTracker(own_deck=OUR_DECK)
    phantom_id = 18
    now = 0.0

    def frame(age, visible):
        return [{"track_id": phantom_id, "name": "giant", "side": "enemy",
                  "bbox": (159, 574, 261, 685), "age": age, "visible": visible, "missed_frames": 0}]

    events = []
    events += t.update(frame(2, True), now := now + 0.1)
    events += t.update(frame(3, True), now := now + 0.1)
    events += t.update(frame(3, False), now := now + 0.1)  # dies for good right after, age frozen at 3
    events += t.update(frame(3, False), now := now + 0.6)  # let the grouping window close

    giant_events = [e for e in events if e["card"] == "giant"]
    assert len(giant_events) == 1, giant_events
    e = giant_events[0]
    assert e["attribution"] == "uncertain" and "persist" in e["flag_reason"], e
    print("test 21 OK: phantom short-lived giant downgraded to", e["attribution"], "-", e["flag_reason"])


def test_22_persistent_real_play_still_confirmed():
    """A genuinely real play whose track keeps growing in age (the same
    shape the one real giant in that match actually had, age 7-8 by its
    own group's close time) must still resolve as 'played' -- the
    persistence check shouldn't over-suppress real, healthy tracks."""
    t = OpponentPlayTracker(own_deck=OUR_DECK)
    real_id = 32
    now = 0.0

    def frame(age):
        return [{"track_id": real_id, "name": "giant", "side": "enemy",
                  "bbox": (141, 509, 286, 655), "age": age, "visible": True, "missed_frames": 0}]

    events = []
    for age in range(2, 9):
        events += t.update(frame(age), now := now + 0.1)
    events += t.update(frame(9), now := now + 0.5)  # let the grouping window close

    giant_events = [e for e in events if e["card"] == "giant"]
    assert len(giant_events) == 1, giant_events
    assert giant_events[0]["attribution"] == "played", giant_events[0]
    print("test 22 OK: persistent real giant still resolves as", giant_events[0]["attribution"])


if __name__ == "__main__":
    test_1_common_single_card()
    test_2_minions_burst()
    test_3_golem_split()
    test_4_tombstone_trickle()
    test_5_skeletons_vs_army()
    test_6_two_of_three_minions_now_resolves_played()
    test_6b_a_single_minion_is_still_below_the_loosened_floor()
    test_two_of_four_goblins_resolves_played()
    test_two_spear_goblins_resolves_played_as_the_real_card()
    test_musketeer_single_and_triple()
    test_7_side_flip_misclassified()
    test_8_real_enemy_play_not_suppressed()
    test_9_archers_burst()
    test_10_own_play_collision_on_our_side()
    test_11_mirror_deck_not_suppressed()
    test_12_flicker_while_still_present()
    test_13_ally_deck_gate_rejects_impossible_card()
    test_14_ally_position_gate_rejects_enemy_territory()
    test_15_building_reacquisition()
    test_16_building_far_away_is_a_real_second_play()
    test_17_building_self_flicker_not_reacquisition()
    test_18_concurrent_ally_duplicate()
    test_19_concurrent_duplicate_age_gate()
    test_20_spawn_source_survives_long_building_gap()
    test_21_phantom_detection_rejected_by_persistence()
    test_22_persistent_real_play_still_confirmed()
    print()
    print("ALL TESTS PASSED")
