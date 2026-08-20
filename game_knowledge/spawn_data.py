"""
Static reference data for opponent-side unit spawn attribution — Phase 4,
part 2 (opponent play detection).

Two distinct kinds of "a new enemy unit appeared, but it wasn't a fresh
hand play" this table exists to explain:

1. SPAWN_TABLE — a card that's already on the board produces MORE units as
   a side effect (dies and splits, or produces units periodically while
   alive). See Project_Status's "Unit-creation/spawn attribution" section
   for the original problem writeup — this is that table, finally built.

2. CARD_BURST_COUNT — a SINGLE hand play produces multiple units of the
   same class all at once (Minions -> 3x "minion", Goblins -> 4x "goblin",
   etc.). Different problem from #1: these units aren't a byproduct of some
   OTHER card, they ARE the played card, just rendered as several
   simultaneous detections instead of one. Without this table,
   opponent_play_tracker.py would miscount one real play as several.

Both tables are hand-curated Clash Royale game knowledge (same spirit as
config/game_constants.yaml's elixir timing or config/cards.json's costs),
NOT derived from anything in this codebase. Corrected directly by Duncan
multiple times as real matches gave better evidence than quick recall did
— notably Goblin Hut, corrected TWICE: a first draft guessed Spear
Goblins, Duncan corrected it to Goblins before any code was written, then
a third live match's direct real-time observation (plus the log itself:
a spear_goblin consistently appeared within ~85px/~0.4s of the hut) showed
the original guess had actually been right and the "correction" wrong —
now Spear Goblins again. Tombstone -> skeleton is still believed correct
(not re-contradicted by any log since). All of CARD_BURST_COUNT was
confirmed correct by Duncan (a strong CR player, per the philosophy doc's
own byline) before being written in. Still: balance patches can change
exact counts over time, and quick recall under real-time observation can
be wrong even for a strong player, so if opponent_play_tracker.py's output
ever looks wrong for a specific card, check here first before assuming
the tracker logic is at fault — but also don't treat a single correction
as necessarily final if a later match's evidence disagrees.

Known, permanent gap: "goblin_giant" is not one of unit_detector.py's 97
detected classes (confirmed by reading CLASS_NAMES directly) — the model
can't see Goblin Giant itself, only its on-death spear_goblin spawns. That
specific attribution case can never find its source track by proximity,
since nothing is ever tracked to check proximity against. Not fixable
without a different/retrained detection model — flagged, not solved here.
"""

# spawner class -> {"on_death": [(spawned_class, count), ...]} or
#                  {"periodic": [(spawned_class, count), ...]}
SPAWN_TABLE = {
    "golem": {"on_death": [("golemite", 2)]},
    "elixir_golem_large": {"on_death": [("elixir_golem_medium", 2)]},
    "elixir_golem_medium": {"on_death": [("elixir_golem_small", 2)]},
    "lava_hound": {"on_death": [("lava_pup", 6)]},
    "phoenix_large": {"on_death": [("phoenix_egg", 1)]},
    "witch": {"periodic": [("skeleton", 2)]},
    "night_witch": {"periodic": [("bat", 2)]},
    "barbarian_hut": {"periodic": [("barbarian", 2)]},
    "goblin_hut": {"periodic": [("spear_goblin", 1)]},   # corrected AGAIN by Duncan (2026-08-16,
    # third live match) after direct real-time observation — the earlier "not spear_goblin"
    # correction (first live match) turns out to have been the wrong read. Also confirmed by the
    # log itself: a spear_goblin consistently appeared within ~85px and ~0.4s of the hut's own
    # position/appearance, both comfortably inside SPAWN_PROXIMITY_PX/GROUPING_WINDOW_SECONDS.
    "furnace": {"periodic": [("fire_spirit", 2)]},
    "tombstone": {"periodic": [("skeleton", 1)]},  # corrected by Duncan
    # Skeleton King's skeleton summon is ability-triggered (on demand), not
    # truly periodic — this table has no "on demand" concept, so it's
    # modeled as periodic anyway. Timing isn't checked closely regardless
    # (see opponent_play_tracker.py's known limitations); this is an
    # approximation, not a claim that it's actually periodic in-game.
    "skeleton_king": {"periodic": [("skeleton", 1)]},
}

# card's own unit class -> how many units ONE hand-play produces
# simultaneously. Any class not listed here defaults to 1 (the
# overwhelmingly common case — most cards are a single unit).
CARD_BURST_COUNT = {
    "archer": 2,            # Archers -- no standalone single "Archer" card exists
    "minion": 3,           # Minions
    "goblin": 4,            # Goblins
    "skeleton": 3,           # Skeletons — see SKELETON_ARMY_BURST_MIN below
    # for how a much larger burst of the same class is told apart as
    # Skeleton Army instead.
    "barbarian": 5,          # Barbarians
    "elite_barbarian": 2,    # Elite Barbarians — own distinct class, no
    # ambiguity with regular Barbarians.
    "royal_recruit": 6,      # Royal Recruits
    "rascal_boy": 1,         # Rascals — 1 boy + 2 girls, two different
    "rascal_girl": 2,        # classes spawned together in one play.
    "guard": 3,              # Guards
    "bat": 5,                 # Bats
    "fire_spirit": 2,        # Fire Spirits
    "spear_goblin": 3,       # Spear Goblins -- added 2026-08-18. Previously NOT in this table
    # at all, so a genuine Spear Goblins play could never resolve "played" via the normal burst
    # path -- a 2-3 count fell straight to the "unexpected multiples of a single-unit card"
    # branch (uncertain), and a 1-count was silently treated as an ordinary single-unit play
    # (technically harmless for cost lookup, since CARD_SLUG_FOR_BURST below maps the class to
    # the real card regardless of count, but mislabeled the play's own reported count). Also
    # still a genuine AMBIGUOUS_SPAWNED_CLASSES entry (goblin_hut's periodic 1-at-a-time
    # trickle, see SPAWN_TABLE above) -- opponent_play_tracker.py's own burst-vs-spawn-source
    # branch ordering means a real hut trickle (always count=1) still falls through to the
    # spawn-source check below since 1 is below even the loosened _burst_min_count floor.
}
# "musketeer" is deliberately NOT in this table. Unlike every other entry
# above, it's valid at TWO different counts as two different real cards —
# 1 = Musketeer, 3 = Three Musketeers — so a single fixed "expected count"
# doesn't fit this table's shape. opponent_play_tracker.py special-cases it
# instead of using this table (caught before shipping: an early version of
# that tracker treated 3 as musketeer's only valid count, which would have
# flagged every ordinary single Musketeer play as "uncertain").

SKELETON_ARMY_BURST_MIN = 8  # a "skeleton" burst at least this large is
# treated as Skeleton Army (real count ~15) rather than the 3-count
# Skeletons card. Picked well clear of both real values so detection noise
# (a few missed/duplicate skeletons in a dense pile) doesn't flip which one
# gets reported.

# Best-effort unit-class -> card-slug mapping for elixir cost lookups
# (game_knowledge.cycle_tracker.load_card_costs() is keyed by card slug,
# e.g. "minions", "three_musketeers", "mini_pekka" — NOT by
# unit_detector.py's class names, e.g. "minion", "musketeer", "minipekka",
# which are frequently abbreviated, singular/plural-mismatched, or missing
# punctuation/underscores compared to the real card slug). Covers the
# multi-unit special cases (this module already needs to distinguish those
# by count anyway) PLUS single-unit name mismatches found so far
# ("minipekka" vs "mini_pekka" — caught while wiring opponent_play_tracker.py's
# own-play collision check, which needs a real slug match to compare against
# game_knowledge.cycle_tracker.py's own play records, not just cost lookups).
# This is NOT a complete class-name -> card-slug mapping for all 97 detector
# classes — building that is a separate, not-yet-done piece of work.
# Elsewhere, opponent_play_tracker.py falls back to trying the unit class
# name directly against the cost table and accepts elixir_cost=None if that
# fails, rather than guessing.
CARD_SLUG_FOR_BURST = {
    "archer": "archers",
    "minion": "minions",
    "goblin": "goblins",
    "barbarian": "barbarians",
    "elite_barbarian": "elite_barbarians",
    "royal_recruit": "royal_recruits",
    "guard": "guards",
    "rascal_boy": "rascals",
    "rascal_girl": "rascals",
    "bat": "bats",
    "fire_spirit": "fire_spirits",
    "minipekka": "mini_pekka",
    "spear_goblin": "spear_goblins",  # matches cycle_tracker.py's _slug("Spear Goblins")
    # convention (plural, underscored) -- same class of fix as archer->archers/minipekka-
    # >mini_pekka above, added alongside the CARD_BURST_COUNT entry so cost lookup resolves
    # correctly instead of falling back to the unmapped singular "spear_goblin".
}
# musketeer and skeleton are count-sensitive (a different card depending on
# how many appeared at once), handled directly in opponent_play_tracker.py
# rather than in this flat table.

# Classes that are NEVER an independently playable card on their own — only
# ever appear as a spawn byproduct. A new track of one of these is always
# attribution="spawned", even if no specific source can be pinned down.
PURE_BYPRODUCT_CLASSES = {
    "golemite", "elixir_golem_medium", "elixir_golem_small",
    "lava_pup", "phoenix_egg", "phoenix_small",
}

# Classes that are BOTH an independently playable card AND a known spawn
# target of something else — genuinely ambiguous without proximity/timing
# context. Derived directly from SPAWN_TABLE rather than hand-listed again,
# so it can't drift out of sync with the table above.
AMBIGUOUS_SPAWNED_CLASSES = {
    spawned
    for rules in SPAWN_TABLE.values()
    for entries in rules.values()
    for spawned, _count in entries
    if spawned not in PURE_BYPRODUCT_CLASSES
}


def spawners_of(spawned_class: str) -> list:
    """Every (spawner_class, kind, count) that could produce spawned_class —
    kind is "on_death" or "periodic". Empty list if nothing in SPAWN_TABLE
    produces this class."""
    result = []
    for spawner, rules in SPAWN_TABLE.items():
        for kind, entries in rules.items():
            for cls, count in entries:
                if cls == spawned_class:
                    result.append((spawner, kind, count))
    return result
