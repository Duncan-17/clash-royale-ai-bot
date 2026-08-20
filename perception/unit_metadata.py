"""
Per-unit metadata not needed for detection itself, but needed for
downstream reasoning about detections. Two fields:
- TRANSPORT (AIR/GROUND) — used by unit_detector.py's red-zone filtering
  (config/red_zones.py) to know whether the river zone applies to a given
  unit (ground troops can't be in the river; flying troops legitimately
  can).
- TARGET_TYPE (added 2026-08-19/20) — used by perception/unit_tracker.py's
  reacquisition search-radius shaping (see TARGET_TYPE's own comment and
  that module's docstring): whether a unit predictably beelines for a
  tower (targets_buildings_only()) or can retarget onto the nearest enemy
  at any moment, which changes how confidently its recent motion predicts
  where it'll be after a tracking gap.

Sourced from vendor/build-a-bot's namespaces/units.py (MIT licensed —
same provenance basis as the detection model itself, see Project_Status).
Cross-checked against known game facts before trusting it wholesale, not
copied blind: one real error found and corrected — their data lists
Ice Wizard as Transport.AIR, but Ice Wizard is a ground unit in the
actual game (it walks; it has never been a flier). Fixed here rather
than propagated. If anything else here later turns out wrong, it means
this cross-check wasn't exhaustive, not that the correction methodology
was skipped.

Only covers the 97 classes unit_detector.py can currently recognize —
see Project_Status for the known 5-card gap (Berserker, Ronin, Spirit
Empress, Suspicious Bush, Void), which by definition can't appear here
either until that gap is closed.
"""

TRANSPORT = {
    "archer": "ground",
    "archer_queen": "ground",
    "baby_dragon": "air",
    "balloon": "air",
    "bandit": "ground",
    "barbarian": "ground",
    "barbarian_hut": "ground",
    "bat": "air",
    "battle_healer": "ground",
    "battle_ram": "ground",
    "bomb_tower": "ground",
    "bomber": "ground",
    "bowler": "ground",
    "brawler": "ground",
    "cannon": "ground",
    "cannon_cart": "ground",
    "dark_prince": "ground",
    "dart_goblin": "ground",
    "electro_dragon": "air",
    "electro_giant": "ground",
    "electro_spirit": "ground",
    "electro_wizard": "ground",
    "elite_barbarian": "ground",
    "elixir_collector": "ground",
    "elixir_golem_large": "ground",
    "elixir_golem_medium": "ground",
    "elixir_golem_small": "ground",
    "executioner": "ground",
    "fire_spirit": "ground",
    "firecracker": "ground",
    "fisherman": "ground",
    "flying_machine": "air",
    "furnace": "ground",
    "giant": "ground",
    "giant_skeleton": "ground",
    "giant_snowball": "ground",
    "goblin": "ground",
    "goblin_cage": "ground",
    "goblin_drill": "ground",
    "goblin_hut": "ground",
    "golden_knight": "ground",
    "golem": "ground",
    "golemite": "ground",
    "guard": "ground",
    "heal_spirit": "ground",
    "hog": "ground",
    "hog_rider": "ground",
    "hunter": "ground",
    "ice_golem": "ground",
    "ice_spirit": "ground",
    "ice_wizard": "ground",  # corrected from build-a-bot's "air" — see module docstring
    "inferno_dragon": "air",
    "inferno_tower": "ground",
    "knight": "ground",
    "lava_hound": "air",
    "lava_pup": "air",
    "little_prince": "ground",
    "lumberjack": "ground",
    "magic_archer": "ground",
    "mega_knight": "ground",
    "mega_minion": "air",
    "mighty_miner": "ground",
    "miner": "ground",
    "minion": "air",
    "minipekka": "ground",
    "monk": "ground",
    "mortar": "ground",
    "mother_witch": "ground",
    "musketeer": "ground",
    "night_witch": "ground",
    "pekka": "ground",
    "phoenix_egg": "ground",
    "phoenix_large": "air",
    "phoenix_small": "air",
    "prince": "ground",
    "princess": "ground",
    "ram_rider": "ground",
    "rascal_boy": "ground",
    "rascal_girl": "ground",
    "royal_ghost": "ground",
    "royal_giant": "ground",
    "royal_guardian": "ground",
    "royal_hog": "ground",
    "royal_recruit": "ground",
    "skeleton": "ground",
    "skeleton_dragon": "air",
    "skeleton_king": "ground",
    "sparky": "ground",
    "spear_goblin": "ground",
    "tesla": "ground",
    "tombstone": "ground",
    "valkyrie": "ground",
    "wall_breaker": "ground",
    "witch": "ground",
    "wizard": "ground",
    "x_bow": "ground",
    "zappy": "ground",
}


def is_air(name: str) -> bool:
    return TRANSPORT.get(name, "ground") == "air"


# Per-unit target behavior -- "ground" (default melee/ranged units), "all" (can hit air
# too), "buildings" (win-condition-style units that beeline for the nearest tower/building
# and ignore troops entirely), or None (non-combat: spawner buildings like Tombstone/
# Goblin Hut/Furnace, the Elixir Collector, and non-attacking entities like a Phoenix Egg
# or the Giant Snowball spell that's a stray detectable class -- see CLAUDE.md-adjacent
# note in game_knowledge/card_reference.json's own entry for it).
#
# Added 2026-08-19/20 for perception/unit_tracker.py's reacquisition search-radius shaping
# (see that module's docstring): a "buildings"-target unit walks a predictable, roughly
# straight line toward a tower, so a lost track can be searched for with a TIGHT,
# DIRECTIONAL radius along its last known heading; a "ground"/"all"-target unit can
# retarget to the nearest enemy at any moment (including via knockback from cards like
# The Log/Giant Snowball/Barbarian Barrel), so its search needs to stay WIDE and non-
# directional instead -- per Duncan's own correction that most units are NOT predictable
# tower-beeliners, only a narrow subset of win-conditions are.
#
# Same source/provenance as TRANSPORT above (vendor/build-a-bot's namespaces/units.py,
# MIT licensed) -- NOT independently cross-checked against every one of the 97 classes the
# way TRANSPORT's Ice Wizard entry was (that correction was found by chance, not an
# exhaustive audit) -- treat as a reasonable starting point, not gospel; a wrong entry here
# only degrades search-radius SHAPE (tight-vs-wide), not side/team correctness, which
# depends on it least of anywhere this module's data is used.
#
# One real error already found and corrected while wiring this up (2026-08-19/20): vendor
# listed Giant as Target.GROUND, not Target.BUILDINGS -- about as close to definitional
# Clash Royale knowledge as exists (Giant is THE textbook building-only tank), so treated
# as a vendor bug and fixed here rather than propagated, same call as the Ice Wizard
# transport fix above. NOT re-verified against every other entry, though -- "giant_skeleton"
# in particular looks worth double-checking (kept at vendor's "ground" value here, since
# unlike Giant this one isn't something I'm confident enough about to silently override).
# Flagged for Duncan to spot-check alongside game_knowledge/card_reference.json's
# target_type column, which was populated from this exact same vendor data.
TARGET_TYPE = {
    "archer": "all",
    "archer_queen": "all",
    "baby_dragon": "all",
    "balloon": "buildings",
    "bandit": "ground",
    "barbarian": "ground",
    "barbarian_hut": None,
    "bat": "all",
    "battle_healer": "ground",
    "battle_ram": "buildings",
    "bomb_tower": "ground",
    "bomber": "ground",
    "bowler": "ground",
    "brawler": "ground",
    "cannon": "ground",
    "cannon_cart": "ground",
    "dark_prince": "ground",
    "dart_goblin": "all",
    "electro_dragon": "all",
    "electro_giant": "buildings",
    "electro_spirit": "all",
    "electro_wizard": "all",
    "elite_barbarian": "ground",
    "elixir_collector": None,
    "elixir_golem_large": "buildings",
    "elixir_golem_medium": "buildings",
    "elixir_golem_small": "buildings",
    "executioner": "all",
    "fire_spirit": "all",
    "firecracker": "all",
    "fisherman": "ground",
    "flying_machine": "all",
    "furnace": None,
    "giant": "buildings",  # corrected from build-a-bot's "ground" -- see comment above
    "giant_skeleton": "ground",  # NOT independently confirmed -- see comment above, may also be wrong
    "giant_snowball": None,  # spell, only detectable by a training-data mistake -- see
    # game_knowledge/card_reference.json's own note on this class
    "goblin": "ground",
    "goblin_cage": None,
    "goblin_drill": "ground",
    "goblin_hut": None,
    "golden_knight": "ground",
    "golem": "buildings",
    "golemite": "buildings",
    "guard": "ground",
    "heal_spirit": "all",
    "hog": "buildings",
    "hog_rider": "buildings",
    "hunter": "all",
    "ice_golem": "buildings",
    "ice_spirit": "all",
    "ice_wizard": "all",
    "inferno_dragon": "all",
    "inferno_tower": None,
    "knight": "ground",
    "lava_hound": "buildings",
    "lava_pup": "buildings",
    "little_prince": "all",
    "lumberjack": "ground",
    "magic_archer": "all",
    "mega_knight": "ground",
    "mega_minion": "all",
    "mighty_miner": "ground",
    "miner": "ground",
    "minion": "all",
    "minipekka": "ground",
    "monk": "ground",
    "mortar": "ground",
    "mother_witch": "all",
    "musketeer": "all",
    "night_witch": "all",
    "pekka": "all",
    "phoenix_egg": None,
    "phoenix_large": "all",
    "phoenix_small": "all",
    "prince": "ground",
    "princess": "all",
    "ram_rider": "buildings",
    "rascal_boy": "ground",
    "rascal_girl": "all",
    "royal_ghost": "ground",
    "royal_giant": "buildings",
    "royal_guardian": "ground",
    "royal_hog": "buildings",
    "royal_recruit": "ground",
    "skeleton": "ground",
    "skeleton_dragon": "all",
    "skeleton_king": "ground",
    "sparky": "ground",
    "spear_goblin": "all",
    "tesla": "all",
    "tombstone": None,
    "valkyrie": "ground",
    "wall_breaker": "buildings",
    "witch": "all",
    "wizard": "all",
    "x_bow": "ground",
    "zappy": "ground",
}


def targets_buildings_only(name: str) -> bool:
    """True for win-condition-style units that beeline for the nearest tower/building and
    never retarget onto troops -- the one case where a track's recent motion vector is a
    genuinely reliable predictor of where it'll be next. Defaults False (the common,
    unpredictable-retargeting case) for anything not in TARGET_TYPE at all."""
    return TARGET_TYPE.get(name) == "buildings"
