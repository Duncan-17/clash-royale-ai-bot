"""
Per-unit metadata not needed for detection itself, but needed for
downstream reasoning about detections — currently just AIR/GROUND
transport, used by unit_detector.py's red-zone filtering (config/
red_zones.py) to know whether the river zone applies to a given unit
(ground troops can't be in the river; flying troops legitimately can).

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
