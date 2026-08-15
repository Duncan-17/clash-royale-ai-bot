"""
Loads config/game_constants.yaml into plain Python values.

These are fixed game rules (elixir timing, phase boundaries) that no API
exposes — see the yaml file's own header for sourcing/caveats.
"""

import os

import yaml

_PATH = os.path.join(os.path.dirname(__file__), "game_constants.yaml")

with open(_PATH) as f:
    _data = yaml.safe_load(f)

ELIXIR_REGEN_SECONDS = _data["elixir"]["regen_seconds"]
PHASE_BOUNDARIES_SECONDS = _data["elixir"]["phase_boundaries_seconds"]


if __name__ == "__main__":
    print("Elixir regen rates (seconds per point):")
    for phase, seconds in ELIXIR_REGEN_SECONDS.items():
        print(f"  {phase:20s} {seconds}")
    print("\nPhase boundaries (seconds into the match):")
    for phase, seconds in PHASE_BOUNDARIES_SECONDS.items():
        print(f"  {phase:24s} {seconds}")
