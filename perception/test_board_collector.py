"""Tests for the detector-retrain data loop: perception/board_collector.py
(banking bound/struggle crops) and training/prepare_board_data.py's card->class
mapping (the underscore->hyphen taxonomy lookup + gap-card hold-aside).

Run: python -m perception.test_board_collector
"""

import importlib.util
import json
import os
import tempfile

import numpy as np

from perception.board_collector import BoardCropCollector, STRUGGLE_MIN_AGE

W, H = 900, 1600


def _u(tid, name, cx, cy, side="ally", bound=False, struggle=False, age=10, w=40, h=60):
    return {"id": tid, "name": name, "side": side, "bound": bound, "struggle": struggle,
            "off_deck": struggle, "age": age,
            "bbox": (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)}


def _load_prepare():
    """Import training/prepare_board_data.py by path (training/ isn't a package)."""
    path = os.path.join(os.path.dirname(__file__), "..", "training", "prepare_board_data.py")
    spec = importlib.util.spec_from_file_location("prepare_board_data", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_collector():
    frame = np.zeros((H, W, 3), np.uint8)
    frame[:] = 40
    qd = tempfile.mkdtemp(prefix="board_q_")
    col = BoardCropCollector(queue_dir=qd)
    col.set_decks(enemy_deck=["hog_rider", "musketeer", "fireball", "ice_spirit",
                              "skeletons", "cannon", "zap", "dart_goblin"],
                  friendly_deck=["knight", "berserker", "archers", "goblins",
                                 "log", "tesla", "fireball", "dart_goblin"])
    units = [
        _u(1, "berserker", 450, 1100, side="ally", bound=True),          # bound -> bank (ground truth)
        _u(2, "mortar", 400, 500, side="enemy", struggle=True, age=10),  # struggle, old enough -> bank
        _u(3, "knight", 500, 1000, side="ally"),                         # plain -> skip
        _u(4, "wizard", 300, 400, side="enemy", struggle=True, age=1),   # struggle too young -> skip
    ]
    col.observe(frame, units, game_id="g1")
    stems = sorted(f[:-4] for f in os.listdir(qd) if f.endswith(".png"))
    assert col.saved == 2, f"expected 2 banked, got {col.saved} ({stems})"
    props = {}
    for s in stems:
        m = json.load(open(os.path.join(qd, s + ".json"), encoding="utf-8"))
        props[m["proposed"]] = m
    assert set(props) == {"berserker", "mortar"}, props.keys()
    # bound crop: candidates = the unit's (ally/bottom=friendly) deck
    assert props["berserker"]["kind"] == "bound" and props["berserker"]["bound"] is True
    assert "berserker" in props["berserker"]["candidates"]           # friendly deck
    # struggle crop: enemy side -> enemy deck candidates
    assert props["mortar"]["kind"] == "struggle"
    assert "hog_rider" in props["mortar"]["candidates"]              # enemy deck
    # crop_box present + normalized
    cb = props["berserker"]["crop_box"]
    assert len(cb) == 4 and all(0 <= v <= 1 for v in cb)

    # once-per-track: a second identical frame banks nothing new
    col.observe(frame, units, game_id="g1")
    assert col.saved == 2, "should not re-bank the same tracks"
    print(f"test collector OK: bound+struggle banked, plain/young skipped, once-per-track "
          f"(STRUGGLE_MIN_AGE={STRUGGLE_MIN_AGE})")


def test_ingest_mapping():
    prep = _load_prepare()
    if not os.path.isfile(prep.V4_YAML):
        print("test ingest mapping SKIPPED (no cr_combined_v4.yaml -- datasets are gitignored)")
        return
    name_to_idx = prep._name_to_idx(prep.V4_YAML)
    assert "archer" in name_to_idx and "spear-goblin" in name_to_idx
    # plural card slug -> singular detector class -> taxonomy index (underscore->hyphen)
    assert prep._class_idx_for_card("goblins", name_to_idx) == name_to_idx["goblin"]
    assert prep._class_idx_for_card("spear_goblins", name_to_idx) == name_to_idx["spear-goblin"]
    assert prep._class_idx_for_card("dart_goblin", name_to_idx) == name_to_idx["dart-goblin"]
    # confusable pair both resolve (the retrain target)
    assert prep._class_idx_for_card("archers", name_to_idx) == name_to_idx["archer"]
    assert prep._class_idx_for_card("musketeer", name_to_idx) == name_to_idx["musketeer"]
    # gap card (no detector class) -> held aside, not a wrong index
    assert prep._class_idx_for_card("berserker", name_to_idx) is None
    assert prep._class_idx_for_card("ronin", name_to_idx) is None
    print("test ingest mapping OK: card->class index (underscore->hyphen), gap cards held aside")


def main():
    test_collector()
    test_ingest_mapping()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
