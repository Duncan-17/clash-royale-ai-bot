# YOLO detector training

Our own Clash Royale unit detector — a standard Ultralytics **YOLOv8l**,
fine-tuned on KataCR's data, replacing the stale `units.onnx` and improving on
stock KataCR. This dir holds the harness + generators; datasets/runs/`*.pt` are
gitignored. See `Project_Status` for the running history.

## Environment (works)
torch 2.11.0+cu128, ultralytics 8.4.123, CUDA on the **RTX 5080 (16GB)**. The
Blackwell `sm_120` setup is done (the `cu128` wheel). A GPU train is ~7 hrs for
~4k images / 100 epochs; ~10k images (v3) is a longer overnight.

## Why standard YOLOv8, not KataCR's model
KataCR's `detector1/2.pt` are a **custom head** (adds an ally/enemy "belonging"
channel) and their training path pulls in **jax** (no good Windows GPU story).
We want KataCR's **DATA**, not their weights. A standard YOLOv8l learns the same
classes, runs on Windows/5080 with plain tooling, and yields a plain `best.pt`.
We lose the native side channel — but the model detects the HP `bar`, and its
**colour gives side** (`perception/bar_reader.side_from_bar_color`), so side is
recovered from the detection, not a separate pixel module.

## Data
- **Synthetic** (`generate_dataset.py`): composites KataCR's ~4,654 pre-cut RGBA
  sprites onto their annotated arena backgrounds → free, perfect YOLO labels.
  Key details baked in: backgrounds resized to KataCR's canonical **568×896** so
  sprite scale matches; units placed in the **playfield only** (no false labels
  on hand/banner UI); **bar+level+clock components attached to units** per
  KataCR's `generation_config` offsets (ally/enemy colour variants), so the
  model detects those natively; spells/objects get no bar.
- **Real** (`prepare_real_data.py`): KataCR's part2 ships **~6,966 real labeled
  gameplay frames** (same 201-class taxonomy, YOLO after dropping their state
  columns). This is the domain-gap fix synthetic can't be — real unit
  appearance, motion, effects, backgrounds. Builds `cr_combined.yaml` = synthetic
  + real.

## Models trained (each `assets/models/cr_yolo_vN/best.pt`, preserved)
- **v1** — synthetic only (units + towers). Proved the loop; better *recall*
  than KataCR live, but no side, no bars/clocks.
- **v2** — added bar/level/clock components → detects them natively; wrapper
  reads bar colour for **side**. Live: friend/foe works, spawn/level visible.
- **v3** (`cr_combined.yaml`, synthetic **+ real**) — prior default, superseded by
  v4. Combined-val mAP50 **0.973** / recall **0.953**. Live (Duncan): "so much
  better". Snapshot `assets/models/cr_yolo_v3/best.pt`.
  ```
  python training/finetune.py --data training/datasets/cr_combined.yaml --name cr_finetune_v3
  ```
- **v4** (`cr_combined_v4.yaml` = **cr_synth_v3 + cr_real**) — **DONE, current
  default (2026-09-10).** cr_synth_v3 adds the **lone level-badge** case
  (`LONE_LEVEL_PROB` in `generate_dataset.py`): before it, `bar-level` was only
  ever pasted alongside a health `bar`, so the model missed a lone badge — the real
  undamaged-enemy case (enemies always show a level, but the health bar only after
  damage). Val: all mAP50 **0.968** / R 0.944 (marginally under v3 only because the
  val now includes lone badges to find); the point metric is **`bar-level` R 0.963 /
  mAP50 0.984** with ~25% lone badges in val. No regression (archer R 0.87, goblin
  0.98, clock 0.985). **Live-confirmed by Duncan** reading levels well in a practice
  game. Snapshot `assets/models/cr_yolo_v4/best.pt`.
  ```
  python training/finetune.py --data training/datasets/cr_combined_v4.yaml --name cr_finetune_v4
  ```

### v3's two residual issues (data-backed)
- **Archers** still weak *live* though val recall (0.86) ≈ goblin (0.86) — a
  residual gap between KataCR's real captures and *Duncan's* emulator (subtle
  archers most sensitive), worsened by **musketeer being 12× over-represented**
  (2,336 vs 184 real instances) skewing the humanoid-ranged boundary. Levers:
  **rebalance** (cap musketeer) + more of Duncan's-emulator real data.
- **goblin↔spear-goblin flip-flop** — near-identical classes wobble frame-to-
  frame; fixed by the **tracker's class debounce**, not the detector.

## Running / rolling back
```
python training/finetune.py --data <data.yaml> --name cr_finetune_vN > training/train_vN.log 2>&1
```
`--batch -1` auto-fits VRAM; `imgsz 896`. Each `--name` writes a separate
`runs/detect/<name>/` and never overwrites another. The inference default is
`perception/yolo_detector.py:DEFAULT_MODEL` — a new model is tested by pointing
at it explicitly and only promoted (copied to `assets/models/`, default flipped)
once it beats the current one.

## Inference wrapper — `perception/yolo_detector.py`
Same `detect()` interface as the other detectors; wired into
`perception/debug_live.py` and `perception/detector_view.py` as `--detector yolo`.
Notable: pads the frame to the training aspect (emulator 900×1600 → 0.634) so
units aren't shrunk; drops detections outside the **arena band** (kills
hand-tray UI → "clock" false positives); reads **side from bar colour**.

## Evaluate — `perception/detector_view.py`
Raw detector output only (no trackers/plays/pixel heuristics) — the honest way
to judge a model. `--detector yolo|katacr`, `--no-ui` (hide towers), `--conf 0.1`
(near-misses), `--static`. Info panel lists units (name/side/conf) + component
tally. Use `model_path=assets/models/cr_yolo_v3/best.pt` to test a new model.

## Next
1. ~~Promote v3~~ → **v4 is promoted** (`DEFAULT_MODEL`), lone-badge fix live-confirmed.
2. **Thin tracker** — DONE (`perception/thin_tracker.py`): identity + class debounce
   (fixes flip-flop) + sticky side + sticky level. Still to do: re-add play detection
   with a deliberate unit→card name mapping (detector: archer/goblin; cards:
   archers/goblins).
3. **Fill the level-digit library** — `detector_view --collect-levels` →
   `review_level_queue` (only 1–4 templates exist; the rest read `Lv?` until collected).
4. **Archer retrain** when worth it: rebalance (cap musketeer's 12× over-rep) +
   Duncan's-emulator real data (also the lever for any residual lone-badge live gap).
5. **Future data engine = SELF-PLAY** (Duncan's call): once the bot plays, it
   knows the card it played → auto-labels its own plays with certainty. TV Royale
   auto-labeling (scaffolded in data/) is the fallback (sparse: one unit/play).

## Duncan's-emulator real data loop (BUILD B+, 2026-09-14)

The "Duncan's-emulator real data" lever above is now a working loop, off the TV
Royale play↔unit binder (`game_knowledge/play_localizer.py`): a play-BOUND unit
carries a **ground-truth** card label, and STRUGGLE/off-deck units are exactly the
confusable/unknown reads to relabel. Crop-level; **RULE #2** — everything queues for
review, nothing auto-files.

```
# 1. collect real crops from your emulator (unattended, or live while watching)
python -m data.tv_royale_watcher --minutes 120 --collect-board
python -m perception.tv_royale_debug --cycles --collect-board   # live alternative

# 2. review + label the queue (mostly Enter to accept the ground-truth label, or a
#    number to pick from that unit's 8-card deck) → assets/board_labeled/<card>/
python -m perception.review_board_queue

# 3. build the combined dataset (labelled crops → YOLO; class idx from cr_combined_v4.yaml)
python training/prepare_board_data.py            # → training/datasets/cr_combined_v5.yaml

# 4. retrain, evaluate raw, promote only if it beats v4
python training/finetune.py --data training/datasets/cr_combined_v5.yaml --name cr_finetune_v5
```

Confusables (archer/musketeer, goblin/spear-goblin, …) **are** in the 201-class
taxonomy, so this targets them directly. The 7 **gap cards** (berserker, ronin,
goblin_gang, goblin_giant, suspicious_bush, void, spirit_empress — `detected_class`
null) can be *labelled* but not *trained* until the taxonomy is extended;
`prepare_board_data.py` holds them aside (reported) rather than dropping them.
