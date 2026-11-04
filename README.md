# Clash Royale AI Bot

A bot that plays Clash Royale from raw screen pixels. It captures frames from an
Android emulator (LDPlayer), works out what's on the board with a custom-trained
YOLOv8 detector, decides what card to play and where, and taps it in through ADB.
No game memory reading or private APIs: it sees the same screen a player does.

I'm building this solo to learn computer vision, ML training, and how to wire
open-source tools into one working system. Work in progress (Aug 2026 – present).

## How it works

```
LDPlayer (emulator) --ADB screenshot--> perception --> game state --> policy --ADB tap--> LDPlayer
```

1. **Perception** (`perception/`)
   - Custom **YOLOv8l unit detector** fine-tuned on synthetic + real frames (v1 → v4).
     Beats the open-source KataCR detector on recall and reads troop side and spawn natively.
   - Health bars and level badges decide each unit's side, HP and level.
   - Thin tracker gives stable unit IDs and smooths class/side flicker between frames.
   - Hand cards (template matching + a YOLOv8-cls classifier), elixir count, tower HP.
2. **Game knowledge** (`game_knowledge/`)
   - Builds a `GameState` (elixir, hand, units, towers, lane zones).
   - Cycle tracker reconstructs the opponent's hidden card rotation from what they play.
   - Play tracker ties newly spawned units back to the card that made them.
3. **Decision + control** (`game_knowledge/policy.py`, `control/`)
   - Rule-based baseline policy picks a card and a placement; the bot loop runs
     perceive → decide → act live (dry-run by default).
4. **Data engine** (`data/`, `training/`)
   - Unattended TV Royale watcher logs top players' matches (both players' plays) as
     training data for imitation learning, and queues uncertain reads for review.
   - Dataset generators and a fine-tune harness to retrain the detector.

## Status

| Stage | State |
|---|---|
| Emulator control + capture | Done |
| Perception (detector, tracker, hand, elixir, towers, levels) | Done, iterating |
| Rule-based bot playing live | Working baseline |
| TV Royale data collection + analysis | Working |
| Learned policy (imitation → RL) | Next |

## Tech

Python · Ultralytics YOLOv8 · PyTorch (CUDA) · ONNX Runtime · OpenCV · ADB · Clash Royale API

## Not in this repo

Trained weights (`*.pt`), generated datasets, logs, and third-party reference code
(`vendor/`) are gitignored because of size. The API key lives in `.env`
(see `.env.example`).

---

## Setup (Windows + LDPlayer)


1. **Enable ADB debugging in LDPlayer.** Settings (wrench icon) → Others/Basic
   settings tab → turn on ADB debugging. Without this, nothing here can
   connect.

2. **Confirm the ADB port.** Open Command Prompt, `cd` into your LDPlayer
   install folder (default `C:\LDPlayer\LDPlayer9`), then run:
   ```
   adb.exe connect 127.0.0.1:5555
   adb.exe devices
   ```
   If a device shows up, you're set — `config/settings.py` already defaults
   to port 5555. If it doesn't show up, or you're running multiple
   instances, check LDPlayer's own settings for the instance's actual ADB
   port and update `ADB_PORT` in `config/settings.py` to match.

3. **Install Python dependencies:**
   ```
   pip install -r requirements.txt
   ```
   If you have an NVIDIA GPU, install the CUDA torch build FIRST — the default
   CPU build runs the newer detector ~23x slower (~2fps, unusable live). For an
   RTX 50-series (Blackwell):
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```

4. **API key (optional, for `fetch_card_roster.py`).** Copy `.env.example` to
   `.env` and paste your Clash Royale API key.

5. **(Optional) Fetch the experimental detector weights.** `perception/katacr_detector.py`
   can run KataCR's YOLOv8 detectors instead of the bundled `units.onnx` — more
   accurate, and the weights are trainable (see `Project_Status`). They're not
   committed (88MB each); download links are in that module's docstring, and it
   also needs the KataCR source for its model class:
   ```
   git clone --depth 1 https://github.com/wty-yy/KataCR.git vendor/katacr
   ```
   Then: `python -m perception.debug_live --detector katacr`

## Verify everything works

```
python test_connection.py
```

This connects to LDPlayer, saves a screenshot as `test_screenshot.png`, and
sends a test tap to the center of the screen. If all three succeed, Phase 1
plumbing is solid and it's safe to start building perception (card/elixir/
unit detection) on top of it.

## Layout

```
control/
    actions.py            — EmulatorController: tap(), swipe()
perception/
    capture.py            — ScreenCapture: screenshot(), save()
    card_detector.py      — cards in hand (template matching; live-play hand)
    hand_classifier.py    — YOLOv8-cls hand-card classifier (assets/models/hand_cls/) — TV Royale hands
    elixir_counter.py     — elixir number + bar fill
    yolo_detector.py      — OUR fine-tuned YOLOv8l (assets/models/cr_yolo_v*/) — the detector.
                            Reads side from bar colour + level from bar-level badges.
    bar_reader.py         — associates detected bars/level-badges to units (side, HP, level)
    level_reader.py       — reads the level digit from a detected bar-level badge (+ its template lib)
    detector_view.py      — the live/eval view: raw detector + optional thin tracker (--track)
    thin_tracker.py       — stable IDs + class debounce + sticky side/level (replaced unit_tracker.py)
    tower_reader.py       — tower HP % from detected tower-bars (destroyed-state now driven by the detector)
game_knowledge/
    cycle_tracker.py      — reconstruct a hidden 4-card queue from hand reads + elixir
    tv_royale_cycles.py   — DUAL cycle tracker (both players) for TV Royale → play events
    tv_royale_analyzer.py — per-frame TV Royale analysis + self-collection (cards/levels)
    play_tracker.py       — own-play ground truth + opponent-play attribution (unit→card)
    zone_classifier.py    — per-lane ally/enemy/neutral zones
    game_state.py         — assembles perception into a GameState (elixir/hand/units/towers)
    policy.py             — rule-based baseline: what/where to play
    bot.py                — the Phase 5 loop: perceive → decide → act (DRY-RUN by default)
data/
    tv_royale_watcher.py  — unattended TV Royale watcher → analyzer (plays log + self-queues)
config/
    settings.py           — ADB host/port, LDPlayer install path
    red_zones.py          — tower/water/arena-bounds geometry
training/                 — OUR detector: generators + fine-tune harness (see training/README.md)
vendor/                   — reference repos, gitignored (see roadmap §3)
test_connection.py        — run this first
```

Most day-to-day work happens through the detector view:
```
python -m perception.detector_view              # OUR detector, raw output (evaluate)
python -m perception.detector_view --track      # + thin tracker + game-state panel
python -m perception.detector_view --track --collect-levels   # + bank level badges to review
```
See `training/README.md` for how the detector is trained (v1→v4), and
`Project_Status` / `TIMELINE.md` / `clash-royale-ai-roadmap.md` for the plan.

## Reference repos (clone into vendor/ when you want to study them)

```
git clone https://github.com/Pbatch/ClashRoyaleBuildABot.git vendor/build-a-bot
git clone https://github.com/pyclashbot/py-clash-bot.git vendor/py-clash-bot
```

See `clash-royale-ai-roadmap.md` in this folder for the full build plan and
resource library.
