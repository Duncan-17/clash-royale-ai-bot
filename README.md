# Clash Royale AI — Phase 1 setup

This is the control + capture plumbing from the roadmap's Phase 1: reliable
screenshot capture and tap/drag control against LDPlayer, no AI yet.

## One-time setup

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

4. **(Optional) Fetch the experimental detector weights.** `perception/katacr_detector.py`
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
    card_detector.py      — cards in hand (template matching)
    elixir_counter.py     — elixir number + bar fill
    unit_detector.py      — board units (bundled units.onnx)
    katacr_detector.py    — EXPERIMENTAL alternative detector, same interface
    unit_tracker.py       — persistent track IDs, side/class resolution
    tower_reader.py       — tower HP % and destroyed state
    side_reader.py        — ally/enemy from the level badge colour
    spawn_icon_reader.py  — side at the moment of placement
    debug_live.py         — the live debug view (start here)
game_knowledge/
    cycle_tracker.py      — our own card cycle
    opponent_play_tracker.py — what the opponent played, and when
    zone_classifier.py    — per-lane ally/enemy/neutral zones
config/
    settings.py           — ADB host/port, LDPlayer install path
    red_zones.py          — tower/water/arena-bounds geometry
vendor/                   — reference repos, gitignored (see roadmap §3)
test_connection.py        — run this first
```

Most day-to-day work happens through the live debug view:
```
python -m perception.debug_live               # current detector
python -m perception.debug_live --detector katacr   # experimental
```

## Reference repos (clone into vendor/ when you want to study them)

```
git clone https://github.com/Pbatch/ClashRoyaleBuildABot.git vendor/build-a-bot
git clone https://github.com/pyclashbot/py-clash-bot.git vendor/py-clash-bot
```

See `clash-royale-ai-roadmap.md` in this folder for the full build plan and
resource library.
