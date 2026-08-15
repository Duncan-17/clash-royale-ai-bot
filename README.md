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
    actions.py      — EmulatorController: tap(), swipe()
perception/
    capture.py       — ScreenCapture: screenshot(), save()
config/
    settings.py       — ADB host/port, LDPlayer install path
vendor/               — clone reference repos here for study (see roadmap §3)
test_connection.py    — run this first
```

## Reference repos (clone into vendor/ when you want to study them)

```
git clone https://github.com/Pbatch/ClashRoyaleBuildABot.git vendor/build-a-bot
git clone https://github.com/pyclashbot/py-clash-bot.git vendor/py-clash-bot
```

See `clash-royale-ai-roadmap.md` in this folder for the full build plan and
resource library.
