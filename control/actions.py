"""
ADB-based control wrapper: taps and drags against the LDPlayer instance.

Uses pure-python-adb (ppadb) per the roadmap's Phase 1 plan, rather than
shelling out to adb.exe for every single action. We do shell out once, on
connect, to make sure the adb server actually has LDPlayer registered —
that's the step people usually forget, and it's why "no devices found"
errors happen even when LDPlayer is clearly running.
"""

import subprocess

from ppadb.client import Client as AdbClient

from config.settings import ADB_HOST, ADB_PATH, ADB_PORT


class EmulatorController:
    def __init__(self, host: str = ADB_HOST, port: int = ADB_PORT):
        self._target = f"{host}:{port}"
        self._ensure_adb_connected()
        # ppadb talks to the local adb *server* (always port 5037),
        # not directly to the emulator's own port.
        self.client = AdbClient(host="127.0.0.1", port=5037)
        self.device = self._get_device()

    def _serial_online(self) -> bool:
        """True only if the target serial (127.0.0.1:5555) is a live 'device' in
        `adb devices` -- NOT just that `adb connect` claimed success. LDPlayer
        sometimes registers as `emulator-5554` instead, and a stale entry can
        make `connect` print 'already connected' while the serial isn't actually
        usable -- which is exactly what starves perception/capture.py (it targets
        this serial for screenrecord) and produced the 'timed out waiting for the
        first frame' failure."""
        out = subprocess.run([ADB_PATH, "devices"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == self._target and parts[1] == "device":
                return True
        return False

    def _ensure_adb_connected(self):
        """Bring the target serial online, recovering from the stale/wrong-serial
        state automatically (connect -> verify -> on failure disconnect + retry)."""
        for attempt in range(3):
            if self._serial_online():
                return
            subprocess.run([ADB_PATH, "connect", self._target], capture_output=True, text=True)
            if self._serial_online():
                print(f"ADB connected: {self._target}")
                return
            # Stale/offline entry -- clear it and retry fresh next loop.
            subprocess.run([ADB_PATH, "disconnect", self._target], capture_output=True, text=True)
        # Last-ditch: it may still be reachable under emulator-XXXX; _get_device
        # will surface a clear error if nothing usable is present.
        print(f"WARNING: could not bring {self._target} online after retries "
              f"(will try whatever `adb devices` shows).")

    def _get_device(self):
        devices = self.client.devices()
        if not devices:
            raise RuntimeError(
                "No ADB devices found. Checklist:\n"
                "  1. ADB debugging is enabled in LDPlayer's settings\n"
                "  2. The LDPlayer instance is actually running\n"
                "  3. The port in config/settings.py matches what LDPlayer "
                "is using (run `adb.exe devices` from the LDPlayer install "
                "folder to check)"
            )
        # Prefer the target serial (127.0.0.1:5555) so the controller and the
        # capture layer act on the SAME device -- capture.py hard-targets it.
        for d in devices:
            if d.serial == self._target:
                print(f"Connected to: {d.serial}")
                return d
        print(f"Connected to: {devices[0].serial} (target {self._target} not "
              f"listed; capture.py may still fail -- see EmulatorController).")
        return devices[0]

    def tap(self, x: int, y: int):
        """Tap a single point on screen, in pixel coordinates."""
        self.device.shell(f"input tap {x} {y}")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300):
        """Drag from (x1, y1) to (x2, y2) over duration_ms milliseconds."""
        self.device.shell(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")

    def play_card(self, slot_index: int, x: int, y: int, select_delay_s: float = 0.06):
        """Play the hand card in slot `slot_index` (0-3, left to right) at arena
        pixel (x, y): tap the card to select it, then tap the placement point.

        This is the bot's one physical action (Phase 5) and the moment its OWN play
        becomes ground truth -- the caller pairs this with
        game_knowledge.play_tracker.PlayTracker.record_own_play(card, (x, y)).
        The short delay lets the client register the selection before the placement
        tap (both are separate `input tap`s)."""
        import time
        from config.hand_region import hand_slot_regions
        regions = hand_slot_regions()
        if not (0 <= slot_index < len(regions)):
            raise ValueError(f"slot_index {slot_index} out of range 0..{len(regions) - 1}")
        rx1, ry1, rx2, ry2 = regions[slot_index]
        self.tap((rx1 + rx2) // 2, (ry1 + ry2) // 2)  # select the card
        time.sleep(select_delay_s)
        self.tap(int(x), int(y))                       # place it


if __name__ == "__main__":
    # Manual smoke test: connect and tap the center of a 1920x1080 screen.
    controller = EmulatorController()
    controller.tap(960, 540)
    print("Tap sent — check the emulator window for a response.")
