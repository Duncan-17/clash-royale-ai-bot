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
        self._ensure_adb_connected(host, port)
        # ppadb talks to the local adb *server* (always port 5037),
        # not directly to the emulator's own port.
        self.client = AdbClient(host="127.0.0.1", port=5037)
        self.device = self._get_device()

    def _ensure_adb_connected(self, host: str, port: int):
        result = subprocess.run(
            [ADB_PATH, "connect", f"{host}:{port}"],
            capture_output=True,
            text=True,
        )
        print(result.stdout.strip() or result.stderr.strip())

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
        print(f"Connected to: {devices[0].serial}")
        return devices[0]

    def tap(self, x: int, y: int):
        """Tap a single point on screen, in pixel coordinates."""
        self.device.shell(f"input tap {x} {y}")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300):
        """Drag from (x1, y1) to (x2, y2) over duration_ms milliseconds."""
        self.device.shell(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")


if __name__ == "__main__":
    # Manual smoke test: connect and tap the center of a 1920x1080 screen.
    controller = EmulatorController()
    controller.tap(960, 540)
    print("Tap sent — check the emulator window for a response.")
