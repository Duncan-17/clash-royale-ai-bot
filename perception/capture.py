"""
Screenshot capture from the LDPlayer instance.

Uses adbnativeblitz's AdbFastScreenshots under the hood: a background
thread runs `adb shell screenrecord` (one continuous H.264 video stream,
not repeated PNG screencaps) and decodes frames as they arrive via PyAV.
screenshot() just returns whatever the most recently decoded frame is —
near-instant, since the stream is already running continuously in the
background, instead of a blocking per-call ADB round-trip.

Benchmarked against this project's real LDPlayer setup (2026-08-15):
~54 distinct frames/second, vs. ~1.5-2 fps for the original plain `adb
screencap` approach this replaced (see git history for that version).
Roughly a 30x improvement — done now rather than deferred, since it was
clear this would eventually be needed (catching near-instant game events
like Zap/Lightning — see clash-royale-ai-pipeline-explained.md — and
fine-grained troop HP tracking) and doing it before more perception code
gets built on top of the old timing assumptions was judged cheaper than
doing it after.

Color channels verified correct without conversion: unlike the old PIL-
based screencap path (which needed an explicit RGB->BGR conversion — see
this file's git history for that bug's story), AdbFastScreenshots' numpy
output already matches what cv2 expects directly. Confirmed by writing a
frame straight to disk via cv2.imwrite and checking known-color UI
elements rendered correctly (gold coins gold, not blue-shifted, etc.).

External API (screenshot(), save()) is deliberately unchanged from the
version this replaced — every other perception module calls these the
same way regardless of what's happening underneath.
"""

import atexit
import threading
import time

import cv2
import numpy as np
from adbnativeblitz import AdbFastScreenshots

from config.settings import ADB_HOST, ADB_PATH, ADB_PORT
from control.actions import EmulatorController

FRAME_WIDTH = 900
FRAME_HEIGHT = 1600
BITRATE = "8M"
GO_IDLE_SECONDS = 0.01  # small poll-throttle so the background thread doesn't busy-spin when idle;
# still far above the ~54fps real content rate, so no real frames are missed by waiting this long
STARTUP_TIMEOUT_SECONDS = 10


class ScreenCapture:
    def __init__(self, controller: EmulatorController):
        self.device = controller.device  # kept for any code that reaches into it directly

        self._latest_frame = None
        self._lock = threading.Lock()
        self._stream = AdbFastScreenshots(
            adb_path=ADB_PATH,
            device_serial=f"{ADB_HOST}:{ADB_PORT}",
            time_interval=179,  # max per-session recording length in seconds (library auto-restarts under the hood before hitting this)
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            bitrate=BITRATE,
            # Lowered 5 -> 2 (2026-08-20) after Duncan reported the debug view running
            # visibly BEHIND the live game. Our own side never queues -- _pump_frames
            # overwrites _latest_frame and screenshot() always takes the freshest one --
            # so buffered frames here are pure added latency, not throughput we use.
            # Some buffer is still wanted so a decode hiccup doesn't starve the stream.
            # NOTE: the larger, irreducible part of that delay is `adb shell screenrecord`
            # being an H.264 ENCODER, which inherently buffers before emitting (typically
            # 100-300ms). That's the price of the ~54fps this replaced ~1.5fps screencap
            # with; removing it entirely would mean a different capture backend (scrcpy),
            # a much bigger change. Cosmetic for a debug view -- every module reads the
            # same frame, so the analysis stays self-consistent -- but it becomes a real
            # problem in Phase 5 when the bot ACTS on what it sees.
            screenshotbuffer=2,
            go_idle=GO_IDLE_SECONDS,
        )

        self._thread = threading.Thread(target=self._pump_frames, daemon=True)
        self._thread.start()
        atexit.register(self.stop)  # best-effort cleanup if a caller never explicitly stops us
        self._wait_for_first_frame()

    def _pump_frames(self):
        for frame in self._stream:
            with self._lock:
                self._latest_frame = frame

    def _wait_for_first_frame(self):
        start = time.time()
        while self._latest_frame is None:
            if time.time() - start > STARTUP_TIMEOUT_SECONDS:
                raise RuntimeError(
                    "Timed out waiting for the first frame from AdbFastScreenshots — "
                    "check LDPlayer is running and ADB debugging is on."
                )
            time.sleep(0.02)

    def screenshot(self) -> np.ndarray:
        """Return the most recently captured frame as a BGR numpy array (H, W, 3)."""
        with self._lock:
            return self._latest_frame.copy()

    def save(self, path: str):
        """Convenience: capture and save directly to disk."""
        cv2.imwrite(path, self.screenshot())

    def stop(self):
        """Stops the background capture stream and its underlying ADB process.
        Not required before process exit (registered via atexit as a safety net),
        but call explicitly if a script keeps running after it's done capturing."""
        self._stream.stop_capture()


if __name__ == "__main__":
    controller = EmulatorController()
    capture = ScreenCapture(controller)
    capture.save("test_screenshot.png")
    print("Saved test_screenshot.png — open it to confirm it shows the emulator screen.")
    capture.stop()
