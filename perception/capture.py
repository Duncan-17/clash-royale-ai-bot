"""
Screenshot capture from the LDPlayer instance via ADB.

Phase 1 baseline: pulls a screencap over the same ADB connection actions.py
uses. This is reliable but not the fastest possible option — the roadmap
flags `adb screencap` polling as a bottleneck for a tight live-perception
loop. Get this working first; swap in adbnativeblitz or scrcpy later
(Phase 3+) if capture speed becomes the limiting factor.
"""

import io

import cv2
import numpy as np
from PIL import Image

from control.actions import EmulatorController


class ScreenCapture:
    def __init__(self, controller: EmulatorController):
        self.device = controller.device

    def screenshot(self) -> np.ndarray:
        """Return the current screen as a BGR numpy array (H, W, 3) — the
        channel order every cv2-based script in this project (card_detector,
        etc.) assumes. ADB's screencap decodes via PIL, which returns RGB,
        so this converts RGB -> BGR before returning. (Previously this
        returned RGB directly, which every OpenCV call downstream silently
        misread as BGR — that's the root cause of the red/blue-swapped
        "blue and yellow" template images; see fix_color_channels.py for
        the one-time repair of files already saved under the old bug.)"""
        raw = self.device.screencap()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        rgb_array = np.array(image)
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    def save(self, path: str):
        """Convenience: capture and save directly to disk. Uses cv2.imwrite
        (not PIL) since screenshot() now returns BGR — cv2.imwrite expects
        that order and will write a correctly-colored file."""
        cv2.imwrite(path, self.screenshot())


if __name__ == "__main__":
    controller = EmulatorController()
    capture = ScreenCapture(controller)
    capture.save("test_screenshot.png")
    print("Saved test_screenshot.png — open it to confirm it shows the emulator screen.")
