"""Grab ONE current frame from the emulator and save it to a named file (and
open it so you can eyeball it). For calibrating multi-screen flows like TV
Royale navigation, where each screen needs its own saved capture.

Deliberately writes ONLY to the path you pass -- it never touches
test_screenshot.png (the canonical capture; see CLAUDE.md). Pass a temp_img/
path so calibration grabs stay out of the project root.

Usage:
    python -m perception.grab_screen temp_img/tvr_01_menu.png
"""

import subprocess
import sys

from perception.capture import ScreenCapture
from control.actions import EmulatorController


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m perception.grab_screen <output_path.png>")
        return
    out = sys.argv[1]
    if out.replace("\\", "/").endswith("test_screenshot.png"):
        print("Refusing to overwrite test_screenshot.png — pass a temp_img/ path instead.")
        return

    capture = ScreenCapture(EmulatorController())
    try:
        capture.save(out)
    finally:
        capture.stop()
    print(f"Saved {out}")
    subprocess.Popen(["mspaint.exe", out.replace("/", "\\")])


if __name__ == "__main__":
    main()
