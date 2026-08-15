"""
Phase 1 smoke test — confirms ADB can connect to LDPlayer, capture a
screenshot, and send a tap. Run this first, before building anything else.

Usage:
    python test_connection.py
"""

from control.actions import EmulatorController
from perception.capture import ScreenCapture


def main():
    print("Connecting to LDPlayer via ADB...")
    controller = EmulatorController()

    print("Capturing a screenshot...")
    capture = ScreenCapture(controller)
    capture.save("test_screenshot.png")
    print("Saved test_screenshot.png — open it and confirm it shows your emulator screen.")

    print("Sending a test tap to the center of the screen...")
    controller.tap(960, 540)
    print("Tap sent. If you're on the LDPlayer home screen, watch for a response.")

    print("\nAll good — Phase 1 plumbing is working.")


if __name__ == "__main__":
    main()
