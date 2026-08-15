"""
Click on any point in test_screenshot.png to print its pixel coordinates
to the terminal. Use this to find the (x, y) of an app icon so you can
target an ADB tap at it.

The displayed window is scaled down to fit your screen if the screenshot
is too large, but the coordinates printed are always translated back to
the original image's real pixel positions — the same coordinate space
tap()/swipe() expect.

Usage:
    python find_coords.py
Then click on the image window that opens. Press any key to close it.
"""

import cv2

MAX_DISPLAY_HEIGHT = 900  # keep the window comfortably on-screen

image = cv2.imread("test_screenshot.png")

if image is None:
    raise FileNotFoundError(
        "test_screenshot.png not found — run test_connection.py or "
        "`python -m perception.capture` first to generate it."
    )

original_h, original_w = image.shape[:2]
scale = min(1.0, MAX_DISPLAY_HEIGHT / original_h)
display_image = cv2.resize(image, (int(original_w * scale), int(original_h * scale)))


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Translate the click on the (possibly scaled-down) display window
        # back to real coordinates in the original screenshot.
        real_x, real_y = int(x / scale), int(y / scale)
        print(f"Clicked at (original image coordinates): x={real_x}, y={real_y}")


cv2.namedWindow("Click to get coordinates")
cv2.setMouseCallback("Click to get coordinates", on_click)

print(f"Original image: {original_w}x{original_h} | Displayed at scale {scale:.2f}")
print("Click on the app icon you want coordinates for. Press any key to close.")
cv2.imshow("Click to get coordinates", display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
