"""
Central config: ADB connection to LDPlayer, and the Clash Royale API.

Update ADB_PATH if your LDPlayer install location differs, and ADB_PORT if
`adb devices` (after connecting) shows a different port than 5555 — LDPlayer
assigns each running instance its own port, and 5555 is only the default for
the first/main instance.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # reads CR_API_KEY out of .env, if present

# --- LDPlayer / ADB ---------------------------------------------------

# Path to LDPlayer's own bundled adb.exe. Using this instead of a separate
# Android platform-tools install keeps things simple — LDPlayer already
# ships a working adb binary.
LDPLAYER_INSTALL_DIR = r"C:\LDPlayer\LDPlayer9"
ADB_PATH = os.path.join(LDPLAYER_INSTALL_DIR, "adb.exe")

# LDPlayer's default main-instance ADB port.
ADB_HOST = "127.0.0.1"
ADB_PORT = 5555

# --- Clash Royale API ---------------------------------------------------

# Routed through RoyaleAPI's proxy rather than api.clashroyale.com directly,
# so the whitelisted IP is RoyaleAPI's fixed 45.79.218.79 instead of your
# own (dynamic, home-connection) IP. See README for the API key setup.
CR_API_BASE = "https://proxy.royaleapi.dev/v1"
CR_API_KEY = os.environ.get("CR_API_KEY", "")
