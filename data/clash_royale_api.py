"""
Thin client for the Clash Royale API, routed through RoyaleAPI's proxy
(so the whitelisted IP is RoyaleAPI's fixed 45.79.218.79, not your own
home IP — see the roadmap notes on why that matters for a home connection).
"""

import requests

from config.settings import CR_API_BASE, CR_API_KEY


class ClashRoyaleAPI:
    def __init__(self, api_key: str = CR_API_KEY, base_url: str = CR_API_BASE):
        if not api_key or api_key == "paste_your_key_here":
            raise RuntimeError(
                "CR_API_KEY isn't set. Copy .env.example to .env and paste "
                "your real key in, then try again."
            )
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def get_cards(self) -> list:
        """Full card roster: name, id, elixir cost, rarity, icon URLs."""
        resp = requests.get(f"{self.base_url}/cards", headers=self.headers, timeout=15)
        resp.raise_for_status()
        return resp.json()["items"]
