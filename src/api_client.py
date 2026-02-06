from __future__ import annotations
import requests
from .config import API_TOKEN, BASE_URL, SPORT_ID, LEAGUE_ID, REQUEST_TIMEOUT

class BetsApiClient:
    def __init__(self, token: str | None = None):
        self.token = token or API_TOKEN
        if not self.token:
            raise ValueError("Missing BETS_API_TOKEN. Set it in your .env file.")

    def ended_events(self, page: int = 1) -> dict:
        """Fetch ended events for NBA.

        Uses BetsAPI Events API: /v3/events/ended
        Required: token, sport_id. Optional: league_id, page.
        """
        url = f"{BASE_URL}v3/events/ended"
        params = {
            "token": self.token,
            "sport_id": SPORT_ID,
            "league_id": LEAGUE_ID,
            "page": page,
        }
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
