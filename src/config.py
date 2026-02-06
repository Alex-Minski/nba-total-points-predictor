import os
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("BETS_API_TOKEN")
BASE_URL = os.getenv("BASE_URL", "https://api.b365api.com/").rstrip("/") + "/"

SPORT_ID = int(os.getenv("SPORT_ID", "18"))
LEAGUE_ID = int(os.getenv("LEAGUE_ID", "2274"))

REQUEST_TIMEOUT = 15
