from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

from .api_client import BetsApiClient
from .features import parse_score, RollingConfig, build_rolling_features

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_games(results: list[dict]) -> list[dict]:
    rows = []
    for g in results:
        ss = g.get("ss")
        score = parse_score(ss) if ss else None
        if not score:
            continue
        home = g.get("home") or {}
        away = g.get("away") or {}

        # BetsAPI typically provides a unix timestamp in `time`; fall back to id ordering if missing
        ts = g.get("time")
        date = None
        try:
            if ts:
                date = datetime.utcfromtimestamp(int(ts)).date().isoformat()
        except Exception:
            date = None

        rows.append({
            "event_id": g.get("id"),
            "date": date,
            "home_id": home.get("id"),
            "away_id": away.get("id"),
            "home_name": home.get("name"),
            "away_name": away.get("name"),
            "home_pts": score[0],
            "away_pts": score[1],
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=20, help="How many pages of ended events to pull (each page ~50)." )
    ap.add_argument("--rolling", type=int, default=10, help="Rolling window size for features.")
    ap.add_argument("--min_games", type=int, default=5, help="Minimum prior games required per team.")
    args = ap.parse_args()

    client = BetsApiClient()

    all_rows = []
    for page in range(1, args.pages + 1):
        cache_path = RAW_DIR / f"ended_page_{page}.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            data = client.ended_events(page=page)
            cache_path.write_text(json.dumps(data), encoding="utf-8")

        results = data.get("results") or []
        all_rows.extend(normalize_games(results))

    df = pd.DataFrame(all_rows)

    # Drop rows without dates; if many are missing, you can still sort by event_id
    if df["date"].notna().any():
        df = df[df["date"].notna()].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", ascending=True)
    else:
        df = df.sort_values("event_id", ascending=True)

    games_path = OUT_DIR / "games.csv"
    df.to_csv(games_path, index=False)
    print(f"Wrote raw games -> {games_path} ({len(df)} rows)")

    feats = build_rolling_features(df, RollingConfig(window=args.rolling, min_games=args.min_games))
    feats_path = OUT_DIR / "features.csv"
    feats.to_csv(feats_path, index=False)
    print(f"Wrote features -> {feats_path} ({len(feats)} rows)")

if __name__ == "__main__":
    main()
