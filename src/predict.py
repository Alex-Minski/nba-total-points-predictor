from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd

FEATURES_PATH = Path("data/processed/features.csv")
MODELS_DIR = Path("models")

FEATURE_COLS = [
    "home_pf_roll","home_pa_roll","home_tot_roll","home_n",
    "away_pf_roll","away_pa_roll","away_tot_roll","away_n",
    "exp_total_mean","exp_total_mix"
]

def _pick_best_model() -> Path:
    ridge = MODELS_DIR / "model_ridge.joblib"
    if ridge.exists():
        return ridge
    candidates = sorted(MODELS_DIR.glob("model_*.joblib"))
    if not candidates:
        raise FileNotFoundError("No saved models found in ./models. Run training first.")
    return candidates[0]

def _latest_team_row(df: pd.DataFrame, team: str, side: str) -> pd.Series:
    team_col = "home_name" if side == "home" else "away_name"
    sub = df[df[team_col] == team].copy()
    if sub.empty:
        # helpful debug: show a few valid names
        valid = sorted(set(df["home_name"]).union(set(df["away_name"])))
        sample = ", ".join(valid[:20])
        raise ValueError(
            f"No historical data found for team '{team}'.\n"
            f"Tip: team names must match exactly what appears in features.csv.\n"
            f"Example valid names: {sample} ..."
        )
    return sub.iloc[-1]

def predict_total(home: str, away: str, model_path: str | None = None) -> float:
    df = pd.read_csv(FEATURES_PATH)

    home_row = _latest_team_row(df, home, side="home")
    away_row = _latest_team_row(df, away, side="away")

    x = pd.DataFrame([{
        "home_pf_roll": home_row["home_pf_roll"],
        "home_pa_roll": home_row["home_pa_roll"],
        "home_tot_roll": home_row["home_tot_roll"],
        "home_n": home_row["home_n"],
        "away_pf_roll": away_row["away_pf_roll"],
        "away_pa_roll": away_row["away_pa_roll"],
        "away_tot_roll": away_row["away_tot_roll"],
        "away_n": away_row["away_n"],
        "exp_total_mean": (home_row["home_tot_roll"] + away_row["away_tot_roll"]) / 2,
        "exp_total_mix": 0.6 * home_row["home_tot_roll"] + 0.4 * away_row["away_tot_roll"],
    }])

    model_file = Path(model_path) if model_path else _pick_best_model()
    model = joblib.load(model_file)

    return float(model.predict(x)[0])

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True, help="Must match home_name in features.csv")
    ap.add_argument("--away", required=True, help="Must match away_name in features.csv")
    ap.add_argument("--model", default=None, help="Optional: path to a model .joblib")
    args = ap.parse_args()

    pred = predict_total(args.home, args.away, args.model)
    print(f"Predicted total points: {pred:.1f}")
