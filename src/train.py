from __future__ import annotations
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
import numpy as np

DATA = Path("data/processed/features.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "home_pf_roll","home_pa_roll","home_tot_roll","home_n",
    "away_pf_roll","away_pa_roll","away_tot_roll","away_n",
    "exp_total_mean","exp_total_mix"
]
TARGET = "y_total"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_ratio", type=float, default=0.2, help="Fraction of most-recent games used for test.")
    args = ap.parse_args()

    df = pd.read_csv(DATA)
    df = df.dropna(subset=FEATURE_COLS + [TARGET]).copy()

    # time-aware split: last X% for test
    split_idx = int(len(df) * (1 - args.test_ratio))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET]

    models = {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0, random_state=42),
    }

    results = []
    best = None
    best_mae = float("inf")

    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))


        results.append((name, mae, rmse))
        if mae < best_mae:
            best_mae = mae
            best = (name, m)

    print("\nEvaluation (most-recent holdout)")
    for name, mae, rmse in results:
        print(f"- {name:13s} | MAE: {mae:6.2f} | RMSE: {rmse:6.2f}")

    assert best is not None
    best_name, best_model = best
    out = MODEL_DIR / f"model_{best_name}.joblib"
    joblib.dump(best_model, out)
    print(f"\nSaved best model -> {out}")

if __name__ == "__main__":
    main()
