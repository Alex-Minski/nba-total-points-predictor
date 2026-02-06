# NBA Total Points Predictor (Starter ML)

This repo trains a **supervised regression** model to predict **NBA game total points (home + away)** using historical game results from **BetsAPI**.

## What it does
1. Pulls **ended NBA games** from BetsAPI (`/v3/events/ended`) and caches raw JSON locally.
2. Builds a dataset with **rolling, pre-game features** (no future leakage):
   - each team's last-N average points scored / allowed
   - last-N average game total
3. Trains a regression model (baseline + Ridge) and reports MAE/RMSE.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set your BetsAPI token. Basketball sport_id=18 and NBA league id is 2274.

## Build dataset
```bash
python -m src.build_dataset --pages 20 --rolling 10
```

## Train + evaluate
```bash
python -m src.train --test_ratio 0.2
```
## Predict(inference)
After training, you can generate a point estimate for an upcoming matchup using each team’s most recent rolling features in features.csv.

```bash
python -m src.predict --home "PHX Suns" --away "NY Knicks"
```
Example output: Predicted total points: 228.4

## Notes / limitations
- Starter features are intentionally simple
- This project is for educational analytics; it is not financial advice.
- Team names must match exactly what appears in features.csv. To view valid names:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/processed/features.csv'); print(sorted(set(df['home_name']).union(set(df['away_name']))))"
```
