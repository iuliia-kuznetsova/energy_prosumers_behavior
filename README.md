# Predict Energy Behavior of Prosumers

Forecast hourly **consumption** and **production** of Estonian energy prosumers so imbalance costs stay low.

Competition: [Enefit — Predict Energy Behavior of Prosumers](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers).  
Metric: **MAE on power**.

---

## Task

A *prediction unit* is a segment of prosumers (`county`, `is_business`, `product_type`). For every hour the model must output two power values: consumption and production.

Kaggle revealed data **day by day**. Features used in training must match what is available at forecast time:

| Information | Latest value when predicting day D |
|---|---|
| Target (revealed power) | all hours of D−2 (48 h) |
| Client capacity and `eic_count` | date D−2 |
| Electricity and gas prices | D−1 |
| Observed weather | shifted 48 h |
| Weather forecast | the day-ahead issue for day D |

The training target is `capacity_factor = power / installed_capacity`. Predictions are converted back to power and clipped in capacity-factor space.

---

## Solution, by step

`python -m src.preds.main_preds` runs the full pipeline.

| Step | What it does | Details |
|---|---|---|
| 1. Load | Download or verify the 7 Kaggle CSVs in `data/raw` | [readme_data_loading.md](readme_data_loading.md) |
| 2. Preprocess | Check columns, parse dates, cast dtypes | [readme_data_preprocessing.md](readme_data_preprocessing.md) |
| 3. Merge | Join capacity, prices and weather onto each power hour; build `capacity_factor` | [readme_data_merging.md](readme_data_merging.md) |
| 4. Features | Calendar, lags, solar position, missing-value flags | [readme_feature_engineering.md](readme_feature_engineering.md) |
| 5. Train | Two CatBoost models (consumption, production), Optuna, April–May 2023 holdout | [readme_modelling.md](readme_modelling.md) |
| 6. Track | Log parameters, holdout MAE and both models to local MLflow | [readme_mlflow_logging.md](readme_mlflow_logging.md) |

```text
data/raw/*.csv
    → preprocess  → merge  → features
    → CatBoost consumption + production
    → models/<name>/{consumption.cbm, production.cbm, metadata.json}
    → results/evaluation_results.json
    → MLflow (optional)
```

---

## Virtual environment

From the project root:

```bash
python -m venv venv_enefit
```

Activate it:

```bash
# Linux / macOS / Git Bash
source venv_enefit/bin/activate

# Windows PowerShell
.\venv_enefit\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Put Kaggle credentials in `~/.kaggle/kaggle.json`, or set `KAGGLE_USERNAME` and `KAGGLE_KEY`. Then:

```bash
python -m src.preds.main_preds
python -m src.preds.main_preds --skip-download    # CSVs already in data/raw
python -m src.preds.main_preds --debug            # four prediction units only
python -m src.preds.main_preds --no-tree-scale    # do not scale the tree count
python -m src.preds.main_preds --no-mlflow        # train without tracking
```

---

## MLflow logging

The tracking server is local: SQLite (`mlflow.db`) and files on disk (`mlartifacts/`). No Docker or cloud account.

1. Start the server from the project root and leave that terminal open:

```powershell
.\start_mlflow_server.ps1
```

```bash
bash start_mlflow_server.sh
```

2. Check that the client can reach it (UI: http://127.0.0.1:5000):

```bash
python -m src.preds.mlflow_logging --check
```

3. A full `main_preds` run logs the trained models after step 5. To log a model that is already saved:

```bash
python -m src.preds.mlflow_logging --model-name catboost_prosumer --run-name my_run
```

If the server is down, training still finishes and the model is written under `models/`. See [readme_mlflow_logging.md](readme_mlflow_logging.md).

---

## Pros of the solution

- **Separate consumption and production models** — the two regimes have different daily shapes and weather sensitivity; `is_consumption` is not forced into one tree.
- **Capacity factor as the target** — `power / installed_capacity` makes large and small units comparable. Training uses capacity as the sample weight, so the loss matches MAE on power. Predictions are clipped in capacity-factor space, then converted back.
- **Features follow the reveal rules** — target lags start at 48 h; capacity is two days old; prices are from D−1; observed weather is shifted 48 h; the forecast is the day-ahead issue. The model cannot train on values it would not have at forecast time.
- **Honest validation** — Optuna scores month-long rolling-origin folds (January–March 2023). April–May 2023 is a holdout that never enters tuning, early stopping or the final fit. Lag-48 and lag-168 baselines sit next to that score.
- **Time encodings that respect cycles and trend** — hour, weekday and month are sin/cos so 23:00 is next to 00:00; `months_since_start` carries growth instead of treating year as a cycle.
- **Signed prices and solar geometry** — negative day-ahead prices stay negative (oversupply hours). County-level solar elevation, azimuth, clear-sky GHI and `forecast GHI × capacity` give the physical production scale.
- **Local experiment tracking** — parameters, holdout MAE and both CatBoost models go to a local MLflow server. Training still finishes if the server is down.

---

**Iuliia Kuznetsova**
