# MLflow logging

Module: [`src/preds/mlflow_logging.py`](src/preds/mlflow_logging.py)

After a successful `fit()`, step 8 of `main_preds` sends one run to a **local** MLflow server. Nothing is uploaded to a cloud account.

## Layout

| Item | Default |
|---|---|
| Tracking URI | `http://127.0.0.1:5000` (`MLFLOW_TRACKING_URI`) |
| Experiment | `energy_prosumers_catboost` (`MLFLOW_EXPERIMENT_NAME`) |
| Metadata | `./mlflow.db` |
| Artifacts | `./mlartifacts/` |
| UI | http://127.0.0.1:5000 |

`mlflow.db`, `mlartifacts/` and `mlruns/` are gitignored.

## Start the server

From the project root, with `venv_enefit` available, leave this terminal open:

```powershell
.\start_mlflow_server.ps1
```

```bash
bash start_mlflow_server.sh
```

The scripts start `mlflow server` with a SQLite backend and `--serve-artifacts`, so the training process only needs the HTTP URI.

Check the connection:

```bash
python -m src.preds.mlflow_logging --check
```

## What one run stores

- **Parameters:** target, seed, Optuna trial count, CV splits, debug / scale-trees flags, feature count, holdout start, and the best CatBoost hyperparameters for consumption and production.
- **Metrics:** holdout power MAE (overall and by segment), rolling-origin CV MAE, lag-48 and lag-168 baselines, per-fold CV MAE.
- **Models:** both CatBoost models, plus the raw `.cbm` files.
- **Artifacts:** `eval_report.json`, `features.json`, feature-importance CSVs.

If the server is down, training still finishes and the model stays under `models/`. `--skip-training` does not log again.

## Commands

```bash
# Train and log (default)
python -m src.preds.main_preds --mlflow-run-name my_run

# Train without tracking
python -m src.preds.main_preds --no-mlflow

# Log a model that is already in models/
python -m src.preds.mlflow_logging --model-name catboost_prosumer --run-name my_run
```

From Python, after `fit()` or `load()`:

```python
from src.preds.mlflow_logging import log_catboost_model_to_mlflow

run_id = log_catboost_model_to_mlflow(
    model=trained_model,
    X_sample=train_df,
    run_name='my_run',
)
```

Optional CLI flags: `--experiment`, `--data-path` (parquet used for input examples), `--register` (Model Registry prefix).
