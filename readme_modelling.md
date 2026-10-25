# Modelling and evaluation

Module: [`src/preds/modelling_catboost.py`](src/preds/modelling_catboost.py)

Class `CatBoostRegressorModel` trains **two** regressors: consumption (`is_consumption == 1`) and production (`== 0`). The fit target is `capacity_factor`. Reported scores are **MAE on power**.

## Why two models

Consumption and production have different daily shapes, weather sensitivity and capacity-factor ranges. `is_consumption` is constant inside each model, so it is not a feature.

## What is not a feature

`datetime`, `row_id`, `data_block_id`, `date`, `power`, `is_consumption`, and the audit `*_block_id` columns. Everything else on the feature frame is a predictor.

Categorical features: `prediction_unit_id`, `county`, `is_business`, `product_type`, `is_holiday`, `is_day_before_holiday`, `is_dst_spring_forward`, `is_dst_fall_backward`.

## Data split

```text
All rows
├── Pre-holdout (before 2023-04-01)
│   └── Rolling-origin months: January, February, March 2023
└── Holdout (2023-04-01 onward) — never used for Optuna, early stopping or the final fit
```

Each CV month uses three slices of whole `data_block_id` days:

| Slice | Role |
|---|---|
| `fit` | Train the temporary early-stopping model |
| `inner` (14 days before the month) | Choose the number of trees (`eval_set` only) |
| `train` = `fit` + `inner` | Refit the model that is scored |
| `val` | Month scored for Optuna; never an `eval_set` |

Example for February 2023: `fit` is history before mid-January, `inner` is the last 14 days of January, `val` is February.

## Loss and conversion back to power

Each CatBoost `Pool` uses `sample_weight=installed_capacity`. Weighted MAE on the capacity factor is then equal to MAE on power.

After `predict`:

1. Clip the predicted capacity factor to `[0, segment upper]`. The upper bound is `1.1 ×` the 99.9th percentile of that segment’s training target, at least 1.
2. Multiply by `installed_capacity`.

## Hyperparameter search

Optuna minimizes the **row-weighted mean power MAE** across the three CV months. Each study uses a seeded `TPESampler`, SQLite persistence (`models/optuna_studies.db` or `OPTUNA_STORAGE`), and `catch=(Exception,)`. Set `OPTUNA_TIMEOUT_SECONDS` to stop on a time budget. A later run continues remaining trials.

Searched fields: `iterations` (100–1000), `depth` (4–12), `learning_rate`, `l2_leaf_reg`, `grow_policy`, `score_function`.  
`score_function` is always chosen from `{Cosine, L2}`. The illegal pair Cosine + `Lossguide` is pruned.

After the best trial, the last 14 pre-holdout days choose a tree count. By default that count is scaled by `n_full_train / n_early_stopping_train`. `--no-tree-scale` keeps the unscaled count. The final model is refit on all pre-holdout rows with no `eval_set`.

## Evaluation

After `fit()`, the holdout is scored once. The report also includes:

- holdout MAE overall, consumption, production
- the same split for the CV months
- naive baselines: power from 48 h ago and from 168 h ago

Written to `results/evaluation_results.json` and stored on the model as `eval_report_`.

## Debug run

`--debug` (or `debug=True`) keeps only `prediction_unit_id` **3, 19, 48, 60**. Use it to exercise the pipeline, not to report performance. The unit list is saved in metadata so a loaded debug model stays limited.

## How to run it

```bash
python -m src.preds.main_preds --n-trials 20 --model-name catboost_prosumer
python -m src.preds.main_preds --debug --no-tree-scale
python -m src.preds.modelling_catboost --debug
```

Saved files: `models/<name>/consumption.cbm`, `production.cbm`, `metadata.json`.
