# Solution Walkthrough: Enefit — Predict Energy Behavior of Prosumers

This document explains, step by step, what the code in [`src/preds`](src/preds) does, why each step exists, what is good about the approach, and what is wrong with it. It ends with a recommended neural network for ensembling with the CatBoost models and a concrete implementation plan.

**TL;DR**

- The pipeline loads the 7 Kaggle tables, joins them onto the hourly power rows, builds calendar and lag features, and trains **two CatBoost regressors** (consumption and production) on `capacity_factor = power / installed_capacity`, tuned with Optuna.
- The engineering is clean (typed schemas, modular sklearn-style transformers, CLI entry point), and the modelling ideas (capacity normalization, separate models per regime, cyclical time encodings) are sound.
- **Critical drawback:** several features are built with information that is *not available at forecast time* in this competition: target lags of 12 h and 24 h, same-hour observed weather, same-day client capacity, same-hour prices and same-day weather forecasts. The model learns to rely on them, so its validation scores are optimistic and the inference path cannot reproduce those features. The fix is to align every join with the competition's `data_block_id` availability rules (section 5.1).
- The second major issue is that **there is no honest performance number**: the reported MAE is computed on the training data, and the Optuna cross-validation is also optimistic.
- Recommended next model for the ensemble: a **global day-ahead encoder–decoder network** (dilated TCN encoder over the past two weeks plus a TiDE-style per-hour decoder fed with weather forecasts). Fix the leakage first, otherwise the ensemble will be tuned on misleading scores.

---

## 1. The competition in one page

**Task.** For every *prediction unit* (a segment of Estonian prosumers defined by `county`, `is_business` and `product_type`, about 69 units), forecast hourly **consumption** and **production** of electricity. The metric is **MAE** on the raw `target` (power).

**Data.** About 2M hourly rows from 2021-09-01 to 2023-05-31 in `train.csv`, plus six context tables: `client.csv` (installed solar capacity and `eic_count` per segment per day), `electricity_prices.csv` (hourly day-ahead prices), `gas_prices.csv` (daily min/max), `historical_weather.csv` (observed weather on a lat/lon grid), `forecast_weather.csv` (weather forecasts on the same grid, issued once a day for 1–48 hours ahead) and `weather_station_to_county_mapping.csv`.

**Timing rules (the most important thing to get right).** Kaggle served the test set day by day via an API. Every table has a `data_block_id`: rows with the same block id are delivered together, and you must predict the whole next day using only data from blocks up to that id. The example test files in `data/raw/example_test_files` show what you actually have when predicting the day `2023-05-28` (block 634):

| Information | Latest value available when predicting day D | Effective minimum lag |
|---|---|---|
| Target (`revealed_targets`) | all hours of D−2 | **48 h** |
| `client` (installed capacity, eic_count) | date D−2 | 2 days |
| Electricity prices | hours of D−1 | 24 h |
| Gas prices | date D−1 | 1 day |
| Historical (observed) weather | D−2 11:00 … D−1 10:00 | ~14–37 h (48 h is always safe) |
| Forecast weather | forecast issued at D−1 02:00, `hours_ahead` 1–48 | target hours of D are `hours_ahead` **22–45** |

Every feature must respect these lags in training too. Otherwise the model learns from information it will never have at prediction time.

---

## 2. Pipeline overview

```text
data/raw/*.csv  (Kaggle)
   │  data_loading.py         download & verify 7 CSVs
   ▼
   │  data_preprocessing.py   DataPreparation: column checks, dtype casting, parquet caching
   ▼
   │  data_merging.py         DataMerging: power ⟵ client, prices, forecast & historical weather
   ▼                          + capacity_factor = power / installed_capacity
   │  feature_engineering.py  FeatureEngineering: calendar, holidays, DST, lags, NA handling
   ▼                          → data/processed/train_features.parquet, power.parquet
   │  modelling_catboost.py   CatBoostRegressorModel: Optuna + TimeSeriesSplit,
   ▼                          one model for consumption, one for production
models/<name>/{consumption.cbm, production.cbm, metadata.json}
results/evaluation_results.json   (train MAE in power units)
```

Everything is orchestrated by [`src/preds/main_preds.py`](src/preds/main_preds.py).

---

## 3. Step-by-step: what each file does

### Step 1 — Setup (`main_preds.py`)

`run_full_pipeline` loads `.env` (via `python-dotenv`), changes the working directory to the project root and creates `data/raw`, `data/processed`, `models` and `results`; each path can be overridden with an environment variable (`RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `MODELS_DIR`, `RESULTS_DIR`). After each heavy stage it deletes intermediate objects and calls `gc.collect()` to keep memory under control, because the merged frame is large.

CLI flags:

- `--skip-download`: only verify that the CSVs exist.
- `--skip-training`: load a saved model instead of fitting.
- `--n-trials`: number of Optuna trials (default 10).
- `--model-name`: folder name under `models/`.

### Step 2 — Data loading (`data_loading.py`)

1. `verify_downloaded_files` checks that all 7 expected CSVs exist in `data/raw` and logs their sizes. If they do, the download is skipped.
2. Otherwise `check_kaggle_credentials` looks for `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`.
3. `download_from_kaggle` calls the Kaggle CLI (`kaggle competitions download -c predict-energy-behavior-of-prosumers`), unzips the archive and deletes the zip.
4. The files are verified again.

### Step 3 — Preprocessing (`data_preprocessing.py`, class `DataPreparation`)

`load_raw_data` reads the CSVs into **Polars** DataFrames. `DataPreparation` then, for each table:

- Checks that the required columns exist (`_check_columns`) and fails loudly if any are missing.
- Parses date/datetime strings explicitly (train mode only; in retrain/test the API already provides typed columns).
- Keeps only the schema columns and casts them to compact dtypes (`Int8`/`Int16` ids, `Float32` weather, `Boolean` flags).
  - Columns dropped here include `data_block_id`, `eic_count`, `origin_date`/`origin_datetime`, `dewpoint`, the cloud-cover layers, `shortwave_radiation` and `snowfall` from the forecast table.
- **Train mode:** saves client, prices and both weather tables to `data/processed/*_train.parquet`.
- **Retrain/test mode:** loads those parquet files and concatenates the new rows. On duplicate keys the new row wins (`unique(..., keep='last')`), so lags can later be computed across the train/test boundary.
- **Test mode:** `prediction_datetime` is renamed to `datetime` and an empty `target` column is added so all modes share one schema.

### Step 4 — Merging (`data_merging.py`, class `DataMerging`)

The power table (`target` renamed to `power`) is the base; all joins are **left joins**, so no power row is lost.

1. **Client:** joined on `product_type, county, is_business, date` and brings in `installed_capacity`.
2. **Electricity price:** joined on `datetime = forecast_date`. The code takes `abs(euros_per_mwh)`.
3. **Gas price:** the daily mid price `(lowest + highest) / 2` is expanded to 24 hourly rows with a cross join, then joined on `datetime`.
4. **Forecast weather:**
   - Keeps rows with `hours_ahead <= 24` and maps grid points to counties via the station mapping.
   - Averages the 7 variables per `(county, forecast_datetime)` and adds the prefix `fw_`.
   - Post-processing: cloud cover is scaled to percent and capped at 100, and precipitation is converted to mm.
   - Derives `fw_diffuse_solar_radiation = max(surface_solar_radiation_downwards − direct_solar_radiation, 0)`.
   - Joined on `county, datetime`; gaps are filled with the median across counties at the same timestamp.
5. **Historical weather:** same county mapping and averaging (prefix `hw_`), plus `hw_total_precipitation = rain + snowfall`. Joined on `county, datetime` with the same median fill.
6. **Target transform:** `capacity_factor = power / installed_capacity`. Columns are put in a fixed order and the frame is sorted by `datetime`.

### Step 5 — Feature engineering (`feature_engineering.py`, class `FeatureEngineering`)

| Group | Features |
|---|---|
| Cyclical time | `hour`, `day` (period 31), `day_of_week` (7), `month` (12), `year` (period 2023), each as `sin`/`cos`; the raw components are dropped |
| Calendar | `is_holiday`, `is_day_before_holiday` (Estonian holidays 2021–2026 from the `holidays` package) |
| DST | `is_dst_spring_forward`, `is_dst_fall_backward` (the 03:00 transition hour, hard-coded for 2021–2026) |
| Lags | `shift(12, 24, 168, 720)` grouped by `(prediction_unit_id, is_consumption)` for `capacity_factor`, both prices and the 8 historical-weather columns, 44 lag columns in total |
| Context | `installed_capacity`, prices, `fw_*`, `hw_*`, identifiers |

Missing-value handling (`_handle_missing_values`):

- Missing `power` in DST hours is set to 0, and other rows without `power` are dropped.
- The first two days (2021-09-01 and 2021-09-02) are removed.
- Prices are forward-filled; capacity and weather are forward-filled per `(prediction_unit_id, is_consumption)`; `capacity_factor` is recomputed where it was null.
- After lagging, `drop_nulls()` removes every row with any null, which discards the first 720 rows (30 days) of every series.

The actual `power` per `row_id` is saved to `data/processed/power.parquet` for evaluation and then dropped. The result is converted to pandas for CatBoost.

Retrain/test mode additionally concatenates the new rows to the previous feature frame (`_merge_historical_data`) so that lags can look back into history, then keeps only rows from the new period.

### Step 6 — Modelling (`modelling_catboost.py`, class `CatBoostRegressorModel`)

- **Features:** every column except `datetime`, `row_id` and `capacity_factor`. Categorical features are `prediction_unit_id`, `is_consumption`, `county`, `is_business`, `product_type` and the four calendar/DST flags.
- **Two independent models:** one for `is_consumption == 1` and one for `== 0`.
- **Tuning:** for each model, an Optuna study (10 trials by default) searches `iterations` (100–1000), `depth` (4–12), `learning_rate`, `l2_leaf_reg`, `score_function` (Cosine/L2) and `grow_policy` (SymmetricTree/Depthwise/Lossguide). It minimizes the mean MAE of `capacity_factor` over a 3-fold `TimeSeriesSplit`.
- **Final fit:** the best parameters are refitted on the whole subset with `loss_function='MAE'`.
- **Prediction (`transform`):**
  - If `capacity_factor_lag_*` columns exist, it runs a row-by-row "recursive" loop that writes each prediction into the lag column of row `idx + lag`; otherwise it does a batch prediction.
  - Power is `max(prediction × installed_capacity, 0)`.
- **Persistence:** `save`/`load` write and read `consumption.cbm`, `production.cbm` and `metadata.json` (features, categorical features and best parameters).

### Step 7 — Evaluation (`main_preds.py` + `compute_mae_power`)

Predictions for the **training rows** are joined to `power.parquet` by `row_id`. The MAE in power units is written to `results/evaluation_results.json`.

### Not part of the flow

- [`src/preds/mlflow_logging.py`](src/preds/mlflow_logging.py) belongs to a different project: it imports `src.recs.modelling_ovr`, which does not exist here, so importing it fails.
- `Enefit.ipynb` is the original Kaggle notebook that the scripts were refactored from.

---

## 4. Pros

- **Capacity-normalized target.** Production scales almost linearly with installed solar capacity, and capacity grows strongly over the 21 months. Predicting `capacity_factor` lets the model share patterns between small and large segments and across time. This is the same normalization most strong public solutions used.
- **Separate consumption and production models.** The two regimes have very different drivers: production depends mostly on solar radiation and daylight, consumption on temperature, calendar and business type. Two models avoid one model having to split on `is_consumption` first.
- **Domain-aware calendar.** Estonian holidays, the day before a holiday, and DST hours are real effects on consumption that a generic feature set would miss.
- **Cyclical encodings** keep hour 23 close to hour 0 and December close to January.
- **Weather is aggregated to the county level**, which matches the granularity of the prediction units. The derived diffuse radiation is a physically meaningful feature for PV output.
- **Engineering quality.**
  - Explicit schemas and column checks.
  - Polars for fast joins.
  - sklearn-style transformers with `train/retrain/test` modes.
  - Parquet caching, a CLI with skip flags, logging, and `gc` between stages.
  - Model and metadata persistence.
- **Chronological CV** (`TimeSeriesSplit`) instead of random K-fold, and **MAE loss**, which matches the competition metric.

---

## 5. Cons and how to fix them

Ordered by impact. Items 5.1 and 5.2 are the ones to fix before doing anything else, including the neural network.

### 5.1 CRITICAL — features use information that is not available at forecast time

Compare the features built in `data_merging.py` and `feature_engineering.py` with the timing table in section 1:

| Feature as implemented | What the competition actually provides | Consequence |
|---|---|---|
| `capacity_factor_lag_12`, `capacity_factor_lag_24` | Targets arrive with a **48 h** delay | The two most predictive features never exist at inference |
| `hw_*` joined at the **same hour** as the target (lag 0) | Observed weather is at least ~14–37 h old | The model "sees" the real sunshine at the hour it predicts; for production this is close to seeing the answer |
| `hw_*_lag_12` (and `lag_24` for hours after ~10:00) | Same as above | Unavailable |
| `fw_*` with `hours_ahead <= 24`, i.e. the forecast issued on the target day itself | Only the forecast issued the day before (`hours_ahead` 22–45) | Training uses fresher and more accurate forecasts. At test time only 3 hours of the day would match the filter and the rest would be median- or forward-filled |
| `electricity_price`, `gas_price` at the target hour/day | Prices arrive with a 1-day lag (`forecast_date` + 1 day) | Mild leakage; at test time the join finds nothing and forward-fill copies yesterday's last price to every hour |
| `installed_capacity` on the same date (also used to build the target) | Client data is 2 days old | Mild leakage; at test time it is forward-filled |

**Why this is the main problem.**

- All three of the model's information sources (recent target history, observed weather, fresh forecasts) are "too good" in training. CatBoost will put most of its trust in them.
- Any score computed this way, including the Optuna CV MAE, is far better than what the model can achieve on real data.
- At inference the same columns are missing, so they get filled with stale values. The model then runs on a feature distribution it has never seen.
- The row-by-row "recursive forecasting" in `transform` was an attempt to work around the missing short lags. It makes predictions slower and noisier (see 5.3) and cannot recover observed weather.

**How to fix it.** Keep `data_block_id` in every schema in `data_preprocessing.py`, and rebuild every join so that a row in block *b* only uses data from blocks ≤ *b*:

```python
# data_merging.py — availability-aligned joins (sketch)
client = client_df.with_columns((pl.col('date') + pl.duration(days=2)).alias('date'))   # D-2 -> D
elec = electricity_df.with_columns(
    (pl.col('forecast_date') + pl.duration(days=1)).alias('datetime')                  # D-1 -> D
)
gas = gas_df.with_columns(
    (pl.col('forecast_date') + pl.duration(days=1)).cast(pl.Date).alias('date')         # join on date, no hourly cross join
)
fw = (
    forecast_weather_df
    .filter(pl.col('hours_ahead').is_between(22, 45))                                   # forecast issued the day before
    .rename({'forecast_datetime': 'datetime'})
    # ... county mapping + mean per (county, datetime) as today
)
hw = historical_weather_df.with_columns(
    (pl.col('datetime') + pl.duration(hours=48)).alias('datetime')                      # observed weather 48 h ago
)
```

Then in `feature_engineering.py`:

- Use **target lags of at least 48 h**, for example `[48, 72, 96, 120, 144, 168, 336]` hours, plus rolling means and standard deviations over lagged windows (e.g. the mean of the same hour over the last 7 available days).
- Use historical-weather lags of 48 h and more (or per-block daily aggregates). Keep prices at lag 24 and more.
- **Make lags time-based, not row-based.** `shift(n)` counts rows, so a missing hour silently changes the meaning of every lag. Build lags with a join on `datetime - duration(hours=n)` per `(prediction_unit_id, is_consumption)`, or reindex each series to a complete hourly grid first.
- **Delete the recursive loop.** Once all lags are at least 48 h, the whole next day can be predicted in one batch, which is exactly how the Kaggle API works.
- Add a unit test that, for a few random rows, asserts that every feature value comes from a `data_block_id` ≤ the row's own block id.

After this change the validation MAE will get noticeably *worse*. That is expected: it becomes the real number.

### 5.2 CRITICAL — no trustworthy evaluation

- **The reported MAE is in-sample.** `compute_mae_power` predicts the same rows the final model was trained on (`results/evaluation_results.json` → `train_mae_power`). It says nothing about generalization, which is why the `Results` table in `README.md` is still "TBD".
- **The Optuna CV is also optimistic.**
  - Each fold passes the validation pool as `eval_set`. CatBoost's default `use_best_model=True` then keeps the iteration that is best *on that same validation fold*, so the number of trees is effectively tuned on the data it is scored on.
  - The final model is then refitted with the full `iterations` value, so it is not even the configuration that was scored.
- **Folds are cut by row position**, so rows from the same hour (about 138 units) can end up on both sides of a split.

**Fix:**

- Build folds from **whole days**, keyed by `data_block_id`. Use a rolling-origin scheme, for example train up to month *m* and validate on month *m+1* for the last 3–4 months.
- Keep the last ~2 months (e.g. 2023-04-01 → 2023-05-31) as a final **holdout** that is never used for tuning.
- Inside folds, use an inner time split for early stopping (`early_stopping_rounds`) and record the best iteration; refit the final model with that iteration count, scaled up for the larger training set.
- Report validation MAE **in power units**, both overall and split by consumption/production. That is the competition metric.
- Add naive baselines (target at lag 48 h; target at lag 168 h) to the report. Every model must beat them.

### 5.3 HIGH — the inference path does not work yet

- `main_preds.py` never builds a test set, calls the Kaggle API (`data/raw/enefit`) or writes a submission.
- In `FeatureEngineering._create_lag_features` for retrain/test mode, `.otherwise(pl.col(col))` fills the lag column with the *current* value of the feature instead of keeping the existing lag.
- The recursive loop in `CatBoostRegressorModel.transform` writes to row `idx + lag` of a frame sorted by `(datetime, prediction_unit_id, is_consumption)`. With about 138 rows per hour, `idx + 24` is a different unit at roughly the same hour, not the same unit 24 hours later. The unit-equality check therefore almost always fails and the lags are never updated.
- The loop also calls `predict` once per row. Because `compute_mae_power` runs on the training frame, Step 7 makes about 2 million single-row predictions, which is extremely slow.

**Fix:** after 5.1 the loop is no longer needed and can be removed. Add a `predict_day(block)` path that reproduces the Kaggle API loop: append new data, rebuild features for the new block, batch-predict, and write `row_id, target`. Run it over the example test files as a smoke test.

### 5.4 HIGH — tuning setup — addressed

- Cosine is suggested only for `SymmetricTree` and `Depthwise`. `Lossguide` is restricted to `L2`. Failed Optuna trials are caught (`catch=(Exception,)`).
- Each study uses a seeded `TPESampler`, inner-slice early stopping, SQLite persistence (`models/optuna_studies.db` or `OPTUNA_STORAGE`), and an optional `OPTUNA_TIMEOUT_SECONDS` budget. Later runs continue remaining trials instead of starting over.
- `is_consumption` is no longer a categorical feature. It only routes rows to the consumption or production model.

### 5.5 MEDIUM — target, metric and features — addressed

- Training pools use `sample_weight=installed_capacity`, so MAE on `capacity_factor` equals MAE on power.
- `eic_count` is kept from `client.csv` and used as a feature alongside `installed_capacity`.
- Predicted capacity factor is clipped to `[0, segment 99.9th-percentile × 1.1]` before multiplying by capacity.
- Electricity prices keep their sign. Negative prices are no longer passed through `abs()`.
- `year_sin` / `year_cos` were replaced by `months_since_start` (months after 2021-09).
- Solar elevation, azimuth and a simple clear-sky GHI are added per county centroid and hour (`pvlib`). Forecast `snowfall`, `cloudcover_low/mid/high` and `fw_surface_solar_radiation_downwards` are kept, plus `fw_ghi_x_capacity`.
- After lags, only rows with a finite target are dropped. CatBoost trains with NaN lags.
- Missing-value handling:
  - Frames are sorted by `(prediction_unit_id, is_consumption, datetime)` before forward-fill.
  - `{col}_is_missing` flags are added before median or forward fills.
  - Zero or non-finite capacity leaves `capacity_factor` null instead of `inf`.
  - Missing rates are logged after the weather and price joins.


### Recommended fix order

1. 5.1 (availability-aligned features) together with 5.2 (day-based rolling CV, holdout, power-MAE report and naive baselines).
2. 5.3: batch day-ahead inference plus a Kaggle-API smoke test.
3. 5.4 and 5.5, keeping only the changes that improve the holdout MAE.

---

## 6. Next step: a neural network to ensemble with CatBoost

### 6.1 What the network must handle

- **Panel of about 138 series:** about 69 units × consumption/production, hourly, about 21 months. One training sample is one *(series, day)*, so there are only about 600 days × 138 ≈ **83k samples**. The network must be small and well regularized.
- **Day-ahead, multi-horizon output:** at block *b* you predict all 24 hours of day D at once, and the last known target is from D−2, so the horizon is 25–48 h after the last observation.
- **Three kinds of inputs:**
  - *Past-only:* target history, observed weather and prices.
  - *Known future:* weather forecasts for the 24 target hours, calendar, solar position and the latest capacity.
  - *Static:* unit, county, product type and business flag.
- **Weather forecasts are the main driver of production.** Any architecture that cannot take known-future covariates (e.g. vanilla N-BEATS, PatchTST, univariate LSTMs) is a poor fit.
- **Diversity for the ensemble:** trees approximate functions with piecewise-constant steps. A network that learns smooth, multiplicative relations (radiation × capacity, temperature curves) and the shape of the recent profile will make different errors, which is what makes blending pay off.

### 6.2 Recommended architecture: global day-ahead encoder–decoder (TCN encoder + TiDE-style decoder)

```text
                       static: unit_id, county, product_type, is_business  ──► embeddings ─────────────┐
                                                                                                        │
past window, L = 336 h (14 days) ending at D-2 23:00                                                    │
  capacity_factor, hw_* (aligned), prices, calendar, missing-mask  ──► dilated causal TCN encoder ──► context c
                                                                       (6 residual blocks, 64 ch,        │
                                                                        dilations 1..32, plus a pooled   │
                                                                        "same hour, last 7 days" vector) │
                                                                                                        ▼
future, 24 target hours of D (h = 1..24)                          for each hour h (shared weights):
  fw_* (hours_ahead 22–45), solar elevation, clear-sky,   ──►   residual MLP([c, static, future_h, hour_emb])
  hour/weekday/holiday/DST, installed_capacity (latest)          ──► softplus ──► capacity_factor_h
                                                                                    × installed_capacity
                                                                                    ──► power_h (24 outputs)
```

**Why this design:**

- **Encoder–decoder split = correct information flow.** The past-only inputs go through the encoder, and the known-future inputs (forecasts, calendar) go directly into each hour's decoder. This matches the competition's timing rules by construction.
- **The TCN encoder** captures daily and weekly patterns in the recent history efficiently. It trains in parallel and is more stable than LSTMs on 336-step windows.
- **The TiDE-style decoder** (one shared MLP applied per hour, conditioned on that hour's forecast) is simple, fast and strong for horizons with rich future covariates. It is the part that turns "forecast radiation at 13:00" into "production at 13:00".
- **Global model with entity embeddings:** one network learns from all units, which helps small and late-appearing units.
- **24 outputs at once:** no recursion and no error accumulation.

**Settings to start with:**

- Separate models for consumption and production, mirroring the CatBoost setup. Later, try one model with an `is_consumption` embedding.
- Target `capacity_factor` (production) and `target / eic_count` or `capacity_factor` (consumption); standardize the inputs with train-fold statistics.
- **Loss:** L1 on power, or L1 on the capacity factor weighted by `installed_capacity` (both match the competition metric).
- **Size:** about 100k–300k parameters, dropout 0.1–0.2, weight decay 1e-4.
- **Training:** AdamW, one-cycle learning rate around 1e-3, batch 256 (series-days), early stopping on the validation month, and the average of 3–5 seeds.

**Alternatives, in order of preference:**

1. **Temporal Fusion Transformer (TFT)** has the same input taxonomy plus variable selection and attention. It is more interpretable but heavier and easier to overfit on about 83k samples. Use it as a second neural model if the TCN works.
2. **Off-the-shelf TiDE / TFT** from [`neuralforecast`](https://nixtlaverse.nixtla.io/neuralforecast/) (`futr_exog_list`, `hist_exog_list`, `stat_exog_list`). This is the fastest way to get a first neural baseline before writing the custom model, and it is worth starting with.
3. **MLP on the same tabular features as CatBoost** (with entity embeddings). It is cheap and blends well, but it does not see the shape of the history.

### 6.3 Implementation plan

1. **Fix the data first (sections 5.1–5.2).** Both models must be trained on availability-aligned features and evaluated on the same day-based folds and holdout. Otherwise ensemble weights are tuned on leaked scores.
2. **Freeze the evaluation protocol.**
   - Rolling-origin folds by month (e.g. validate on 2023-01, 2023-02 and 2023-03), plus the holdout 2023-04 → 2023-05.
   - One shared script that, given a model's predictions keyed by `row_id`, reports power MAE overall, by `is_consumption` and by hour.
3. **Produce out-of-fold (OOF) CatBoost predictions** for every validation fold and the holdout. Save them as `oof_catboost.parquet` (`row_id`, `fold`, `pred`).
4. **Build the sequence dataset** (new module `src/preds/nn_dataset.py`):
   - Reindex each series to a full hourly grid and store dense arrays of shape `[n_series, n_hours, n_features]` (past features), `[n_series, n_hours, n_future_features]` (known-future features) and static ids.
   - A PyTorch `Dataset` indexes `(series, day)` and slices the past window ending at D−2 23:00 and the 24 future hours of D, including masks for missing values.
   - Fit the normalization on training folds only.
5. **Quick neural baseline:** `neuralforecast` TiDE (and/or TFT) with the same splits. This shows quickly whether a neural model adds value.
6. **Custom TCN + TiDE-decoder model** (`src/preds/modelling_nn.py`, PyTorch or Lightning) with the same `fit / transform / save / load` interface as `CatBoostRegressorModel`, so `main_preds.py` can switch models with a flag.
7. **Train and select:** early stopping per fold, then a small Optuna search (lookback 168 vs 336, channels, dropout, learning rate), then a 3–5-seed average.
8. **Blend:**
   - Start with a simple weighted average, with a weight `w` per `is_consumption` (optionally per hour of day) chosen by minimizing MAE on the **OOF** predictions.
   - Check the result on the **holdout**, which was never used to choose `w`.
   - Only try stacking (e.g. a small model on `[pred_cb, pred_nn, hour, capacity]`) if the simple blend clearly helps.
9. **Day-ahead inference loop:** run both models through the same `predict_day(block)` path over the example test files, blend, clip at 0 and write the submission.
10. **Tracking:** log parameters, fold metrics, OOF files and model artifacts to MLflow (after rewriting `mlflow_logging.py` for this project). Then fill in the `Results` table in `README.md` with holdout MAE for CatBoost, the network and the blend.

---

## 7. Running the current pipeline

From the project root:

```powershell
python -m src.preds.main_preds                                              # full run
python -m src.preds.main_preds --skip-download                              # data already in data/raw
python -m src.preds.main_preds --n-trials 20 --model-name catboost_prosumer
python -m src.preds.main_preds --skip-training --model-name catboost_prosumer
```

Outputs:

- `data/processed/train_features.parquet` and `power.parquet`
- `models/<model-name>/{consumption.cbm, production.cbm, metadata.json}`
- `results/evaluation_results.json`, which holds the **in-sample** MAE; see 5.2

Notes:

- `--skip-training` still runs loading, preprocessing, merging and feature engineering; it only replaces fitting with loading a saved model.
- Kaggle credentials are only needed when the raw CSVs are missing.
