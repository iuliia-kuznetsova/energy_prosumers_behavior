# Feature engineering

Module: [`src/preds/feature_engineering.py`](src/preds/feature_engineering.py)

Class `FeatureEngineering` starts from the merged Polars frame and writes a pandas table for CatBoost. The target is still `capacity_factor`. Actual `power` is saved to `data/processed/power.parquet` and dropped from the feature frame.

Lags are built by joining each series to itself at `datetime − n hours`, not by `shift(n)` on rows. A missing hour therefore does not misalign later lags.

Target lags shorter than **48 h** are rejected: revealed targets are two days old.

## Engineered features

Raw `hour`, `day`, `day_of_week`, `month` and `year` are encoded and then dropped.

**Calendar and time**

- `hour_sin`, `hour_cos` — 24-hour cycle as sine and cosine. Hour 23 must sit next to hour 0. A raw `hour` integer treats them as far apart; one-hot encoding loses that they are neighbours on a clock.
- `day_sin`, `day_cos` — day-of-month cycle (period 31). Month-end should be close to month-start for the same reason.
- `day_of_week_sin`, `day_of_week_cos` — weekly cycle (period 7). Consumption and production both have weekday/weekend shapes; Sunday must be next to Monday.
- `month_sin`, `month_cos` — annual season cycle (period 12). December is next to January. Solar production and heating demand both follow this.
- `months_since_start` — integer months after 2021-09. This is a trend (fleet growth, more units appearing), not a cycle. Encoding `year` as sin/cos with period 2023 would only label the calendar year.
- `is_holiday` — Estonian public holiday (2021–2026). Household and business load, and sometimes production, change on holidays.
- `is_day_before_holiday` — the calendar day before a holiday. Demand often shifts on that eve.
- `is_dst_spring_forward` — the Estonia spring-forward 03:00 hour, when a clock hour is missing and power can be null.
- `is_dst_fall_backward` — the fall-back 03:00 hour, when a clock hour is repeated.

**Lag features**

Lags are joined at `datetime − n hours`, not shifted by row count, so a gap does not mislabel later hours. Target lags shorter than 48 h are rejected: revealed power is two days old.

- `capacity_factor_lag_n` — the same series’ capacity factor `n` hours earlier, for `n` in 48, 72, 96, 120, 144, 168, 336. This is the autoregressive signal: yesterday-minus-one, the same weekday last week, two weeks ago. 48 h is the shortest lag that is known at forecast time.
- `capacity_factor_same_hour_mean_7d` — mean capacity factor at this clock hour over the last seven *revealed* days (lags 48…192). It is the typical daily profile (the usual 13:00 production or 19:00 consumption) without using hours that have not been revealed yet.
- `capacity_factor_same_hour_std_7d` — spread of those same seven hours. A stable profile and a volatile one should not be treated the same.
- `electricity_price_lag_24`, `electricity_price_lag_168` — price one day and one week further back from the already legal (D−1) price. Recent and weekly price levels move both demand and, indirectly, production.
- `gas_price_lag_24`, `gas_price_lag_168` — the same extra steps for the daily gas midpoint, which tracks heating-related consumption.
- `hw_*_lag_24`, `hw_*_lag_168` — observed weather one day and one week further back from the already +48 h legal observation, for temperature, pressure, cloud cover, precipitation, wind and irradiance. They give persistence and a weekly weather analogue without using same-hour observed weather.

## Passed through from merging

These stay as model inputs (plus `{name}_is_missing` where a fill was applied). They are not created in this step.

**Size and prices:** `installed_capacity`, `eic_count`, `electricity_price`, `gas_price`.

**Solar:** `solar_elevation`, `solar_azimuth`, `clearsky_ghi`, `fw_ghi_x_capacity`.

**Forecast weather (`fw_`):** temperature, cloud layers, precipitation, snowfall, wind components, direct / surface / diffuse solar radiation.

**Observed weather (`hw_`), already shifted +48 h:** temperature, pressure, cloud cover, rain, snowfall, total precipitation, wind, direct and diffuse radiation.

**Identifiers kept on the frame, not used as predictors:** `row_id`, `datetime`, `data_block_id`, `prediction_unit_id`, `is_consumption`, `county`, `is_business`, `product_type`.

`is_consumption` only routes a row to the consumption or production model. Audit `*_block_id` columns are dropped before pandas conversion.

## Missing values

1. Sort by `(prediction_unit_id, is_consumption, datetime)` so forward-fill follows time inside a series.
2. In train mode, fill missing `power` with 0 on DST transition hours; drop other rows without `power`; drop 2021-09-01 and 2021-09-02.
3. Add `{col}_is_missing` before filling, if merging did not already add it.
4. Forward-fill prices globally; forward-fill capacity, `eic_count` and weather per series.
5. Recompute `capacity_factor` only when capacity is `> 0` and finite.
6. After lags, **drop only rows whose target is missing or non-finite**. NaN lags stay; CatBoost handles them.

## Retrain / test

History is concatenated so lags can look back across the train boundary. Only the new period is kept after lagging.

## How to run it

```bash
python -m src.preds.feature_engineering
```

The full pipeline writes `data/processed/train_features.parquet`.
