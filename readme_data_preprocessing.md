# Data preprocessing

Module: [`src/preds/data_preprocessing.py`](src/preds/data_preprocessing.py)

Class `DataPreparation` turns raw CSVs into typed Polars frames that merging can join. It does not create features.

## Modes

| Mode | When | Extra behaviour |
|---|---|---|
| `train` | First full run | Parse date strings; save client, prices and weather to `data/processed/*_train.parquet` |
| `retrain` | New revealed rows during inference | Concatenate onto the saved train parquets; newest row wins on duplicate keys |
| `test` | Predict a new day | Rename `prediction_datetime` → `datetime`; add an empty `target` so the schema matches train |

## Columns after this step

Each table is checked, reduced to this schema and cast to compact dtypes. Names below are the columns merging receives.

**Power**

- `row_id` — unique id of one hour of one series. It is the key used later to join predictions back to saved power.
- `datetime` — timestamp of the consumption or production hour.
- `data_block_id` — competition batch (one calendar day). Later joins may use this batch or an older one, never a newer one.
- `prediction_unit_id` — prosumer segment (county × business flag × product type).
- `is_consumption` — `True` for consumption, `False` for production. The two regimes are modelled separately.
- `county` — Estonian county of the segment.
- `is_business` — whether the segment is business or household.
- `product_type` — contract / product group of the segment.
- `target` — hourly power. Renamed to `power` in merging.

**Client**

- `date` — calendar date of the client snapshot (two days behind the target day of the same batch).
- `data_block_id` — batch that revealed this snapshot.
- `county`, `is_business`, `product_type` — keys that match a prediction unit.
- `eic_count` — number of consumption points in the segment. A more natural size measure for demand than solar capacity.
- `installed_capacity` — installed solar capacity of the segment, used to build `capacity_factor`.

**Electricity**

- `forecast_date` — hour the day-ahead price refers to (shifted +1 day in merging).
- `data_block_id` — batch that revealed this price.
- `euros_per_mwh` — day-ahead electricity price. The sign is kept. Negative prices are real Nord Pool hours (oversupply, often sunny or windy). They are a different regime from high positive prices; `abs()` would make −20 and +20 look the same. Merging later names this column `electricity_price`.

**Gas**

- `forecast_date` — day the gas quote refers to (shifted +1 day in merging).
- `data_block_id` — batch that revealed this quote.
- `lowest_price_per_mwh`, `highest_price_per_mwh` — daily range. Merging uses their midpoint as `gas_price`.

**Forecast weather**

- `latitude`, `longitude` — forecast grid point.
- `data_block_id` — batch of the forecast issue (used to keep only the day-ahead issue).
- `forecast_datetime` — hour the forecast is for.
- `hours_ahead` — lead time of that forecast row.
- `temperature` — forecast air temperature.
- `cloudcover_high`, `cloudcover_low`, `cloudcover_mid`, `cloudcover_total` — cloud cover by layer and overall. Layers matter for solar production; total alone hides whether clouds sit in front of the sun.
- `total_precipitation`, `snowfall` — forecast rain/snow. Snow covers panels; rain and wet weather change both production and consumption.
- `10_metre_u_wind_component`, `10_metre_v_wind_component` — forecast wind vectors (cooling of panels, wind-related demand).
- `direct_solar_radiation` — forecast beam irradiance.
- `surface_solar_radiation_downwards` — forecast global horizontal irradiance, the main driver of PV output.

**Historical weather**

- `datetime` — observation time (shifted +48 h in merging so it is always revealed).
- `data_block_id` — batch that revealed the observation.
- `latitude`, `longitude` — station / grid point.
- `temperature`, `surface_pressure` — observed temperature and pressure.
- `rain`, `snowfall` — observed precipitation components.
- `cloudcover_total` — observed total cloud cover.
- `windspeed_10m`, `winddirection_10m` — observed wind.
- `direct_solar_radiation`, `diffuse_radiation` — observed irradiance split.

**Mapping**

- `county` — Estonian county id.
- `latitude`, `longitude` — weather-station coordinates used to average the grid into that county.

Dropped here on purpose: `origin_date` / `origin_datetime`, `dewpoint`, forecast `shortwave_radiation`. Those are unused.

## How to run it

```bash
python -m src.preds.data_preprocessing
```

The full pipeline loads the CSVs with `load_raw_data` and then calls `DataPreparation(mode='train').fit_transform(...)`.
