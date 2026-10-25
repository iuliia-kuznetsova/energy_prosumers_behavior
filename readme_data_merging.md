# Data merging

Module: [`src/preds/data_merging.py`](src/preds/data_merging.py)

Class `DataMerging` left-joins every context table onto the hourly power rows. Shifts follow the competition reveal rules, so a feature on day D is one that would have been known when predicting D.

Power (`target` renamed to `power`) is the base frame. No power row is dropped by a join.

## Joins

| Source | Shift | Join keys | Columns brought in |
|---|---|---|---|
| Client | `date + 2 days` | `product_type`, `county`, `is_business`, `date` | `installed_capacity`, `eic_count` |
| Electricity | `forecast_date + 1 day` | `datetime` | `electricity_price` (signed) |
| Gas | `forecast_date + 1 day` | `date` | `gas_price` = midpoint of low and high |
| Forecast weather | same `data_block_id`, valid time = target hour | `county`, `datetime`, `data_block_id` | `fw_*` |
| Historical weather | `datetime + 48 hours` | `county`, `datetime` | `hw_*` |

Weather grid points are mapped to counties and averaged. Cloud cover is scaled to percent and capped at 100. Forecast precipitation is converted to mm. Diffuse forecast radiation is  
`max(surface − direct, 0)`.

Historical precipitation is `hw_rain + hw_snowfall`.

Source batch ids are stored as `client_block_id`, `electricity_block_id`, `gas_block_id`, `fw_block_id`, `hw_block_id` for leakage checks. Feature engineering drops them before training.

## Derived columns

- `capacity_factor` — `power / installed_capacity` for that hour, using the capacity that was already known two days earlier. It is the training target. Raw power is not comparable across units: a 5 MW plant and a 50 kW rooftop can have the same weather response but errors of very different size. Dividing by capacity puts every series on a utilization scale (production is typically 0–1; consumption is power relative to the segment’s solar size). The model then learns *how fully the capacity is used*, not *how large the segment is*. At prediction time the capacity factor is multiplied back by the known `installed_capacity` to recover power. If capacity is missing, zero or non-finite, the ratio is left null instead of `inf`.
- `solar_elevation`, `solar_azimuth`, `clearsky_ghi` — sun height, compass direction and a simple clear-sky irradiance at the county centroid (`pvlib`, `Europe/Tallinn`). They tell the model whether that hour is physically able to produce solar power, independently of the weather forecast.
- `fw_ghi_x_capacity` — forecast surface solar radiation times installed capacity. Expected generation scales with both sunlight and plant size; the product gives the model that interaction in one column.

## Missing values at this step

Before any fill, the code writes `{col}_is_missing` and logs missing rates. County gaps in weather are then filled with the **median across counties at the same timestamp**. Those fills stay visible to the model through the indicator flags.

## How to run it

```bash
python -m src.preds.data_merging
```

Standalone usage first runs preprocessing, then `DataMerging(mode='train').fit_transform(processed_data)`.
