# Data loading

Module: [`src/preds/data_loading.py`](src/preds/data_loading.py)

This step puts the seven competition CSVs into `data/raw`. The pipeline calls it first unless `--skip-download` is set.

## Files

| File | Role |
|---|---|
| `train.csv` | Hourly consumption and production (`target` = power) per prediction unit |
| `client.csv` | Daily `installed_capacity` and `eic_count` per segment |
| `electricity_prices.csv` | Hourly day-ahead electricity price (`euros_per_mwh`) |
| `gas_prices.csv` | Daily gas price range |
| `historical_weather.csv` | Observed weather on a lat/lon grid |
| `forecast_weather.csv` | Weather forecasts on the same grid, issued once a day |
| `weather_station_to_county_mapping.csv` | Grid point → Estonian county |

The public Kaggle data is used for education only. Manual download: [competition data page](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/data).

## What the code does

1. `verify_downloaded_files` checks that every name in `EXPECTED_FILES` exists under `RAW_DATA_DIR` (default `./data/raw`) and logs sizes.
2. If a file is missing, `check_kaggle_credentials` looks for `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`.
3. `download_from_kaggle` runs  
   `kaggle competitions download -c predict-energy-behavior-of-prosumers`,  
   unzips into `data/raw`, and deletes the zip.
4. The file list is verified again.

A later stage never starts if a required CSV is absent.

## How to run it

```bash
python -m src.preds.data_loading
# or, from the full pipeline:
python -m src.preds.main_preds
python -m src.preds.main_preds --skip-download
```

`--skip-download` only verifies the CSVs. Use it when `data/raw` is already populated.
