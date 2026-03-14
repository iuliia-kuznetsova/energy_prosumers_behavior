# Predict Energy Behavior of Prosumers
Predict energy consumption and production patterns of prosumers (consumers who also produce energy) in Estonia to minimize energy imbalance costs for energy companies.
Solution to the Kaggle competition [Enefit - Predict Energy Behavior of Prosumers](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers).


## Problem Statement
### Business Context
Energy companies face significant costs from energy imbalance — the difference between predicted and actual energy usage. Prosumers, who both consume and produce energy, make forecasting especially challenging because their behavior depends on weather, time of day, installed solar capacity, and other factors.

### ML Objective
Build a model that predicts hourly energy consumption and production (in power units) for each prosumer segment, evaluated by **Mean Absolute Error (MAE)**.

Strategy:
- Train two separate **CatBoost** regressors — one for consumption, one for production
- Use `capacity_factor = power / installed_capacity` as the target variable during training
- Convert predictions back to power by multiplying with `installed_capacity`
- Optimize hyperparameters with **Optuna** and **TimeSeriesSplit** cross-validation


## High-Level Solution Architecture

```
┌─────────────────────┐
│   DataPreparation   │  → Schema validation & type conversion
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│    DataMerging      │  → Join all datasets on appropriate keys
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│ FeatureEngineering  │  → Create temporal, lag & weather features
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│  CatBoostRegressor  │  → Train/Predict with Optuna optimization
└─────────────────────┘
```


## Project Structure

```
energy_prosumers_behavior/
├── Enefit.ipynb                    # Original Kaggle notebook
├── README.md                       # This file
├── .env                            # Environment variables
├── requirements.txt                # Python dependencies
├── start_mlflow_server.sh          # MLflow server launch script
│
├── src/
│   ├── logging_setup.py            # Logging configuration
│   └── preds/
│       ├── main_preds.py           # Main orchestrator
│       ├── data_loading.py         # Download from Kaggle
│       ├── data_preprocessing.py   # DataPreparation class
│       ├── data_merging.py         # DataMerging class
│       ├── feature_engineering.py  # FeatureEngineering class
│       ├── modelling_catboost.py   # CatBoostRegressorModel class
│       └── mlflow_logging.py       # MLflow experiment tracking
│
├── data/
│   ├── raw/                        # Raw CSV files from Kaggle
│   └── preprocessed/               # Intermediate processed data
│
├── models/                         # Saved trained models
│   └── catboost_prosumer/
│       ├── consumption.cbm
│       ├── production.cbm
│       └── metadata.json
│
├── results/                        # Evaluation results
│   └── evaluation_results.json
│
└── logs/                           # Log files
```


## Libraries
- **Polars** — high-performance data manipulation
- **Pandas** — data manipulation and CatBoost compatibility
- **NumPy** — numerical computations
- **CatBoost** — gradient boosting regression
- **Scikit-learn** — pipeline utilities and cross-validation
- **Optuna** — hyperparameter optimization
- **Holidays** — Estonian public holiday calendar
- **Kaggle** — competition data download
- **MLflow** — experiment tracking (optional)
- **python-dotenv** — environment variable management


## Data
The dataset contains public data from the Kaggle competition and is used for educational purposes only.

### Download Links
The data can be downloaded automatically via Kaggle API or manually from the competition page and placed into `./data/raw` directory:
- [Enefit - Predict Energy Behavior of Prosumers](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/data)

### Dataset Descriptions
| File | Description |
|------|-------------|
| `train.csv` | Hourly energy consumption/production records per prosumer segment |
| `client.csv` | Client metadata including installed capacity and EIC count |
| `electricity_prices.csv` | Hourly electricity prices (euros per MWh) |
| `gas_prices.csv` | Daily gas price ranges (lowest/highest per MWh) |
| `historical_weather.csv` | Observed weather data (temperature, pressure, wind, solar radiation) |
| `forecast_weather.csv` | Weather forecasts (temperature, cloud cover, wind, solar radiation) |
| `weather_station_to_county_mapping.csv` | Mapping between weather stations and Estonian counties |


## Quick Start
1. Clone the repository
   ```bash
   git clone https://github.com/iuliia-kuznetsova/energy_prosumers_behavior.git
   cd energy_prosumers_behavior
   ```

2. Prepare virtual environment

    Create virtual environment
    ```bash
    python3 -m venv venv_enefit
    ```
    Activate virtual environment
    ```bash
    source venv_enefit/bin/activate
    ```  # Linux/Mac

    or

    ```bash
    .\venv_enefit\Scripts\activate
    ```   # Windows

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the data (see Data section) and place it in `./data/raw` directory, or configure Kaggle API credentials:
   ```bash
   # Option A: Create ~/.kaggle/kaggle.json with {"username":"xxx","key":"xxx"}
   # Option B: Set environment variables in .env (see env.template)
   ```

5. Start MLflow Server (optional)
```bash
# Start MLflow server (requires PostgreSQL and S3/MinIO)
bash start_mlflow_server.sh
# Or start with local backend
mlflow server --host 0.0.0.0 --port 5000
```

6. Run the ML pipeline:
```bash
# Run full pipeline (download data, preprocess, train, evaluate)
python3 -m src.preds.main_preds

# Skip data download if already present
python3 -m src.preds.main_preds --skip-download

# Skip model training (load existing model)
python3 -m src.preds.main_preds --skip-training

# Custom number of Optuna trials
python3 -m src.preds.main_preds --n-trials 20
```

7. Stopping Services
```bash
# Stop virtual environment
deactivate
# Stop MLflow server
# (Ctrl+C if running in MLflow running terminal)
```


## Approach
1. **Data Loading**:
   - Automated download from Kaggle API with file verification
2. **Data Preprocessing**:
   - Schema validation and type conversion for all datasets
   - Mode-aware processing (train / retrain / test)
3. **Data Merging**:
   - Joining power, client, electricity prices, gas prices, forecast weather, and historical weather into a unified DataFrame via county mapping
   - Computing the target as `capacity_factor = power / installed_capacity`
4. **Feature Engineering**:
   - Cyclical datetime encoding (hour, day, day-of-week, month, year as sin/cos)
   - Estonian holiday and day-before-holiday flags
   - Daylight saving time transition flags
   - Lag features (12 h, 24 h, 1 week, 30 days) for capacity factor, prices, and weather
   - Forward-fill missing values per prediction unit
5. **Model Training and Hyperparameter Optimization**:
   - Separate CatBoost regressors for energy consumption and production 
   - Optuna search (iterations, depth, learning rate, L2 regularization, score function, grow policy)
   - TimeSeriesSplit cross-validation (3 folds) with MAE loss
6. **Prediction and Evaluation**:
   - Recursive forecasting with lag feature updates
   - Batch prediction mode when recursive updates are not needed
   - MAE evaluation on power (`capacity_factor × installed_capacity`)


## Service URLs
MLflow — Experiment tracking: http://localhost:5000


## Results

| Metric | Value |
|--------|-------|
| Train MAE (Power) | TBD |
| Public LB Score | TBD |
| Private LB Score | TBD |


## Pros of my solution
- **Separate models for consumption and production** — captures their distinct behavioral patterns instead of forcing a single model to learn both
- **Capacity factor as target** — normalizing by installed capacity makes predictions comparable across prosumer segments of different sizes
- **Cyclical encoding of time features** — sin/cos encoding preserves continuity (e.g. hour 23 is close to hour 0), unlike one-hot or ordinal encoding
- **Rich multi-source feature set** — combines electricity prices, gas prices, forecast weather, historical weather, and calendar effects in a single model
- **Type safety** — explicit schema definitions ensure data consistency across all pipeline stages
- **Estonian-specific calendar** — public holidays and day-before-holiday flags improve local prediction accuracy
- **DST awareness** — handling daylight saving time transitions prevents errors during March/October clock shifts
- **Temporal cross-validation** — TimeSeriesSplit respects chronological order, preventing data leakage from future observations
- **Modular sklearn-compatible pipeline** — each step (preprocessing, merging, feature engineering, modelling) is a standalone transformer, easy to test and swap
- **Recursive forecasting** — lag features are updated row-by-row during inference so predictions feed into subsequent forecasts, matching real-world sequential deployment
- **Batch prediction fallback** — when recursive updates are not needed, vectorized batch prediction runs significantly faster


## Further Improvements
Currently working on:
- Creation of derived weather features (wind chill, heat index, "feels like" temperature);
- Implementation of feature importance analysis;
- Implementation of target encoding for categorical features;
- Explicit outlier handling for extreme values;
- Increasage of Optuna trials to 50–100 for better hyperparameter search;
- Usage of neural network architectures - Temporal Fusion Transformer, Autoregressive RNN, N-HiTS / N-BEATS, Patch Time Series Transformer, LSTM or Temporal Convolutional Network;
- Ensembling of gradient boosting model and  neural network in order to differently learn data patterns.


## Author
**Iuliia Kuznetsova**
March 2026
