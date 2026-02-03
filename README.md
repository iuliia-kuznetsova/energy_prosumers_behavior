# Enefit - Predict Energy Behavior of Prosumers

## 🎯 Competition Overview

This repository contains my solution for the [Enefit - Predict Energy Behavior of Prosumers](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers) Kaggle competition.

**Objective:** Predict energy consumption and production patterns of prosumers (consumers who also produce energy) in Estonia to minimize energy imbalance costs for energy companies.

**Evaluation Metric:** Mean Absolute Error (MAE)

---

## 📊 Solution Architecture

### Target Variable
- **capacity_factor** = power / installed_capacity
- Final predictions are converted back to power by multiplying with installed_capacity

### Model Strategy
Two separate **CatBoost** models trained for:
1. **Consumption** predictions (`is_consumption == 1`)
2. **Production** predictions (`is_consumption == 0`)

### Pipeline Structure

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

---

## 🔧 Feature Engineering

### Data Sources Merged
| Dataset | Key Features |
|---------|-------------|
| **train.csv** | datetime, prediction_unit_id, county, is_business, product_type, target |
| **client.csv** | installed_capacity, eic_count |
| **electricity_prices.csv** | euros_per_mwh |
| **gas_prices.csv** | lowest/highest_price_per_mwh |
| **historical_weather.csv** | temperature, pressure, precipitation, solar radiation, wind |
| **forecast_weather.csv** | temperature, cloud cover, wind, solar radiation forecasts |

### Engineered Features

#### 1. Temporal Features (Cyclical Encoding)
```python
# Cyclical encoding to preserve continuity
hour_sin, hour_cos     # 24-hour cycle
day_sin, day_cos       # 31-day cycle
day_of_week_sin/cos    # 7-day cycle
month_sin, month_cos   # 12-month cycle
year_sin, year_cos     # Year encoding
```

#### 2. Calendar Features
- `is_holiday` - Estonian public holidays (2021-2025)
- `is_day_before_holiday` - Day before holidays
- `is_dst_spring_forward` - Daylight saving time shift (March)
- `is_dst_fall_backward` - Daylight saving time shift (October)

#### 3. Lag Features
Lag periods: **12, 24, 168 (1 week), 720 (1 month)** hours for:
- `capacity_factor`
- `electricity_price`
- `gas_price`
- All historical weather features

#### 4. Weather Features
- Forecast weather (`fw_*`): temperature, cloud cover, precipitation, wind components, solar radiation
- Historical weather (`hw_*`): temperature, pressure, precipitation, wind speed/direction, solar radiation

---

## 🤖 Model Details

### Hyperparameter Optimization
- **Optuna** with 10 trials per model (consumption/production)
- **TimeSeriesSplit** with 3 folds for cross-validation
- **Loss Function:** MAE

### Search Space
```python
params = {
    'iterations': (100, 1000),
    'depth': (4, 12),
    'learning_rate': (0.01, 0.3),  # log scale
    'l2_leaf_reg': (0.01, 10),     # log scale
    'score_function': ['Cosine', 'L2'],
    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']
}
```

### Categorical Features
```python
['prediction_unit_id', 'is_consumption', 'county', 'is_business', 
 'product_type', 'is_holiday', 'is_day_before_holiday',
 'is_dst_spring_forward', 'is_dst_fall_backward']
```

---

## ✅ Pros of This Solution

| Aspect | Description |
|--------|-------------|
| **Separate Models** | Training separate models for consumption and production captures their distinct patterns |
| **Cyclical Encoding** | Proper handling of periodic features (hour, day, month) preserves temporal continuity |
| **Rich Feature Set** | Comprehensive feature engineering including weather forecasts, lags, and calendar effects |
| **Type Safety** | Explicit schema definitions ensure data consistency across pipeline stages |
| **Holiday Handling** | Estonian-specific holiday calendar improves local predictions |
| **DST Awareness** | Handling daylight saving time shifts prevents errors during time transitions |
| **Cross-Validation** | TimeSeriesSplit respects temporal ordering (no data leakage from future) |
| **Modular Pipeline** | sklearn-compatible transformers allow easy experimentation |
| **Missing Value Strategy** | Forward-fill within prediction units maintains temporal consistency |

---

## ❌ Cons of This Solution

| Aspect | Description |
|--------|-------------|
| **Limited Optuna Trials** | Only 10 trials may not find optimal hyperparameters |
| **No Ensemble** | Single model type (CatBoost) - no blending with LightGBM/XGBoost |
| **Fixed Lag Windows** | Hardcoded lags [12, 24, 168, 720] may not be optimal for all patterns |
| **No Feature Selection** | All features used without importance-based filtering |
| **Slow Inference** | Row-by-row recursive forecasting is computationally expensive |
| **No Target Encoding** | Could improve categorical feature handling |
| **Limited Weather Processing** | No derived features like "feels like" temperature or heat index |
| **No Outlier Handling** | Extreme values not explicitly addressed |

---

## 🐛 Critical Logical Problems Found (✅ FIXED)

### 1. ~~**CRITICAL: Broken Recursive Forecasting**~~ ✅ FIXED

The lag feature update logic in `transform()` was using wrong lags `[1, 2, 3]` instead of `[12, 24, 168, 720]` and incorrect column naming.

**Fix Applied:** Updated to use correct lag values and column naming pattern `capacity_factor_lag_{lag}`.

---

### 2. ~~**Missing `self.features` When Loading Model**~~ ✅ FIXED

When loading a pre-trained model, `self.features` was never set, causing `AttributeError`.

**Fix Applied:** 
- Added `features` parameter to `load()` method
- Added `set_features()` method for manual feature setting
- Model now tries to infer features from loaded model or warns if not set

---

### 3. **Potential Data Leakage in Forward-Fill** ⚠️ (Minor - Unchanged)

Forward-fill for `electricity_price` and `gas_price` is applied globally. This is acceptable since these prices are the same across all prediction units at a given time, but the data should be sorted by datetime first (which it is).

---

### 4. ~~**Inconsistent `fit()` Methods**~~ ✅ FIXED

All transformer classes had `fit()` methods that returned copied data instead of `self`.

**Fix Applied:** All `fit()` methods now return `self` as per sklearn convention.

---

### 5. ~~**Inefficient Row-by-Row Prediction**~~ ✅ OPTIMIZED

The transform method iterated row-by-row which is slow.

**Fix Applied:** Added batch processing mode when recursive forecasting is not needed, with automatic detection of whether lag features require updating.

---

### 6. ~~**Model Filename Inconsistency**~~ ✅ FIXED

Inconsistent naming between consumption and production model files.

**Fix Applied:** Both now use consistent naming pattern `_model_11_consumption` and `_model_11_production`.

---

## 🔄 Recommended Improvements

1. ~~**Fix the recursive forecasting bug**~~ ✅ Done
2. **Increase Optuna trials** to 50-100 for better hyperparameter search
3. **Add model ensemble** - Blend CatBoost with LightGBM/XGBoost
4. ~~**Implement vectorized prediction**~~ ✅ Done (batch mode when possible)
5. **Add feature importance analysis** - Remove low-importance features
6. **Create derived weather features** - Wind chill, heat index, etc.
7. **Add cross-validation on full pipeline** - Not just the model
8. ~~**Implement proper error handling**~~ ✅ Done (validation checks added)

---

## 📁 Project Structure

```
energy_prosumers_behavior/
├── Enefit.ipynb                    # Original Kaggle notebook
├── README.md                       # This file
├── .env.example                    # Environment variables template
├── requirements.txt                # Python dependencies
│
├── src/
│   ├── logging_setup.py            # Logging configuration
│   └── preds/
│       ├── main_preds.py           # 🎯 Main orchestrator
│       ├── data_loading.py         # Download from Kaggle
│       ├── data_preprocessing.py   # DataPreparation class
│       ├── data_merging.py         # DataMerging class
│       ├── feature_engineering.py  # FeatureEngineering class
│       └── modelling_catboost.py   # CatBoostRegressorModel class
│
├── data/
│   ├── raw/                        # Raw CSV files from Kaggle
│   └── processed/                  # Intermediate processed data
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

## 🔧 Module Overview

| Module | Class | Description |
|--------|-------|-------------|
| `data_loading.py` | - | Downloads competition data from Kaggle API |
| `data_preprocessing.py` | `DataPreparation` | Schema validation, type conversion |
| `data_merging.py` | `DataMerging` | Joins all datasets into unified DataFrame |
| `feature_engineering.py` | `FeatureEngineering` | Creates temporal, calendar, lag features |
| `modelling_catboost.py` | `CatBoostRegressorModel` | Trains consumption/production models |
| `main_preds.py` | - | Orchestrates the full pipeline |

---

## 🚀 How to Run

### Option 1: Run the Full Pipeline (Recommended)

```bash
# Install dependencies
pip install polars pandas numpy catboost optuna scikit-learn holidays kaggle python-dotenv

# Setup Kaggle credentials (get from https://www.kaggle.com/settings)
# Option A: Create ~/.kaggle/kaggle.json with {"username":"xxx","key":"xxx"}
# Option B: Set environment variables KAGGLE_USERNAME and KAGGLE_KEY

# Run full pipeline
python -m src.preds.main_preds

# Or with custom options
python -m src.preds.main_preds --n-trials 20              # More optimization trials
python -m src.preds.main_preds --skip-download            # Skip data download
python -m src.preds.main_preds --skip-training            # Load existing model
```

### Option 2: Run Individual Modules

```python
from src.preds.data_loading import run_data_loading
from src.preds.data_preprocessing import DataPreparation, load_raw_data
from src.preds.data_merging import DataMerging
from src.preds.feature_engineering import FeatureEngineering
from src.preds.modelling_catboost import CatBoostRegressorModel

# Step 1: Download data
run_data_loading()

# Step 2: Preprocess
raw_data = load_raw_data('./data/raw')
preprocessor = DataPreparation(mode='train')
processed_data = preprocessor.fit_transform(raw_data)

# Step 3: Merge
merger = DataMerging(mode='train')
merged_df = merger.fit_transform(processed_data)

# Step 4: Feature engineering
engineer = FeatureEngineering(mode='train')
train_df = engineer.fit_transform(merged_df)

# Step 5: Train model
model = CatBoostRegressorModel(n_trials=10)
model.fit(train_df)
model.save('catboost_prosumer')

# Step 6: Evaluate
mae = model.compute_mae_power(train_df)
print(f'MAE: {mae:.4f}')
```

### Option 3: Run the Original Notebook

```bash
jupyter notebook Enefit.ipynb
```

### Environment Variables

Create a `.env` file in the project root:

```env
RANDOM_STATE=3112
RAW_DATA_DIR=./data/raw
PROCESSED_DATA_DIR=./data/processed
MODELS_DIR=./models
RESULTS_DIR=./results
OPTUNA_N_TRIALS=10
CV_SPLITS=3
```

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Train MAE (Power) | TBD |
| Public LB Score | TBD |
| Private LB Score | TBD |

---

## 📚 References

- [Competition Page](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [Optuna Documentation](https://optuna.readthedocs.io/)

---

## 📝 License

This project is for educational purposes as part of the Kaggle competition.

## Quick Start

### 1. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv_enefit

# Activate virtual environment
source venv_enefit/bin/activate  # Linux/Mac
# or
.\venv_enefit\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Start MLflow Server

```bash
# Start MLflow server (requires PostgreSQL and S3/MinIO)
bash start_mlflow_server.sh

# Or start with local backend
mlflow server --host 0.0.0.0 --port 5000
```

### 3. Run ML Pipeline

```bash
# Run full pipeline (download data, preprocess, train, evaluate)
python3 -m src.recs.main_recs

# Skip data download if already present
python3 -m src.recs.main_recs --skip-download

```

### 4. Build and Start API Containers

```bash
# Build and start all services (API, Prometheus, Grafana)
docker compose -f src/api/docker-compose.yml up -d --build

# Check service status
docker compose -f src/api/docker-compose.yml ps

# View logs
docker compose -f src/api/docker-compose.yml logs -f recommender

# Rebuild after code changes
docker compose -f src/api/docker-compose.yml build --no-cache recommender
docker compose -f src/api/docker-compose.yml up -d

# Fix logs permissions if falls with error
sudo chown -R mle-user:mle-user ./logs/
```

### 5. Test the API

```bash
# Health check
curl -s http://localhost:8080/health | jq

# Model info
curl -s http://localhost:8080/model/info | jq

# Single prediction
curl -s -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "12345678",
    "features": {
      "age": 35,
      "renta": 50000.0,
      "customer_period": 12,
      "ind_nuevo": false,
      "indresi": true,
      "indfall": false,
      "ind_actividad_cliente": true,
      "ind_empleado": "A",
      "pais_residencia": "ES",
      "sexo": "H",
      "indrel": "1",
      "indrel_1mes": "1",
      "tiprel_1mes": "A",
      "canal_entrada": "KHE",
      "cod_prov": "28",
      "segmento": "02 - PARTICULARES"
    },
    "top_k": 7
  }' | jq

# Run API tests
python3 -m src.api.test_api --sample --limit 100 --sleep 0.1
```


## Service URLs
Recommender API - Main API: http://localhost:8080
API Documentation - Swagger UI: http://localhost:8080/docs
API Metrics - Prometheus metrics: http://localhost:8080/metrics
Prometheus - Metrics database: http://localhost:9090
Grafana - Dashboards (admin/admin): http://localhost:3000
MLflow - Experiment tracking: http://localhost:5000


## Stopping Services
```bash
# Stop Docker containers
docker compose -f src/api/docker-compose.yml down

# Stop and remove volumes
docker compose -f src/api/docker-compose.yml down -v

# Stop MLflow server
# (Ctrl+C if running in MlFlow running terminal)
```

## Additional Documentation
- Recommendation Pipeline Documentation: src/recs/README_RECS.md
- API Documentation: src/api/README_API.md
