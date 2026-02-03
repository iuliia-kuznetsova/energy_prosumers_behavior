'''
    Energy Prosumers Behavior Prediction Package

    This package provides tools for predicting energy consumption and production
    patterns of prosumers (consumers who also produce energy).

    Modules:
    - data_loading: Download competition data from Kaggle
    - data_preprocessing: DataPreparation class for schema validation
    - data_merging: DataMerging class for joining datasets
    - feature_engineering: FeatureEngineering class for feature creation
    - modelling_catboost: CatBoostRegressorModel for training/prediction
    - main_preds: Main orchestrator for the full pipeline
'''

from src.preds.data_loading import run_data_loading
from src.preds.data_preprocessing import DataPreparation, load_raw_data
from src.preds.data_merging import DataMerging
from src.preds.feature_engineering import FeatureEngineering
from src.preds.modelling_catboost import CatBoostRegressorModel

__all__ = [
    'run_data_loading',
    'DataPreparation',
    'load_raw_data',
    'DataMerging',
    'FeatureEngineering',
    'CatBoostRegressorModel',
]
