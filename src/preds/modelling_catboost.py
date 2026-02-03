'''
    CatBoost Regression Model for Energy Prosumer Prediction

    This module provides the CatBoostRegressor class for training and predicting
    energy consumption/production using capacity_factor as target.

    Strategy:
    - Train separate models for consumption and production
    - Use Optuna for hyperparameter optimization
    - Support recursive forecasting with lag feature updates
    - Convert capacity_factor predictions back to power

    Input:
    - Pandas DataFrame with features from FeatureEngineering
    - Target: capacity_factor

    Output:
    - Power predictions (capacity_factor * installed_capacity)

    Usage:
    python -m src.preds.modelling_catboost
'''

# ---------- Imports ---------- #
import os
import gc
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from dotenv import load_dotenv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostRegressor as CatBoost, Pool
import optuna

from src.logging_setup import setup_logging


# ---------- Logging setup ---------- #
logger = setup_logging('modelling_catboost')


# ---------- Config ---------- #
load_dotenv()
PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)


# ---------- Constants ---------- #
RANDOM_STATE = int(os.getenv('RANDOM_STATE', 3112))
MODELS_DIR = os.getenv('MODELS_DIR', './models')
PROCESSED_DATA_DIR = os.getenv('PROCESSED_DATA_DIR', './data/processed')

# Default hyperparameter search space
DEFAULT_N_TRIALS = int(os.getenv('OPTUNA_N_TRIALS', 10))
DEFAULT_CV_SPLITS = int(os.getenv('CV_SPLITS', 3))

# Target and categorical features
TARGET = 'capacity_factor'
CAT_FEATURES = [
    'prediction_unit_id', 'is_consumption', 
    'county', 'is_business', 'product_type',  
    'is_holiday', 'is_day_before_holiday',
    'is_dst_spring_forward', 'is_dst_fall_backward'
]

# Lag values must match FeatureEngineering
LAG_VALUES = [12, 24, 168, 720]


# ---------- CatBoostRegressor Class ---------- #
class CatBoostRegressorModel(BaseEstimator, TransformerMixin):
    '''
        CatBoost regression model for energy prosumer prediction.
        
        Trains separate models for consumption and production data,
        uses Optuna for hyperparameter tuning, and supports recursive
        forecasting for proper lag feature handling.
    '''
    
    def __init__(
        self,
        random_state: int = RANDOM_STATE,
        target: str = TARGET, 
        cat_features: List[str] = None,
        n_trials: int = DEFAULT_N_TRIALS,
        cv_splits: int = DEFAULT_CV_SPLITS,
        models_dir: str = MODELS_DIR
    ):
        '''
            Initialize the CatBoostRegressor model.
            
            Args:
                random_state: Random seed for reproducibility
                target: Target column name
                cat_features: List of categorical feature names
                n_trials: Number of Optuna trials for hyperparameter optimization
                cv_splits: Number of cross-validation splits
                models_dir: Directory to save/load models
        '''
        self.random_state = random_state
        self.target = target
        self.cat_features = cat_features or CAT_FEATURES
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.models_dir = Path(models_dir)
        
        # Model components
        self.final_model_cb_consumption = None
        self.final_model_cb_production = None
        self.features = None
        
        # Best hyperparameters
        self.best_params_consumption = None
        self.best_params_production = None
        
        # Create models directory
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f'CatBoostRegressorModel initialized')
        logger.info(f'Target: {target}, Categorical features: {len(self.cat_features)}')

    def _objective(
        self,
        trial: optuna.Trial, 
        df: pd.DataFrame, 
        mask: pd.Series
    ) -> float:
        '''
            Optuna objective function for hyperparameter optimization.
            
            Uses TimeSeriesSplit cross-validation and MAE as the metric.
        '''
        df_subset = df[mask]
        features = df_subset.columns.difference(['datetime', self.target, 'row_id'])

        tscv = TimeSeriesSplit(n_splits=self.cv_splits)

        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.01, 10, log=True),
            'score_function': trial.suggest_categorical('score_function', ['Cosine', 'L2']),
            'loss_function': 'MAE',
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
            'random_seed': self.random_state,
            'logging_level': 'Silent'
        }

        mae_scores = []
        for train_index, test_index in tscv.split(df_subset):
            train_df, test_df = df_subset.iloc[train_index], df_subset.iloc[test_index]
            X_train, y_train = train_df[features], train_df[self.target]
            X_test, y_test = test_df[features], test_df[self.target]
            
            train_pool = Pool(X_train, y_train, cat_features=self.cat_features)
            test_pool = Pool(X_test, y_test, cat_features=self.cat_features)
            
            model = CatBoost(**params)
            model.fit(train_pool, eval_set=test_pool, verbose=0)
            
            predictions = model.predict(test_pool)
            mae = np.mean(np.abs(y_test - predictions))
            mae_scores.append(mae)

        mean_mae = np.mean(mae_scores)
        logger.info(f'Trial {trial.number + 1}: MAE = {mean_mae:.6f}')
        return mean_mae

    def fit(self, df: pd.DataFrame, y=None) -> 'CatBoostRegressorModel':
        '''
            Fit the CatBoost models for consumption and production.
            
            Runs Optuna hyperparameter optimization separately for each,
            then trains final models with best parameters on full data.
        '''
        logger.info('Starting model training')
        
        # Define features
        self.features = df.columns.difference(['datetime', self.target, 'row_id'])
        logger.info(f'Features: {len(self.features)}')

        # Optimize and train consumption model
        logger.info('Optimizing CONSUMPTION model...')
        study_consumption = optuna.create_study(direction='minimize')
        mask_consumption = df['is_consumption'] == 1
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study_consumption.optimize(
            lambda trial: self._objective(trial, df=df, mask=mask_consumption), 
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        self.best_params_consumption = study_consumption.best_params
        self.best_params_consumption.update({
            'loss_function': 'MAE',
            'random_seed': self.random_state,
            'logging_level': 'Silent'
        })
        logger.info(f'Best consumption params: {self.best_params_consumption}')
        logger.info(f'Best consumption MAE: {study_consumption.best_value:.6f}')

        # Optimize and train production model
        logger.info('Optimizing PRODUCTION model...')
        study_production = optuna.create_study(direction='minimize')
        mask_production = df['is_consumption'] == 0
        
        study_production.optimize(
            lambda trial: self._objective(trial, df=df, mask=mask_production), 
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        self.best_params_production = study_production.best_params
        self.best_params_production.update({
            'loss_function': 'MAE',
                            'random_seed': self.random_state,
            'logging_level': 'Silent'
        })
        logger.info(f'Best production params: {self.best_params_production}')
        logger.info(f'Best production MAE: {study_production.best_value:.6f}')

        # Train final models on full data
        logger.info('Training final CONSUMPTION model on full data...')
        self.final_model_cb_consumption = CatBoost(**self.best_params_consumption)
        consumption_pool = Pool(
            df[self.features][mask_consumption], 
            df[self.target][mask_consumption], 
            cat_features=self.cat_features
        )
        self.final_model_cb_consumption.fit(consumption_pool, verbose=100)

        logger.info('Training final PRODUCTION model on full data...')
        self.final_model_cb_production = CatBoost(**self.best_params_production)
        production_pool = Pool(
            df[self.features][mask_production], 
            df[self.target][mask_production], 
            cat_features=self.cat_features
        )
        self.final_model_cb_production.fit(production_pool, verbose=100)

        logger.info('DONE: Model training completed')
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        '''
            Generate predictions using trained models.
            
            Implements recursive forecasting with lag feature updates
            when capacity_factor lag features are present.
        '''
        if self.final_model_cb_consumption is None or self.final_model_cb_production is None:
            raise RuntimeError("Models not trained. Call `fit` before `transform`.")
        
        if self.features is None:
            raise RuntimeError("Features not defined. Call `fit` or `load` with features first.")
        
        X_test = X.copy()
        X_test['predictions'] = np.nan
        
        # Sort for sequential forecasting
        X_test = X_test.sort_values(
            by=['datetime', 'prediction_unit_id', 'is_consumption']
        ).reset_index(drop=True)
        
        # Check if recursive forecasting is needed
        capacity_lag_cols = [
            f'capacity_factor_lag_{lag}' for lag in LAG_VALUES 
            if f'capacity_factor_lag_{lag}' in X_test.columns
        ]
        needs_recursive = len(capacity_lag_cols) > 0
        
        n_rows = len(X_test)
        consumption_mask = X_test['is_consumption'] == 1
        production_mask = X_test['is_consumption'] == 0
        
        if needs_recursive:
            # Recursive forecasting - process row by row
            logger.info(f'Running recursive forecasting for {n_rows} rows...')
            
            for idx in range(n_rows):
                current_features = X_test[self.features].iloc[idx:idx+1]
                
                if X_test.at[idx, 'is_consumption'] == 1:
                    prediction = self.final_model_cb_consumption.predict(current_features)[0]
                else:
                    prediction = self.final_model_cb_production.predict(current_features)[0]
                
                X_test.at[idx, 'predictions'] = prediction
                X_test.at[idx, 'capacity_factor'] = prediction
                
                # Update lag features for future rows
                for lag in LAG_VALUES:
                    future_idx = idx + lag
                    if future_idx < n_rows:
                        lag_column = f'capacity_factor_lag_{lag}'
                        if lag_column in X_test.columns:
                            if (X_test.at[future_idx, 'prediction_unit_id'] == X_test.at[idx, 'prediction_unit_id'] and
                                X_test.at[future_idx, 'is_consumption'] == X_test.at[idx, 'is_consumption']):
                                X_test.at[future_idx, lag_column] = prediction
                
                if (idx + 1) % 1000 == 0:
                    logger.info(f'Processed {idx + 1}/{n_rows} rows')
        else:
            # Batch prediction (no recursive forecasting needed)
            logger.info('Running batch prediction...')
            
            if consumption_mask.any():
                X_test.loc[consumption_mask, 'predictions'] = \
                    self.final_model_cb_consumption.predict(X_test.loc[consumption_mask, self.features])
            
            if production_mask.any():
                X_test.loc[production_mask, 'predictions'] = \
                    self.final_model_cb_production.predict(X_test.loc[production_mask, self.features])
        
        # Calculate power from capacity_factor
        installed_capacity = X_test['installed_capacity'].values
        X_test['target'] = X_test['predictions'] * installed_capacity
        X_test['target'] = np.maximum(X_test['target'], 0)  # Non-negative
        
        submission = X_test[['row_id', 'target']]
        
        logger.info(f'DONE: Generated {len(submission)} predictions')
        return submission
    
    def compute_mae_power(self, df: pd.DataFrame) -> float:
        '''
            Compute MAE for power predictions against actual values.
            
            Loads actual power from saved file and compares with predictions.
        '''
        # Load actual power values
        power_path = Path(PROCESSED_DATA_DIR) / 'power.parquet'
        if not power_path.exists():
            raise FileNotFoundError(f"Power file not found: {power_path}")
        
        revealed_power = pd.read_parquet(power_path)
        
        # Get predictions
        predicted_power = self.transform(df)
        
        # Merge and compute MAE
        mae_calculation = pd.merge(
            revealed_power[['row_id', 'power']], 
            predicted_power[['row_id', 'target']], 
            on='row_id', 
            how='left'
        )
        mae_calculation = mae_calculation.dropna(subset=['power', 'target'])
        mae_power = np.mean(np.abs(mae_calculation['target'] - mae_calculation['power']))
        
        logger.info(f'Power MAE: {mae_power:.4f}')
        return mae_power

    def save(self, model_name: str = 'catboost_prosumer') -> str:
        '''
            Save trained models and metadata to disk.
            
            Args:
                model_name: Base name for saved files
                
            Returns:
                Path to saved model directory
        '''
        if self.final_model_cb_consumption is None or self.final_model_cb_production is None:
            raise RuntimeError("No models to save. Train first.")
        
        model_path = self.models_dir / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        self.final_model_cb_consumption.save_model(str(model_path / 'consumption.cbm'))
        self.final_model_cb_production.save_model(str(model_path / 'production.cbm'))
        
        # Save metadata
        metadata = {
            'features': list(self.features),
            'cat_features': self.cat_features,
            'target': self.target,
            'random_state': self.random_state,
            'best_params_consumption': self.best_params_consumption,
            'best_params_production': self.best_params_production
        }
        with open(model_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f'Models saved to {model_path}')
        return str(model_path)
    
    def load(
        self, 
        model_path: str = None,
        model_name: str = 'catboost_prosumer'
    ) -> 'CatBoostRegressorModel':
        '''
            Load trained models and metadata from disk.
            
            Args:
                model_path: Full path to model directory (takes precedence)
                model_name: Model name in models_dir
                
            Returns:
                Self with loaded models
        '''
        if model_path:
            load_path = Path(model_path)
        else:
            load_path = self.models_dir / model_name
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")
        
        # Load models
        self.final_model_cb_consumption = CatBoost()
        self.final_model_cb_consumption.load_model(str(load_path / 'consumption.cbm'))
        
        self.final_model_cb_production = CatBoost()
        self.final_model_cb_production.load_model(str(load_path / 'production.cbm'))
        
        # Load metadata
        with open(load_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.features = pd.Index(metadata['features'])
        self.cat_features = metadata['cat_features']
        self.target = metadata['target']
        self.best_params_consumption = metadata.get('best_params_consumption')
        self.best_params_production = metadata.get('best_params_production')
        
        logger.info(f'Models loaded from {load_path}')
        return self
    
    def set_features(self, features: Union[List[str], pd.Index]) -> None:
        '''Set features manually (useful when loading without metadata)'''
        self.features = pd.Index(features) if not isinstance(features, pd.Index) else features
        logger.info(f'Features set: {len(self.features)} features')


# ---------- Helper Functions ---------- #
def run_modelling(
    train_df: pd.DataFrame,
    n_trials: int = DEFAULT_N_TRIALS,
    save_model: bool = True,
    model_name: str = 'catboost_prosumer'
) -> CatBoostRegressorModel:
    '''
        Run the modelling pipeline.
        
        Args:
            train_df: Training DataFrame with features
            n_trials: Number of Optuna trials
            save_model: Whether to save the trained model
            model_name: Name for saved model
            
        Returns:
            Trained CatBoostRegressorModel
    '''
    logger.info('Starting modelling pipeline')
    
    model = CatBoostRegressorModel(n_trials=n_trials)
    model.fit(train_df)
    
    if save_model:
        model.save(model_name)
    
    logger.info('Modelling pipeline completed successfully')
    
    gc.collect()
    return model


# ---------- Main function ---------- #
if __name__ == '__main__':
    # For standalone testing
    from src.preds.data_preprocessing import run_preprocessing
    from src.preds.data_merging import run_merging
    from src.preds.feature_engineering import run_feature_engineering
    
    processed_data = run_preprocessing()
    merged_df = run_merging(processed_data)
    features_df = run_feature_engineering(merged_df)
    
    model = run_modelling(features_df, n_trials=5)
    mae = model.compute_mae_power(features_df)
    print(f'Train MAE: {mae:.4f}')


# ---------- All exports ---------- #
__all__ = [
    'CatBoostRegressorModel', 
    'run_modelling', 
    'TARGET', 
    'CAT_FEATURES',
    'LAG_VALUES'
]
