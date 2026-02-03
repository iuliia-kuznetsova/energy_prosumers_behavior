'''
    Modelling with CatBoost

    Strategy:


    Input:
    - X_train, y_train - Training features and targets
    - X_test, y_test - Test features and targets

    Output:


    Usage:
    python -m src.preds.modelling_ovr
'''

# ---------- Imports ---------- #
import polars as pl
import numpy as np
import pandas as pd
import os
import json
import pickle
import gc
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime

from catboost import CatBoostClassifier, Pool
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    log_loss
)
from dotenv import load_dotenv
from tqdm import tqdm
import optuna
from optuna.samplers import TPESampler

import mlflow
import mlflow.catboost

from src.logging_setup import setup_logging


# ---------- Logging setup ---------- #
logger = setup_logging('modelling_ovr')
logging.getLogger('catboost').setLevel(logging.ERROR)


# ---------- Config ---------- #
load_dotenv()
PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)


# ---------- Constants ---------- #
# Helper to parse boolean env vars
# As bool() of any empty string is True
def _parse_bool_env(key: str, default: bool = True) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ('true', '1', 'yes', 'on')

# Random state
RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))

# Directories
DATA_DIR = os.getenv('DATA_DIR', './data')
RESULTS_DIR = os.getenv('RESULTS_DIR', './results')
MODELS_DIR = os.getenv('MODELS_DIR', './models')
# Preprocessed data file
PREPROCESSED_DATA_FILE = os.getenv('PREPROCESSED_DATA_FILE', 'data_preprocessed.parquet')

# MLflow experiment name
MLFLOW_EXPERIMENT = os.getenv('MLFLOW_EXPERIMENT', 'bank_products_recommendation')

# Hyperparameter optimization parameters
# Run Optuna hyperparameter optimization
OPTIMIZE = _parse_bool_env('OPTIMIZE', True)
# Number of Optuna trials per group
OPT_N_TRIALS = int(os.getenv('OPT_N_TRIALS', 15))
# Optuna timeout per group optimization (seconds)
OPT_OPTUNA_TIMEOUT = int(os.getenv('OPT_OPTUNA_TIMEOUT', 180))
# Number of CV folds for optimization
OPT_N_SPLITS = int(os.getenv('OPT_N_SPLITS', 3))
# Lower iterations during optimization
OPT_MAX_ITERATIONS = int(os.getenv('OPT_MAX_ITERATIONS', 300))
# Early stopping for optimization
OPT_EARLY_STOPPING_ROUNDS = int(os.getenv('OPT_EARLY_STOPPING_ROUNDS', 30))
# CatBoost Pool pre-quantization for optimization
OPT_USE_QUANTIZED_POOL = _parse_bool_env('OPT_USE_QUANTIZED_POOL', True)

# Training parameters
# Retraining final models on full preprocessed data
RETRAIN_ON_FULL_DATA = _parse_bool_env('RETRAIN_ON_FULL_DATA', True)
# Boosting type: 'Plain' is faster, 'Ordered' may be slightly better quality
MODEL_BOOSTING_TYPE = os.getenv('MODEL_BOOSTING_TYPE', 'Plain')
# Border count (max_bin) - lower = faster but less precise
MODEL_BORDER_COUNT = int(os.getenv('MODEL_BORDER_COUNT', 128))
# Thread count for CatBoost (-1 = auto)
MODEL_THREAD_COUNT = int(os.getenv('MODEL_THREAD_COUNT', -1))
# CTR complexity (1-2 for speed, higher for quality with many categoricals)
MODEL_MAX_CTR_COMPLEXITY = int(os.getenv('MODEL_MAX_CTR_COMPLEXITY', 1))
# One-hot encoding max size for low-cardinality categoricals
MODEL_ONE_HOT_MAX_SIZE = int(os.getenv('MODEL_ONE_HOT_MAX_SIZE', 10))
# Early stopping rounds for final training
MODEL_EARLY_STOPPING_ROUNDS = int(os.getenv('MODEL_EARLY_STOPPING_ROUNDS', 50))


# ---------- Helper Functions ---------- #
def load_training_data(
    data_dir: str = DATA_DIR
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
        Load train/test as pandas DataFrames.
        Uses Polars to read parquet (handles Categorical types) then converts to pandas.
    '''

    # Load with Polars first to handle Categorical columns properly
    X_train = pl.read_parquet(f"{data_dir}/X_train.parquet").to_pandas()
    X_test = pl.read_parquet(f"{data_dir}/X_test.parquet").to_pandas()
    y_train = pl.read_parquet(f"{data_dir}/y_train.parquet").to_pandas()
    y_test = pl.read_parquet(f"{data_dir}/y_test.parquet").to_pandas()
    
    logger.info(f'Train data loaded: X_train={X_train.shape[0]:,} rows x {X_train.shape[1]} columns, y_train={y_train.shape[0]:,} rows x {y_train.shape[1]} columns')
    logger.info(f'Test data loaded: X_test={X_test.shape[0]:,} rows x {X_test.shape[1]} columns, y_test={y_test.shape[0]:,} rows x {y_test.shape[1]} columns')

    return X_train, X_test, y_train, y_test

def get_product_names(
    y_train: pd.DataFrame
) -> List[str]:
    '''
        Extract product names from target columns.
    '''

    return [col for col in y_train.columns if col.startswith('target_')]


# ---------- OvR Group Model Class ---------- #
class Model:
    '''
        Model with CatBoost.
    '''
    
    def __init__(
        self,
        cat_features: Optional[List[str]] = None,
        random_state: int = RANDOM_STATE,
        models_dir: str = MODELS_DIR,
        results_dir: str = RESULTS_DIR,
        data_dir: str = DATA_DIR,
        mlflow_experiment: Optional[str] = MLFLOW_EXPERIMENT,

    ):
        '''
            Initialize model:

            
            Speed optimizations:
                - HPO uses TimeSeriesSplit CV (hpo_n_splits folds) with early stopping
                - Pre-quantized CatBoost Pool reused across trials
                - Configurable boosting_type, border_count, thread_count
                - Optional feature selection based on importance
        '''
        self.cat_features = cat_features or []
        self.random_state = random_state
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.data_dir = Path(data_dir)
        self.mlflow_experiment = mlflow_experiment
        self.frequent_threshold = frequent_threshold
        self.rare_threshold = rare_threshold
        self.optimize = optimize
        self.n_trials = n_trials
        self.optuna_timeout = optuna_timeout
        self.top_k = top_k
        self.retrain_on_full_data = retrain_on_full_data
        self.preprocessed_data_file = preprocessed_data_file
        
        # HPO speed parameters
        self.hpo_n_splits = hpo_n_splits
        self.hpo_max_iterations = hpo_max_iterations
        self.hpo_early_stopping_rounds = hpo_early_stopping_rounds
        self.use_quantized_pool = use_quantized_pool
        
        # Training speed parameters
        self.boosting_type = boosting_type
        self.border_count = border_count
        self.thread_count = thread_count
        self.max_ctr_complexity = max_ctr_complexity
        self.one_hot_max_size = one_hot_max_size
        self.early_stopping_rounds = early_stopping_rounds
        
        # Feature selection
        self.use_feature_selection = use_feature_selection
        self.min_feature_importance = min_feature_importance
        self.selected_features: Optional[List[str]] = None  # Will be set during training

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # OvR models per group: {'frequent': OvR, 'mid': OvR, 'rare': OvR}
        self.models: Dict[str, OneVsRestClassifier] = {}
        
        # Products in each group: {'frequent': [products], 'mid': [...], 'rare': [...]}
        self.product_groups: Dict[str, List[str]] = {}
        
        # Best hyperparameters per group
        self.best_params: Dict[str, Dict[str, Any]] = {}
        
        # Thresholds per product (across all groups)
        self.thresholds: Dict[str, float] = {}
        
        # CV scores per group
        self.cv_scores: Dict[str, List[float]] = {}
        
        # All products (ordered)
        self.all_products: List[str] = []
        
        # Feature metadata
        self.feature_names: List[str] = []
        self.cat_feature_indices: List[int] = []
        
        # Cached quantized pools for HPO (reused across trials)
        self._quantized_pools: Dict[str, Pool] = {}
        
        # Setup MLflow
        if mlflow_experiment:
            # Set tracking URI to MLflow server (default: localhost:5000)
            tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f'MLflow tracking URI: {tracking_uri}')
            mlflow.set_experiment(mlflow_experiment)
            logger.info(f'MLflow experiment set to: {mlflow_experiment}')
        
        logger.info('OvRGroupModel initialized')
        logger.info(f'Hyperparameters optimization settings: n_splits={hpo_n_splits}, max_iter={hpo_max_iterations}, early_stop={hpo_early_stopping_rounds}')
        logger.info(f'Models training settings: boosting={boosting_type}, border_count={border_count}, threads={thread_count}')
        logger.info(f'Feature selection settings: enabled={use_feature_selection}, min_importance={min_feature_importance}')
    
    # ----------- Data Loading ------------ #
    def _load_full_preprocessed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
            Load full preprocessed data for final model retraining.
            Returns features (X) and targets (y) as pandas DataFrames.
        '''
        full_data_path = self.data_dir / self.preprocessed_data_file
        
        # Load with Polars first
        df_full = pl.read_parquet(str(full_data_path))
        logger.info(f'Full data loaded: {df_full.height:,} rows x {df_full.width} columns')
        
        # Identify target columns
        target_cols = [col for col in df_full.columns if col.startswith('target_')]
        
        # Identify feature columns (exclude IDs, targets, and datetime columns)
        id_cols = ['ncodpers', 'fecha_dato']
        datetime_cols = [
            col for col in df_full.columns 
            if df_full[col].dtype in (pl.Date, pl.Datetime, pl.Time)
        ]
        feature_cols = [
            col for col in df_full.columns 
            if col not in id_cols 
            and col not in target_cols
            and col not in datetime_cols
        ]
        
        # Extract features and targets
        X_full = df_full.select(feature_cols).to_pandas()
        y_full = df_full.select(target_cols).to_pandas()
        
        logger.info(f'DONE: Full data prepared for model retraining')
        logger.info(f'Full data features: {X_full.shape}, targets: {y_full.shape}')
        
        del df_full
        gc.collect()
        
        return X_full, y_full

    # ----------- Data Preparation ------------ #
    def _prepare_data(
        self,
        X: Union[pl.DataFrame, pd.DataFrame],
        y: Optional[Union[pl.DataFrame, pd.DataFrame]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
            Convert data to pandas DataFrames.
            Handles NaN values in categorical columns if present.
        '''

        # Convert X
        if isinstance(X, pl.DataFrame):
            X_pd = X.to_pandas()
        else:
            X_pd = X.copy()
        
        # Convert categorical columns to string for CatBoost compatibility
        # CatBoost cannot handle None/NaN in categorical features
        for col in X_pd.columns:
            if X_pd[col].dtype.name == 'category' or X_pd[col].dtype == 'object':
                # Convert to string first (NaN becomes 'nan'), then replace with 'missing'
                X_pd[col] = X_pd[col].astype(str).replace({'nan': 'missing', 'None': 'missing', '<NA>': 'missing'})
        
        # Convert y
        if y is None:
            y_pd = None
        elif isinstance(y, pl.DataFrame):
            y_pd = y.to_pandas()
        else:
            y_pd = y.copy()
        
        y_shape = y_pd.shape if y_pd is not None else None
        logger.info(f'DONE: Data prepared for model training')

        return X_pd, y_pd
    
    def _get_cat_feature_indices(
        self, X: pd.DataFrame
    ) -> List[int]:
        '''
            Get indices of categorical features.
        '''

        if not self.cat_features:
            cat_indices = [
                i for i, col in enumerate(X.columns)
                if X[col].dtype == 'object' or X[col].dtype.name == 'category'
            ]
        else:
            cat_indices = [
                i for i, col in enumerate(X.columns)
                if col in self.cat_features
            ]

        logger.info(f'DONE: Categorical features and their indices retrieved')
        logger.info(f'Categorical features: {self.cat_features}')
        logger.info(f'Categorical feature indices: {cat_indices}')

        return cat_indices
    
    def _group_products_by_prevalence(
        self, 
        y: pd.DataFrame
    ) -> Dict[str, List[str]]:
        '''
            Group products into frequent, mid, and rare based on prevalence.
        '''

        # Calculate prevalence as a percentage of total customers 
        # who don't have product current month, but will newly acquire it next month
        prevalence = y.fillna(0).mean() * 100
        
        # Sort products by prevalence descending
        sorted_prevalence = prevalence.sort_values(ascending=False)

        frequent_products = []
        mid_products = []
        rare_products = []

        for product in sorted_prevalence.index:
            logger.info(f"Product: {product}, prevalence: {sorted_prevalence[product]:.4f}%")
            if sorted_prevalence[product] > self.frequent_threshold:
                frequent_products.append(product)
            elif sorted_prevalence[product] < self.rare_threshold:
                rare_products.append(product)
            else:
                mid_products.append(product)
        
        groups = {
            'frequent': frequent_products,
            'mid': mid_products,
            'rare': rare_products
        }
        
        logger.info('DONE: Product groups by prevalence created')
        logger.info(f"Frequent group: {len(groups['frequent'])} products with prevalence >= {self.frequent_threshold:.4f}%")
        logger.info(f"Mid group: {len(groups['mid'])} products with prevalence {self.rare_threshold:.4f}%-{self.frequent_threshold:.4f}%")
        logger.info(f"Rare group: {len(groups['rare'])} products with prevalence < {self.rare_threshold:.4f}%")
        
        for group_name, products in groups.items():
            if products:
                prevalences = [prevalence[p] for p in products]             
                logger.info(f"{group_name}: computed prevalence range [{min(prevalences):.4f}%, {max(prevalences):.4f}%]")
        
        return groups
      
    # ----------- Hyperparameter Optimization ------------ #
    def _create_quantized_pool(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        cat_feature_indices: List[int],
        pool_key: str
    ) -> Pool:
        '''
            Create and cache a quantized CatBoost Pool for faster optimization.
            Quantization is done once and reused across all trials.
        '''

        if pool_key in self._quantized_pools:
            logger.info(f'Using cached quantized pool: {pool_key}')
            return self._quantized_pools[pool_key]
        
        logger.info(f'Creating quantized pool: {pool_key}')
        pool = Pool(
            data=X,
            label=y if len(y.shape) == 1 else y[:, 0],  # Use first column for quantization
            cat_features=cat_feature_indices if cat_feature_indices else None
        )
        
        # Cache the pool structure (not the full quantized data which is per-model)
        self._quantized_pools[pool_key] = pool
        logger.info(f'DONE: Quantized pool created and cached: {pool_key}')

        return pool
    
    def _create_optuna_objective(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        cat_feature_indices: List[int]
    ):
        '''
            Create Optuna objective with TimeSeriesSplit CV for OvR.
            
            Speed optimizations:
            - Uses fewer CV folds during optimization
            - Early stopping to avoid unnecessary iterations
            - Fixed "heavy" categorical params for speed
            - Memory cleanup after each fold
        '''

        def objective(trial: optuna.Trial) -> float:

            # Hyperparameter search space - focus on most impactful params
            # Fixed heavy params early for speed: max_ctr_complexity, one_hot_max_size, boosting_type
            params = {
                # Tunable params (most impactful)
                'iterations': trial.suggest_int('iterations', 100, self.hpo_max_iterations, step=50), # Lower iterations for speed
                'depth': trial.suggest_int('depth', 4, 7),  # Smaller depth for speed
                'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.2, log=True), # Lower learning rate for speed
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True), # Lower L2 regularization for speed
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0), # Lower bagging temperature for speed
                # Fixed params to speed up optimization
                'loss_function': 'Logloss', # Loss function to use for optimization
                'random_seed': self.random_state, # Random seed for reproducibility
                'logging_level': 'Silent', # Suppress all CatBoost output
                'thread_count': self.thread_count, # Number of threads to use for training (lower = faster)
                'auto_class_weights': 'Balanced', # Balanced class weights - automatically adjust class weights in order to deal with class imbalance
                'boosting_type': self.boosting_type,  # Type of gradient boosting algorithm ('Plain' is faster than 'Ordered')
                'border_count': self.border_count,  # Maximum number of bins for categorical features (lower = faster)
                'max_ctr_complexity': self.max_ctr_complexity,  # Reduce CTR complexity 
                'one_hot_max_size': self.one_hot_max_size, # Limit one-hot encoding size
                #'used_ram_limit': f'{self.max_ram_gb}gb', # Limit the amount of RAM used for training (not obligatory for CatBoost)
                'early_stopping_rounds': self.hpo_early_stopping_rounds # Stop training when metric plateaus
            }
            
            # Split data into folds for optimization
            tscv = TimeSeriesSplit(n_splits=self.hpo_n_splits)
            auc_scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                
                # Log fold progress
                if fold_idx == 0:
                    logger.info(
                        f'Trial {trial.number + 1} | {self.hpo_n_splits} folds | '
                        f'iterations={params["iterations"]}, depth={params["depth"]}, learning_rate={params["learning_rate"]:.3f}'
                    )
                
                # Create OvR with CatBoost base estimator
                base_estimator = CatBoostClassifier(
                    cat_features=cat_feature_indices if cat_feature_indices else None,
                    **params
                )
                ovr = OneVsRestClassifier(base_estimator, n_jobs=1)
                
                # Train with early stopping using validation set
                try:
                    ovr.fit(X_tr, y_tr)
                except Exception as e:
                    logger.warning(f'Trial {trial.number + 1} | Fold {fold_idx + 1} failed: {e}')
                    # Clean up and continue
                    del base_estimator, ovr
                    gc.collect()
                    continue
                
                # Predict probabilities
                y_proba = ovr.predict_proba(X_val)
                
                # Calculate mean AUC across products
                fold_aucs = []
                for i in range(y_val.shape[1]):
                    positives = y_val[:, i].sum()
                    if positives > 0 and positives < len(y_val):
                        auc = roc_auc_score(y_val[:, i], y_proba[:, i])
                        fold_aucs.append(auc)

                if fold_aucs:
                    fold_mean_auc = np.mean(fold_aucs)
                    auc_scores.append(fold_mean_auc)
                
                # Memory cleanup after each fold
                del X_tr, X_val, y_tr, y_val, base_estimator, ovr, y_proba
                gc.collect()
            
            mean_auc = np.mean(auc_scores) if auc_scores else 0.0
            logger.info(f'Trial {trial.number + 1} completed: AUC={mean_auc:.4f}')
            
            return mean_auc
        
        return objective
    
    def _optimize_group_hyperparameters(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        group_name: str
    ) -> Dict[str, Any]:
        '''
            Run Optuna hyperparameter optimization for a group.
        '''
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        # Create Optuna objective
        objective = self._create_optuna_objective(
            X, y, self.cat_feature_indices
        )
        
        # Callback to log intermediate trial results
        def logging_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
            logger.info(
                f'[{group_name}]  | '
                f'Trial {trial.number + 1}/{self.n_trials}: AUC={trial.value:.4f}  | '
                f'Params: depth={trial.params.get("depth")}, learning_rate={trial.params.get("learning_rate", 0):.4f}, iterations={trial.params.get("iterations")}   | '
                f'Best AUC so far: {study.best_value:.4f}'
            )
        
        # Don't show Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        logger.info(f'Starting Optuna optimization for {group_name}: {self.n_trials} trials, timeout={self.optuna_timeout}s')
        
        # Optimize with timeout for faster execution
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.optuna_timeout,
            show_progress_bar=True,
            callbacks=[logging_callback]
        )
        
        # Build full params
        best_params = study.best_params.copy()
        best_params.update({
            'loss_function': 'Logloss',
            'random_seed': self.random_state,
            'logging_level': 'Silent',
            'thread_count': -1,
            'auto_class_weights': 'Balanced'
        })
        
        logger.info(f'DONE: Hyperparameter optimization for {group_name} group completed')
        logger.info(f'Best CV AUC of {group_name} group: {study.best_value:.4f}')
        
        return best_params

    # ----------- Feature Selection ------------ #
    def _select_features_by_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        min_importance: float = 0.001
    ) -> List[str]:
        '''
            Train a quick model to identify low-importance features,
            then return a list of features to keep.
            
            This speeds up subsequent training by dropping uninformative features.
        '''

        # Train a quick model on a subset of data for feature importance
        sample_size = min(50000, len(X))
        sample_idx = np.random.RandomState(self.random_state).choice(
            len(X), size=sample_size, replace=False
        )
        X_sample = X.iloc[sample_idx]
        
        # Use sum of all targets for a quick importance estimate
        y_sample = (y[sample_idx].sum(axis=1) > 0).astype(int)
        
        # Quick model with few iterations
        quick_model = CatBoostClassifier(
            iterations=50,
            depth=4,
            learning_rate=0.1,
            loss_function='Logloss',
            random_seed=self.random_state,
            logging_level='Silent',
            thread_count=self.thread_count,
            cat_features=self.cat_feature_indices if self.cat_feature_indices else None
        )
        quick_model.fit(X_sample, y_sample)
        
        # Get feature importances
        importances = quick_model.get_feature_importance()
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Normalize importances
        total_importance = importance_df['importance'].sum()
        importance_df['importance_norm'] = importance_df['importance'] / total_importance
        
        # Select features above threshold
        selected = importance_df[importance_df['importance_norm'] >= min_importance]['feature'].tolist()
        dropped = importance_df[importance_df['importance_norm'] < min_importance]['feature'].tolist()
        
        logger.info(f'DONE: Feature selection completed')
        logger.info(f'Number of features kept: {len(selected)} out of {len(X.columns)}')
        if dropped:
            logger.info(f'Dropped low-importance features: {dropped[:10]}...' if len(dropped) > 10 else f'Dropped features: {dropped}')
        
        # Free up memory
        del quick_model, X_sample, y_sample, importance_df
        gc.collect()
        
        return selected

    # ----------- Threshold Optimization ------------ #
    def _optimize_thresholds_for_group(
        self,
        products: List[str],
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        '''
            Optimize classification thresholds (probability, above which the product is recommended)
            per product using F1 score.
        '''
        thresholds = {}

        # Optimize thresholds for each product
        for i, product in enumerate(products):
            y_true_i = y_true[:, i]  # True labels for this product
            y_proba_i = y_proba[:, i]  # Predicted probabilities for this product

            # If no positive cases, set threshold to 0.5
            if y_true_i.sum() == 0:
                thresholds[product] = 0.5
                continue
            
            # Calculate precision, recall, and thresholds
            precision, recall, thresh = precision_recall_curve(y_true_i, y_proba_i)
            
            # SAFE F1 calculation using np.divide (handles 0/0 → 0)
            numerator = 2 * precision[:-1] * recall[:-1]
            denominator = precision[:-1] + recall[:-1]
            f1_scores = np.divide(
                numerator, 
                denominator, 
                out=np.zeros_like(denominator),  # Fill NaN with 0
                where=(denominator != 0)
            )
            
            # Find best threshold
            if len(f1_scores) > 0 and np.max(f1_scores) > 0:
                best_idx = np.argmax(f1_scores)
                thresholds[product] = float(thresh[best_idx])
            else:
                thresholds[product] = 0.5
            
        logger.info(f'DONE: Thresholds optimization for {len(products)} products completed')
        logger.info(f'Product thresholds: {thresholds}')

        return thresholds
    
    # ----------- Training ------------ #
    def fit(
        self,
        X_train: Union[pl.DataFrame, pd.DataFrame],
        y_train: Union[pl.DataFrame, pd.DataFrame],
        X_val: Optional[Union[pl.DataFrame, pd.DataFrame]] = None,
        y_val: Optional[Union[pl.DataFrame, pd.DataFrame]] = None,
        target_names: Optional[List[str]] = None
    ) -> 'OvRGroupModel':
        '''
            Train OvR models for all product groups.
        '''

        # Prepare data
        X_pd, y_pd = self._prepare_data(X_train, y_train)
        self.feature_names = list(X_pd.columns)
        self.cat_feature_indices = self._get_cat_feature_indices(X_pd)
        
        # Optional feature selection to speed up training
        if self.use_feature_selection:
            # Get target array for feature selection
            y_for_selection = y_pd.astype(int).values
            self.selected_features = self._select_features_by_importance(
                X_pd, y_for_selection, self.min_feature_importance
            )
            
            # Filter to selected features
            X_pd = X_pd[self.selected_features]
            self.feature_names = list(X_pd.columns)
            # Recompute cat indices for filtered features
            self.cat_feature_indices = self._get_cat_feature_indices(X_pd)
               
        # Get target names
        if target_names is not None:
            self.all_products = target_names
        elif y_pd is not None:
            self.all_products = list(y_pd.columns)
        logger.info(f'Data after feature selection: |'
                    f'Total products: {len(self.all_products)} |'
                    f'Total features: {len(self.feature_names)} |'
                    f'Categorical features: {len(self.cat_feature_indices)}'
                )
        
        # Group products by prevalence
        self.product_groups = self._group_products_by_prevalence(y_pd)
        
        # Prepare validation data
        X_val_pd, y_val_pd = None, None
        if X_val is not None and y_val is not None:
            X_val_pd, y_val_pd = self._prepare_data(X_val, y_val)
            # Apply same feature selection to validation data
            if self.use_feature_selection and self.selected_features:
                X_val_pd = X_val_pd[self.selected_features]
        
        # Start MLflow run
        mlflow_run = None
        if self.mlflow_experiment:
            mlflow_run = mlflow.start_run(
                run_name=f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
            mlflow.log_param('n_products', len(self.all_products))
            mlflow.log_param('n_features', len(self.feature_names))
            mlflow.log_param('n_train_samples', len(X_pd))
            mlflow.log_param('n_frequent', len(self.product_groups['frequent']))
            mlflow.log_param('n_mid', len(self.product_groups['mid']))
            mlflow.log_param('n_rare', len(self.product_groups['rare']))
            mlflow.log_param('optimize', self.optimize)
            mlflow.log_param('n_trials', self.n_trials)
            mlflow.log_param('optuna_timeout', self.optuna_timeout)
        
        # Train model for each group with progress bar
        group_names = ['frequent', 'mid', 'rare']
        group_progress = tqdm(group_names, desc='Training Groups', unit='group')
        
        try:
            for group_name in group_progress:
                products = self.product_groups[group_name]
                
                if not products:
                    logger.info(f'Skipping {group_name} group: no products')
                    continue
                
                group_progress.set_postfix({'group': group_name, 'products': len(products)})
                logger.info(f'Training {group_name} group ({len(products)} products)')
                
                # Get target columns for this group (fill nulls with 0, convert to int)
                y_group = y_pd[products].astype(int).values
                
                # Start nested MLflow run for this group
                group_mlflow_run = None
                if mlflow_run:
                    group_mlflow_run = mlflow.start_run(
                        run_name=f'group_{group_name}',
                        nested=True
                    )
                    mlflow.log_param('group_name', group_name)
                    mlflow.log_param('n_products', len(products))
                    mlflow.log_param('products', products[:10])  # Log first 10
                
                try:
                    # Optuna optimization
                    if self.optimize:
                        best_params = self._optimize_group_hyperparameters(
                            X_pd, y_group, group_name
                        )
                        self.best_params[group_name] = best_params
                        
                        if group_mlflow_run:
                            for k, v in best_params.items():
                                if k not in ['verbose', 'thread_count']:
                                    mlflow.log_param(f'hp_{k}', v)
                        
                        # Memory cleanup after optimization
                        gc.collect()

                    else:
                        logger.info(f'Skipping Optuna optimization, using default params for {group_name}')
                        best_params = {
                            'iterations': 200,
                            'depth': 6,
                            'learning_rate': 0.1,
                            'loss_function': 'Logloss',
                            'random_seed': self.random_state,
                            'logging_level': 'Silent',
                            'thread_count': self.thread_count,
                            'auto_class_weights': 'Balanced',
                            # Speed optimizations
                            'boosting_type': self.boosting_type,
                            'border_count': self.border_count,
                            'max_ctr_complexity': self.max_ctr_complexity,
                            'one_hot_max_size': self.one_hot_max_size,
                            'early_stopping_rounds': self.early_stopping_rounds,
                            'store_all_simple_ctr': False,
                        }
                        self.best_params[group_name] = best_params
                        logger.info(f' Default params: iterations={best_params["iterations"]}, depth={best_params["depth"]}, '
                                   f'learning_rate={best_params["learning_rate"]}, boosting_type={self.boosting_type}')
                                        
                    # Train products one at a time to minimize memory usage
                    n_products_group = len(products)
                    logger.info(f'Training {group_name} group: {n_products_group} products (one at a time for memory efficiency)')
                    
                    # Create temp directory for intermediate saves
                    temp_model_dir = self.models_dir / f'temp_{group_name}'
                    temp_model_dir.mkdir(parents=True, exist_ok=True)
                    
                    estimators = []
                    skipped_products = []  # Products with no positive examples
                    trained_products = []  # Products that were actually trained
                    
                    for prod_idx, product in enumerate(products):
                        # Get binary target for this product
                        y_product = y_group[:, prod_idx]
                        n_positive = y_product.sum()
                        n_negative = len(y_product) - n_positive
                        
                        # Check if we have both classes
                        if n_positive == 0 or n_negative == 0:
                            logger.warning(f'[{prod_idx + 1}/{n_products_group}] Skipped: {product} (only {n_positive} positives, {n_negative} negatives)')
                            skipped_products.append(product)
                            continue
                        
                        logger.info(f'[{prod_idx + 1}/{n_products_group}] Training: {product} ({n_positive:,} positives, {n_negative:,} negatives)')
                        
                        # Create and train single CatBoost model
                        estimator = CatBoostClassifier(
                            cat_features=self.cat_feature_indices if self.cat_feature_indices else None,
                            **best_params
                        )
                        estimator.fit(X_pd, y_product)
                        
                        # Save model immediately to disk
                        model_path = temp_model_dir / f'{product}.cbm'
                        estimator.save_model(str(model_path))
                        trained_products.append(product)
                        logger.info(f'[{prod_idx + 1}/{n_products_group}] Saved: {product}')
                        
                        # Free up memory
                        del estimator
                        gc.collect()
                    
                    if skipped_products:
                        logger.warning(f'Skipped {len(skipped_products)} products with no positive examples: {skipped_products}')
                    
                    logger.info(f'Trained {len(trained_products)}/{n_products_group} products')
                    
                    # Reload all trained models
                    for product in trained_products:
                        model_path = temp_model_dir / f'{product}.cbm'
                        estimator = CatBoostClassifier()
                        estimator.load_model(str(model_path))
                        estimators.append(estimator)
                    
                    # Update products list to only include trained ones
                    self.product_groups[group_name] = trained_products
                    
                    # Skip if no products were trained
                    if not trained_products:
                        logger.warning(f'No products trained for {group_name} group (all skipped)')
                        continue
                    
                    # Create OvR-like wrapper with proper sklearn attributes
                    from sklearn.preprocessing import LabelBinarizer
                    ovr_model = OneVsRestClassifier(CatBoostClassifier())
                    ovr_model.estimators_ = estimators
                    ovr_model.classes_ = np.array([0, 1])
                    # Initialize label_binarizer_ to avoid predict_proba errors
                    ovr_model.label_binarizer_ = LabelBinarizer(sparse_output=False)
                    ovr_model.label_binarizer_.fit(range(len(trained_products)))
                    
                    # Store model in registry
                    self.models[group_name] = ovr_model
                    logger.info(f'DONE: Final OvR model for {group_name} group trained')
                    
                    # Optimize thresholds on validation set
                    if X_val_pd is not None and y_val_pd is not None and trained_products:
                        y_val_group = y_val_pd[trained_products].astype(int).values
                        # Manually get probabilities from each estimator (bypass OvR wrapper issues)
                        y_proba_val = np.column_stack([
                            est.predict_proba(X_val_pd)[:, 1] for est in estimators
                        ])
                        group_thresholds = self._optimize_thresholds_for_group(
                            trained_products, y_val_group, y_proba_val
                        )
                        self.thresholds.update(group_thresholds)
                        logger.info(f'DONE: {group_name} thresholds optimized')
                    else:
                        # Default thresholds
                        for product in trained_products:
                            self.thresholds[product] = 0.5
                    
                    # Set default thresholds for skipped products
                    for product in skipped_products:
                        self.thresholds[product] = 1.0  # High threshold = never recommend
                    
                    # Log model to MLflow
                    if group_mlflow_run:
                        for i, (product, estimator) in enumerate(zip(trained_products, ovr_model.estimators_)):
                            mlflow.catboost.log_model(estimator, name=f'model_{product}')
                
                finally:
                    if group_mlflow_run:
                        mlflow.end_run()
            
            # Log overall metrics
            if mlflow_run:
                mlflow.log_metric('n_groups_trained', len(self.models))
        
        finally:
            if mlflow_run:
                mlflow.end_run()
        
        # ----------- Retrain on Full Data (Optional) ------------ #
        if self.retrain_on_full_data and self.best_params:
            logger.info('Starting retraining on full data')
            
            # Load full preprocessed data
            X_full, y_full = self._load_full_preprocessed_data()
            X_full_pd, y_full_pd = self._prepare_data(X_full, y_full)
            
            # Apply feature selection if it was used during optimization
            if self.use_feature_selection and self.selected_features:
                X_full_pd = X_full_pd[self.selected_features]
                logger.info(f'Applied feature selection to full data: {len(self.selected_features)} features')
            
            # Update cat feature indices for full data (should be same as sampled)
            self.cat_feature_indices = self._get_cat_feature_indices(X_full_pd)
            
            # Retrain each group with optimized hyperparameters
            for group_name in ['frequent', 'mid', 'rare']:
                products = self.product_groups.get(group_name, [])
                if not products or group_name not in self.best_params:
                    logger.info(f'Skipping {group_name} group: no products or no optimized params')
                    continue
                
                logger.info(f'Retraining {group_name.upper()} group ({len(products)} products) on full data')
                
                best_params = self.best_params[group_name]
                y_group = y_full_pd[products].astype(int).values
                
                # Create temp directory for intermediate saves
                temp_model_dir = self.models_dir / f'temp_{group_name}_full'
                temp_model_dir.mkdir(parents=True, exist_ok=True)
                
                estimators = []
                trained_products = []
                
                for prod_idx, product in enumerate(products):
                    y_product = y_group[:, prod_idx]
                    n_positive = y_product.sum()
                    n_negative = len(y_product) - n_positive
                    
                    if n_positive == 0 or n_negative == 0:
                        logger.warning(f'[{prod_idx + 1}/{len(products)}] Skipped: {product} (only {n_positive} positives)')
                        continue
                    
                    logger.info(f'[{prod_idx + 1}/{len(products)}] Retraining: {product} ({n_positive:,} positives)')
                    
                    estimator = CatBoostClassifier(
                        cat_features=self.cat_feature_indices if self.cat_feature_indices else None,
                        **best_params
                    )
                    estimator.fit(X_full_pd, y_product)
                    
                    model_path = temp_model_dir / f'{product}.cbm'
                    estimator.save_model(str(model_path))
                    trained_products.append(product)
                    
                    del estimator
                    gc.collect()
                
                # Reload all trained models
                estimators = []
                for product in trained_products:
                    model_path = temp_model_dir / f'{product}.cbm'
                    estimator = CatBoostClassifier()
                    estimator.load_model(str(model_path))
                    estimators.append(estimator)
                
                # Update product groups with trained products only
                self.product_groups[group_name] = trained_products
                
                if not trained_products:
                    logger.warning(f'No products trained for {group_name} group on full data')
                    continue
                
                # Create OvR-like wrapper
                ovr_model = OneVsRestClassifier(CatBoostClassifier())
                ovr_model.estimators_ = estimators
                ovr_model.classes_ = np.array([0, 1])
                ovr_model.label_binarizer_ = LabelBinarizer(sparse_output=False)
                ovr_model.label_binarizer_.fit(range(len(trained_products)))
                
                # Replace model in registry
                self.models[group_name] = ovr_model
                logger.info(f'{group_name} group retrained on full data: {len(trained_products)} products')
            
            logger.info('DONE: Retraining on full data completed')
            
            del X_full, y_full, X_full_pd, y_full_pd
            gc.collect()
        
        # Save thresholds
        thresholds_file = self.results_dir / 'ovr_group_thresholds.json'
        with open(thresholds_file, 'w') as f:
            json.dump(self.thresholds, f, indent=2)
        logger.info(f'Thresholds saved to: {thresholds_file}')
        
        logger.info(f'DONE: Training completed: {len(self.models)} group models')
        
        # Final memory cleanup
        # Clear cached quantized pools (no longer needed after training)
        self._quantized_pools.clear()
        gc.collect()
        
        return self

    # ----------- Inference ------------ #
    def predict_proba(
        self,
        X: Union[pl.DataFrame, pd.DataFrame]
    ) -> pd.DataFrame:
        '''
            Get probability predictions for all products for all customers in test data.
        '''

        # Check if models are trained
        if not self.models:
            raise ValueError('No models trained')
        
        # Prepare data
        X_pd, _ = self._prepare_data(X)
        
        # Apply feature selection if it was used during training
        if self.use_feature_selection and self.selected_features:
            X_pd = X_pd[self.selected_features]
        
        # Collect predictions from all groups
        all_probas = {}
        
        for group_name, model in self.models.items(): # Loop over groups
            products = self.product_groups[group_name]
            # Get probabilities from each estimator directly (avoid OvR wrapper issues)
            for i, product in enumerate(products): # Loop over products
                # CatBoost predict_proba returns [P(class=0), P(class=1)]
                all_probas[product] = model.estimators_[i].predict_proba(X_pd)[:, 1] 
        
        # Order by original product order
        ordered_probas = {p: all_probas[p] for p in self.all_products if p in all_probas} # Filter products that are in the model
        
        logger.info(f'DONE: Predictions for all products completed')
        
        return pd.DataFrame(ordered_probas)
    

    # ----------- Top-K Recommendations ------------ #
    def predict_top_k(
        self,
        X: Union[pl.DataFrame, pd.DataFrame],
        k: Optional[int] = None,
        apply_thresholds: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[List[str]]]:
        '''
            Get top-K product recommendations per customer.
            Combines all group predictions, ranks by probability.
        '''
        # Use provided k or default to self.top_k
        k = k if k is not None else self.top_k

        # Get probabilities for all products
        proba_df = self.predict_proba(X)
        # Convert to matrix
        proba_matrix = proba_df.values
        # Get product names
        product_names = list(proba_df.columns)
        # Get number of customers
        n_customers = proba_matrix.shape[0]
        
        # Rank products by probability
        # Get indices of top-k products for each customer
        top_k_indices = np.argsort(-proba_matrix, axis=1)[:, :k]
        # Get corresponding probabilities for the top-k products
        top_k_probas = np.take_along_axis(proba_matrix, top_k_indices, axis=1)
        
        # Apply threshold filtering
        if apply_thresholds:
            for i in range(n_customers): # Loop over customers
                for j in range(k): # Loop over top-k products
                    product_idx = top_k_indices[i, j]
                    product_name = product_names[product_idx]
                    threshold = self.thresholds.get(product_name, 0.5)
                    if top_k_probas[i, j] < threshold: # If probability is less than threshold, set to 0
                        top_k_probas[i, j] = 0.0
        
        # Get product names
        top_k_names = [
            [product_names[idx] for idx in customer_indices]
            for customer_indices in top_k_indices
        ]

        logger.info(f'DONE: Top-K recommendations for {n_customers} customers completed')
        
        return top_k_indices, top_k_probas, top_k_names
    
    def recommend(
        self,
        X: Union[pl.DataFrame, pd.DataFrame],
        return_probas: bool = True,
        apply_thresholds: bool = True
    ) -> pd.DataFrame:
        '''
            Generate product recommendations for customers.
        '''

        # Get top-k products for each customer
        top_k_indices, top_k_probas, top_k_names = self.predict_top_k(
            X, self.top_k, apply_thresholds
        )
        
        # Build recommendations
        recommendations = []
        for i in range(len(top_k_names)): # Loop over customers 
            rec = {
                'customer_idx': i,
                'recommendations': top_k_names[i]
            }
            if return_probas: # If return probabilities, add them to the recommendation
                rec['probabilities'] = top_k_probas[i].tolist()
            recommendations.append(rec)
        
        logger.info(f'DONE: Recommendations for {len(recommendations)} customers completed')

        return pd.DataFrame(recommendations)
    

    # ----------- Evaluation ------------ #
    def _calculate_map_at_k(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> float:
        '''
            Calculate Mean Average Precision @ K.
        '''
        aps = []
        
        for idx in range(len(y_true)): # Loop over samples
            true_labels = y_true[idx]
            pred_proba = y_pred_proba[idx] # Predicted probabilities for this sample
            
            # Skip if no positive labels
            if true_labels.sum() == 0:
                continue
            
            # Get top-k indices
            top_k_indices = np.argsort(-pred_proba)[:self.top_k]
            
            hits = 0
            precision_sum = 0
            for i, idx_pred in enumerate(top_k_indices): # Loop over top-k indices
                if true_labels[idx_pred] == 1: # If true label is 1, increment hits
                    hits += 1
                    precision_sum += hits / (i + 1)
            
            # Number of relevant items
            n_relevant = min(self.top_k, int(true_labels.sum()))
            # Average precision
            ap = precision_sum / n_relevant if n_relevant > 0 else 0
            # Append average precision to list
            aps.append(ap)
        
        logger.info(f'DONE: MAP@K calculation completed')

        return np.mean(aps) if aps else 0.0
    
    def _calculate_precision_at_k(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> float:
        '''
            Calculate Precision @ K.
        '''
        precisions = []
        
        for idx in range(len(y_true)): # Loop over samples
            true_labels = y_true[idx] # True labels for this sample
            pred_proba = y_pred_proba[idx] # Predicted probabilities for this sample
            
            # Get top-k indices
            top_k_indices = np.argsort(-pred_proba)[:self.top_k]
            # Count hits
            hits = sum(true_labels[i] for i in top_k_indices)
            # Append precision to list
            precisions.append(hits / self.top_k)
        
        logger.info(f'DONE: Precision@K calculation completed')
        
        return np.mean(precisions) if precisions else 0.0
    
    def _calculate_precision_at_k_with_thresholds(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        products: List[str]
    ) -> float:
        '''
            Calculate Precision @ K with threshold filtering.
        '''
        precisions = []
        
        for idx in range(len(y_true)): # Loop over samples
            true_labels = y_true[idx] # True labels for this sample
            pred_proba = y_pred_proba[idx] # Predicted probabilities for this sample
            
            # Get sorted indices
            sorted_indices = np.argsort(-pred_proba)
            
            selected = []
            for i in sorted_indices: # Loop over sorted indices
                # If selected products are greater than or equal to k, break
                if len(selected) >= self.top_k: 
                    break
                product = products[i] # Get product name
                threshold = self.thresholds.get(product, 0.5) # Get threshold for this product
                # If probability is greater than threshold, add to selected
                if pred_proba[i] >= threshold:
                    selected.append(i)
            
             # If selected products are less than k, add remaining products
            if len(selected) < self.top_k:
                selected = list(sorted_indices[:self.top_k])
            
            # Count hits
            hits = sum(true_labels[i] for i in selected)
            # Append precision to list
            precisions.append(hits / self.top_k)
        
        logger.info(f'DONE: Precision@K with thresholds calculation completed')
        
        return np.mean(precisions) if precisions else 0.0

    def evaluate(
        self,
        X_test: Union[pl.DataFrame, pd.DataFrame],
        y_test: Union[pl.DataFrame, pd.DataFrame]
    ) -> Dict[str, Any]:
        '''
            Evaluate model performance on test data.
        '''

        logger.info('Evaluating OvR Grouped model')
        
        # Convert data to pandas DataFrames if needed
        X_pd, y_pd = self._prepare_data(X_test, y_test)
        
        # Get probabilities for all products
        y_pred_proba = self.predict_proba(X_pd)
        
        # Align columns
        # Get common products
        common_products = [p for p in self.all_products if p in y_pd.columns and p in y_pred_proba.columns]
        # Get true labels and probabilities for common products (fill nulls, convert to int)
        y_true_aligned = y_pd[common_products].fillna(False).astype(int).values # True labels
        y_proba_aligned = y_pred_proba[common_products].values # Predicted probabilities
        
        # Per-product metrics
        metrics_per_product = {}
        for i, product in enumerate(common_products): # Loop over common products
            y_true_i = y_true_aligned[:, i] # True labels for this product
            y_proba_i = y_proba_aligned[:, i] # Predicted probabilities for this product
            
            if y_true_i.sum() == 0: # If no positive cases, skip
                continue
            
            # Get threshold for this product
            threshold = self.thresholds.get(product, 0.5) 
            # Convert probabilities to binary predictions
            y_pred_i = (y_proba_i >= threshold).astype(int)
            
            # Calculate metrics for this product
            metrics_per_product[product] = {
                'auc_roc': roc_auc_score(y_true_i, y_proba_i),
                'avg_precision': average_precision_score(y_true_i, y_proba_i),
                'precision': precision_score(y_true_i, y_pred_i, zero_division=0),
                'recall': recall_score(y_true_i, y_pred_i, zero_division=0),
                'f1': f1_score(y_true_i, y_pred_i, zero_division=0),
                'log_loss': log_loss(y_true_i, y_proba_i),
                'positive_rate': float(y_true_i.mean()),
                'threshold': threshold
            }
        
        # Aggregate metrics
        aucs = [m['auc_roc'] for m in metrics_per_product.values()]
        avg_precs = [m['avg_precision'] for m in metrics_per_product.values()]
        
        # Mean Average Precision @ K - Average of precision at each rank
        map_at_k = self._calculate_map_at_k(y_true_aligned, y_proba_aligned)
        
        # Precision @ K - Ratio of true positives at each rank
        precision_at_k = self._calculate_precision_at_k(y_true_aligned, y_proba_aligned)
        
        # Precision @ K with thresholds - Ratio of true positives at each rank with threshold filtering
        precision_at_k_thresh = self._calculate_precision_at_k_with_thresholds(
            y_true_aligned, y_proba_aligned, common_products
        )
        
        # Overall metrics
        overall_metrics = {
            'mean_auc': np.mean(aucs) if aucs else 0.0,
            'std_auc': np.std(aucs) if aucs else 0.0,
            'mean_avg_precision': np.mean(avg_precs) if avg_precs else 0.0,
            f'map_at_{self.top_k}': map_at_k,
            f'precision_at_{self.top_k}': precision_at_k,
            f'precision_at_{self.top_k}_with_thresholds': precision_at_k_thresh,
            'n_products_evaluated': len(metrics_per_product)
        }
        
        # Group-level metrics
        group_metrics = {}
        for group_name, products in self.product_groups.items(): # Loop over groups
            group_products = [p for p in products if p in metrics_per_product] # Get products that are in this group and have metrics
            if group_products: # If there are products in this group
                group_aucs = [metrics_per_product[p]['auc_roc'] for p in group_products] # Get AUCs for this group
                group_metrics[group_name] = { # Store metrics for this group
                    'mean_auc': np.mean(group_aucs),
                    'n_products': len(group_products)
                }
        
        results = {
            'overall': overall_metrics,
            'per_group': group_metrics,
            'per_product': metrics_per_product
        }
        
        logger.info('DONE: Evaluation completed')
        mean_auc = overall_metrics["mean_auc"]
        std_auc = overall_metrics["std_auc"]
        logger.info(f'Metrics: Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}  | '
                    f'MAP@{self.top_k}: {map_at_k:.4f}  | '
                    f'Precision@{self.top_k}: {precision_at_k:.4f}  | '
                    f'Precision@{self.top_k} (with thresholds): {precision_at_k_thresh:.4f}'
                    )
        
        for group_name, g_metrics in group_metrics.items():
            logger.info(f'DONE: Evaluation done for group: {group_name}')
            g_mean_auc = g_metrics["mean_auc"]
            g_n_products = g_metrics["n_products"]
            logger.info(f"{group_name}: AUC={g_mean_auc:.4f} ({g_n_products} products)")
        
        # Save results locally
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f'ovr_grouped_evaluation_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f'Evaluation results saved to: {results_file}')
        
        return results
    
    # ----------- Results Access ------------ #
    def get_group_model(
        self, 
        group_name: str
    ) -> OneVsRestClassifier:
        '''
            Get the trained OvR model for a specific group.
        '''

        # If group not found, raise error
        if group_name not in self.models:
            logger.error(f'ERROR: Group {group_name} not found.')
            raise ValueError(f'Group {group_name} not found.')

        return self.models[group_name]
    
    def get_group_products(
        self, group_name: str) -> List[str]:
        '''
            Get list of products in a specific group.
        '''

        # If group not found, raise error
        if group_name not in self.product_groups:
            logger.error(f'ERROR: Group {group_name} not found')
            raise ValueError(f'Group {group_name} not found')

        return self.product_groups[group_name]
    
    def get_feature_importance(
        self,
        group_name: Optional[str] = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        '''
            Get feature importance, optionally for a specific group.
        '''

        # If no models trained, raise error
        if not self.models:
            logger.error(f'ERROR: No models trained')
            raise ValueError('No models trained')
        
        # Get groups to use
        if group_name is not None: # If group name is provided, use only that group
            if group_name not in self.models: # If group not found, raise error
                logger.error(f'ERROR: Group {group_name} not found')
                raise ValueError(f'Group {group_name} not found')
            groups_to_use = {group_name: self.models[group_name]}
            products_to_use = {group_name: self.product_groups[group_name]}
        else:
            groups_to_use = self.models
            products_to_use = self.product_groups
        
        importance_data = []
        for g_name, model in groups_to_use.items(): # Loop over groups
            products = products_to_use[g_name] # Get products for this group
            for product, estimator in zip(products, model.estimators_): # Loop over products and estimators
                importance = estimator.get_feature_importance(type='FeatureImportance') # Get feature importance
                for feat_name, imp_value in zip(self.feature_names, importance): # Loop over feature names and importances
                    importance_data.append({
                        'group': g_name,
                        'product': product,
                        'feature': feat_name,
                        'importance': imp_value
                    })
        
        df_imp = pd.DataFrame(importance_data)
        
        mean_imp = (
            df_imp
                .groupby('feature')['importance']
                .mean()
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index()
        )
        mean_imp.columns = ['feature', 'mean_importance']
        
        return mean_imp
    
    # ----------- Save / Load ------------ #
    def save(
        self, 
        model_name: str = 'ovr_grouped_catboost'
    ) -> str:
        '''
            Save all group models and metadata.
        '''

        # If no models trained, raise error
        if not self.models:
            logger.error(f'ERROR: No models to save')
            raise ValueError('No models to save')
        
        # Create model directory if it doesn't exist
        model_dir = self.models_dir / f'{model_name}'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each group model
        for group_name in tqdm(self.models.keys(), desc='Saving groups'):            
            model = self.models[group_name]
            products = self.product_groups[group_name]
            
            # Save estimators
            for product, estimator in zip(products, model.estimators_):
                estimator.save_model(str(model_dir / f'{group_name}_{product}.cbm'))
            
            # Save group metadata
            group_meta = {
                'products': products,
                'best_params': self.best_params.get(group_name, {}),
                'thresholds': {p: self.thresholds.get(p, 0.5) for p in products}
            }
            with open(model_dir / f'{group_name}_metadata.json', 'w') as f:
                json.dump(group_meta, f, indent=2)
        
        # Save main metadata
        metadata = {
            'all_products': self.all_products,
            'product_groups': self.product_groups,
            'thresholds': self.thresholds,
            'best_params': self.best_params,
            'feature_names': self.feature_names,
            'cat_features': self.cat_features,
            'cat_feature_indices': self.cat_feature_indices,
            'cv_scores': self.cv_scores,
            'random_state': self.random_state,
            'frequent_threshold': self.frequent_threshold,
            'rare_threshold': self.rare_threshold,
            'n_groups': len(self.models),
            # Feature selection metadata
            'use_feature_selection': self.use_feature_selection,
            'selected_features': self.selected_features,
        }
        
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f'DONE: Models saved to: {model_dir}')

        return str(model_dir)
    
    def load(
        self, 
        model_path: str
    ) -> 'OvRGroupModel':
        '''
            Load group models from disk.
        '''
        model_dir = Path(model_path)
        
        # If model directory not found, raise error
        if not model_dir.exists():
            raise ValueError(f'Model directory not found: {model_path}')
        
        # Load main metadata
        with open(model_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.all_products = metadata['all_products']
        self.product_groups = metadata['product_groups']
        self.thresholds = metadata['thresholds']
        self.best_params = metadata.get('best_params', {})
        self.feature_names = metadata['feature_names']
        self.cat_features = metadata.get('cat_features', [])
        self.cat_feature_indices = metadata.get('cat_feature_indices', [])
        self.cv_scores = metadata.get('cv_scores', {})
        
        # Feature selection metadata
        self.use_feature_selection = metadata.get('use_feature_selection', False)
        self.selected_features = metadata.get('selected_features', None)
        
        # Load each group
        self.models = {}
        
        for group_name in tqdm(['frequent', 'mid', 'rare'], desc='Loading groups'):
            # Load group metadata
            with open(model_dir / f'{group_name}_metadata.json', 'r') as f:
                group_meta = json.load(f)
            
            products = group_meta['products']
            
            # Load estimators
            estimators = []
            for product in products:
                model = CatBoostClassifier()
                model.load_model(str(model_dir / f'{group_name}_{product}.cbm'))
                estimators.append(model)
            
            # Reconstruct OvR
            ovr = OneVsRestClassifier(CatBoostClassifier())
            ovr.estimators_ = estimators
            ovr.classes_ = np.array([0, 1])
            
            self.models[group_name] = ovr
        
        logger.info(f'DONE: Loaded {len(self.models)} group models from: {model_path}')
        return self

def run_modelling_ovr():
    '''
        Run OvR Group Model training pipeline.
    '''
    logger.info('Starting OvR Group Model training pipeline')
    
    # Load data
    X_train, X_test, y_train, y_test = load_training_data(DATA_DIR)
    
    # Get product names
    target_names = get_product_names(y_train)
    
    # Detect categorical features
    cat_features = [
        col for col in X_train.columns
        if X_train[col].dtype == 'category' or X_train[col].dtype == 'object'
    ]
    
    # Initialize model with speed optimizations
    model = OvRGroupModel(
        cat_features=cat_features,
        random_state=RANDOM_STATE,
        models_dir=MODELS_DIR,
        results_dir=RESULTS_DIR,
        data_dir=DATA_DIR,
        mlflow_experiment=MLFLOW_EXPERIMENT,
        frequent_threshold=FREQUENT_THRESHOLD,
        rare_threshold=RARE_THRESHOLD,
        optimize=OPTIMIZE,
        n_trials=OPT_N_TRIALS,
        optuna_timeout=OPT_OPTUNA_TIMEOUT,
        top_k=TOP_K,
        retrain_on_full_data=RETRAIN_ON_FULL_DATA,
        preprocessed_data_file=PREPROCESSED_DATA_FILE,
        # HPO speed parameters
        hpo_n_splits=OPT_N_SPLITS,
        hpo_max_iterations=OPT_MAX_ITERATIONS,
        hpo_early_stopping_rounds=OPT_EARLY_STOPPING_ROUNDS,
        use_quantized_pool=OPT_USE_QUANTIZED_POOL,
        # Training speed parameters
        boosting_type=MODEL_BOOSTING_TYPE,
        border_count=MODEL_BORDER_COUNT,
        thread_count=MODEL_THREAD_COUNT,
        max_ctr_complexity=MODEL_MAX_CTR_COMPLEXITY,
        one_hot_max_size=MODEL_ONE_HOT_MAX_SIZE,
        early_stopping_rounds=MODEL_EARLY_STOPPING_ROUNDS,
        # Feature selection
        use_feature_selection=USE_FEATURE_SELECTION,
        min_feature_importance=MIN_FEATURE_IMPORTANCE,
    )
    
    # Train all group models
    model.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        target_names=target_names
    )
    
    # Evaluate on test set
    results = model.evaluate(X_test, y_test)
    
    # Get top-7 recommendations for test customers
    recommendations = model.recommend(X_test)
    
    # Show sample recommendations
    logger.info('Sample recommendations (first 5 customers):')
    for i in range(min(5, len(recommendations))):
        rec = recommendations.iloc[i]
        logger.info(f'Customer {i}: {rec["recommendations"][:3]}')
    
    # Feature importance
    importance = model.get_feature_importance(top_n=10)
    logger.info('Top 10 features mean importance:')
    logger.info(importance.to_string(index=False))
    
    # Save models
    model_path = model.save('ovr_grouped_catboost')
    logger.info(f'Model saved to: {model_path}')
    
    logger.info('OvR Group Model training pipeline completed successfully')


# ---------- Main ---------- #
if __name__ == '__main__':
    run_modelling_ovr()


# ---------- All exports ---------- #
__all__ = ['run_modelling_ovr', 'OvRGroupModel', 'load_training_data', 'get_product_names'] 
