'''
    CatBoost Regression Model for Energy Prosumer Prediction

    This module provides the CatBoostRegressor class for training and predicting
    energy consumption/production using capacity_factor as target.

    Strategy:
    - Train separate models for consumption and production.

    - Use CatBoostRegressor class for training and predicting.

    - Test data is the April–May 2023 holdout.
      This data is separated before model training and is never used for Optuna,
      early stopping, tree-count selection, or final model fitting.
      Test data is used only for evaluation of the final model performance.

    - Train / validation data is split into three time slices using Rolling-origin CV logic, that is 
        for each of validation months (the last 3 months January, February, and March 2023) 
        the code creates three time slices:
        -- fit for training with early stopping, 
        -- inner for selection of the number of trees through early stopping, 
        -- val for hyperparameters optimization, that is for calculating the MAE that Optuna minimizes.
        Fit and inner together form train set.
        Fit and inner together form the training set for the model that is
        Val is never passed as eval_set.
        For example, for Fabruary 2023 as a holdout month:
        -- fit: everything before roughly mid-January,
        -- inner: the last 14 days of January.
        -- val: all of February;

    - Use Optuna to optimize CatBoost hyperparameters by minimizing the
      row-weighted mean power MAE across the three CV folds.

    - After hyperparameter selection, refit each segment on all pre-holdout data and evaluate once on the test data.

    - Fit with sample_weight=installed_capacity so MAE on capacity_factor
      matches MAE on power.
    - Clip the predicted capacity_factor to [0, segment upper] before
      converting it back to power.

    Input:
    - Pandas DataFrame with features from FeatureEngineering
    - Target: capacity_factor

    Output:
    - Power predictions (capacity_factor * installed_capacity)

    Usage:
        python -m src.preds.modelling_catboost
        python -m src.preds.modelling_catboost --debug
        python -m src.preds.modelling_catboost --no-tree-scale
'''

# ---------- Imports ---------- #
import os
import gc
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from dotenv import load_dotenv
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostRegressor as CatBoost, Pool
import optuna
from optuna.samplers import TPESampler

from src.logging_setup import setup_logging
from src.preds.feature_engineering import TARGET_LAG_HOURS


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
DEFAULT_N_TRIALS = int(os.getenv('OPTUNA_N_TRIALS', 2))
# Optional wall-clock budget for each Optuna study, in seconds.
_OPTUNA_TIMEOUT = os.getenv('OPTUNA_TIMEOUT_SECONDS')
DEFAULT_OPTUNA_TIMEOUT = float(_OPTUNA_TIMEOUT) if _OPTUNA_TIMEOUT else None
# How many month-long folds to score, counting backwards from the holdout.
DEFAULT_CV_SPLITS = int(os.getenv('CV_SPLITS', 3))
# Last two months of the competition window. Optuna never sees these rows.
HOLDOUT_START = pd.Timestamp('2023-04-01')
# Inner slice used only to decide how many trees to keep.
INNER_ES_DAYS = 14
# Stop the inner fit after this many rounds without a better MAE.
EARLY_STOPPING_ROUNDS = 50
# Passed to fit() and transform() when debug=True, so a short run uses the same units in both.
DEBUG_PREDICTION_UNITS = [3, 19, 48, 60]

# Target and categorical features
TARGET = 'capacity_factor'
# is_consumption is constant inside each segment model, so it is not a feature.
CAT_FEATURES = [
    'prediction_unit_id',
    'county', 'is_business', 'product_type',
    'is_holiday', 'is_day_before_holiday',
    'is_dst_spring_forward', 'is_dst_fall_backward',
]

# Target lags the saved metadata advertises. 12h and 24h are not revealed in time, so they are gone.
LAG_VALUES = TARGET_LAG_HOURS

# Columns that identify a row or a fold. None of them is a predictor.
NON_FEATURE_COLUMNS = [
    'datetime', 'row_id', 'data_block_id', 'date', 'power',
    'is_consumption',
    'client_block_id', 'electricity_block_id', 'gas_block_id', 'fw_block_id', 'hw_block_id',
]


# ---------- Fold helpers ---------- #
def block_day(df: pd.DataFrame) -> pd.Series:
    '''
        One competition batch is one calendar day. Return that day, midnight, for every row.
    '''
    # Min timestamp of the batch, then midnight, so a missing first hour cannot change the day.
    return df.groupby('data_block_id')['datetime'].transform('min').dt.normalize()

def holdout_mask(df: pd.DataFrame, holdout_start: pd.Timestamp = HOLDOUT_START) -> pd.Series:
    '''
        True for every row whose batch day is on or after the holdout. No batch is split.
    '''
    day = block_day(df)
    # The cut is on the batch day, so every hour of a day stays on the same side.
    mask = day >= holdout_start
    # A batch id on both sides would mean the score mixes training hours with scored hours.
    overlap = set(df.loc[mask, 'data_block_id']) & set(df.loc[~mask, 'data_block_id'])
    if overlap:
        raise ValueError(f'Holdout boundary splits data_block_id values: {sorted(overlap)[:5]}')
    return mask

def validation_month_starts(
    cv_splits: int,
    holdout_start: pd.Timestamp = HOLDOUT_START,
) -> list:
    '''
        Month starts of the rolling-origin folds, oldest first, all strictly before the holdout.
    '''
    starts = []
    for months_back in range(cv_splits, 0, -1):
        # April holdout and 3 folds -> January, February, March.
        month_start = (holdout_start - pd.DateOffset(months=months_back)).replace(day=1)
        starts.append(pd.Timestamp(month_start))
    return starts

def rolling_origin_masks(df: pd.DataFrame, val_start: pd.Timestamp, inner_days: int = INNER_ES_DAYS) -> dict:
    '''
        Masks for one month fold.

        'val' is the scored month. It is never passed as an eval_set.
        'inner' is only for early stopping. 'train' is what the scored model is fit on.
    '''
    day = block_day(df)
    # The scored window is one calendar month of whole batches.
    val_end = val_start + pd.DateOffset(months=1)
    # The two weeks before that month choose the tree count and nothing else.
    es_start = val_start - pd.Timedelta(days=inner_days)
    fit = day < es_start
    inner = (day >= es_start) & (day < val_start)
    train = day < val_start
    val = (day >= val_start) & (day < val_end)
    return {'fit': fit, 'inner': inner, 'train': train, 'val': val}

def scaled_tree_count(
    best_iteration_index: Optional[int],
    n_es_train: int,
    n_full_train: int,
    fallback: int,
    scale_trees: bool = True,
) -> int:
    '''
        Turn CatBoost's 0-based best iteration into a tree count.

        If enabled, scaling multiplies CatBoost best iteration by the ratio of
        the full training rows to the early-stopping rows, which adds trees the
        inner slice never checked.
    '''
    # None means early stopping did not report an iteration, so keep the search budget.
    if best_iteration_index is None or best_iteration_index < 0:
        n_trees = fallback
    else:
        # best_iteration_ is the index of the best tree, so the count is one more.
        n_trees = best_iteration_index + 1
    if n_es_train <= 0:
        raise ValueError('Early-stopping train split has no rows')
    if not scale_trees:
        logger.info(f'Tree count left unscaled at {n_trees}')
        return max(1, int(n_trees))
    # More rows need proportionally more trees to reach the same leaf size.
    scale = n_full_train / n_es_train
    scaled = max(1, int(np.ceil(n_trees * scale)))
    logger.info(f'Tree count scaled from {n_trees} to {scaled} (x{scale:.3f})')
    return scaled


def score_functions_for_grow_policy(grow_policy: str) -> List[str]:
    '''
        Cosine is not valid with Lossguide. Restrict the search to the legal pairs.
    '''
    if grow_policy == 'Lossguide':
        return ['L2']
    return ['Cosine', 'L2']


def suggest_catboost_params(trial: optuna.Trial, random_state: int) -> dict:
    '''
        CatBoost hyperparameters for one Optuna trial.

        Both categoricals use a fixed choice set. Cosine + Lossguide is pruned
        because Optuna cannot change the score_function space mid-study.
    '''
    grow_policy = trial.suggest_categorical(
        'grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']
    )
    score_function = trial.suggest_categorical('score_function', ['Cosine', 'L2'])
    if score_function not in score_functions_for_grow_policy(grow_policy):
        raise optuna.TrialPruned(
            f'{score_function} is not valid with grow_policy={grow_policy}'
        )
    return {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.01, 10, log=True),
        'score_function': score_function,
        # Weighted MAE on capacity_factor equals MAE on power when the weight is capacity.
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'grow_policy': grow_policy,
        'random_seed': random_state,
        'logging_level': 'Silent',
    }


def capacity_factor_clip_upper(values: pd.Series) -> float:
    '''
        Upper clip for predicted capacity_factor: 1.1 times the 99.9th training percentile, at least 1.
    '''
    finite = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    return float(max(1.0, np.quantile(finite, 0.999) * 1.1))


def limit_to_prediction_units(df: pd.DataFrame, units: List[int]) -> pd.DataFrame:
    '''
        Limit data only to a number of prediction_unit_id.

        If enabled, keeps only data rows whose prediction_unit_id is in `units`,
        raises an error if none of them are present.
    '''
    if 'prediction_unit_id' not in df.columns:
        raise ValueError('prediction_unit_id is required when debug units are set')
    # int() so an int16 id from the CSV matches the Python ints in DEBUG_PREDICTION_UNITS.
    present = {int(unit) for unit in pd.unique(df['prediction_unit_id'])}
    missing = [unit for unit in units if unit not in present]
    if missing:
        logger.warning(f'Debug prediction_unit_id values not in the frame: {missing}')
    kept = [unit for unit in units if unit in present]
    if not kept:
        raise ValueError(f'None of the debug units {list(units)} are in the frame')
    limited = df.loc[df['prediction_unit_id'].isin(kept)].copy()
    logger.info(
        f'Debug mode: {len(limited)} rows for prediction_unit_id {kept} '
        f'(dropped {len(df) - len(limited)} rows)'
    )
    return limited


def segment_power_mae(power_true: np.ndarray, power_pred: np.ndarray, is_consumption: np.ndarray) -> dict:
    '''
        MAE in power units, overall and split into consumption and production.
    '''
    error = np.abs(np.asarray(power_pred) - np.asarray(power_true))
    # Boolean view of the 0/1 flag. True is consumption.
    consumption = np.asarray(is_consumption).astype(bool)

    def _mean(mask: np.ndarray):
        # An empty side has no MAE. JSON can store null.
        if mask.sum() == 0:
            return None
        return float(np.mean(error[mask]))

    return {
        'mae_power': float(np.mean(error)) if len(error) else None,
        'mae_power_consumption': _mean(consumption),
        'mae_power_production': _mean(~consumption),
        'n': int(len(error)),
    }


def _weighted_mae(folds: list) -> Optional[float]:
    '''
        Row-weighted mean of fold MAEs. Equal-weighting months would hide a longer month.
    '''
    total_n = sum(fold['n'] for fold in folds)
    if total_n == 0:
        return None
    total = sum(fold['mae_power'] * fold['n'] for fold in folds)
    return float(total / total_n)


def baseline_power_mae(df: pd.DataFrame, score_mask: pd.Series) -> dict:
    '''
        MAE of predicting each scored hour with the power from 48h ago and from 168h ago.

        The history has to be in `df`, not only the scored rows, because April 1 looks back into March.
    '''
    power_path = Path(PROCESSED_DATA_DIR) / 'power.parquet'
    # Actual power saved before it was dropped from the feature frame.
    actual = pd.read_parquet(power_path)
    history = df[['row_id', 'datetime', 'prediction_unit_id', 'is_consumption']].merge(
        actual, on='row_id', how='inner'
    )
    scored_keys = df.loc[score_mask, ['row_id', 'datetime', 'prediction_unit_id', 'is_consumption']]
    results = {}
    for lag in (48, 168):
        shifted = history.copy()
        # Place each past target on the later timestamp that is allowed to use it.
        shifted['datetime'] = shifted['datetime'] + pd.Timedelta(hours=lag)
        # Two rows at the same series-hour would duplicate the scored rows.
        shifted = shifted.drop_duplicates(
            subset=['datetime', 'prediction_unit_id', 'is_consumption'],
            keep='last',
        )
        pred = scored_keys.merge(
            shifted[['datetime', 'prediction_unit_id', 'is_consumption', 'power']].rename(
                columns={'power': 'pred_power'}
            ),
            on=['datetime', 'prediction_unit_id', 'is_consumption'],
            how='inner',
        )
        # Actual power of the scored hour, not of the lagged hour.
        pred = pred.merge(actual, on='row_id', how='inner')
        # The competition target is non-negative.
        pred['pred_power'] = np.maximum(pred['pred_power'].to_numpy(), 0)
        stats = segment_power_mae(
            pred['power'].to_numpy(),
            pred['pred_power'].to_numpy(),
            pred['is_consumption'].to_numpy(),
        )
        results[f'baseline_lag_{lag}_mae_power'] = stats['mae_power']
        results[f'baseline_lag_{lag}_mae_power_consumption'] = stats['mae_power_consumption']
        results[f'baseline_lag_{lag}_mae_power_production'] = stats['mae_power_production']
        results[f'baseline_lag_{lag}_n'] = stats['n']
    return results


# ---------- CatBoostRegressor Class ---------- #
class CatBoostRegressorModel(BaseEstimator, TransformerMixin):
    '''
        CatBoost regression model for energy prosumer prediction.
        
        Trains separate models for consumption and production data,
        tunes them on month-long rolling folds, and reports MAE on a holdout
        that is excluded from training.
    '''
    
    def __init__(
        self,
        random_state: int = RANDOM_STATE,
        target: str = TARGET, 
        cat_features: List[str] = None,
        n_trials: int = DEFAULT_N_TRIALS,
        cv_splits: int = DEFAULT_CV_SPLITS,
        models_dir: str = MODELS_DIR,
        debug: bool = False,
        scale_trees: bool = True,
    ):
        '''
            Initialize the CatBoostRegressor model.
            
            Args:
                random_state: Random seed for reproducibility
                target: Target column name
                cat_features: List of categorical feature names
                n_trials: Number of Optuna trials for hyperparameter optimization
                cv_splits: Number of month-long rolling folds immediately before the holdout
                models_dir: Directory to save/load models
                debug: Train and predict only DEBUG_PREDICTION_UNITS
                scale_trees: Grow the early-stopping tree count in proportion to the larger train set
        '''
        self.random_state = random_state
        self.target = target
        self.cat_features = list(cat_features or CAT_FEATURES)
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.models_dir = Path(models_dir)
        # False keeps every prediction unit. True limits fit and transform to DEBUG_PREDICTION_UNITS.
        self.debug = debug
        self.prediction_units = list(DEBUG_PREDICTION_UNITS) if debug else None
        # False keeps the tree count early stopping chose, instead of multiplying it by the row ratio.
        self.scale_trees = scale_trees
        self.optuna_timeout = DEFAULT_OPTUNA_TIMEOUT
        
        # Model components
        self.final_model_cb_consumption = None
        self.final_model_cb_production = None
        self.features = None
        
        # Best hyperparameters
        self.best_params_consumption = None
        self.best_params_production = None
        # Holdout metrics. None until fit() has scored the months it did not train on.
        self.eval_report_ = None
        # Clip predicted capacity_factor to [0, upper] before converting to power.
        self.cf_clip_upper_consumption_ = 1.0
        self.cf_clip_upper_production_ = 1.0
        
        # Create models directory
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f'CatBoostRegressorModel initialized')
        logger.info(f'Target: {target}, Categorical features: {len(self.cat_features)}')
        logger.info(f'Debug units: {self.prediction_units}. Scale trees: {self.scale_trees}')

    def _feature_columns(self, df: pd.DataFrame) -> pd.Index:
        '''
            Predictors only. The batch id stays on the frame for the folds and is not a feature.
        '''
        excluded = list(NON_FEATURE_COLUMNS) + [self.target]
        return df.columns.difference(excluded)

    def _pool(self, frame: pd.DataFrame, mask: pd.Series = None) -> Pool:
        '''
            CatBoost pool with capacity as the sample weight.

            |capacity_factor error| * installed_capacity = |power error|.
        '''
        data = frame if mask is None else frame.loc[mask]
        weights = pd.to_numeric(data['installed_capacity'], errors='coerce').to_numpy(dtype=float)
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
        return Pool(
            data[self.features],
            data[self.target],
            cat_features=self.cat_features,
            weight=weights,
        )

    def _clip_to_power(
        self,
        capacity_pred: np.ndarray,
        installed_capacity: np.ndarray,
        is_consumption: np.ndarray,
    ) -> np.ndarray:
        '''
            Clip predicted capacity_factor to [0, segment upper], then convert to power.
        '''
        pred = np.asarray(capacity_pred, dtype=float)
        capacity = np.asarray(installed_capacity, dtype=float)
        consumption = np.asarray(is_consumption).astype(bool)
        upper = np.where(consumption, self.cf_clip_upper_consumption_, self.cf_clip_upper_production_)
        clipped = np.clip(pred, 0, upper)
        return clipped * np.where(np.isfinite(capacity), np.maximum(capacity, 0), 0)

    def _create_study(self, segment: str) -> optuna.Study:
        '''
            Seeded TPE study persisted to SQLite so a later run can continue the same search.
        '''
        storage_path = Path(os.getenv('OPTUNA_STORAGE', self.models_dir / 'optuna_studies.db'))
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f'sqlite:///{storage_path.as_posix()}'
        study_name = f'{segment}_debug{int(self.debug)}_scale{int(self.scale_trees)}'
        return optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state),
            storage=storage,
            study_name=study_name,
            load_if_exists=True,
        )

    def _optimize_study(self, study: optuna.Study, objective) -> None:
        '''
            Run only the remaining trials, and stop early if a time budget is set.
        '''
        n_done = len([
            trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ])
        n_remain = max(0, self.n_trials - n_done)
        if n_remain == 0 and self.optuna_timeout is None:
            logger.info(f'Study {study.study_name} already has {n_done} complete trials')
            return
        kwargs = {
            'show_progress_bar': True,
            'catch': (Exception,),
        }
        if n_remain:
            kwargs['n_trials'] = n_remain
        if self.optuna_timeout is not None:
            kwargs['timeout'] = self.optuna_timeout
        study.optimize(objective, **kwargs)

    def _scaled_iterations(
        self,
        params: dict,
        frame: pd.DataFrame,
        train_mask: pd.Series,
        fit_mask: pd.Series,
        inner_mask: pd.Series,
    ) -> int:
        '''
            Fit on the early-stopping train slice, read the best tree index, scale it.

            The inner slice is the only eval_set. The caller refits without it.
        '''
        if int(fit_mask.sum()) == 0 or int(inner_mask.sum()) == 0:
            raise ValueError('Early-stopping split has an empty side')
        # Copy so the search budget on this trial is not overwritten by the scaled count.
        es_params = dict(params)
        model = CatBoost(**es_params)
        model.fit(
            self._pool(frame, fit_mask),
            # Inner slice only. The month we report is not in this pool.
            eval_set=self._pool(frame, inner_mask),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            # Keep the best inner-slice tree so get_best_iteration is that tree, not the last one.
            use_best_model=True,
            verbose=0,
        )
        return scaled_tree_count(
            model.get_best_iteration(),
            int(fit_mask.sum()),
            int(train_mask.sum()),
            fallback=int(es_params['iterations']),
            scale_trees=self.scale_trees,
        )

    def _objective(
        self,
        trial: optuna.Trial, 
        df: pd.DataFrame, 
        mask: pd.Series
    ) -> float:
        '''
            Mean power MAE over month-long folds.

            Each fold early-stops on the two weeks before the month, then refits
            on everything before the month and scores the month. The month is not an eval_set.
        '''
        # This frame is already the pre-holdout rows of one segment.
        df_subset = df.loc[mask]
        params = suggest_catboost_params(trial, self.random_state)

        fold_stats = []
        for val_start in validation_month_starts(self.cv_splits):
            masks = rolling_origin_masks(df_subset, val_start)
            if int(masks['val'].sum()) == 0 or int(masks['train'].sum()) == 0:
                raise ValueError(f'Fold {val_start.date()} has no train or validation rows')
            n_trees = self._scaled_iterations(
                params, df_subset, masks['train'], masks['fit'], masks['inner']
            )
            refit_params = dict(params)
            refit_params['iterations'] = n_trees
            model = CatBoost(**refit_params)
            model.fit(
                self._pool(df_subset, masks['train']),
                verbose=0,
            )
            val = df_subset.loc[masks['val']]
            capacity_pred = model.predict(val[self.features])
            # Clip in capacity-factor space, then convert to power.
            power_pred = self._clip_to_power(
                capacity_pred,
                val['installed_capacity'].to_numpy(),
                val['is_consumption'].to_numpy(),
            )
            power_true = (val[self.target] * val['installed_capacity']).to_numpy()
            stats = segment_power_mae(power_true, power_pred, val['is_consumption'].to_numpy())
            stats['val_start'] = str(val_start.date())
            fold_stats.append(stats)

        # Stored on the trial so the winning trial's months can be written to the report.
        trial.set_user_attr('folds', fold_stats)
        mean_mae = _weighted_mae(fold_stats)
        logger.info(f'Trial {trial.number + 1}: power MAE = {mean_mae:.6f}')
        return mean_mae

    def _final_iterations(self, params: dict, frame: pd.DataFrame) -> int:
        '''
            Tree count for the model.
            Early-stop on the last 14 pre-holdout days, then scale it up to the full train set.
        '''
        day = block_day(frame)
        # These 14 days sit inside the training period. The holdout has already been removed from `frame`.
        es_start = HOLDOUT_START - pd.Timedelta(days=INNER_ES_DAYS)
        fit_mask = day < es_start
        inner_mask = day >= es_start
        # Scale up from the early-stopping rows to every pre-holdout row of this segment.
        train_mask = pd.Series(True, index=frame.index)
        return self._scaled_iterations(params, frame, train_mask, fit_mask, inner_mask)

    def _fit_segment(self, study: optuna.Study, frame: pd.DataFrame) -> Tuple[CatBoost, dict, list]:
        '''
            Refit one segment on all of its pre-holdout rows, with the scaled tree count.
            No eval_set: there is no scored month left in this frame.
        '''
        params = dict(study.best_params)
        params.update({
            'loss_function': 'MAE',
            'eval_metric': 'MAE',
            'random_seed': self.random_state,
            'logging_level': 'Silent',
        })
        # Replace the searched ceiling with the tree count chosen on the inner slice.
        params['iterations'] = self._final_iterations(params, frame)
        model = CatBoost(**params)
        # No eval_set: there is no scored month left in this frame.
        model.fit(
            self._pool(frame),
            verbose=100,
        )
        return model, params, study.best_trial.user_attrs['folds']

    def fit(self, df: pd.DataFrame, y=None) -> 'CatBoostRegressorModel':
        '''
            Train the consumption and production models.
        '''
        logger.info('Starting model training')

        # Check if the required columns are present.
        required = {'datetime', 'data_block_id', self.target, 'installed_capacity', 'is_consumption', 'row_id'}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f'Missing columns required for day folds and power MAE: {sorted(missing)}')

        # If in the debug mode, limit the training data to the prediction units.
        if self.debug:
            df = limit_to_prediction_units(df, self.prediction_units)

        # Exclude batch id, so the model cannot use the fold key as a feature.
        self.features = self._feature_columns(df)
        self.cat_features = [col for col in self.cat_features if col in self.features]
        logger.info(f'Features: {len(self.features)}')

        # Check if the holdout data is present.
        is_holdout = holdout_mask(df)
        tune_df = df.loc[~is_holdout]
        holdout_df = df.loc[is_holdout]
        if len(tune_df) == 0 or len(holdout_df) == 0:
            raise ValueError(
                f'Need rows on both sides of {HOLDOUT_START.date()}. '
                f'Got {len(tune_df)} train rows and {len(holdout_df)} holdout rows.'
            )
        logger.info(f'Pre-holdout rows: {len(tune_df)}. Holdout rows: {len(holdout_df)}')

        mask_consumption = tune_df['is_consumption'] == 1
        mask_production = tune_df['is_consumption'] == 0
        self.cf_clip_upper_consumption_ = capacity_factor_clip_upper(tune_df.loc[mask_consumption, self.target])
        self.cf_clip_upper_production_ = capacity_factor_clip_upper(tune_df.loc[mask_production, self.target])
        logger.info(
            f'Capacity-factor clip uppers: consumption {self.cf_clip_upper_consumption_:.4f}, '
            f'production {self.cf_clip_upper_production_:.4f}'
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Perform consumption Optuna study for hyperparameters optimization.
        logger.info('Optimizing consumption model')
        study_consumption = self._create_study('consumption')
        self._optimize_study(
            study_consumption,
            lambda trial: self._objective(trial, df=tune_df, mask=mask_consumption),
        )
        logger.info(f'Best consumption power MAE: {study_consumption.best_value:.6f}')
        consumption_frame = tune_df.loc[mask_consumption]
        (
            self.final_model_cb_consumption,
            self.best_params_consumption,
            consumption_folds,
        ) = self._fit_segment(study_consumption, consumption_frame)
        logger.info(f'Best consumption params: {self.best_params_consumption}')

        # Perform production Optuna study for hyperparameters optimization.
        logger.info('Optimizing production model')
        study_production = self._create_study('production')
        self._optimize_study(
            study_production,
            lambda trial: self._objective(trial, df=tune_df, mask=mask_production),
        )
        logger.info(f'Best production power MAE: {study_production.best_value:.6f}')
        production_frame = tune_df.loc[mask_production]
        (
            self.final_model_cb_production,
            self.best_params_production,
            production_folds,
        ) = self._fit_segment(study_production, production_frame)
        logger.info(f'Best production params: {self.best_params_production}')

        # Compute the MAE for the holdout data.
        holdout_stats = self.compute_mae_power(holdout_df)

        # Compute the baselines for the holdout data.
        baselines = baseline_power_mae(df, is_holdout)

        # Create the evaluation report.
        self.eval_report_ = {
            'holdout_start': str(HOLDOUT_START.date()),
            # The saved model was not fit on these rows. The MAE below is not an in-sample score.
            'holdout_excluded_from_training': True,
            'holdout_mae_power': holdout_stats['mae_power'],
            'holdout_mae_power_consumption': holdout_stats['mae_power_consumption'],
            'holdout_mae_power_production': holdout_stats['mae_power_production'],
            'holdout_n': holdout_stats['n'],
            'cv_mae_power_consumption': _weighted_mae(consumption_folds),
            'cv_mae_power_production': _weighted_mae(production_folds),
            'cv_mae_power': _weighted_mae(consumption_folds + production_folds),
            'cv_folds_consumption': consumption_folds,
            'cv_folds_production': production_folds,
            'n_train_samples': int(len(tune_df)),
            'n_trials': self.n_trials,
            'debug': self.debug,
            'debug_prediction_units': self.prediction_units,
            'scale_trees': self.scale_trees,
            'cf_clip_upper_consumption': self.cf_clip_upper_consumption_,
            'cf_clip_upper_production': self.cf_clip_upper_production_,
        }
        self.eval_report_.update(baselines)
        logger.info(f"Holdout power MAE: {self.eval_report_['holdout_mae_power']:.4f}")
        logger.info(f"Baseline lag-48 power MAE: {baselines['baseline_lag_48_mae_power']:.4f}")
        logger.info(f"Baseline lag-168 power MAE: {baselines['baseline_lag_168_mae_power']:.4f}")
        logger.info('DONE: Model training completed')

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        '''
            Predict every row in one batch.
        '''
        if self.final_model_cb_consumption is None or self.final_model_cb_production is None:
            raise RuntimeError("Models not trained. Call `fit` before `transform`.")
        
        if self.features is None:
            raise RuntimeError("Features not defined. Call `fit` or `load` with features first.")

        # Same unit list as training, so a debug model is not asked to score every series.
        if self.debug:
            X = limit_to_prediction_units(X, self.prediction_units)
        
        X_test = X.copy()
        X_test['predictions'] = np.nan
        consumption_mask = X_test['is_consumption'] == 1
        production_mask = X_test['is_consumption'] == 0

        # One predict call per segment. A row-by-row loop is not needed once lags are >= 48h.
        if consumption_mask.any():
            X_test.loc[consumption_mask, 'predictions'] = \
                self.final_model_cb_consumption.predict(X_test.loc[consumption_mask, self.features])
        if production_mask.any():
            X_test.loc[production_mask, 'predictions'] = \
                self.final_model_cb_production.predict(X_test.loc[production_mask, self.features])
        
        # Clip in capacity-factor space, then convert to power.
        X_test['target'] = self._clip_to_power(
            X_test['predictions'].to_numpy(),
            X_test['installed_capacity'].to_numpy(),
            X_test['is_consumption'].to_numpy(),
        )
        
        submission = X_test[['row_id', 'target']]
        logger.info(f'DONE: Generated {len(submission)} predictions')
        return submission
    
    def compute_mae_power(self, df: pd.DataFrame) -> dict:
        '''
            Power MAE of `df` against the saved target, overall and by segment.

            Pass the holdout rows, not the training rows. The return is a dict, not a single in-sample float.
        '''
        power_path = Path(PROCESSED_DATA_DIR) / 'power.parquet'
        if not power_path.exists():
            raise FileNotFoundError(f"Power file not found: {power_path}")
        
        revealed_power = pd.read_parquet(power_path)
        predicted_power = self.transform(df)
        # Inner join: only hours that have both a prediction and a saved target.
        scored = predicted_power.merge(revealed_power[['row_id', 'power']], on='row_id', how='inner')
        # The segment flag lives on the feature frame, not in power.parquet.
        scored = scored.merge(df[['row_id', 'is_consumption']], on='row_id', how='left')
        scored = scored.dropna(subset=['power', 'target'])
        stats = segment_power_mae(
            scored['power'].to_numpy(),
            scored['target'].to_numpy(),
            scored['is_consumption'].to_numpy(),
        )
        logger.info(
            f"Power MAE: {stats['mae_power']:.4f} "
            f"(consumption {stats['mae_power_consumption']}, production {stats['mae_power_production']})"
        )
        return stats

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
            'best_params_production': self.best_params_production,
            # The holdout score travels with the model so a later load does not recompute it on training rows.
            'eval_report': self.eval_report_,
            # Reloading a debug model keeps prediction on the units it was trained on.
            'debug': self.debug,
            'debug_prediction_units': self.prediction_units,
            'scale_trees': self.scale_trees,
            'cf_clip_upper_consumption': self.cf_clip_upper_consumption_,
            'cf_clip_upper_production': self.cf_clip_upper_production_,
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
        # Absent on models saved before the holdout split existed.
        self.eval_report_ = metadata.get('eval_report')
        # A model trained on the debug units must keep that limit when it is loaded for prediction.
        if metadata.get('debug'):
            self.debug = True
            self.prediction_units = metadata.get('debug_prediction_units', list(DEBUG_PREDICTION_UNITS))
        if 'scale_trees' in metadata:
            self.scale_trees = bool(metadata['scale_trees'])
        self.cf_clip_upper_consumption_ = float(metadata.get('cf_clip_upper_consumption', 1.0))
        self.cf_clip_upper_production_ = float(metadata.get('cf_clip_upper_production', 1.0))
        
        logger.info(f'Models loaded from {load_path}')
        return self
    
    def set_features(self, features: Union[List[str], pd.Index]) -> None:
        '''Set features manually (useful when loading without metadata)'''
        self.features = pd.Index(features) if not isinstance(features, pd.Index) else features
        logger.info(f'Features set: {len(self.features)} features')


# ---------- Argument Parser ---------- #
def parse_args(argv: list = None) -> argparse.Namespace:
    '''
        Parse command line arguments for a standalone modelling run.
    '''
    parser = argparse.ArgumentParser(
        description='Train CatBoost consumption and production models'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help=(
            'Train and predict only prediction_unit_id '
            + ', '.join(str(unit) for unit in DEBUG_PREDICTION_UNITS)
        ),
    )
    parser.add_argument(
        '--no-tree-scale',
        action='store_true',
        help='Keep the early-stopping tree count instead of scaling it to the full training set',
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=5,
        help='Number of Optuna trials (default: 5)',
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='catboost_prosumer',
        help='Name for saved model (default: catboost_prosumer)',
    )
    return parser.parse_args() if argv is None else parser.parse_args(argv)


# ---------- Helper Functions ---------- #
def run_modelling(
    train_df: pd.DataFrame,
    n_trials: int = DEFAULT_N_TRIALS,
    save_model: bool = True,
    model_name: str = 'catboost_prosumer',
    debug: bool = False,
    scale_trees: bool = True,
) -> CatBoostRegressorModel:
    '''
        Run the modelling pipeline.
        
        Args:
            train_df: Training DataFrame with features
            n_trials: Number of Optuna trials
            save_model: Whether to save the trained model
            model_name: Name for saved model
            debug: Train and predict only DEBUG_PREDICTION_UNITS
            scale_trees: Scale the early-stopping tree count up to the full training set
            
        Returns:
            Trained CatBoostRegressorModel
    '''
    logger.info('Starting modelling pipeline')
    
    model = CatBoostRegressorModel(n_trials=n_trials, debug=debug, scale_trees=scale_trees)
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

    args = parse_args()
    processed_data = run_preprocessing()
    merged_df = run_merging(processed_data)
    features_df = run_feature_engineering(merged_df)

    model = run_modelling(
        features_df,
        n_trials=args.n_trials,
        debug=args.debug,
        scale_trees=not args.no_tree_scale,
        model_name=args.model_name,
    )
    # Printed number is the holdout, which fit() did not train on.
    print(model.eval_report_)


# ---------- All exports ---------- #
__all__ = [
    'CatBoostRegressorModel',
    'run_modelling',
    'parse_args',
    'TARGET',
    'CAT_FEATURES',
    'LAG_VALUES',
    'DEBUG_PREDICTION_UNITS',
    'suggest_catboost_params',
    'score_functions_for_grow_policy',
]
