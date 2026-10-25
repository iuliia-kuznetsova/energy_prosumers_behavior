'''
    Feature Engineering

    This module provides the FeatureEngineering class for creating features
    from merged data, including temporal, calendar, and lag features.

    Strategy:
    - Add cyclical datetime features (hour, day, month encoded as sin/cos)
    - Add calendar features (holidays, day before holiday, DST shifts)
    - Create time-based lag features (target lags start at 48h)
    - Handle missing values appropriately

    Input:
    - Merged Polars DataFrame from DataMerging

    Output:
    - Pandas DataFrame with all features ready for model training

    Usage:
    python -m src.preds.feature_engineering
'''

# ---------- Imports ---------- #
import os
import gc
import copy
import numpy as np
import polars as pl
import pandas as pd
import holidays
from pathlib import Path
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv
from sklearn.base import BaseEstimator, TransformerMixin

from src.logging_setup import setup_logging


# ---------- Logging setup ---------- #
logger = setup_logging('feature_engineering')


# ---------- Config ---------- #
load_dotenv()
PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)


# ---------- Constants ---------- #
PROCESSED_DATA_DIR = os.getenv('PROCESSED_DATA_DIR', './data/processed')

# Targets are revealed 48h later, so nothing shorter than 48h is knowable at forecast time.
MIN_TARGET_LAG_HOURS = 48
# 48h is two days back; 336h is two weeks. 12h and 24h were removed because they are not revealed yet.
TARGET_LAG_HOURS = [48, 72, 96, 120, 144, 168, 336]
# Kept as the name older imports used. It is no longer [12, 24, 168, 720].
DEFAULT_LAG_PERIODS = TARGET_LAG_HOURS
# Seven revealed same-hours: the target day minus 2 days, back through minus 8 days.
SAME_HOUR_LAG_HOURS = [48 + 24 * day for day in range(7)]

# Prices and observed weather are already shifted onto the latest legal timestamp in data_merging.
# These extra steps go further back from that legal value, so they are legal too.
COVARIATE_LAG_HOURS = [24, 168]
COVARIATE_LAG_COLUMNS = [
    'electricity_price', 'gas_price',
    'hw_temperature', 'hw_surface_pressure',
    'hw_cloudcover_total', 'hw_total_precipitation',
    'hw_windspeed_10m', 'hw_winddirection_10m',
    'hw_direct_solar_radiation', 'hw_diffuse_radiation',
]
# Old name. capacity_factor is lagged on its own, starting at 48h, so it is not in this list.
LAG_COLUMNS = COVARIATE_LAG_COLUMNS

# Written by data_merging so a test can check the source batch. Not model inputs.
AUDIT_BLOCK_COLUMNS = [
    'client_block_id', 'electricity_block_id', 'gas_block_id', 'fw_block_id', 'hw_block_id',
]


# ---------- FeatureEngineering Class ---------- #
class FeatureEngineering(BaseEstimator, TransformerMixin):
    '''
        Generate features from merged data including temporal, calendar,
        and lag features.
        
        Features created:
        - Cyclical datetime encoding (hour, day, month) plus months since 2021-09
        - Estonian holidays and day-before-holiday flags
        - DST transition flags
        - Lag features for capacity_factor (48h and older), prices, and observed weather
    '''

    def __init__(
        self, 
        mode: str = 'train', 
        n_lag: List[int] = None, 
        train_df: pd.DataFrame = None, 
        retrain_df: pd.DataFrame = None
    ):
        '''
            Initialize the FeatureEngineering transformer.
            
            Args:
                mode: One of 'train', 'retrain', or 'test'
                n_lag: Target lag hours. Each one must be >= 48. Default: 48, 72, 96, 120, 144, 168, 336.
                train_df: Training DataFrame for retrain/test modes (for lag calculation)
                retrain_df: Retrain DataFrame for test mode (for lag calculation)
        '''
        if mode not in ['train', 'retrain', 'test']:
            raise ValueError(f"Mode must be 'train', 'retrain', or 'test', got: {mode}")
        
        self.mode = mode
        # A 12h or 24h target lag would be the target itself, which is not revealed until 48h later.
        if n_lag is not None and any(lag < MIN_TARGET_LAG_HOURS for lag in n_lag):
            raise ValueError(
                f'Target lags must be >= {MIN_TARGET_LAG_HOURS}h because revealed targets are 2 days old. Got: {n_lag}'
            )
        self.target_lags = list(n_lag) if n_lag is not None else list(TARGET_LAG_HOURS)
        # Alias used by the log line and by callers that still read n_lag.
        self.n_lag = self.target_lags
        self.min_test_datetime = None
        self.min_retrain_datetime = None
        
        # Convert pandas DataFrames to Polars if provided
        self.train_df_pl = pl.from_pandas(train_df) if train_df is not None else None
        self.retrain_df_pl = pl.from_pandas(retrain_df) if retrain_df is not None else None
        
        # Estonian holidays (2021-2026)
        self.estonian_holidays = list(
            holidays.country_holidays("EE", years=range(2021, 2027)).keys()
        )
        self.day_before_holidays = (
            pl.Series("day_before_holidays", self.estonian_holidays)
            .cast(pl.Date) 
            - pl.duration(days=1)
        )
        
        # DST shift dates for Estonia (2021-2026)
        self.march_dsts = (
            pl.Series("march_dsts", [
                "2021-03-28 03:00:00",
                "2022-03-27 03:00:00",
                "2023-03-26 03:00:00",
                "2024-03-31 03:00:00",
                "2025-03-30 03:00:00",
                "2026-03-29 03:00:00"
            ])
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
            .cast(pl.Datetime("us"))
        )

        self.october_dsts = (
            pl.Series("october_dsts", [
                "2021-10-31 03:00:00",
                "2022-10-30 03:00:00",
                "2023-10-29 03:00:00",
                "2024-10-27 03:00:00",
                "2025-10-26 03:00:00",
                "2026-10-25 03:00:00"
            ])
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
            .cast(pl.Datetime("us"))
        )
        
        logger.info(f'FeatureEngineering initialized with mode: {mode}, lags: {self.n_lag}')

    def _is_holiday(self, date: pl.Series) -> pl.Series:
        '''Check if date is an Estonian holiday'''
        return date.is_in(self.estonian_holidays)
    
    def _is_day_before_holiday(self, date: pl.Series) -> pl.Series:
        '''Check if date is day before an Estonian holiday'''
        return date.is_in(self.day_before_holidays)
    
    def _is_spring_forward(self, datetime_col: pl.Series) -> pl.Series:
        '''Check if datetime is during spring-forward DST shift'''
        return datetime_col.is_in(self.march_dsts)

    def _is_fall_backward(self, datetime_col: pl.Series) -> pl.Series:
        '''Check if datetime is during fall-backward DST shift'''
        return datetime_col.is_in(self.october_dsts)

    def _add_datetime_features(self, df: pl.DataFrame) -> pl.DataFrame:
        '''Add cyclical datetime and calendar features'''

        df = (
            df
                # Add datetime components
                .with_columns([
                    pl.col('datetime').dt.year().alias('year'),
                    pl.col('datetime').dt.month().alias('month'),
                    pl.col('datetime').dt.day().alias('day'),
                    pl.col('datetime').dt.weekday().alias('day_of_week'),
                    pl.col('datetime').dt.hour().alias('hour'),
                    # Calendar features
                    pl.col('datetime').dt.date().map_batches(self._is_holiday).alias('is_holiday'),
                    pl.col('datetime').dt.date().map_batches(self._is_day_before_holiday).alias('is_day_before_holiday'),
                    pl.col('datetime').map_batches(self._is_spring_forward).alias('is_dst_spring_forward'),
                    pl.col('datetime').map_batches(self._is_fall_backward).alias('is_dst_fall_backward')
                ])
                # Add cyclical encodings
                .with_columns([
                    (2 * np.pi * pl.col('hour') / 24).sin().alias('hour_sin'),
                    (2 * np.pi * pl.col('hour') / 24).cos().alias('hour_cos'),
                    (2 * np.pi * pl.col('day') / 31).sin().alias('day_sin'),
                    (2 * np.pi * pl.col('day') / 31).cos().alias('day_cos'),
                    (2 * np.pi * pl.col('day_of_week') / 7).sin().alias('day_of_week_sin'),
                    (2 * np.pi * pl.col('day_of_week') / 7).cos().alias('day_of_week_cos'),
                    (2 * np.pi * pl.col('month') / 12).sin().alias('month_sin'),
                    (2 * np.pi * pl.col('month') / 12).cos().alias('month_cos'),
                    ((pl.col('year') - 2021) * 12 + pl.col('month') - 9).alias('months_since_start'),
                ])
                # Drop raw datetime components. Year is a numeric trend, not a cycle.
                .drop(['hour', 'day', 'day_of_week', 'month', 'year'])
        )

        logger.info('Datetime features added')
        return df
    
    def _merge_historical_data(self, df: pl.DataFrame) -> pl.DataFrame:
        '''Merge with historical data for lag feature calculation'''
        
        # Store min datetime for filtering later
        self.min_retrain_datetime = df['datetime'].min() if self.mode == 'retrain' else None
        self.min_test_datetime = df['datetime'].min() if self.mode == 'test' else None

        # For retraining, merge with training data
        if self.mode == 'retrain' and self.train_df_pl is not None:
            missing_columns = set(self.train_df_pl.columns) - set(df.columns)
            df = df.with_columns([pl.lit(None).alias(col) for col in missing_columns])
            df = df.select(self.train_df_pl.columns)
            
            df = (
                pl.concat([self.train_df_pl, df], how='vertical')
                    .unique(
                        subset=['datetime', 'prediction_unit_id', 'is_consumption',  
                                'county', 'is_business', 'product_type'], 
                        keep='last'
                    )
                    .sort('datetime', 'prediction_unit_id', 'is_consumption', 'product_type')
            )
            logger.info('Merged with training data for retrain mode')

        # For testing, merge with both training and retrain data
        if self.mode == 'test':
            if self.train_df_pl is not None and self.retrain_df_pl is not None:
                merged_train = pl.concat([self.train_df_pl, self.retrain_df_pl], how='vertical')
            elif self.train_df_pl is not None:
                merged_train = self.train_df_pl
            elif self.retrain_df_pl is not None:
                merged_train = self.retrain_df_pl
            else:
                merged_train = None
            
            if merged_train is not None:
                missing_columns = set(self.train_df_pl.columns) - set(df.columns)
                df = df.with_columns([pl.lit(None).alias(col) for col in missing_columns])
                df = df.select(self.train_df_pl.columns)
                
                df = (
                    pl.concat([merged_train, df], how='vertical')
                        .unique(
                            subset=['datetime', 'prediction_unit_id', 'is_consumption',  
                                    'county', 'is_business', 'product_type'], 
                            keep='last'
                        )
                        .sort('datetime', 'prediction_unit_id', 'is_consumption', 'product_type')
                )
            logger.info('Merged with training and retrain data for test mode')
        
        return df
    
    def _handle_missing_values(self, df: pl.DataFrame) -> pl.DataFrame:
        '''Handle missing values in the data'''

        # Forward-fill needs chronological order inside each series.
        df = df.sort(['prediction_unit_id', 'is_consumption', 'datetime'])

        if self.mode != 'test':
            # Fill missing power with 0 during DST transitions
            df = df.with_columns(
                pl.when(
                    ((pl.col('is_dst_spring_forward') == True) | 
                     (pl.col('is_dst_fall_backward') == True)) & 
                    (pl.col('power').is_null())
                )
                .then(0)
                .otherwise(pl.col('power'))
                .alias('power')
            )

        if self.mode == 'train':
            df = (
                df
                    # Drop rows with missing power
                    .drop_nulls(['power'])
                    # Filter out problematic date range
                    .filter(~(pl.col('datetime') 
                            .is_between(pl.datetime(2021, 9, 1, 0, 0, 0), 
                                       pl.datetime(2021, 9, 2, 23, 59, 59))))
            )
            logger.info('Dropped rows with missing power in train mode')

        fill_candidates = [
            col for col in df.columns
            if col in ('electricity_price', 'gas_price', 'installed_capacity', 'eic_count')
            or (
                col.startswith(('fw_', 'hw_'))
                and not col.endswith('_is_missing')
                and not col.endswith('_block_id')
            )
        ]
        for col in fill_candidates:
            flag = f'{col}_is_missing'
            if flag not in df.columns:
                df = df.with_columns(pl.col(col).is_null().alias(flag))

        n = df.height
        for col in fill_candidates:
            nulls = df[col].null_count()
            if nulls:
                logger.info(f'feature missing {col}: {nulls}/{n} ({100.0 * nulls / n:.2f}%)')

        series_fill = [
            col for col in fill_candidates
            if col not in ('electricity_price', 'gas_price')
        ]
        fill_exprs = []
        if 'electricity_price' in df.columns:
            fill_exprs.append(pl.col('electricity_price').fill_null(strategy='forward'))
        if 'gas_price' in df.columns:
            fill_exprs.append(pl.col('gas_price').fill_null(strategy='forward'))
        fill_exprs.extend([
            pl.col(col)
                .fill_null(strategy='forward')
                .over(partition_by=(pl.col('prediction_unit_id'), pl.col('is_consumption')))
            for col in series_fill
        ])
        if fill_exprs:
            df = df.with_columns(fill_exprs)

        if self.mode != 'test':
            # Recompute capacity_factor only when capacity is positive and finite.
            df = df.with_columns(
                pl.when(
                    pl.col('capacity_factor').is_null()
                    & (pl.col('installed_capacity') > 0)
                    & pl.col('power').is_finite()
                )
                .then(pl.col('power') / pl.col('installed_capacity'))
                .otherwise(pl.col('capacity_factor'))
                .alias('capacity_factor')
            )
            df = df.with_columns(
                pl.when(pl.col('capacity_factor').is_finite())
                .then(pl.col('capacity_factor'))
                .otherwise(None)
                .alias('capacity_factor')
            )

        logger.info('Missing values handled')
        return df

    def _create_lag_features(self, df: pl.DataFrame) -> pl.DataFrame:
        '''
            Create lag features by joining each series to itself at an exact timestamp.

            shift(n) counts rows, so a missing hour would relabel every later lag.
            Joining on datetime - n hours keeps the lag equal to n hours across gaps and DST.
        '''
        series_keys = ['prediction_unit_id', 'is_consumption', 'datetime']
        # One row per series-hour. A duplicate hour would multiply the join.
        df = df.unique(subset=series_keys, keep='last')

        # The concatenated train/test frame may already carry lag columns from an earlier run.
        # Drop them and rebuild, so a non-null current value cannot be copied into its own lag.
        stale_lag_columns = [
            col for col in df.columns
            if '_lag_' in col or col.startswith('capacity_factor_same_hour_')
        ]
        if stale_lag_columns:
            df = df.drop(stale_lag_columns)

        # Hours to fetch, and which source columns are read at each hour.
        columns_by_lag = {}
        for lag in self.target_lags:
            columns_by_lag.setdefault(lag, []).append('capacity_factor')
        # 192h is the 7th revealed same-hour. It is only an input to the rolling stats below.
        for lag in SAME_HOUR_LAG_HOURS:
            columns_by_lag.setdefault(lag, [])
            if 'capacity_factor' not in columns_by_lag[lag]:
                columns_by_lag[lag].append('capacity_factor')
        for lag in COVARIATE_LAG_HOURS:
            columns_by_lag.setdefault(lag, []).extend(COVARIATE_LAG_COLUMNS)

        # Read lags from this frame, not from the frame we are adding columns to.
        base = df
        for lag, columns in sorted(columns_by_lag.items()):
            lagged = (
                base
                    .select(series_keys + columns)
                    # A source row at T-lag is moved onto T, which is the row allowed to see it.
                    .with_columns(
                        (pl.col('datetime') + pl.duration(hours=lag)).alias('datetime')
                    )
                    .rename({col: f'{col}_lag_{lag}' for col in columns})
            )
            df = df.join(lagged, on=series_keys, how='left')

        same_hour_columns = [f'capacity_factor_lag_{lag}' for lag in SAME_HOUR_LAG_HOURS]
        df = df.with_columns([
            # Mean of the same clock hour over the last 7 days whose target is already known.
            pl.mean_horizontal(same_hour_columns).alias('capacity_factor_same_hour_mean_7d'),
            # Spread of those same 7 hours. Nulls inside the list would make std null, so drop them first.
            pl.concat_list(same_hour_columns).list.drop_nulls().list.std()
                .alias('capacity_factor_same_hour_std_7d'),
        ])
        # Do not keep a lag that was built only to feed the 7-day stats.
        temporary_lags = [
            f'capacity_factor_lag_{lag}'
            for lag in SAME_HOUR_LAG_HOURS
            if lag not in self.target_lags
        ]
        if temporary_lags:
            df = df.drop(temporary_lags)

        logger.info(
            f'Lag features created: target lags {self.target_lags}, covariate lags {COVARIATE_LAG_HOURS}'
        )
        return df
    
    def _drop_rows(self, df: pl.DataFrame) -> pl.DataFrame:
        '''Drop rows after lag feature creation'''

        if self.mode == 'retrain' and self.min_retrain_datetime is not None:
            df = df.filter(pl.col('datetime') >= self.min_retrain_datetime)

        if self.mode == 'test' and self.min_test_datetime is not None:
            df = df.filter(pl.col('datetime') >= self.min_test_datetime)

        # Keep rows whose target is known. CatBoost can train with NaN lags.
        if self.mode != 'test':
            df = df.filter(
                pl.col('capacity_factor').is_not_null() & pl.col('capacity_factor').is_finite()
            )
        
        logger.info(f'Rows filtered. Current shape: {df.shape}')
        return df
    
    def _save_power_and_drop(self, df: pl.DataFrame) -> pl.DataFrame:
        '''Save power column and drop it from features'''

        if self.mode != 'test':
            # Save power for later evaluation
            processed_dir = Path(PROCESSED_DATA_DIR)
            processed_dir.mkdir(parents=True, exist_ok=True)
            df.select(['row_id', 'power']).write_parquet(processed_dir / 'power.parquet')
            logger.info(f'Power saved to {processed_dir}/power.parquet')
        
        df = df.drop('power')
        return df
    
    def _convert_to_pandas(self, df: pl.DataFrame) -> pd.DataFrame:
        '''Convert to Pandas DataFrame for CatBoost compatibility'''
        # Source batch ids are for the leakage check. They are not predictors.
        audit_columns = [col for col in AUDIT_BLOCK_COLUMNS if col in df.columns]
        if audit_columns:
            df = df.drop(audit_columns)
        return df.to_pandas()
    
    def fit(self, X: pl.DataFrame, y=None) -> 'FeatureEngineering':
        '''Fit method (required for sklearn pipeline compatibility)'''
        return self

    def transform(self, X: pl.DataFrame) -> pd.DataFrame:
        '''Apply feature engineering pipeline'''

        if self.mode == 'train':
            df = self._add_datetime_features(X)
            df = self._handle_missing_values(df)
            df = self._create_lag_features(df)
            df = self._drop_rows(df)
            df = self._save_power_and_drop(df)
            df = self._convert_to_pandas(df)
        else:
            df = self._add_datetime_features(X)
            df = self._merge_historical_data(df)
            df = self._handle_missing_values(df)
            df = self._create_lag_features(df)
            df = self._drop_rows(df)
            df = self._save_power_and_drop(df)
            df = self._convert_to_pandas(df)

        logger.info(f'DONE: Feature engineering completed. Final shape: {df.shape}')
        return df

    def fit_transform(self, X: pl.DataFrame, y=None) -> pd.DataFrame:
        '''Fit and transform in one step'''
        self.fit(X)
        return self.transform(X)


# ---------- Helper Functions ---------- #
def run_feature_engineering(
    merged_df: pl.DataFrame,
    mode: str = 'train',
    train_df: pd.DataFrame = None,
    retrain_df: pd.DataFrame = None
) -> pd.DataFrame:
    '''
        Run feature engineering pipeline.
        
        Args:
            merged_df: Merged DataFrame from DataMerging
            mode: Processing mode ('train', 'retrain', 'test')
            train_df: Training DataFrame for retrain/test modes
            retrain_df: Retrain DataFrame for test mode
            
        Returns:
            Pandas DataFrame with engineered features
    '''
    logger.info('Starting feature engineering pipeline')
    
    engineer = FeatureEngineering(
        mode=mode,
        train_df=train_df,
        retrain_df=retrain_df
    )
    features_df = engineer.fit_transform(merged_df)
    
    logger.info('Feature engineering pipeline completed successfully')
    
    gc.collect()
    return features_df


# ---------- Main function ---------- #
if __name__ == '__main__':
    # For standalone testing
    from src.preds.data_preprocessing import run_preprocessing
    from src.preds.data_merging import run_merging
    
    processed_data = run_preprocessing()
    merged_df = run_merging(processed_data)
    features_df = run_feature_engineering(merged_df)
    print(f'Features DataFrame shape: {features_df.shape}')
    print(f'Features columns: {list(features_df.columns)[:20]}...')


# ---------- All exports ---------- #
__all__ = [
    'FeatureEngineering',
    'run_feature_engineering',
    'LAG_COLUMNS',
    'DEFAULT_LAG_PERIODS',
    'TARGET_LAG_HOURS',
    'COVARIATE_LAG_HOURS',
    'AUDIT_BLOCK_COLUMNS',
]
