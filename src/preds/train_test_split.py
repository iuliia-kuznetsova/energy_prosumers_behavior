'''
    Train/Test Split

    This module provides functionality to split data into train and test sets.

    Strategy:

    Usage:
    python -m src.preds.train_test_split
'''

# ---------- Imports ---------- #
import os
import gc
import polars as pl
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Tuple

from src.logging_setup import setup_logging


# ---------- Logging setup ---------- #
logger = setup_logging('train_test_split')


# ---------- Config ---------- #
# Load environment variables
load_dotenv()
# Set working directory to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent  # src/recs -> src -> project_root
os.chdir(PROJECT_ROOT)


# ---------- Constants ---------- #
# Random state
RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))
# Data directory
DATA_DIR = os.getenv('DATA_DIR', './data')
# Results directory
RESULTS_DIR = os.getenv('RESULTS_DIR', './results')
# Preprocessed data file
PREPROCESSED_DATA_FILE = os.getenv('PREPROCESSED_DATA_FILE', 'data_preprocessed.parquet')
# Ids columns
ID_COLS = os.getenv('ID_COLS', ['ncodpers', 'fecha_dato'])
# Sample ratio
SAMPLE_RATIO = float(os.getenv('SAMPLE_RATIO', 0.1))


# ---------- Functions ---------- #
def get_feature_and_target_columns(
    df: pl.DataFrame,
    id_cols: List[str] = ID_COLS
) -> Tuple[List[str], List[str]]:
    '''
        Identify feature columns and target columns from the dataframe.
        Excludes datetime columns (CatBoost can't handle them directly).
    '''

    # Target columns start with 'target_'
    target_cols = [col for col in df.columns if col.startswith('target_')]
    
    # Get datetime column names (Date and Datetime types)
    datetime_cols = [
        col for col in df.columns 
        if df[col].dtype in (pl.Date, pl.Datetime, pl.Time)
    ]
    
    # Feature columns are everything except ids, targets, and datetime columns
    feature_cols = [
        col for col in df.columns 
        if col not in id_cols 
        and col not in target_cols
        and col not in datetime_cols
    ]
    
    if datetime_cols:
        logger.info(f'Datetime features excluded: {datetime_cols}')
    
    return feature_cols, target_cols

def sample_data(
    data_dir: str = DATA_DIR,
    results_dir: str = RESULTS_DIR,
    preprocessed_data_file: str = PREPROCESSED_DATA_FILE,
    seed: int = RANDOM_STATE,
    sample_ratio: float = SAMPLE_RATIO
) -> pl.DataFrame:
    '''
        Sample data from the preprocessed data file.
        
        Strategy:
        - Keep ALL rows with at least one positive target (any product will be acquired)
        - Sample negatives to reach ~10% of total data
    '''

    # Load preprocessed, feature and target-engineered data
    df = pl.read_parquet(f"{data_dir}/{preprocessed_data_file}")
    logger.info(f'Loaded data from: {data_dir}/{preprocessed_data_file}')
    logger.info(f'Data size: {df.height:,} rows x {df.width} columns')

    # Get target columns
    target_cols = [col for col in df.columns if col.startswith('target_')]
    logger.info(f'Number of target columns: {len(target_cols)}')

    # All Positives + Random Negatives Sampling
    # Get all positives : rows with at least one positive target
    positive_mask = df.select(pl.sum_horizontal(target_cols)).to_series() > 0
    positives = df.filter(positive_mask)
    logger.info(f'Sampling: all positive rows {len(positives):,} ({len(positives)/len(df):.1%} of total data)')
    
    # Sample negatives : random rows with negative target
    # Target ~10% of total data, but keep all positives
    frac = max(0, sample_ratio - len(positives)/len(df))
    negatives = df.filter(~positive_mask)
    if frac > 0:
        negatives_sampled = negatives.sample(fraction=frac, seed=seed)
    else:
        # If positives already exceed 10%, don't sample any negatives
        negatives_sampled = pl.DataFrame(schema=negatives.schema)
    logger.info(f'Sampling: random negative rows {len(negatives_sampled):,} ({len(negatives_sampled)/len(df):.1%} of total data)')
    
    # Combine positives and sampled negatives
    df_sampled = pl.concat([positives, negatives_sampled]).sort(['ncodpers', 'fecha_dato'])
    logger.info(f'Sampling: total sampled rows {len(df_sampled):,} ({len(df_sampled)/len(df):.1%} of total data)')
    logger.info(f'DONE: Sampling completed')

    del df, positives, negatives, negatives_sampled
    gc.collect()

    return df_sampled

def temporal_train_test_split(
    df_sampled: pl.DataFrame,
    data_dir: str = DATA_DIR,
    test_months: int = 1
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    '''
        Split data into train and test sets using temporal splitting.
    '''

    # Get unique dates and sort them
    dates = df_sampled.select('fecha_dato').unique().sort('fecha_dato')
    unique_dates = dates['fecha_dato'].to_list()
    logger.info(f'Data date range: {unique_dates[0]} to {unique_dates[-1]} ({len(unique_dates)} months)')
    
    # Define train/test split date
    # Test set: last `test_months` month(s)
    # Train set: all earlier months
    test_start_idx = len(unique_dates) - test_months
    train_dates = unique_dates[:test_start_idx]
    test_dates = unique_dates[test_start_idx:]
    logger.info(f'Train dates: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} months)')
    logger.info(f'Test dates: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} lastmonth)')
    
    # Split data by date
    df_train = df_sampled.filter(pl.col('fecha_dato').is_in(train_dates))
    df_test = df_sampled.filter(pl.col('fecha_dato').is_in(test_dates))
    logger.info(f'Train rows: {df_train.height:,}')
    logger.info(f'Test rows: {df_test.height:,}')
    
    # Get feature and target columns
    feature_cols, target_cols = get_feature_and_target_columns(df_sampled)
    logger.info(f'Number of features: {len(feature_cols)}')
    logger.info(f'Number of targets: {len(target_cols)}')
       
    # Separate features (X) and targets (y)
    X_train = df_train.select(feature_cols)
    X_test = df_test.select(feature_cols)
    y_train = df_train.select(target_cols)
    y_test = df_test.select(target_cols)
    logger.info('DONE: Train/test split completed')

    # Save splits
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    X_train.write_parquet(f"{data_dir}/X_train.parquet")
    X_test.write_parquet(f"{data_dir}/X_test.parquet")
    y_train.write_parquet(f"{data_dir}/y_train.parquet")
    y_test.write_parquet(f"{data_dir}/y_test.parquet")
    logger.info(f'Train/test splits saved to: {data_dir}/')

    del df_train, df_test, X_train, X_test, y_train, y_test
    gc.collect()

    return None

def run_train_test_split():
    '''
        Run train/test split pipeline.
    '''
    logger.info('Starting sampling and temporal train/test split pipeline')
    df_sampled = sample_data(
        data_dir=DATA_DIR,
        results_dir=RESULTS_DIR,
        preprocessed_data_file=PREPROCESSED_DATA_FILE,
        seed=RANDOM_STATE,
        sample_ratio=SAMPLE_RATIO
    )
    temporal_train_test_split(df_sampled, DATA_DIR, test_months=1)
    logger.info('DONE: Sampling and temporal train/test split pipeline completed successfully')


# ---------- Main function ---------- #
if __name__ == '__main__':
    run_train_test_split()


# ---------- All exports ---------- #
__all__ = ['run_train_test_split']