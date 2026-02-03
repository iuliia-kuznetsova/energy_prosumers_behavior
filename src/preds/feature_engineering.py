'''
    Feature Engineering

    This module provides functionality to engineer features from preprocessed data.

    Input:
    - data_dir - Directory with preprocessed data
    - results_dir - Output directory for results parquet files

    Output:
    - data_engineered.parquet - Data file after feature engineering

    Usage:
    python -m src.preds.feature_engineering
'''

# ---------- Imports ---------- #
import os
import gc
import polars as pl
from dotenv import load_dotenv
from pathlib import Path

from src.logging_setup import setup_logging


# ---------- Logging setup ---------- #
logger = setup_logging('feature_engineering')


# ---------- Config ---------- #
# Load environment variables
load_dotenv()
# Set working directory to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent  # src/recs -> src -> project_root
os.chdir(PROJECT_ROOT)


# ---------- Constants ---------- #
# Data directory
DATA_DIR = os.getenv('DATA_DIR', './data')
# Results directory
RESULTS_DIR = os.getenv('RESULTS_DIR', './results')
# Preprocessed data file
PREPROCESSED_DATA_FILE = os.getenv('PREPROCESSED_DATA_FILE', 'data_preprocessed.parquet')
# Preprocessed data summary file
PREPROCESSED_DATA_SUMMARY_FILE = os.getenv('PREPROCESSED_DATA_SUMMARY_FILE', 'data_preprocessed_summary.parquet')


# ---------- Functions ---------- #
def engineer_features( 
    data_dir: str = DATA_DIR, 
    results_dir: str = RESULTS_DIR,
    preprocessed_data_file: str = PREPROCESSED_DATA_FILE,
    preprocessed_data_summary_file: str = PREPROCESSED_DATA_SUMMARY_FILE,
) -> pl.DataFrame:
    '''
        Engineer features from preprocessed data.
        
        Strategy:

    '''

    # Load preprocessed data
    df_preprocessed = pl.read_parquet(f"{data_dir}/{preprocessed_data_file}")
    logger.info(f'Preprocessed data loaded from: {data_dir}/{preprocessed_data_file}')
    

    return df_engineered

def run_feature_engineering():
    '''
        Run feature engineering pipeline.
    '''
    logger.info('Starting feature engineering pipeline')
    df = engineer_features(DATA_DIR, RESULTS_DIR, PREPROCESSED_DATA_FILE, PREPROCESSED_DATA_SUMMARY_FILE)
    del df
    gc.collect()
    logger.info('Feature engineering pipeline completed successfully')


# ---------- Main function ---------- #
if __name__ == '__main__':
    run_feature_engineering()


# ---------- All exports ---------- #
__all__ = ['run_feature_engineering', 'engineer_features']