'''
    Data Preprocessing

    This module provides functionality to preprocess raw data 
    into a format suitable for model training.

    Strategy:


    Input:


    Output:


    Usage:
    python -m src.preds.data_preprocessing
'''

# ---------- Imports ---------- #
import os
import gc
import polars as pl
from dotenv import load_dotenv
from pathlib import Path
import json

from src.logging_setup import setup_logging


# ---------- Logging setup ---------- #
logger = setup_logging('data_preprocessing')


# ---------- Config ---------- #
# Load environment variables
load_dotenv()
# Set working directory to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)


# ---------- Constants ---------- #
# Data directory
DATA_DIR = os.getenv('DATA_DIR', './data')
# Results directory
RESULTS_DIR = os.getenv('RESULTS_DIR', './results')
# Raw data file
RAW_DATA_FILE = os.getenv('RAW_DATA_FILE', 'train_ver2.csv')
# Preprocessed data file
PREPROCESSED_DATA_FILE = os.getenv('PREPROCESSED_DATA_FILE', 'data_preprocessed.parquet')
# Preprocessed data summary file
PREPROCESSED_DATA_SUMMARY_FILE = os.getenv('PREPROCESSED_DATA_SUMMARY_FILE', 'data_preprocessed_summary.parquet')



# ---------- Functions ---------- #
def load_data(
    data_dir: str = DATA_DIR,
    results_dir: str = RESULTS_DIR,
    raw_file: str = RAW_DATA_FILE
) -> pl.DataFrame:
    '''
        Load CSV data.
        
        Strategy:

    '''
       
    


    
    return df

def preprocess_data(
    df: pl.DataFrame, 
    data_dir: str = DATA_DIR, 
    results_dir: str = RESULTS_DIR,
    preprocessed_data_file: str = PREPROCESSED_DATA_FILE,
    preprocessed_data_summary_file: str = 'data_preprocessed_summary.parquet'
) -> pl.DataFrame:
    
    
    
    logger.info('DONE: Data preprocessing completed')
    # Save preprocessed data
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    df_preprocessed.write_parquet(f"{data_dir}/{preprocessed_data_file}")
    logger.info(f"Preprocessed data saved to: {data_dir}/{preprocessed_data_file}")

    # Save preprocessed data summary
    if preprocessed_data_summary_file is not None:
        df_preprocessed_summary = df_preprocessed.describe()
        df_preprocessed_summary.write_parquet(f"{results_dir}/{preprocessed_data_summary_file}")
        logger.info(f"Preprocessed data summary saved to: {results_dir}/{preprocessed_data_summary_file}")
   
    del df_preprocessed_summary
    gc.collect()

    return df_preprocessed

def run_preprocessing():
    logger.info('Starting data preprocessing pipeline')
    df_encoded = load_data(DATA_DIR, RESULTS_DIR, RAW_DATA_FILE)
    df = preprocess_data(df_encoded, DATA_DIR, RESULTS_DIR, PREPROCESSED_DATA_FILE, PREPROCESSED_DATA_SUMMARY_FILE)
    del df_encoded, df
    gc.collect()
    logger.info('Data preprocessing pipeline completed successfully')


# ---------- Main function ---------- #
if __name__ == '__main__':
    run_preprocessing()


# ---------- All exports ---------- #
__all__ = ['run_preprocessing', 'load_data', 'preprocess_data']