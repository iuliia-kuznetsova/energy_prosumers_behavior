'''
    Main Orchestrator for Energy Prosumers Behavior Prediction

    This module provides the main entry point for running the complete
    prediction pipeline from data loading to model training and evaluation.

    Pipeline Steps:
    1. Load environment variables and create directories
    2. Download raw data from Kaggle (if needed)
    3. Preprocess data (schema validation, type conversion)
    4. Merge all datasets into unified DataFrame
    5. Engineer features (datetime, calendar, lags)
    6. Train CatBoost models (consumption & production)
    7. Evaluate model performance
    8. Save trained models

    Usage:
        python -m src.preds.main_preds                    # Full pipeline
        python -m src.preds.main_preds --skip-download    # Skip Kaggle download
        python -m src.preds.main_preds --skip-training    # Skip model training
        python -m src.preds.main_preds --n-trials 20      # Custom Optuna trials
'''

# ---------- Imports ---------- #
import os
import sys
import gc
import argparse
import traceback
from pathlib import Path
from dotenv import load_dotenv

from src.logging_setup import setup_logging
from src.preds.data_loading import run_data_loading, verify_downloaded_files
from src.preds.data_preprocessing import DataPreparation, load_raw_data
from src.preds.data_merging import DataMerging
from src.preds.feature_engineering import FeatureEngineering
from src.preds.modelling_catboost import CatBoostRegressorModel, run_modelling


# ---------- Logging setup ---------- #
logger = setup_logging('main_preds')


# ---------- Config ---------- #
load_dotenv()
PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)


# ---------- Constants ---------- #
RAW_DATA_DIR = os.getenv('RAW_DATA_DIR', './data/raw')
PROCESSED_DATA_DIR = os.getenv('PROCESSED_DATA_DIR', './data/processed')
MODELS_DIR = os.getenv('MODELS_DIR', './models')
RESULTS_DIR = os.getenv('RESULTS_DIR', './results')


# ---------- Memory Helper ---------- #
def log_memory_cleanup(step_name: str) -> None:
    '''Force garbage collection and log cleanup'''
    collected = gc.collect()
    logger.info(f'Memory cleanup after {step_name}: {collected} objects collected')


# ---------- Argument Parser ---------- #
def parse_args() -> argparse.Namespace:
    '''Parse command line arguments'''
    parser = argparse.ArgumentParser(
        description='Energy Prosumers Behavior Prediction Pipeline'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip Kaggle data download if data already exists'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training (load existing model)'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=10,
        help='Number of Optuna hyperparameter optimization trials (default: 10)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='catboost_prosumer',
        help='Name for saved model (default: catboost_prosumer)'
    )
    return parser.parse_args()


# ---------- Pipeline Functions ---------- #
def run_full_pipeline(args: argparse.Namespace) -> None:
    '''
        Run the complete prediction pipeline.
        
        Steps:
        1. Setup directories
        2. Download data (optional)
        3. Preprocess data
        4. Merge datasets
        5. Engineer features
        6. Train models
        7. Evaluate performance
    '''
    
    # ========== STEP 1: Setup ==========
    print('\n' + '=' * 80)
    logger.info('STEP 1: Setting up directories')
    print('=' * 80)
    
    # Create directories
    for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f'Directory ready: {dir_path}')
    
    logger.info('STEP 1 DONE: Setup completed')
    
    # ========== STEP 2: Download Data ==========
    print('\n' + '=' * 80)
    logger.info('STEP 2: Data Loading')
    print('=' * 80)
    
    if args.skip_download:
        logger.info('Skipping download (--skip-download flag)')
        if not verify_downloaded_files(RAW_DATA_DIR):
            logger.error('Required data files not found. Remove --skip-download flag.')
            sys.exit(1)
    else:
        try:
            run_data_loading(skip_if_exists=True)
        except Exception as e:
            logger.error(f'Data loading failed: {e}')
            traceback.print_exc()
            sys.exit(1)
    
    log_memory_cleanup('data_loading')
    logger.info('STEP 2 DONE: Data loading completed')
    
    # ========== STEP 3: Preprocess Data ==========
    print('\n' + '=' * 80)
    logger.info('STEP 3: Data Preprocessing')
    print('=' * 80)
    
    try:
        raw_data = load_raw_data(RAW_DATA_DIR)
        preprocessor = DataPreparation(mode='train')
        processed_data = preprocessor.fit_transform(raw_data)
        
        del raw_data
        log_memory_cleanup('preprocessing')
        logger.info('STEP 3 DONE: Data preprocessing completed')
        
    except Exception as e:
        logger.error(f'Preprocessing failed: {e}')
        traceback.print_exc()
        sys.exit(1)
    
    # ========== STEP 4: Merge Data ==========
    print('\n' + '=' * 80)
    logger.info('STEP 4: Data Merging')
    print('=' * 80)
    
    try:
        merger = DataMerging(mode='train')
        merged_df = merger.fit_transform(processed_data)
        
        del processed_data
        log_memory_cleanup('merging')
        logger.info('STEP 4 DONE: Data merging completed')
        
    except Exception as e:
        logger.error(f'Data merging failed: {e}')
        traceback.print_exc()
        sys.exit(1)
    
    # ========== STEP 5: Feature Engineering ==========
    print('\n' + '=' * 80)
    logger.info('STEP 5: Feature Engineering')
    print('=' * 80)
    
    try:
        engineer = FeatureEngineering(mode='train')
        train_df = engineer.fit_transform(merged_df)
        
        # Save train_df for later use
        train_df.to_parquet(Path(PROCESSED_DATA_DIR) / 'train_features.parquet')
        logger.info(f'Features saved: {train_df.shape}')
        
        del merged_df
        log_memory_cleanup('feature_engineering')
        logger.info('STEP 5 DONE: Feature engineering completed')
        
    except Exception as e:
        logger.error(f'Feature engineering failed: {e}')
        traceback.print_exc()
        sys.exit(1)
    
    # ========== STEP 6: Model Training ==========
    print('\n' + '=' * 80)
    logger.info('STEP 6: Model Training')
    print('=' * 80)
    
    if args.skip_training:
        logger.info('Skipping training (--skip-training flag)')
        try:
            model = CatBoostRegressorModel()
            model.load(model_name=args.model_name)
        except FileNotFoundError:
            logger.error(f'Model not found: {args.model_name}. Remove --skip-training flag.')
            sys.exit(1)
    else:
        try:
            model = CatBoostRegressorModel(n_trials=args.n_trials)
            model.fit(train_df)
            model.save(args.model_name)
            
            log_memory_cleanup('training')
            logger.info('STEP 6 DONE: Model training completed')
            
        except Exception as e:
            logger.error(f'Model training failed: {e}')
            traceback.print_exc()
            sys.exit(1)
    
    # ========== STEP 7: Evaluation ==========
    print('\n' + '=' * 80)
    logger.info('STEP 7: Model Evaluation')
    print('=' * 80)
    
    try:
        mae = model.compute_mae_power(train_df)
        logger.info(f'Train MAE (Power): {mae:.4f}')
        
        # Save evaluation results
        results = {
            'train_mae_power': mae,
            'model_name': args.model_name,
            'n_trials': args.n_trials,
            'n_samples': len(train_df)
        }
        
        import json
        results_path = Path(RESULTS_DIR) / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f'Evaluation results saved to {results_path}')
        
        log_memory_cleanup('evaluation')
        logger.info('STEP 7 DONE: Evaluation completed')
        
    except Exception as e:
        logger.error(f'Evaluation failed: {e}')
        traceback.print_exc()
        sys.exit(1)
    
    # ========== DONE ==========
    print('\n' + '=' * 80)
    logger.info('PIPELINE COMPLETED SUCCESSFULLY!')
    print('=' * 80)
    
    logger.info(f'Model saved to: {MODELS_DIR}/{args.model_name}')
    logger.info(f'Train MAE (Power): {mae:.4f}')
    
    # Final cleanup
    del train_df, model
    gc.collect()


# ---------- Main function ---------- #
def main():
    '''Main entry point'''
    args = parse_args()
    
    logger.info('=' * 80)
    logger.info('Energy Prosumers Behavior Prediction Pipeline')
    logger.info('=' * 80)
    logger.info(f'Arguments: {vars(args)}')
    
    run_full_pipeline(args)


if __name__ == '__main__':
    main()


# ---------- All exports ---------- #
__all__ = ['main', 'run_full_pipeline', 'parse_args']
