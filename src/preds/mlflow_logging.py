'''
    MLflow logging for OVR recommendation models.

    This module provides functions to log trained OVR models and their metadata to MLflow.
    
    Usage:
    from src.recs.mlflow_logging import log_ovr_model_to_mlflow
    
    # Log a trained model
    run_id = log_ovr_model_to_mlflow(
        model=trained_model,
        X_sample=X_test[:100],
        experiment_name='bank_products_recommendation',
        run_name='my_run',
        metrics=evaluation_results
    )
'''

# ---------- Imports ---------- #
import os
import json
import tempfile
import numpy as np
import pandas as pd
import mlflow
import mlflow.catboost
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from dotenv import load_dotenv
from mlflow.models import infer_signature
import polars as pl
import argparse

from src.logging_setup import setup_logging
from src.recs.modelling_ovr import OvRGroupModel

# ---------- Logging setup ---------- #
logger = setup_logging('mlflow_logging')


# ---------- Config ---------- #
load_dotenv()
PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)


# ---------- Constants ---------- #
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MLFLOW_S3_ENDPOINT_URL = os.getenv('MLFLOW_S3_ENDPOINT_URL')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
RESULTS_DIR = Path(os.getenv('RESULTS_DIR', './results'))


# ----------- Helper Functions ------------ #
def _setup_mlflow_env():
    '''
        Setup MLflow tracking URI and S3 credentials.
    '''
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
    
    # Set S3 credentials if available
    if MLFLOW_S3_ENDPOINT_URL:
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = MLFLOW_S3_ENDPOINT_URL
    if AWS_ACCESS_KEY_ID:
        os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
    if AWS_SECRET_ACCESS_KEY:
        os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
    
    logger.info(f'MLflow tracking URI: {MLFLOW_TRACKING_URI}')


def _sanitize_params(params: Dict[str, Any]) -> Dict[str, str]:
    '''
        Sanitize parameters for MLflow logging.
        Converts nested dicts/lists to JSON strings, handles None values.
    '''
    sanitized = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            sanitized[key] = json.dumps(value)
        else:
            sanitized[key] = str(value)
    return sanitized


def _filter_numeric_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    '''
        Filter metrics to only include numeric values (MLflow requirement).
    '''
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


def _save_artifact_json(data: Any, artifact_name: str, artifact_path: str = 'metadata'):
    '''
        Save data as JSON artifact to MLflow.
    '''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f, indent=2, default=str)
        temp_path = f.name
    mlflow.log_artifact(temp_path, artifact_path)
    os.remove(temp_path)
    logger.info(f'Logged artifact: {artifact_path}/{artifact_name}')


def _save_artifact_csv(df: pd.DataFrame, artifact_name: str, artifact_path: str = 'data'):
    '''
        Save DataFrame as CSV artifact to MLflow.
    '''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
        temp_path = f.name
    mlflow.log_artifact(temp_path, artifact_path)
    os.remove(temp_path)
    logger.info(f'Logged artifact: {artifact_path}/{artifact_name}')


# ----------- Main MLflow Logging Function ------------ #
def log_ovr_model_to_mlflow(
    model,  # OvRGroupModel instance
    X_sample: pd.DataFrame,
    experiment_name: str = 'bank_products_recommendation',
    run_name: Optional[str] = None,
    registry_model_name: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    feature_importance_path: Optional[str] = None
) -> Optional[str]:
    '''
        Log trained OVR model to MLflow with all metadata.
        
        Args:
            model: Trained OvRGroupModel instance
            X_sample: Sample of features for input example and signature inference
            experiment_name: MLflow experiment name
            run_name: Optional name for the MLflow run
            registry_model_name: Optional name for model registry (if None, models won't be registered)
            metrics: Optional dict of evaluation metrics to log
            feature_importance_path: Optional path to feature importance plot
            
        Returns:
            MLflow run ID if successful, None otherwise
    '''
    # Check model has been trained
    if not hasattr(model, 'models') or not model.models:
        logger.warning('No models to log. Call fit() first')
        return None
    
    # Setup MLflow environment
    _setup_mlflow_env()
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    
    # Generate run name if not provided
    run_name = run_name or f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    # End any active run
    if mlflow.active_run():
        mlflow.end_run()
    
    run_id = None
    
    try:
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            logger.info(f'Starting MLflow logging (run_id: {run_id})')
            
            # Prepare input example
            input_example = X_sample[:10] if len(X_sample) >= 10 else X_sample
            
            # Log Parameters
            params = {
                'n_products': len(model.all_products),
                'n_features': len(model.feature_names),
                'n_groups': len(model.models),
                'frequent_threshold': model.frequent_threshold,
                'rare_threshold': model.rare_threshold,
                'random_state': model.random_state,
                'top_k': model.top_k,
                'use_feature_selection': model.use_feature_selection,
                'n_frequent_products': len(model.product_groups.get('frequent', [])),
                'n_mid_products': len(model.product_groups.get('mid', [])),
                'n_rare_products': len(model.product_groups.get('rare', [])),
            }
            
            # Add best hyperparameters per group
            for group_name, group_params in model.best_params.items():
                for param_name, param_value in group_params.items():
                    if param_name not in ['verbose', 'thread_count', 'logging_level']:
                        params[f'{group_name}_hp_{param_name}'] = param_value
            
            # Sanitize and log params
            sanitized_params = _sanitize_params(params)
            mlflow.log_params(sanitized_params)
            logger.info(f'Logged {len(sanitized_params)} parameters')
            
            # Log Metrics
            if metrics:
                # Filter and log numeric metrics
                numeric_metrics = _filter_numeric_metrics(metrics)
                if numeric_metrics:
                    mlflow.log_metrics(numeric_metrics)
                    logger.info(f'Logged {len(numeric_metrics)} metrics')
            
            # Log CV scores if available
            if hasattr(model, 'cv_scores') and model.cv_scores:
                all_cv_aucs = []
                for group_name, scores in model.cv_scores.items():
                    if scores:
                        mlflow.log_metric(f'{group_name}_cv_mean_auc', np.mean(scores))
                        mlflow.log_metric(f'{group_name}_cv_std_auc', np.std(scores))
                        all_cv_aucs.extend(scores)
                
                if all_cv_aucs:
                    mlflow.log_metric('overall_cv_mean_auc', np.mean(all_cv_aucs))
                    mlflow.log_metric('overall_cv_std_auc', np.std(all_cv_aucs))
            
            # Log CatBoost Models
            n_models_logged = 0
            requirements_path = PROJECT_ROOT / 'requirements.txt'
            
            # Check if requirements.txt is valid UTF-8
            pip_requirements = None
            if requirements_path.exists():
                try:
                    with open(requirements_path, 'r', encoding='utf-8') as f:
                        f.read()  # Test if readable as UTF-8
                    pip_requirements = str(requirements_path)
                except UnicodeDecodeError:
                    logger.warning(f'requirements.txt has invalid encoding, skipping pip_requirements')
            
            for group_name, ovr_model in model.models.items():
                products = model.product_groups[group_name]
                
                for product, estimator in zip(products, ovr_model.estimators_):
                    # Clean product name for artifact path
                    product_clean = product.replace('target_', '').replace('.', '_')
                    artifact_name = f'{group_name}_{product_clean}'
                    
                    # Infer signature from model predictions
                    try:
                        predictions = estimator.predict_proba(input_example)[:, 1]
                        signature = infer_signature(input_example, predictions)
                    except Exception:
                        signature = None
                    
                    # Log model with signature and input example
                    mlflow.catboost.log_model(
                        cb_model=estimator,
                        artifact_path=artifact_name,
                        registered_model_name=f'{registry_model_name}_{artifact_name}' if registry_model_name else None,
                        input_example=input_example,
                        signature=signature,
                        pip_requirements=pip_requirements
                    )
                    n_models_logged += 1
                    logger.info(f'  Logged model: {artifact_name}')
            
            logger.info(f'Logged {n_models_logged} CatBoost models')
            
            # Log Metadata Artifacts
            # Thresholds
            _save_artifact_json(model.thresholds, 'thresholds.json', 'metadata')
            
            # Product groups
            _save_artifact_json(model.product_groups, 'product_groups.json', 'metadata')
            
            # Best params
            _save_artifact_json(model.best_params, 'best_params.json', 'metadata')
            
            # Model metadata
            metadata = {
                'all_products': model.all_products,
                'feature_names': model.feature_names,
                'cat_features': model.cat_features,
                'cat_feature_indices': model.cat_feature_indices,
                'selected_features': model.selected_features,
                'cv_scores': model.cv_scores if hasattr(model, 'cv_scores') else {}
            }
            _save_artifact_json(metadata, 'model_metadata.json', 'metadata')
            
            # Log Feature Importance
            if feature_importance_path and os.path.exists(feature_importance_path):
                mlflow.log_artifact(feature_importance_path, artifact_path='plots')
                logger.info(f'Logged feature importance plot')
            
            # Try to generate and log feature importance from model
            try:
                importance_df = model.get_feature_importance(top_n=50)
                _save_artifact_csv(importance_df, 'feature_importance.csv', 'feature_importance')
            except Exception as e:
                logger.warning(f'Could not log feature importance: {e}')
            
            # Log Tags
            mlflow.set_tag('model_type', 'OvR_CatBoost')
            mlflow.set_tag('n_groups', len(model.models))
            mlflow.set_tag('n_total_models', n_models_logged)
            
            # Save Model Info Locally
            model_info = {
                'run_id': run_id,
                'experiment_id': experiment_id,
                'experiment_name': experiment_name,
                'run_name': run_name,
                'registry_model_name': registry_model_name,
                'model_uri': f'runs:/{run_id}/model',
                'n_models': n_models_logged,
                'timestamp': datetime.now().isoformat()
            }
            
            if metrics:
                model_info['metrics'] = metrics
            
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            info_file = RESULTS_DIR / f'mlflow_model_info_{run_name}.json'
            with info_file.open('w') as f:
                json.dump(model_info, f, indent=2)
            logger.info(f'Saved model info to: {info_file}')
            
            logger.info(f'DONE: Model logged to MLflow (run_id: {run_id})')
    
    except Exception as e:
        logger.error(f'Error logging to MLflow: {e}')
        import traceback
        traceback.print_exc()
        return None
    
    # End run explicitly
    mlflow.end_run()
    
    return run_id


# ----------- Convenience Function for Loading and Logging ------------ #
def log_saved_model_to_mlflow(
    model_path: str,
    X_sample_path: str,
    experiment_name: str = 'bank_products_recommendation',
    run_name: Optional[str] = None,
    registry_model_name: Optional[str] = None
) -> Optional[str]:
    '''
        Load a saved OvRGroupModel and log it to MLflow.
        
        Args:
            model_path: Path to saved model directory
            X_sample_path: Path to sample features parquet file (e.g., X_test.parquet)
            experiment_name: MLflow experiment name
            run_name: Optional run name
            registry_model_name: Optional model registry name
            
        Returns:
            MLflow run ID if successful, None otherwise
    '''
    
    # Load model
    model = OvRGroupModel()
    model.load(model_path)
    logger.info(f'Loaded model from: {model_path}')
    
    # Load sample data
    X_sample = pl.read_parquet(X_sample_path).to_pandas()
    
    # Apply feature selection if model uses it
    if model.use_feature_selection and model.selected_features:
        X_sample = X_sample[model.selected_features]
    
    logger.info(f'Loaded sample data: {X_sample.shape}')
    
    # Log to MLflow
    return log_ovr_model_to_mlflow(
        model=model,
        X_sample=X_sample,
        experiment_name=experiment_name,
        run_name=run_name,
        registry_model_name=registry_model_name
    )


# ---------- CLI Entry Point ---------- #
if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='Log OvR model to MLflow')
    parser.add_argument('--model-path', type=str, default='models/ovr_grouped_catboost',
                        help='Path to saved model directory')
    parser.add_argument('--data-path', type=str, default='data/X_test.parquet',
                        help='Path to sample features parquet')
    parser.add_argument('--experiment', type=str, default='bank_products_recommendation',
                        help='MLflow experiment name')
    parser.add_argument('--run-name', type=str, default=None,
                        help='MLflow run name')
    parser.add_argument('--register', type=str, default=None,
                        help='Model registry name (optional)')
    
    args = parser.parse_args()
    
    run_id = log_saved_model_to_mlflow(
        model_path=args.model_path,
        X_sample_path=args.data_path,
        experiment_name=args.experiment,
        run_name=args.run_name,
        registry_model_name=args.register
    )
    
    if run_id:
        print(f'Successfully logged model to MLflow (run_id: {run_id})')
    else:
        print('Failed to log model to MLflow')


# ---------- All exports ---------- #
__all__ = ['log_ovr_model_to_mlflow', 'log_saved_model_to_mlflow']