'''
    MLflow logging for the CatBoost energy prosumer models.

    Sends the parameters, holdout / CV metrics and both trained CatBoost models
    (consumption and production) of one training run to an MLflow tracking server.

    The server is expected to run locally (SQLite backend, artifacts on local disk):
        .\\start_mlflow_server.ps1        # PowerShell
        bash start_mlflow_server.sh       # Git Bash / WSL / Linux

    Usage:
    from src.preds.mlflow_logging import log_catboost_model_to_mlflow
    run_id = log_catboost_model_to_mlflow(model=trained_model, X_sample=train_df)

    CLI (log an already saved model):
    python -m src.preds.mlflow_logging --model-name catboost_prosumer
'''

# ---------- Imports ---------- #
import os
import sys
import tempfile
import argparse
import numpy as np
import pandas as pd
import requests
import mlflow
import mlflow.catboost
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from src.logging_setup import setup_logging
from src.preds.modelling_catboost import CatBoostRegressorModel


# ---------- Logging setup ---------- #
logger = setup_logging('mlflow_logging')


# ---------- Config ---------- #
load_dotenv()
PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)


# ---------- Constants ---------- #
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'energy_prosumers_catboost')
PROCESSED_DATA_DIR = os.getenv('PROCESSED_DATA_DIR', './data/processed')
# Rows kept as the input example stored next to each model.
INPUT_EXAMPLE_ROWS = 5


# ----------- Helper Functions ------------ #
def setup_mlflow(tracking_uri: str = MLFLOW_TRACKING_URI, timeout: float = 5.0) -> bool:
    '''
        Point the MLflow client at the tracking server and check that it answers.

        Returns False instead of raising an error, so a stopped server does not fail a finished training run.
    '''
    mlflow.set_tracking_uri(tracking_uri)

    # MLflow prints an emoji when a run ends. A cp1252 Windows console cannot encode it, and the
    # resulting UnicodeEncodeError leaves the run stuck in RUNNING.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(errors='replace')

    # Only HTTP servers have a health endpoint. A file or sqlite URI is used directly by the client.
    if not tracking_uri.startswith(('http://', 'https://')):
        logger.info(f'MLflow tracking URI: {tracking_uri}')
        return True
    try:
        response = requests.get(f"{tracking_uri.rstrip('/')}/health", timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning(
            f'MLflow server is not reachable at {tracking_uri} ({e}). '
            f'Start it with .\\start_mlflow_server.ps1 or bash start_mlflow_server.sh'
        )
        return False
    logger.info(f'Connected to MLflow server: {tracking_uri}')
    return True


def _numeric_metrics(report: Dict[str, Any]) -> Dict[str, float]:
    '''
        Finite numbers only. bool is an int subclass, so it is excluded explicitly.
    '''
    metrics = {}
    for key, value in report.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            continue
        if np.isfinite(value):
            metrics[key] = float(value)
    return metrics


def _segment_params(prefix: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    '''
        Prefix each CatBoost hyperparameter with its segment, skipping logging switches.
    '''
    if not params:
        return {}
    return {
        f'{prefix}_{name}': value
        for name, value in params.items()
        if name not in ('logging_level', 'verbose')
    }


def _log_segment_model(
    cb_model,
    name: str,
    registered_model_name: Optional[str],
) -> None:
    '''
        Log one CatBoost model.

        The .cbm file is always stored as an artifact. The CatBoost flavor is logged
        without an input example: pandas 3 cannot compare a RangeIndex, and MLflow's
        signature path raises `eq not implemented for <class 'range'>`.
    '''
    with tempfile.TemporaryDirectory() as tmp:
        cbm_path = Path(tmp) / f'{name}.cbm'
        cb_model.save_model(str(cbm_path))
        mlflow.log_artifact(str(cbm_path), artifact_path='models')

    mlflow.catboost.log_model(
        cb_model=cb_model,
        name=name,
        registered_model_name=f'{registered_model_name}_{name}' if registered_model_name else None,
    )

    importance = pd.DataFrame({
        'feature': list(cb_model.feature_names_),
        'importance': cb_model.get_feature_importance(),
    }).sort_values('importance', ascending=False)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        importance.to_csv(f, index=False)
        importance_path = f.name
    mlflow.log_artifact(importance_path, artifact_path='feature_importance')
    os.remove(importance_path)
    logger.info(f'Logged model: {name}')


# ----------- Main MLflow Logging Function ------------ #
def log_catboost_model_to_mlflow(
    model: CatBoostRegressorModel,
    X_sample: pd.DataFrame,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    run_name: Optional[str] = None,
    registered_model_name: Optional[str] = None,
) -> Optional[str]:
    '''
        Log a trained CatBoostRegressorModel to MLflow.

        Args:
            model: Trained (or loaded) CatBoostRegressorModel
            X_sample: Feature frame with both segments; only a few rows are stored as input examples
            experiment_name: MLflow experiment name
            run_name: Optional MLflow run name
            registered_model_name: If set, both models are registered as <name>_consumption / <name>_production

        Returns:
            MLflow run ID, or None if the server is unreachable or logging failed
    '''
    if model.final_model_cb_consumption is None or model.final_model_cb_production is None:
        raise RuntimeError('Models not trained. Call `fit` or `load` before logging.')

    if not setup_mlflow():
        return None

    mlflow.set_experiment(experiment_name)
    run_name = run_name or f'catboost_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    report = model.eval_report_ or {}

    # A leftover run from an earlier failure would swallow this one.
    if mlflow.active_run():
        mlflow.end_run()

    try:
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            logger.info(f'Started MLflow run {run_id} in experiment "{experiment_name}"')

            # ---- Parameters ---- #
            params = {
                'target': model.target,
                'random_state': model.random_state,
                # A loaded model has the default n_trials; the report keeps the value it was trained with.
                'n_trials': report.get('n_trials', model.n_trials),
                'cv_splits': model.cv_splits,
                'debug': model.debug,
                'scale_trees': model.scale_trees,
                'n_features': len(model.features),
                'holdout_start': report.get('holdout_start'),
            }
            params.update(_segment_params('consumption', model.best_params_consumption))
            params.update(_segment_params('production', model.best_params_production))
            mlflow.log_params({k: str(v) for k, v in params.items() if v is not None})

            # ---- Metrics ---- #
            mlflow.log_metrics(_numeric_metrics(report))
            # One step per CV month, so the UI can plot the folds in order.
            for segment in ('consumption', 'production'):
                for step, fold in enumerate(report.get(f'cv_folds_{segment}', [])):
                    if fold.get('mae_power') is not None:
                        mlflow.log_metric(f'cv_fold_mae_power_{segment}', fold['mae_power'], step=step)

            # ---- Models ---- #
            features = list(model.features)
            _log_segment_model(
                model.final_model_cb_consumption, 'consumption', registered_model_name,
            )
            _log_segment_model(
                model.final_model_cb_production, 'production', registered_model_name,
            )

            # ---- Artifacts ---- #
            mlflow.log_dict(report, 'eval_report.json')
            mlflow.log_dict(
                {'features': features, 'cat_features': model.cat_features},
                'features.json',
            )
            mlflow.set_tags({'model_type': 'CatBoostRegressor', 'segments': 'consumption,production'})

        logger.info(f'DONE: Run logged to MLflow (run_id: {run_id})')
        return run_id

    except Exception as e:
        logger.error(f'Error logging to MLflow: {e}')
        if mlflow.active_run():
            mlflow.end_run(status='FAILED')
        return None


# ---------- CLI Entry Point ---------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Log a saved CatBoost prosumer model to MLflow')
    parser.add_argument('--model-name', type=str, default='catboost_prosumer',
                        help='Saved model directory name inside MODELS_DIR')
    parser.add_argument('--data-path', type=str,
                        default=str(Path(PROCESSED_DATA_DIR) / 'train_features.parquet'),
                        help='Feature parquet used for input examples')
    parser.add_argument('--experiment', type=str, default=MLFLOW_EXPERIMENT_NAME,
                        help='MLflow experiment name')
    parser.add_argument('--run-name', type=str, default=None, help='MLflow run name')
    parser.add_argument('--register', type=str, default=None,
                        help='Register both models in the Model Registry under this prefix')
    parser.add_argument('--check', action='store_true',
                        help='Only check the connection to the MLflow server')
    args = parser.parse_args()

    if args.check:
        raise SystemExit(0 if setup_mlflow() else 1)

    loaded = CatBoostRegressorModel().load(model_name=args.model_name)
    sample = pd.read_parquet(args.data_path)
    run_id = log_catboost_model_to_mlflow(
        model=loaded,
        X_sample=sample,
        experiment_name=args.experiment,
        run_name=args.run_name,
        registered_model_name=args.register,
    )
    print(f'Logged to MLflow (run_id: {run_id})' if run_id else 'Failed to log to MLflow')


# ---------- All exports ---------- #
__all__ = ['setup_mlflow', 'log_catboost_model_to_mlflow']
