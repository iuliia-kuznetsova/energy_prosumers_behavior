#!/usr/bin/env bash
# Local MLflow tracking server: SQLite backend (mlflow.db) and local artifact store (mlartifacts/).
# No Docker, PostgreSQL or cloud storage needed.
# Usage (from the project root, Git Bash / WSL / Linux):  bash start_mlflow_server.sh
# On Windows PowerShell use .\start_mlflow_server.ps1 instead.
# UI and tracking URI: http://localhost:5000  (must match MLFLOW_TRACKING_URI in .env)

set -e  # Exit on any error

cd "$(dirname "$0")"

# Prefer the project venv; fall back to whatever python is on PATH.
if [ -x "venv_enefit/Scripts/python.exe" ]; then
  PYTHON="venv_enefit/Scripts/python.exe"
elif [ -x "venv_enefit/bin/python" ]; then
  PYTHON="venv_enefit/bin/python"
else
  PYTHON="python"
fi

MLFLOW_HOST="${MLFLOW_HOST:-127.0.0.1}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
# Relative paths: resolved against the project root, which avoids /c/... vs C:\... mismatches in Git Bash.
MLFLOW_BACKEND_URI="sqlite:///mlflow.db"
MLFLOW_ARTIFACTS_DIR="./mlartifacts"

mkdir -p "$MLFLOW_ARTIFACTS_DIR"

echo "Starting MLflow server..."
echo "Backend:   $MLFLOW_BACKEND_URI"
echo "Artifacts: $MLFLOW_ARTIFACTS_DIR"
echo "UI:        http://$MLFLOW_HOST:$MLFLOW_PORT"

# --serve-artifacts proxies artifact uploads through the server, so the training script only needs the HTTP URI.
"$PYTHON" -m mlflow server \
  --backend-store-uri "$MLFLOW_BACKEND_URI" \
  --artifacts-destination "$MLFLOW_ARTIFACTS_DIR" \
  --serve-artifacts \
  --host "$MLFLOW_HOST" \
  --port "$MLFLOW_PORT"
