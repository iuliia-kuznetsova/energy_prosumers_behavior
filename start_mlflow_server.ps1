# Local MLflow tracking server: SQLite backend (mlflow.db) and local artifact store (mlartifacts/).
# Usage (from the project root):  .\start_mlflow_server.ps1
# UI and tracking URI: http://127.0.0.1:5000  (must match MLFLOW_TRACKING_URI in .env)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot 'venv_enefit\Scripts\python.exe'
$backendUri = 'sqlite:///mlflow.db'
$artifactsDir = './mlartifacts'
New-Item -ItemType Directory -Force -Path $artifactsDir | Out-Null

Write-Host 'Starting MLflow server...'
Write-Host "Backend:   $backendUri"
Write-Host "Artifacts: $artifactsDir"

# --serve-artifacts proxies artifact uploads through the server, so clients need no direct filesystem access.
& $python -m mlflow server `
    --backend-store-uri $backendUri `
    --artifacts-destination $artifactsDir `
    --serve-artifacts `
    --host 127.0.0.1 `
    --port 5000
