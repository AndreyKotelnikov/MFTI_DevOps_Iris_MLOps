# scripts/all.ps1
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path) | Out-Null
Set-Location ..

Write-Host "==> Обучаем модель..."
docker compose run --rm app python -m src.models.train

Write-Host "==> Генерируем отчёты Deepchecks..."
docker compose run --rm app python -m src.monitoring.data_quality_deepchecks

Write-Host "==> Генерируем отчёты Evidently..."
docker compose run --rm app python -m src.monitoring.data_drift_evidently

Write-Host "==> Готово. MLflow: http://localhost:5000, отчёты: ./reports"
