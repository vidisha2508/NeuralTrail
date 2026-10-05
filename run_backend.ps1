# Neural Trail ML Backend Launch Script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ScriptDir "backend\.venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment not found at $VenvPython. Run installation first."
    exit 1
}

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  NEURAL TRAIL — ML BACKEND SERVER           " -ForegroundColor Magenta
Write-Host "  Starting FastAPI + PyTorch Inference Engine " -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan

Set-Location (Join-Path $ScriptDir "backend")
& $VenvPython run.py
