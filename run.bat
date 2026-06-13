@echo off
title MediAI — Telemedicine ML Platform
color 0A

echo.
echo  =========================================
echo    MediAI - Telemedicine ML Platform
echo  =========================================
echo.

:: Check Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python is not installed or not in PATH.
    echo  Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

cd /d "%~dp0telemedicine_ml"

:: Check project folder exists
if not exist "model.py" (
    echo  [ERROR] Could not find telemedicine_ml folder.
    pause
    exit /b 1
)

echo  [1/3] Installing dependencies...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo  [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo  Done.
echo.

:: Skip training if models already exist
if exist "models\churn_model.pkl" if exist "models\risk_model.pkl" (
    echo  [2/3] Models already trained. Skipping retraining.
) else (
    echo  [2/3] Training ML models...
    python model.py
    if errorlevel 1 (
        echo  [ERROR] Model training failed.
        pause
        exit /b 1
    )
    echo  Done.
)
echo.

echo  [3/3] Launching Desktop UI...
start python ui.py
if errorlevel 1 (
    echo  [ERROR] Failed to launch UI.
    pause
    exit /b 1
)

echo.
echo  =========================================
echo    MediAI is running!
echo  =========================================
echo.
pause
