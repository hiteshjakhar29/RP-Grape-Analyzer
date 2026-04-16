@echo off
REM run.bat - One-click launcher for Grape Analyzer on Windows
REM Handles: venv setup, pip install, app launch
REM Mac / Linux users: use run.sh instead

echo.
echo ============================================
echo    Grape Analyzer - Starting Setup
echo ============================================
echo.

REM Go to project folder (same directory as this script)
cd /d "%~dp0"

REM Virtual environment
if not exist ".venv" (
    echo Creating Python virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Python not found. Install Python 3.10+ from https://python.org
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install dependencies
echo Installing/updating dependencies...
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt

echo.
echo Setup complete.
echo Note: Fiji must be installed (e.g. C:\Fiji.app or C:\Program Files\Fiji.app)
echo       or locate it via the Settings dialog on first run.
echo.

REM Launch app
echo Launching Grape Analyzer...
echo.
python main.py

pause
