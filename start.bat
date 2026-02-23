@echo off
setlocal enabledelayedexpansion

color 0B
echo ============================================
echo Photo Scanner - GPU Accelerated
echo ============================================
echo.

cd /d "%~dp0"

REM Auto-update from GitHub
echo Checking for updates...
for %%f in ("Crop Split Rotate Upscale.py" "requirements.txt") do (
    powershell -Command "& {try { Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/iman-hussain/Scan-Tools/main/%%~f' -OutFile '%%~f' -UseBasicParsing -ErrorAction Stop } catch { }}" >nul 2>&1
)
echo Up to date.
echo.

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        color 0C
        echo ERROR: Failed to create virtual environment
        echo Make sure Python 3.10+ is installed and in your PATH
        pause
        exit /b 1
    )
    echo Virtual environment created.
    echo.
)

REM Activate venv
call "venv\Scripts\activate.bat"

REM Check if requirements are installed
python -c "import cv2" 2>nul
if errorlevel 1 (
    echo Installing requirements - this may take a few minutes...
    python -m pip install --upgrade pip --quiet
    pip install -r requirements.txt
    if errorlevel 1 (
        color 0C
        echo ERROR: Failed to install requirements
        pause
        exit /b 1
    )
    echo Requirements installed.
    echo.
)

REM Run the script
echo Starting Photo Scanner...
echo.
python "Crop Split Rotate Upscale.py"

echo.
echo ============================================
echo Script finished.
echo ============================================
pause
