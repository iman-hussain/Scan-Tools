@echo off
echo ============================================
echo Photo Scanner - GPU Accelerated
echo ============================================
echo.

cd /d "%~dp0"

REM Auto-update from GitHub
echo Checking for updates...
powershell -ExecutionPolicy Bypass -File "update.ps1" 2>nul
if %errorlevel% neq 0 (
    echo Continuing without update check...
)
echo.

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Make sure Python 3.10+ is installed and in your PATH
        pause
        exit /b 1
    )
    echo Virtual environment created.
    echo.
)

REM Activate venv
call venv\Scripts\activate.bat

REM Check if requirements are installed (check for opencv)
python -c "import cv2" 2>nul
if errorlevel 1 (
    echo Installing requirements (this may take a few minutes)...
    pip install --upgrade pip --quiet
    pip install -r requirements.txt
    if errorlevel 1 (
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
echo Script finished. Press any key to exit.
pause >nul