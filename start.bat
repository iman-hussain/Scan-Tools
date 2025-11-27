@echo off
echo ============================================
echo Photo Scanner Processing Script - Launcher
echo GPU Accelerated with CUDA (RTX 2080Ti)
echo ============================================
echo.

cd /d "%~dp0"

REM Add CUDA and cuDNN to PATH for this session
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set "CUDNN_PATH=C:\Program Files\NVIDIA\CUDNN\v9.16\bin\12.9"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%CUDNN_PATH%;%PATH%"

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Make sure Python is installed and in your PATH
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
    echo Installing requirements...
    pip install --upgrade pip
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
python "Crop Split Rotate.py"

echo.
echo ============================================
echo Script finished. Press any key to exit.
pause >nul
