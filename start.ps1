# Photo Scanner - GPU Accelerated
# Double-click to run, or right-click -> "Run with PowerShell"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Photo Scanner - GPU Accelerated" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $PSScriptRoot

# Auto-update from GitHub
Write-Host "Checking for updates..." -ForegroundColor Yellow
$repo = 'iman-hussain/Scan-Tools'
$files = @('Crop Split Rotate Upscale.py', 'requirements.txt')
foreach ($file in $files) {
    try {
        $url = "https://raw.githubusercontent.com/$repo/main/$file"
        Invoke-WebRequest -Uri $url -OutFile $file -UseBasicParsing -ErrorAction Stop
    } catch { }
}
Write-Host "Up to date." -ForegroundColor Green
Write-Host ""

# Check if venv exists
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        Write-Host "Make sure Python 3.10+ is installed and in your PATH"
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "Virtual environment created." -ForegroundColor Green
    Write-Host ""
}

# Activate venv
& "venv\Scripts\Activate.ps1"

# Check if requirements are installed
python -c "import cv2" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing requirements - this may take a few minutes..." -ForegroundColor Yellow
    pip install --upgrade pip --quiet
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install requirements" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "Requirements installed." -ForegroundColor Green
    Write-Host ""
}

# Run the script
Write-Host "Starting Photo Scanner..." -ForegroundColor Cyan
Write-Host ""
python "Crop Split Rotate Upscale.py"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Script finished." -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Read-Host "Press Enter to exit"
