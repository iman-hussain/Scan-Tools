# Auto-update script files from GitHub
$repo = 'iman-hussain/Scan-Tools'
$files = @('Crop Split Rotate Upscale.py', 'requirements.txt')

foreach ($file in $files) {
    try {
        $url = "https://raw.githubusercontent.com/$repo/main/$file"
        $temp = "$file.tmp"
        Invoke-WebRequest -Uri $url -OutFile $temp -UseBasicParsing -ErrorAction Stop
        if (Test-Path $temp) { 
            Move-Item -Force $temp $file 
        }
    } catch { 
        # Silently continue if download fails (offline, etc.)
    }
}
Write-Host "Up to date."
