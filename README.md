# 📷 Photo Scanner Tools

GPU-accelerated batch processing for scanned photos. Automatically extracts, rotates, straightens, and AI-upscales photos from flatbed scanner images.

## ⚠️ Requirements

- **Windows 10/11**
- **NVIDIA GPU** (CUDA-compatible, 4GB+ VRAM recommended)
- **Python 3.10+**

> **Note:** This tool uses CUDA for GPU acceleration. An NVIDIA GPU is required for best performance.

## ✨ Features

- 🔍 **Auto-detect photos** - Finds 1-2 photos per scan (black cloth backdrop)
- 🔄 **Smart rotation** - Uses AI face detection to orient photos correctly
- 📐 **Deskew/Straighten** - Fixes tilted scans automatically
- ✂️ **Border removal** - Trims dark borders tightly
- 🚀 **2x AI Upscaling** - Real-ESGAN enhancement for crisp, lifelike results
- 📁 **Preserves folder structure** - Organizes output matching your input folders

## 🚀 Quick Start

### 1. Install Prerequisites

**Python 3.10+**

- Download from [python.org](https://www.python.org/downloads/)
- ✅ Check "Add Python to PATH" during installation

**NVIDIA GPU Drivers & CUDA** (required for GPU acceleration)

1. Install latest [NVIDIA GPU Drivers](https://www.nvidia.com/drivers)
2. Install [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads)
   - Choose your Windows version
   - Select "exe (local)" installer
   - Default installation options are fine
3. Install [cuDNN 9.x](https://developer.nvidia.com/cudnn-downloads)
   - Requires free NVIDIA Developer account
   - Choose "Windows" → "x86_64" → "exe (local)"
   - Run the installer (installs automatically to CUDA folder)

> 💡 **Tip:** After installing CUDA and cuDNN, restart your computer.

### 2. Clone or Download

```bash
git clone https://github.com/iman-hussain/Scan-Tools.git
cd Scan-Tools
```

Or download and extract the ZIP.

### 3. Add Your Scans

Place your scanned images in the `Input/` folder:

```
Input/
├── scan001.png
├── scan002.jpg
├── Holidays/
│   ├── beach_scan.png
│   └── mountains.jpg
└── Family/
    └── reunion.tiff
```

**Supported formats:** PNG, JPG, JPEG, TIFF, TIF, BMP

### 4. Run

Double-click **`start.bat`**

That's it! The script will:

1. Set up a Python virtual environment (first run only)
2. Install all dependencies automatically
3. Process your photos

### 5. Get Your Photos

Processed photos appear in `Output/` with the same folder structure:

```
Output/
├── scan001_p1.png
├── scan001_p2.png      (if 2 photos detected)
├── Holidays/
│   ├── beach_scan_p1.png
│   └── mountains_p1.png
└── Family/
    └── reunion_p1.png
```

## ⚙️ Configuration

Edit `Crop Split Rotate Upscale.py` to change settings:

```python
TEST_MODE = True    # Set to False to process ALL files
TEST_LIMIT = 20     # Number of files in test mode
UPSCALE_2X = True   # Enable/disable AI upscaling
```

## 📋 Scanning Tips

For best results:

- Use a **black cloth** as backdrop on your scanner
- Place **1-2 photos** per scan
- Photos can be at any angle - rotation is auto-detected
- 300 DPI recommended for good quality

## 🛠️ Troubleshooting

### "CUDA not available"

- Ensure you have an NVIDIA GPU
- Install [NVIDIA drivers](https://www.nvidia.com/drivers)
- The script will still work on CPU but much slower

### "No photos found"

- Check your scan has a dark background
- Ensure photos aren't touching the scanner edges
- Try adjusting the threshold in the script

### First run is slow

- Dependencies are being downloaded (~2GB for PyTorch + CUDA)
- ML models are downloaded automatically
- Subsequent runs will be much faster

## 📊 Performance

On RTX 2080 Ti (11GB VRAM):

- ~4-5 seconds per photo with 2x AI upscaling
- ~1-2 seconds per photo without upscaling

## 📄 License

MIT License - feel free to use and modify.

## 🙏 Credits

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - AI upscaling
- [OpenCV](https://opencv.org/) - Image processing
- [ONNX Runtime](https://onnxruntime.ai/) - Face detection inference
- [Spandrel](https://github.com/chaiNNer-org/spandrel) - Model loading
