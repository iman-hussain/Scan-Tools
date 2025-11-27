"""
Photo Scanner Processing Script
================================
Optimized for A4 flatbed scanner with black cloth backdrop.
- Each scan contains 1 or 2 photos
- Crops photos from dark background
- Auto-rotates based on face detection
- GPU accelerated using CUDA (RTX 2080Ti)

Author: Photo Scanner Assistant
"""

import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
from PIL import Image, ImageOps
import urllib.request
import time

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "Input")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Output")

# Scanner-specific settings (A4 flatbed with black cloth)
BG_THRESHOLD = 55          # Threshold for dark cloth (higher = more tolerant of grey cloth)
MIN_PHOTO_AREA = 80000     # Minimum photo size in pixels^2
MAX_PHOTOS_PER_SCAN = 2    # Each scan has 1 or 2 photos
BORDER_MARGIN = 15         # Pixels to add around detected photo edges

# Face detection settings
DETECTION_CONFIDENCE = 0.4 # Lower = more sensitive detection

# Testing (set to True to only process a subset)
TEST_MODE = True           # Change to False for full processing
TEST_LIMIT = 20            # Number of files to process in test mode
# ---------------------

print("=" * 60)
print("PHOTO SCANNER PROCESSING SCRIPT")
print("=" * 60)
print(f"Scanner: A4 flatbed with black cloth backdrop")
print(f"Photos per scan: 1-2")
print()

# === GPU / ONNX Runtime Setup (CUDA ONLY) ===
ONNX_GPU_AVAILABLE = False
GPU_PROVIDER = None
ort = None

try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"ONNX Runtime providers: {providers}")

    if 'CUDAExecutionProvider' in providers:
        ONNX_GPU_AVAILABLE = True
        GPU_PROVIDER = 'CUDAExecutionProvider'
        print("✓ CUDA GPU acceleration enabled (RTX 2080Ti)")
    else:
        print("✗ CUDA not available - check CUDA 12.6 installation")
        print("  Make sure CUDA bin folder is in PATH")
except ImportError:
    print("✗ ONNX Runtime not installed")

# === CuPy for GPU image operations ===
CUPY_AVAILABLE = False
cp = None
try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    CUPY_AVAILABLE = True
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    print(f"✓ CuPy GPU acceleration enabled ({gpu_name})")
except Exception as e:
    print(f"⚠ CuPy not available (using CPU for image ops)")

# --- MODEL PATHS ---
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

YUNET_MODEL_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

ULTRAFACE_MODEL_PATH = os.path.join(MODELS_DIR, "version-RFB-320.onnx")
ULTRAFACE_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.onnx"

DNN_PROTO_PATH = os.path.join(MODELS_DIR, "deploy.prototxt")
DNN_MODEL_PATH = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
DNN_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
DNN_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


def download_model(url, path, name):
    """Download model file if not present."""
    if not os.path.exists(path):
        print(f"  Downloading {name}...")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"  ✓ Downloaded {name}")
            return True
        except Exception as e:
            print(f"  ✗ Could not download {name}: {e}")
            return False
    return True


# === Initialize Face Detection Models ===
print("\nInitializing face detection models...")

# ONNX Runtime GPU Face Detector
ONNX_FACE_SESSION = None
if ort is not None and ONNX_GPU_AVAILABLE:
    if download_model(ULTRAFACE_URL, ULTRAFACE_MODEL_PATH, "Ultra-Light Face Detector"):
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            ONNX_FACE_SESSION = ort.InferenceSession(
                ULTRAFACE_MODEL_PATH,
                sess_options=sess_options,
                providers=[GPU_PROVIDER, 'CPUExecutionProvider']
            )
            actual = ONNX_FACE_SESSION.get_providers()[0]
            print(f"  ✓ ONNX Face Detector loaded ({actual})")
        except Exception as e:
            print(f"  ✗ Could not load ONNX face detector: {e}")

# YuNet face detector (best for orientation)
YUNET_DETECTOR = None
if download_model(YUNET_URL, YUNET_MODEL_PATH, "YuNet Face Detector"):
    try:
        YUNET_DETECTOR = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH, "", (320, 320), 0.5, 0.3, 5000
        )
        print("  ✓ YuNet Face Detector loaded")
    except Exception as e:
        print(f"  ✗ Could not load YuNet: {e}")

# SSD DNN detector (backup)
DNN_FACE_DETECTOR = None
if download_model(DNN_PROTO_URL, DNN_PROTO_PATH, "SSD prototxt"):
    if download_model(DNN_MODEL_URL, DNN_MODEL_PATH, "SSD model"):
        try:
            DNN_FACE_DETECTOR = cv2.dnn.readNetFromCaffe(DNN_PROTO_PATH, DNN_MODEL_PATH)
            print("  ✓ SSD Face Detector loaded")
        except Exception as e:
            print(f"  ✗ Could not load SSD detector: {e}")

print()


# ============================================================
# PHOTO EXTRACTION (optimized for black cloth backdrop)
# ============================================================

def find_photos_in_scan(image):
    """Find 1-2 photos in scan with black/dark cloth background.
    
    Uses adaptive thresholding to handle varying darkness of the cloth.
    Returns list of cropped photo images.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise from cloth texture
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try multiple threshold levels to handle varying cloth darkness
    best_contours = []
    best_count = 0
    
    for threshold in [BG_THRESHOLD - 15, BG_THRESHOLD, BG_THRESHOLD + 15, BG_THRESHOLD + 30]:
        if threshold < 20:
            continue
            
        _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        valid = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_PHOTO_AREA:
                continue
                
            x, y, cw, ch = cv2.boundingRect(cnt)
            
            # Aspect ratio check (photos aren't super skinny)
            aspect = min(cw, ch) / max(cw, ch) if max(cw, ch) > 0 else 0
            if aspect < 0.3:
                continue
            
            # Size check - photo shouldn't be tiny fraction of scan
            if area < (w * h) * 0.08:
                continue
                
            valid.append((cnt, area))
        
        # Sort by area and limit to MAX_PHOTOS_PER_SCAN
        valid.sort(key=lambda x: x[1], reverse=True)
        valid = valid[:MAX_PHOTOS_PER_SCAN]
        
        # Keep best threshold result (1-2 good contours)
        if len(valid) >= 1:
            if len(valid) > best_count or (len(valid) == best_count and len(valid) <= 2):
                best_contours = valid
                best_count = len(valid)
    
    # Extract photos from best contours
    photos = []
    for cnt, _ in best_contours:
        photo = extract_photo(image, cnt, w, h)
        if photo is not None:
            photos.append(photo)
    
    return photos


def extract_photo(image, contour, scan_w, scan_h):
    """Extract a single photo using bounding rectangle with margin."""
    x, y, w, h = cv2.boundingRect(contour)
    
    # Add margin
    x1 = max(0, x - BORDER_MARGIN)
    y1 = max(0, y - BORDER_MARGIN)
    x2 = min(scan_w, x + w + BORDER_MARGIN)
    y2 = min(scan_h, y + h + BORDER_MARGIN)
    
    cropped = image[y1:y2, x1:x2].copy()
    
    if cropped.size == 0:
        return None
    
    # Clean up dark borders
    cropped = trim_dark_borders(cropped)
    
    return cropped


def trim_dark_borders(image, threshold=45):
    """Remove dark borders while preserving photo content."""
    if image is None or image.size == 0:
        return image
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Find rows/cols where majority is dark
    row_dark = np.mean(gray < threshold, axis=1)
    col_dark = np.mean(gray < threshold, axis=0)
    
    # Find first/last non-dark row/col (where less than 70% is dark)
    valid_rows = np.where(row_dark < 0.7)[0]
    valid_cols = np.where(col_dark < 0.7)[0]
    
    if len(valid_rows) < 2 or len(valid_cols) < 2:
        return image
    
    y1, y2 = valid_rows[0], valid_rows[-1] + 1
    x1, x2 = valid_cols[0], valid_cols[-1] + 1
    
    # Safety: don't trim more than 15% from any side
    max_trim = 0.15
    if y1 > h * max_trim:
        y1 = 0
    if (h - y2) > h * max_trim:
        y2 = h
    if x1 > w * max_trim:
        x1 = 0
    if (w - x2) > w * max_trim:
        x2 = w
    
    return image[y1:y2, x1:x2]


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces_onnx(image, conf=0.5):
    """CUDA-accelerated face detection."""
    if ONNX_FACE_SESSION is None:
        return []

    h, w = image.shape[:2]
    
    img_resized = cv2.resize(image, (320, 240))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = (img_rgb - 127.0) / 128.0
    img_batch = img_norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    input_name = ONNX_FACE_SESSION.get_inputs()[0].name
    confidences, boxes = ONNX_FACE_SESSION.run(None, {input_name: img_batch})

    results = []
    for i in range(boxes.shape[1]):
        face_conf = confidences[0, i, 1]
        if face_conf > conf:
            x1 = max(0, int(boxes[0, i, 0] * w))
            y1 = max(0, int(boxes[0, i, 1] * h))
            x2 = min(w, int(boxes[0, i, 2] * w))
            y2 = min(h, int(boxes[0, i, 3] * h))
            if x2 - x1 > 10 and y2 - y1 > 10:
                results.append({'conf': float(face_conf)})

    return results


def detect_faces_yunet(image, conf=0.5):
    """YuNet face detection - most accurate for orientation."""
    if YUNET_DETECTOR is None:
        return []

    h, w = image.shape[:2]
    YUNET_DETECTOR.setInputSize((w, h))
    _, faces = YUNET_DETECTOR.detect(image)

    if faces is None:
        return []

    results = []
    for face in faces:
        confidence = float(face[14]) if len(face) > 14 else float(face[-1])
        if confidence >= conf:
            results.append({'conf': confidence})

    return results


def detect_faces_dnn(image, conf=0.5):
    """SSD DNN face detection (backup)."""
    if DNN_FACE_DETECTOR is None:
        return []

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    DNN_FACE_DETECTOR.setInput(blob)
    detections = DNN_FACE_DETECTOR.forward()

    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf:
            results.append({'conf': float(confidence)})

    return results


def get_rotation_score(image, conf_threshold):
    """Get face detection score for a single orientation."""
    faces_yunet = detect_faces_yunet(image, conf_threshold)
    faces_onnx = detect_faces_onnx(image, conf_threshold)
    faces_dnn = detect_faces_dnn(image, conf_threshold)
    
    # YuNet is most reliable for orientation, weight it higher
    score = sum(f['conf'] for f in faces_yunet) * 3.0
    score += sum(f['conf'] for f in faces_onnx) * 1.0
    score += sum(f['conf'] for f in faces_dnn) * 1.0
    
    # Bonus for number of faces detected
    total_faces = len(faces_yunet) + len(faces_onnx) + len(faces_dnn)
    score += total_faces * 0.3
    
    return score, len(faces_yunet)


def get_best_rotation(image):
    """Find the rotation that gives the best face detection results."""
    h, w = image.shape[:2]
    
    # Scale down for faster detection
    max_dim = 800
    scale = min(1.0, max_dim / max(h, w))
    
    if scale < 1.0:
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = image
    
    best_angle = 0
    best_score = 0
    best_faces = 0
    
    for angle in [0, 90, 180, 270]:
        if angle == 0:
            test_img = small
        elif angle == 90:
            test_img = cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            test_img = cv2.rotate(small, cv2.ROTATE_180)
        else:
            test_img = cv2.rotate(small, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        score, faces = get_rotation_score(test_img, DETECTION_CONFIDENCE)
        
        if score > best_score:
            best_score = score
            best_angle = angle
            best_faces = faces
    
    return best_angle, best_score, best_faces


def apply_rotation(image, angle):
    """Apply rotation to image."""
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


# ============================================================
# SAVE
# ============================================================

def save_photo(image, path):
    """Save photo with slight auto-contrast enhancement."""
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img = ImageOps.autocontrast(pil_img, cutoff=0.5)
        pil_img.save(path, quality=95)
        return True
    except Exception as e:
        print(f"  Error saving: {e}")
        return False


# ============================================================
# MAIN
# ============================================================

def process_scans():
    """Main processing loop."""
    print("=" * 60)
    print("PROCESSING SCANS")
    print("=" * 60)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    if TEST_MODE:
        print(f"*** TEST MODE: Processing only {TEST_LIMIT} files ***")
    print()

    # Find all input files
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif', '*.bmp')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(INPUT_DIR, '**', ext), recursive=True))
    
    files.sort()
    
    if TEST_MODE:
        files = files[:TEST_LIMIT]
    
    total = len(files)
    print(f"Found {total} scan files to process.")
    print()

    if total == 0:
        print("No files to process!")
        return

    start = time.time()
    extracted = 0
    processed = 0

    pbar = tqdm(files, unit="scan", desc="Processing")
    
    for file_path in pbar:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        
        # Output folder
        rel = os.path.relpath(os.path.dirname(file_path), INPUT_DIR)
        out_folder = os.path.join(OUTPUT_DIR, rel)
        os.makedirs(out_folder, exist_ok=True)
        
        # Skip if processed
        if os.path.exists(os.path.join(out_folder, f"{base_name}_p1.png")):
            pbar.set_postfix_str(f"Skip: {filename[:20]}")
            continue
        
        pbar.set_postfix_str(filename[:25])
        
        # Load
        img = cv2.imread(file_path)
        if img is None:
            pbar.write(f"  ✗ Could not load: {filename}")
            continue
        
        # Find photos
        photos = find_photos_in_scan(img)
        
        if not photos:
            pbar.write(f"  ⚠ No photos found: {filename}")
            continue
        
        # Process each photo
        for i, photo in enumerate(photos, 1):
            angle, score, faces = get_best_rotation(photo)
            final = apply_rotation(photo, angle)
            h, w = final.shape[:2]
            
            if angle != 0:
                pbar.write(f"  [{base_name} p{i}] Rotated {angle}° ({faces} faces) -> {w}x{h}")
            else:
                pbar.write(f"  [{base_name} p{i}] ({faces} faces) -> {w}x{h}")
            
            out_path = os.path.join(out_folder, f"{base_name}_p{i}.png")
            if save_photo(final, out_path):
                extracted += 1
        
        processed += 1

    pbar.close()
    
    elapsed = time.time() - start
    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Scans processed:  {processed}")
    print(f"Photos extracted: {extracted}")
    print(f"Time: {elapsed:.1f}s ({elapsed/max(processed,1):.2f}s/scan)")


if __name__ == "__main__":
    process_scans()

