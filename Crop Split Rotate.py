"""
Photo Scanner Processing Script v2.0
=====================================
Optimized for A4 flatbed scanner with black cloth backdrop.
- Each scan contains 1-2 photos (standard print sizes)
- Crops photos from dark background with precise boundaries
- Corrects skewed angles (photos scanned at weird angles)
- Auto-rotates to upright based on face detection
- GPU accelerated using CUDA (RTX 2080Ti)

Standard photo print sizes (inches): 4x6, 5x7, 6x8, 8x10
"""

import os
import sys
import cv2
import numpy as np
import glob
from tqdm import tqdm
from PIL import Image, ImageOps
import urllib.request
import time
import logging

# === SUPPRESS ALL WARNINGS ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ONNXRUNTIME_LOG_LEVEL'] = '3'  # Suppress ONNX C++ warnings
logging.getLogger('onnxruntime').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "Input")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Output")

# Scanner-specific settings (A4 flatbed with black cloth)
MIN_PHOTO_AREA = 200000    # Minimum photo area in pixels^2 (about 500x400)
MAX_PHOTOS_PER_SCAN = 2    # Each scan has 1 or 2 photos
PHOTO_PADDING = 5          # Extra pixels around detected photo (trim black edges)

# Standard photo aspect ratios (width/height, either orientation)
STANDARD_RATIOS = [
    4/6, 6/4,    # 4x6
    5/7, 7/5,    # 5x7 
    6/8, 8/6,    # 6x8
    8/10, 10/8,  # 8x10
    3.5/5, 5/3.5, # 3.5x5
    1.0,         # Square
]

# Testing
TEST_MODE = True
TEST_LIMIT = 20  # Test first 20 files (includes problematic 14, 15)
# ---------------------

print("=" * 60)
print("PHOTO SCANNER v2.0 - GPU Accelerated")
print("=" * 60)
print()

# === GPU Setup ===
ONNX_GPU_AVAILABLE = False
ort = None

try:
    import onnxruntime as ort
    # Suppress ONNX logging
    ort.set_default_logger_severity(3)  # ERROR level only
    
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        ONNX_GPU_AVAILABLE = True
        print("✓ CUDA GPU acceleration enabled")
    else:
        print("✗ CUDA not available")
except ImportError:
    print("✗ ONNX Runtime not installed")

# CuPy for GPU image ops
CUPY_AVAILABLE = False
try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    CUPY_AVAILABLE = True
    print("✓ CuPy GPU acceleration enabled")
except:
    pass

# --- MODEL PATHS ---
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# RetinaFace - heavy/accurate face detection
RETINAFACE_MODEL_PATH = os.path.join(MODELS_DIR, "retinaface_resnet50.onnx")
RETINAFACE_URL = "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"

# YuNet - fast face detection
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

# Ultra-Light ONNX
ULTRAFACE_MODEL_PATH = os.path.join(MODELS_DIR, "version-RFB-320.onnx")
ULTRAFACE_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.onnx"

# SSD ResNet
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
            return True
        except Exception as e:
            print(f"  ✗ Could not download {name}: {e}")
            return False
    return True


# === Initialize Face Detectors ===
print("\nLoading face detection models...")

ONNX_FACE_SESSION = None
if ort is not None and ONNX_GPU_AVAILABLE:
    if download_model(ULTRAFACE_URL, ULTRAFACE_MODEL_PATH, "Ultra-Light Face"):
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.log_severity_level = 3  # Suppress warnings
            
            ONNX_FACE_SESSION = ort.InferenceSession(
                ULTRAFACE_MODEL_PATH,
                sess_options=sess_options,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            print("  ✓ ONNX Face Detector (GPU)")
        except Exception as e:
            print(f"  ✗ ONNX Face Detector failed: {e}")

YUNET_DETECTOR = None
if download_model(YUNET_URL, YUNET_MODEL_PATH, "YuNet Face"):
    try:
        YUNET_DETECTOR = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH, "", (320, 320), 0.5, 0.3, 5000
        )
        print("  ✓ YuNet Face Detector")
    except:
        pass

DNN_FACE_DETECTOR = None
if download_model(DNN_PROTO_URL, DNN_PROTO_PATH, "SSD proto"):
    if download_model(DNN_MODEL_URL, DNN_MODEL_PATH, "SSD model"):
        try:
            DNN_FACE_DETECTOR = cv2.dnn.readNetFromCaffe(DNN_PROTO_PATH, DNN_MODEL_PATH)
            print("  ✓ SSD Face Detector")
        except:
            pass

print()


# ============================================================
# PHOTO DETECTION & EXTRACTION
# ============================================================

def analyze_scan_background(gray):
    """Determine if scan has dark (cloth) or light background."""
    h, w = gray.shape
    
    # Sample corners
    corners = [
        gray[0:50, 0:50].mean(),
        gray[0:50, -50:].mean(),
        gray[-50:, 0:50].mean(),
        gray[-50:, -50:].mean(),
    ]
    corner_avg = np.mean(corners)
    
    # Sample center
    center = gray[h//3:2*h//3, w//3:2*w//3].mean()
    
    # Sample edges (middle of each edge)
    edge_samples = [
        gray[h//2, 0:50].mean(),      # left edge
        gray[h//2, -50:].mean(),      # right edge
        gray[0:50, w//2].mean(),      # top edge
        gray[-50:, w//2].mean(),      # bottom edge
    ]
    edge_avg = np.mean(edge_samples)
    
    # Dark background: edges/corners are dark, there's content in the middle
    # Light background: everything is light (blank/document scan)
    if (corner_avg < 100 or edge_avg < 80):
        return "dark", int(min(corner_avg, edge_avg))
    elif corner_avg > 150 and center > 150:
        return "light", int(corner_avg)
    else:
        return "dark", int(corner_avg)  # Default to trying to find photos


def find_photos_in_scan(image):
    """
    Find 1-2 photos in a scan with black cloth background.
    Uses multiple strategies to detect photo boundaries.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Analyze background
    bg_type, bg_level = analyze_scan_background(gray)
    
    # If light background (blank/document), skip
    if bg_type == "light":
        return []
    
    # Calculate adaptive threshold based on background darkness
    # Darker cloth = lower threshold needed
    base_thresh = max(35, min(70, bg_level + 15))
    
    # Apply Gaussian blur to smooth cloth texture
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    best_photos = []
    best_score = 0
    
    # Try multiple thresholds
    for thresh_offset in [-15, -5, 0, 10, 20]:
        threshold = base_thresh + thresh_offset
        if threshold < 25 or threshold > 90:
            continue
        
        # Binary threshold
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        
        # Clean up with morphological operations
        kernel = np.ones((7, 7), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and score contours
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_PHOTO_AREA:
                continue
            
            # Skip if contour is basically the whole scan
            if area > (w * h) * 0.9:
                continue
            
            x, y, cw, ch = cv2.boundingRect(cnt)
            
            # Aspect ratio check
            aspect = min(cw, ch) / max(cw, ch) if max(cw, ch) > 0 else 0
            if aspect < 0.4:  # Photos aren't super skinny
                continue
            
            # Score based on how "photo-like" this contour is
            # Prefer contours that:
            # 1. Match standard photo aspect ratios
            # 2. Are reasonably sized
            # 3. Have good rectangularity
            
            rect_area = cw * ch
            rectangularity = area / rect_area if rect_area > 0 else 0
            
            # Check aspect ratio match to standard photo sizes
            ratio = cw / ch if ch > 0 else 1
            ratio_score = min(abs(ratio - r) for r in STANDARD_RATIOS)
            
            score = rectangularity * 100 - ratio_score * 50 + (area / (w*h)) * 100
            
            candidates.append({
                'contour': cnt,
                'bbox': (x, y, cw, ch),
                'area': area,
                'score': score,
                'rect_score': rectangularity
            })
        
        # Sort by area (largest first), take top 2
        candidates.sort(key=lambda c: c['area'], reverse=True)
        candidates = candidates[:MAX_PHOTOS_PER_SCAN]
        
        # Calculate total score for this threshold
        total_score = sum(c['score'] for c in candidates)
        
        if len(candidates) >= 1 and total_score > best_score:
            best_score = total_score
            best_photos = candidates
    
    # Extract photos
    results = []
    for photo_info in best_photos:
        x, y, cw, ch = photo_info['bbox']
        cnt = photo_info['contour']
        
        # Use minAreaRect for potential angle detection
        rect = cv2.minAreaRect(cnt)
        angle = rect[2]
        
        # Extract with bounding rect first
        pad = PHOTO_PADDING
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + cw + pad)
        y2 = min(h, y + ch + pad)
        
        cropped = image[y1:y2, x1:x2].copy()
        
        if cropped.size == 0:
            continue
        
        # Trim any remaining dark borders
        cropped = trim_dark_edges(cropped)
        
        # Check if significantly skewed and correct
        if cropped is not None and cropped.size > 0:
            cropped = correct_skew(cropped)
        
        if cropped is not None and cropped.size > 0:
            results.append(cropped)
    
    return results


def trim_dark_edges(image, dark_thresh=40):
    """
    Precisely trim dark edges from a photo.
    Uses edge analysis to find the actual photo boundary.
    """
    if image is None or image.size == 0:
        return image
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Find first/last rows and cols that have substantial bright content
    # Use percentile to be robust to noise
    
    def find_content_start(line_means, thresh=50):
        """Find first index where content appears."""
        for i, val in enumerate(line_means):
            if val > thresh:
                return max(0, i - 2)  # Small margin
        return 0
    
    def find_content_end(line_means, thresh=50):
        """Find last index where content appears."""
        for i in range(len(line_means) - 1, -1, -1):
            if line_means[i] > thresh:
                return min(len(line_means), i + 3)  # Small margin
        return len(line_means)
    
    # Calculate row and column means
    row_means = np.mean(gray, axis=1)
    col_means = np.mean(gray, axis=0)
    
    # Find content bounds
    y1 = find_content_start(row_means, dark_thresh + 10)
    y2 = find_content_end(row_means, dark_thresh + 10)
    x1 = find_content_start(col_means, dark_thresh + 10)
    x2 = find_content_end(col_means, dark_thresh + 10)
    
    # Safety: don't trim more than 10% from any edge
    max_trim = 0.10
    if y1 > h * max_trim:
        y1 = 0
    if (h - y2) > h * max_trim:
        y2 = h
    if x1 > w * max_trim:
        x1 = 0
    if (w - x2) > w * max_trim:
        x2 = w
    
    # Ensure we have content
    if y2 <= y1 or x2 <= x1:
        return image
    
    return image[y1:y2, x1:x2]


def correct_skew(image, max_angle=15):
    """
    Detect and correct skew in photo (for photos scanned at angles).
    Only corrects small angles (up to max_angle degrees).
    """
    if image is None or image.size == 0:
        return image
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=w//4, maxLineGap=10)
    
    if lines is None or len(lines) < 3:
        return image
    
    # Calculate angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Normalize to -45 to 45 range
        while angle > 45:
            angle -= 90
        while angle < -45:
            angle += 90
        angles.append(angle)
    
    if not angles:
        return image
    
    # Use median angle (robust to outliers)
    median_angle = np.median(angles)
    
    # Only correct if angle is small but noticeable
    if abs(median_angle) < 0.5 or abs(median_angle) > max_angle:
        return image
    
    # Rotate to correct skew
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    
    # Calculate new bounding box size
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    # Adjust rotation matrix
    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2
    
    # Apply rotation with white border fill
    rotated = cv2.warpAffine(image, matrix, (new_w, new_h), 
                              borderMode=cv2.BORDER_REPLICATE)
    
    # Trim any new dark edges introduced
    return trim_dark_edges(rotated)


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces_onnx(image, conf=0.4):
    """CUDA-accelerated face detection using Ultra-Light model."""
    if ONNX_FACE_SESSION is None:
        return []
    
    h, w = image.shape[:2]
    
    # Preprocess
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
            results.append({'conf': float(face_conf)})
    
    return results


def detect_faces_yunet(image, conf=0.4):
    """YuNet face detection."""
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


def detect_faces_dnn(image, conf=0.4):
    """SSD DNN face detection."""
    if DNN_FACE_DETECTOR is None:
        return []
    
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    DNN_FACE_DETECTOR.setInput(blob)
    detections = DNN_FACE_DETECTOR.forward()
    
    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf:
            results.append({'conf': float(confidence)})
    
    return results


def count_faces_all_methods(image, conf=0.4):
    """Get face count using all available methods."""
    faces = []
    
    # YuNet (most reliable for orientation)
    yunet_faces = detect_faces_yunet(image, conf)
    
    # ONNX GPU
    onnx_faces = detect_faces_onnx(image, conf)
    
    # SSD backup  
    dnn_faces = detect_faces_dnn(image, conf)
    
    # Weighted score - YuNet is best for orientation detection
    score = len(yunet_faces) * 3 + len(onnx_faces) * 2 + len(dnn_faces) * 1
    total = len(yunet_faces) + len(onnx_faces) + len(dnn_faces)
    
    return score, total


def find_best_rotation(image):
    """
    Find the rotation angle that produces the most face detections.
    Tests 0°, 90°, 180°, 270° rotations.
    """
    h, w = image.shape[:2]
    
    # Scale down for faster processing
    max_dim = 640
    scale = min(1.0, max_dim / max(h, w))
    
    if scale < 1.0:
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = image
    
    best_angle = 0
    best_score = 0
    best_count = 0
    
    for angle in [0, 90, 180, 270]:
        if angle == 0:
            test_img = small
        elif angle == 90:
            test_img = cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            test_img = cv2.rotate(small, cv2.ROTATE_180)
        else:
            test_img = cv2.rotate(small, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        score, count = count_faces_all_methods(test_img)
        
        if score > best_score:
            best_score = score
            best_angle = angle
            best_count = count
    
    return best_angle, best_count


def apply_rotation(image, angle):
    """Apply 90-degree rotation."""
    if angle == 90:
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
    """Save photo with auto-contrast enhancement."""
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img = ImageOps.autocontrast(pil_img, cutoff=0.5)
        pil_img.save(path, quality=95)
        return True
    except Exception as e:
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
        print(f"*** TEST MODE: Processing {TEST_LIMIT} files ***")
    print()
    
    # Find files
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif', '*.bmp')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(INPUT_DIR, '**', ext), recursive=True))
    
    files.sort()
    
    if TEST_MODE:
        files = files[:TEST_LIMIT]
    
    total = len(files)
    print(f"Found {total} scan files.\n")
    
    if total == 0:
        print("No files to process!")
        return
    
    start = time.time()
    extracted = 0
    skipped = 0
    
    pbar = tqdm(files, unit="scan", desc="Processing")
    
    for file_path in pbar:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        
        # Output folder
        rel = os.path.relpath(os.path.dirname(file_path), INPUT_DIR)
        out_folder = os.path.join(OUTPUT_DIR, rel)
        os.makedirs(out_folder, exist_ok=True)
        
        pbar.set_postfix_str(filename[:25])
        
        # Load image
        img = cv2.imread(file_path)
        if img is None:
            pbar.write(f"  ✗ Could not load: {filename}")
            continue
        
        # Find photos in scan
        photos = find_photos_in_scan(img)
        
        if not photos:
            skipped += 1
            pbar.write(f"  ⚠ Skipped (no photos): {filename}")
            continue
        
        # Process each photo
        for i, photo in enumerate(photos, 1):
            # Find best rotation
            angle, face_count = find_best_rotation(photo)
            
            # Apply rotation
            final = apply_rotation(photo, angle)
            h, w = final.shape[:2]
            
            # Log
            rot_str = f"Rot {angle}°" if angle != 0 else "No rot"
            pbar.write(f"  [{base_name} p{i}] {rot_str} ({face_count} faces) -> {w}x{h}")
            
            # Save
            out_path = os.path.join(out_folder, f"{base_name}_p{i}.png")
            if save_photo(final, out_path):
                extracted += 1
    
    pbar.close()
    
    elapsed = time.time() - start
    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Photos extracted: {extracted}")
    print(f"Scans skipped:    {skipped}")
    print(f"Time: {elapsed:.1f}s")


if __name__ == "__main__":
    process_scans()

