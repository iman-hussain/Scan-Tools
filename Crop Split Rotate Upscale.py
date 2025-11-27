"""
Photo Scanner Processing Script v4.2 (Parallel)
=====================================
Simple, clear workflow:
1. Load input image
2. Detect photos (bordered by dark cloth, possibly with white scanner edge)
3. Split into individual photos (1 or 2 per scan)
4. Rotate each photo until faces are upright
5. Save
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
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ONNXRUNTIME_LOG_LEVEL'] = '3'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import logging
logging.getLogger('onnxruntime').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "Input")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Output")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Config
TEST_MODE = False  # Process limited files for testing
TEST_LIMIT = 10
UPSCALE_2X = True  # AI upscale photos by 2x using Real-ESRGAN
PARALLEL_WORKERS = 8  # Number of parallel workers

# Thread locks for thread safety
GPU_LOCK = threading.Lock()      # For Real-ESRGAN GPU operations
DETECTOR_LOCK = threading.Lock() # For YuNet and Haar (not thread-safe)

print("=" * 60)
print("PHOTO SCANNER v4.2 (Parallel)")
print("=" * 60)

# ============================================================
# FACE DETECTION SETUP
# ============================================================

# Download models
def download_file(url, path):
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(url, path)
            return True
        except:
            return False
    return True

# YuNet model
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
download_file(
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    YUNET_PATH
)

# ONNX models
ONNX_320_PATH = os.path.join(MODELS_DIR, "version-RFB-320.onnx")
ONNX_640_PATH = os.path.join(MODELS_DIR, "version-RFB-640.onnx")
download_file(
    "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.onnx",
    ONNX_320_PATH
)
download_file(
    "https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-640.onnx",
    ONNX_640_PATH
)

# Initialize detectors
print("Loading face detectors...")

YUNET = None
if os.path.exists(YUNET_PATH):
    try:
        YUNET = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), 0.5, 0.3, 5000)
        print("  ✓ YuNet")
    except Exception as e:
        print(f"  ✗ YuNet: {e}")

# ONNX Runtime
ONNX_SESSION = None
try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    
    # Force CPU-only for comparison test
    providers = ['CPUExecutionProvider']
    
    if os.path.exists(ONNX_640_PATH):
        ONNX_SESSION = ort.InferenceSession(ONNX_640_PATH, providers=providers)
        print("  ✓ UltraFace 640 (CPU)")
except Exception as e:
    print(f"  ✗ ONNX: {e}")

# Haar cascades
HAAR_FRONT = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
HAAR_PROFILE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
print("  ✓ Haar Cascades")

# Real-ESRGAN upscaler (2x, real photos - not anime)
UPSCALER = None
UPSCALER_DEVICE = None
UPSCALER_SCALE = 2
UPSCALER_TILE_SIZE = 512  # Process in tiles to save VRAM

if UPSCALE_2X:
    print("\nLoading AI upscaler...")
    try:
        import torch
        import spandrel
        
        # Use RealESRGAN_x2plus - best for real photos at 2x
        model_path = os.path.join(MODELS_DIR, "RealESRGAN_x2plus.pth")
        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        
        if not os.path.exists(model_path):
            print("  Downloading RealESRGAN_x2plus model (67MB)...")
            download_file(model_url, model_path)
        
        if os.path.exists(model_path):
            # Use spandrel to load the model (no basicsr dependency issues)
            UPSCALER_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_descriptor = spandrel.ModelLoader().load_from_file(model_path)
            UPSCALER = model_descriptor.model
            UPSCALER_SCALE = model_descriptor.scale
            UPSCALER = UPSCALER.eval().to(UPSCALER_DEVICE)
            
            # Use half precision on GPU for speed
            if UPSCALER_DEVICE.type == 'cuda':
                UPSCALER = UPSCALER.half()
            
            device_str = 'GPU (CUDA)' if UPSCALER_DEVICE.type == 'cuda' else 'CPU'
            print(f"  ✓ Real-ESRGAN x{UPSCALER_SCALE} Upscaler ({device_str})")
    except Exception as e:
        print(f"  ✗ Real-ESRGAN: {e}")
        UPSCALER = None

print()


# ============================================================
# STEP 1: DETECT AND SPLIT PHOTOS
# ============================================================

def find_photos(image):
    """
    Find 1 or 2 photos in a scan.
    Photos are surrounded by dark cloth (and possibly white scanner edge).
    Returns list of cropped photo images.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Check if this is a blank/light scan (no dark background)
    corner_brightness = [
        gray[0:50, 0:50].mean(),
        gray[0:50, -50:].mean(),
        gray[-50:, 0:50].mean(),
        gray[-50:, -50:].mean()
    ]
    if min(corner_brightness) > 200:
        return []  # Blank white scan
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive approach - find the best threshold that gives us 1-2 photos
    best_contours = []
    best_score = -1
    
    for thresh in range(25, 100, 5):
        _, binary = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)
        
        # Clean up
        kernel = np.ones((7, 7), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter valid photo-sized contours
        valid = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, cw, ch = cv2.boundingRect(cnt)
            
            # Photo must be reasonably sized (not tiny, not whole scan)
            min_area = (w * h) * 0.05  # At least 5% of scan
            max_area = (w * h) * 0.85  # Not more than 85% of scan
            
            if area < min_area or area > max_area:
                continue
            
            # Aspect ratio must be reasonable for a photo
            aspect = min(cw, ch) / max(cw, ch) if max(cw, ch) > 0 else 0
            if aspect < 0.4:  # Not too skinny
                continue
            
            # Must not be touching all edges (that's the whole scan)
            if x < 10 and y < 10 and x + cw > w - 10 and y + ch > h - 10:
                continue
                
            valid.append({
                'contour': cnt,
                'bbox': (x, y, cw, ch),
                'area': area
            })
        
        # Score this threshold
        # Prefer: 1-2 good-sized photos, not overlapping
        if len(valid) == 0:
            continue
        
        # Take top 2 by area
        valid.sort(key=lambda c: c['area'], reverse=True)
        valid = valid[:2]
        
        # Score based on total coverage and number of photos
        total_area = sum(c['area'] for c in valid)
        coverage = total_area / (w * h)
        
        # Best coverage between 15% and 70%
        if coverage < 0.1 or coverage > 0.8:
            continue
        
        score = coverage * len(valid)
        
        if score > best_score:
            best_score = score
            best_contours = valid
    
    # Extract photos from best contours
    photos = []
    for item in best_contours:
        x, y, cw, ch = item['bbox']
        
        # Add small padding
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + cw + pad)
        y2 = min(h, y + ch + pad)
        
        crop = image[y1:y2, x1:x2].copy()
        
        # Trim any remaining dark borders
        crop = trim_dark_borders(crop)
        
        if crop is not None and crop.shape[0] > 100 and crop.shape[1] > 100:
            photos.append(crop)
    
    return photos


def trim_dark_borders(image, threshold=50):
    """Remove dark borders from image edges. More aggressive trimming."""
    if image is None or image.size == 0:
        return image
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Use multiple passes with increasing threshold for tighter trim
    for thresh in [threshold, threshold + 20, threshold + 40]:
        # Find rows/cols that have content (not all dark)
        row_max = np.max(gray, axis=1)
        col_max = np.max(gray, axis=0)
        
        # Also check row/col means - dark borders have low mean
        row_mean = np.mean(gray, axis=1)
        col_mean = np.mean(gray, axis=0)
        
        # A row/col is "content" if it has bright pixels AND reasonable mean
        content_rows = np.where((row_max > thresh) & (row_mean > thresh * 0.5))[0]
        content_cols = np.where((col_max > thresh) & (col_mean > thresh * 0.5))[0]
        
        if len(content_rows) < 10 or len(content_cols) < 10:
            continue
        
        y1 = content_rows[0]
        y2 = content_rows[-1] + 1
        x1 = content_cols[0]
        x2 = content_cols[-1] + 1
        
        # Validate we're not trimming too much (max 15% per side)
        if y1 <= h * 0.15 and (h - y2) <= h * 0.15 and x1 <= w * 0.15 and (w - x2) <= w * 0.15:
            gray = gray[y1:y2, x1:x2]
            image = image[y1:y2, x1:x2]
            h, w = gray.shape
    
    return image


# ============================================================
# STEP 2: DETECT FACES AND FIND BEST ROTATION
# ============================================================

def detect_faces_yunet(image):
    """Detect faces using YuNet. Returns list of confidence scores."""
    if YUNET is None:
        return []
    
    h, w = image.shape[:2]
    
    # Work at reasonable size
    max_dim = 800
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        small = cv2.resize(image, None, fx=scale, fy=scale)
    else:
        small = image
    
    sh, sw = small.shape[:2]
    YUNET.setInputSize((sw, sh))
    
    _, faces = YUNET.detect(small)
    
    if faces is None:
        return []
    
    return [float(f[-1]) for f in faces]


def detect_faces_onnx(image):
    """Detect faces using ONNX UltraFace. Returns list of confidence scores."""
    if ONNX_SESSION is None:
        return []
    
    # Preprocess for 640x480 model
    img_resized = cv2.resize(image, (640, 480))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = (img_rgb - 127.0) / 128.0
    img_batch = img_norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    
    # Run inference
    input_name = ONNX_SESSION.get_inputs()[0].name
    confidences, boxes = ONNX_SESSION.run(None, {input_name: img_batch})
    
    # Get face confidences above threshold
    results = []
    for i in range(confidences.shape[1]):
        conf = confidences[0, i, 1]
        if conf > 0.5:
            results.append(float(conf))
    
    return results


def detect_faces_haar(image):
    """Detect faces using Haar cascades. Returns list of placeholder scores."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize for speed
    max_dim = 600
    h, w = gray.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale)
    
    results = []
    
    # Frontal faces
    faces = HAAR_FRONT.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    results.extend([0.7] * len(faces))
    
    # Profile faces
    profiles = HAAR_PROFILE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    results.extend([0.6] * len(profiles))
    
    return results


def compute_face_score(image):
    """
    Compute a score for this orientation based on face detection.
    Higher score = more likely to be correct orientation.
    Uses confidence-weighted scoring - quality over quantity.
    """
    # YuNet and Haar are NOT thread-safe, need lock
    with DETECTOR_LOCK:
        yunet_confs = detect_faces_yunet(image)
        haar_confs = detect_faces_haar(image)
    
    # ONNX is thread-safe
    onnx_confs = detect_faces_onnx(image)
    
    score = 0.0
    
    # YuNet - most reliable, high weight
    for conf in yunet_confs:
        score += (conf ** 2) * 5.0
        if conf > 0.8:
            score += 3.0  # Bonus for high confidence
    
    # ONNX UltraFace - also good
    for conf in onnx_confs:
        score += (conf ** 2) * 4.0
        if conf > 0.8:
            score += 2.0
    
    # Haar - less reliable but useful
    for conf in haar_confs:
        score += (conf ** 2) * 1.5
    
    total_faces = len(yunet_confs) + len(onnx_confs) + len(haar_confs)
    
    return score, total_faces


def find_best_rotation(image):
    """
    Test 0°, 90°, 180°, 270° and return the angle with best face score.
    """
    scores = {}
    
    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rotated = image
        elif angle == 90:
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(image, cv2.ROTATE_180)
        else:
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        score, faces = compute_face_score(rotated)
        scores[angle] = (score, faces)
    
    # Find best angle
    best_angle = max(scores, key=lambda a: scores[a][0])
    
    return best_angle, scores


def apply_rotation(image, angle):
    """Apply rotation to image."""
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


# ============================================================
# STEP 3: DESKEW (STRAIGHTEN TILTED PHOTOS)
# ============================================================

def deskew(image, max_angle=25):
    """
    Detect and correct image tilt using multiple methods for best accuracy.
    Combines: minAreaRect, Hough lines on edges, and gradient analysis.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    angles_detected = []
    
    # === METHOD 1: MinAreaRect of photo content ===
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > (w * h) * 0.3:  # Must be significant
            rect = cv2.minAreaRect(largest)
            angle1 = rect[2]
            rect_w, rect_h = rect[1]
            if rect_w < rect_h:
                angle1 = angle1 + 90
            if abs(angle1) <= max_angle:
                angles_detected.append(('minAreaRect', angle1, 3))  # weight 3
    
    # === METHOD 2: Hough lines on photo edges ===
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Dilate edges slightly to connect broken lines
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60,
                            minLineLength=min(w, h) // 8, maxLineGap=15)
    
    if lines is not None and len(lines) > 0:
        line_angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Weight longer lines more
            if length < 50:
                continue
            
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
            # Normalize to -45 to 45
            while angle > 45:
                angle -= 90
            while angle < -45:
                angle += 90
            
            # Weight by line length
            line_angles.extend([angle] * int(length / 50))
        
        if line_angles:
            angle2 = np.median(line_angles)
            if abs(angle2) <= max_angle:
                angles_detected.append(('hough', angle2, 2))
    
    # === METHOD 3: Border edge analysis ===
    # Look at the edges of the image specifically
    border_angle = detect_border_angle(gray)
    if border_angle is not None and abs(border_angle) <= max_angle:
        angles_detected.append(('border', border_angle, 4))  # Highest weight
    
    # === METHOD 4: Gradient orientation (for subtle tilts) ===
    gradient_angle = detect_gradient_angle(gray)
    if gradient_angle is not None and abs(gradient_angle) <= max_angle:
        angles_detected.append(('gradient', gradient_angle, 1))
    
    if not angles_detected:
        return image
    
    # Combine angles using weighted average
    total_weight = sum(a[2] for a in angles_detected)
    weighted_angle = sum(a[1] * a[2] for a in angles_detected) / total_weight
    
    # Only correct if angle is noticeable (> 0.3 degrees)
    if abs(weighted_angle) < 0.3:
        return image
    
    # Apply rotation
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, weighted_angle, 1.0)
    
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2
    
    rotated = cv2.warpAffine(image, matrix, (new_w, new_h),
                              borderMode=cv2.BORDER_CONSTANT, 
                              borderValue=(0, 0, 0))  # Black border for detection
    
    # Now crop to the largest rectangle inside the rotated image
    # This removes the triangular artifacts from rotation
    cropped = crop_to_content_rectangle(rotated)
    
    return cropped


def crop_to_content_rectangle(image):
    """
    After rotation, crop to the largest axis-aligned rectangle 
    containing only photo content (no black rotation artifacts).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Find the content (non-black areas)
    _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Get the largest contour
    largest = max(contours, key=cv2.contourArea)
    
    # Get bounding rect
    x, y, cw, ch = cv2.boundingRect(largest)
    
    # Now find the largest rectangle INSIDE the content
    # We need to avoid the black triangular corners from rotation
    
    # Create a mask of the content
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    
    # Find the largest inscribed rectangle
    # Start from bounding rect and shrink until all corners are inside content
    x1, y1, x2, y2 = x, y, x + cw, y + ch
    
    # Shrink from each side until we're fully inside the photo
    margin = 5
    
    # Check corners and shrink iteratively
    for _ in range(50):  # Max iterations
        corners_ok = True
        
        # Check if corners are in content (not black)
        test_points = [
            (x1 + margin, y1 + margin),
            (x2 - margin, y1 + margin),
            (x1 + margin, y2 - margin),
            (x2 - margin, y2 - margin),
        ]
        
        for px, py in test_points:
            px = max(0, min(w-1, int(px)))
            py = max(0, min(h-1, int(py)))
            if mask[py, px] == 0:
                corners_ok = False
                break
        
        if corners_ok:
            break
        
        # Shrink all sides slightly
        x1 += 3
        y1 += 3
        x2 -= 3
        y2 -= 3
        
        if x2 - x1 < 100 or y2 - y1 < 100:
            # Don't shrink too much
            x1, y1, x2, y2 = x, y, x + cw, y + ch
            break
    
    # Ensure bounds are valid
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    
    if x2 - x1 < 100 or y2 - y1 < 100:
        return image
    
    return image[y1:y2, x1:x2]


def detect_border_angle(gray):
    """
    Detect angle by analyzing the borders of the image.
    Looks for the transition from dark border to photo content.
    """
    h, w = gray.shape
    
    # Sample the edges to find where the photo content starts
    # This works well when there's still some dark border visible
    
    edges = []
    
    # Top edge: scan down to find first bright row
    for x in range(w // 4, 3 * w // 4, w // 20):
        for y in range(min(h // 4, 100)):
            if gray[y, x] > 60:
                edges.append((x, y, 'top'))
                break
    
    # Bottom edge: scan up
    for x in range(w // 4, 3 * w // 4, w // 20):
        for y in range(h - 1, max(h - h // 4, h - 100), -1):
            if gray[y, x] > 60:
                edges.append((x, y, 'bottom'))
                break
    
    # Left edge: scan right
    for y in range(h // 4, 3 * h // 4, h // 20):
        for x in range(min(w // 4, 100)):
            if gray[y, x] > 60:
                edges.append((x, y, 'left'))
                break
    
    # Right edge: scan left
    for y in range(h // 4, 3 * h // 4, h // 20):
        for x in range(w - 1, max(w - w // 4, w - 100), -1):
            if gray[y, x] > 60:
                edges.append((x, y, 'right'))
                break
    
    # Fit lines to top/bottom edges
    angles = []
    
    top_points = [(e[0], e[1]) for e in edges if e[2] == 'top']
    if len(top_points) >= 3:
        xs = [p[0] for p in top_points]
        ys = [p[1] for p in top_points]
        if max(xs) - min(xs) > 50:  # Need spread
            slope, _ = np.polyfit(xs, ys, 1)
            angles.append(math.degrees(math.atan(slope)))
    
    bottom_points = [(e[0], e[1]) for e in edges if e[2] == 'bottom']
    if len(bottom_points) >= 3:
        xs = [p[0] for p in bottom_points]
        ys = [p[1] for p in bottom_points]
        if max(xs) - min(xs) > 50:
            slope, _ = np.polyfit(xs, ys, 1)
            angles.append(math.degrees(math.atan(slope)))
    
    left_points = [(e[0], e[1]) for e in edges if e[2] == 'left']
    if len(left_points) >= 3:
        xs = [p[0] for p in left_points]
        ys = [p[1] for p in left_points]
        if max(ys) - min(ys) > 50:
            # For vertical lines, swap x and y
            slope, _ = np.polyfit(ys, xs, 1)
            angle = math.degrees(math.atan(slope))
            angles.append(angle)
    
    right_points = [(e[0], e[1]) for e in edges if e[2] == 'right']
    if len(right_points) >= 3:
        xs = [p[0] for p in right_points]
        ys = [p[1] for p in right_points]
        if max(ys) - min(ys) > 50:
            slope, _ = np.polyfit(ys, xs, 1)
            angle = math.degrees(math.atan(slope))
            angles.append(angle)
    
    if not angles:
        return None
    
    return np.median(angles)


def detect_gradient_angle(gray):
    """
    Detect angle using image gradients - good for subtle tilts.
    """
    h, w = gray.shape
    
    # Compute gradients
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    
    # Find strong gradient points (edges)
    magnitude = np.sqrt(gx**2 + gy**2)
    threshold = np.percentile(magnitude, 90)
    
    strong_points = magnitude > threshold
    
    # Get angles at strong gradient points
    angles = np.arctan2(gy[strong_points], gx[strong_points])
    angles = np.degrees(angles)
    
    # We want horizontal/vertical lines, so look for angles near 0, 90, -90, 180
    # Normalize to -45 to 45
    normalized = []
    for a in angles:
        while a > 45:
            a -= 90
        while a < -45:
            a += 90
        normalized.append(a)
    
    if not normalized:
        return None
    
    # Use median
    return np.median(normalized)


# ============================================================
# MAIN PROCESSING
# ============================================================

def upscale_photo(image):
    """Upscale photo 2x using Real-ESRGAN AI with tiled processing."""
    if UPSCALER is None:
        return image
    
    try:
        import torch
        
        h, w = image.shape[:2]
        tile_size = UPSCALER_TILE_SIZE
        tile_pad = 16  # Overlap for seamless blending
        
        # Output image (2x size)
        scale = UPSCALER_SCALE
        output = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
        
        # Process in tiles to save VRAM
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                # Calculate tile bounds with padding
                y1 = max(0, y - tile_pad)
                x1 = max(0, x - tile_pad)
                y2 = min(h, y + tile_size + tile_pad)
                x2 = min(w, x + tile_size + tile_pad)
                
                # Extract tile
                tile = image[y1:y2, x1:x2]
                
                # Convert BGR to RGB, then to tensor
                tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                tile_tensor = torch.from_numpy(tile_rgb).permute(2, 0, 1).float() / 255.0
                tile_tensor = tile_tensor.unsqueeze(0).to(UPSCALER_DEVICE)
                
                # Half precision on GPU
                if UPSCALER_DEVICE.type == 'cuda':
                    tile_tensor = tile_tensor.half()
                
                # Upscale (GPU operation - serialize for VRAM safety)
                with GPU_LOCK:
                    with torch.no_grad():
                        upscaled_tensor = UPSCALER(tile_tensor)
                
                # Convert back to numpy
                upscaled = upscaled_tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
                upscaled = (upscaled * 255).clip(0, 255).astype(np.uint8)
                upscaled = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
                
                # Calculate output region (remove padding from result)
                pad_y = (y - y1) * scale
                pad_x = (x - x1) * scale
                out_h = min(tile_size, h - y) * scale
                out_w = min(tile_size, w - x) * scale
                
                # Copy to output (avoiding padding region)
                output[y*scale:y*scale+out_h, x*scale:x*scale+out_w] = \
                    upscaled[pad_y:pad_y+out_h, pad_x:pad_x+out_w]
        
        return output
        
    except Exception as e:
        # Fall back to original if upscaling fails
        print(f"      Upscale error: {e}")
        return image


def save_photo(image, path, do_upscale=True):
    """Save photo with optional AI upscaling and auto-contrast."""
    # Upscale if enabled
    if do_upscale and UPSCALER is not None:
        image = upscale_photo(image)
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    pil_img = ImageOps.autocontrast(pil_img, cutoff=0.3)
    pil_img.save(path, quality=95)


def process_single_scan(args):
    """Process a single scan file. Used for parallel processing."""
    filepath, out_folder = args
    filename = os.path.basename(filepath)
    basename = os.path.splitext(filename)[0]
    
    # Load image
    image = cv2.imread(filepath)
    if image is None:
        return {"status": "error", "file": filename, "msg": "Could not load", "photos": 0, "times": []}
    
    # STEP 1: Find and split photos
    photos = find_photos(image)
    
    if not photos:
        return {"status": "skip", "file": filename, "msg": "No photos found", "photos": 0, "times": []}
    
    photo_count = 0
    log_lines = []
    photo_times = []
    
    # Process each photo
    for i, photo in enumerate(photos, 1):
        photo_start = time.time()
        
        # STEP 2: Deskew (straighten if tilted)
        photo = deskew(photo)
        
        # STEP 3: Find best rotation (faces upright)
        best_angle, scores = find_best_rotation(photo)
        
        # Apply rotation
        if best_angle != 0:
            photo = apply_rotation(photo, best_angle)
        
        # STEP 4: Save (with upscaling if enabled)
        out_path = os.path.join(out_folder, f"{basename}_p{i}.png")
        orig_h, orig_w = photo.shape[:2]
        save_photo(photo, out_path, do_upscale=UPSCALE_2X)
        photo_count += 1
        
        photo_time = time.time() - photo_start
        photo_times.append(photo_time)
        
        # Log result
        scores_str = " ".join([f"{a}°:{s[0]:.1f}" for a, s in scores.items()])
        rot_str = f"→{best_angle}°" if best_angle != 0 else "OK"
        upscale_str = f" →{orig_w*2}x{orig_h*2}" if UPSCALE_2X and UPSCALER else ""
        log_lines.append(f"  [{basename}_p{i}] {orig_w}x{orig_h}{upscale_str} {rot_str} ({photo_time:.1f}s) [{scores_str}]")
    
    return {"status": "ok", "file": filename, "photos": photo_count, "logs": log_lines, "times": photo_times}


def process_all():
    """Main processing loop with parallel execution."""
    print("=" * 60)
    print("PROCESSING")
    print("=" * 60)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Workers: {PARALLEL_WORKERS}")
    
    if TEST_MODE:
        print(f"*** TEST MODE: {TEST_LIMIT} files ***")
    print()
    
    # Find all input files
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif', '*.bmp')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(INPUT_DIR, '**', ext), recursive=True))
    files.sort()
    
    if TEST_MODE:
        files = files[:TEST_LIMIT]
    
    print(f"Found {len(files)} input files.\n")
    
    if not files:
        return
    
    start_time = time.time()
    total_photos = 0
    skipped = 0
    errors = 0
    photo_times = []  # Track individual photo processing times
    
    # Prepare work items (filepath, output_folder)
    work_items = []
    for filepath in files:
        rel_path = os.path.relpath(os.path.dirname(filepath), INPUT_DIR)
        out_folder = os.path.join(OUTPUT_DIR, rel_path)
        os.makedirs(out_folder, exist_ok=True)
        work_items.append((filepath, out_folder))
    
    # Process in parallel
    pbar = tqdm(total=len(work_items), unit="scan")
    
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {executor.submit(process_single_scan, item): item for item in work_items}
        
        for future in as_completed(futures):
            result = future.result()
            pbar.update(1)
            
            if result["status"] == "ok":
                total_photos += result["photos"]
                photo_times.extend(result.get("times", []))
                for line in result.get("logs", []):
                    pbar.write(line)
            elif result["status"] == "skip":
                skipped += 1
                pbar.write(f"  ⚠ {result['msg']}: {result['file']}")
            else:
                errors += 1
                pbar.write(f"  ✗ {result['msg']}: {result['file']}")
    
    pbar.close()
    
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Photos saved: {total_photos}")
    print(f"Scans skipped: {skipped}")
    if errors:
        print(f"Errors: {errors}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(files):.1f}s/scan)")
    if photo_times:
        avg_time = sum(photo_times) / len(photo_times)
        min_time = min(photo_times)
        max_time = max(photo_times)
        print(f"Photo times: avg {avg_time:.1f}s, min {min_time:.1f}s, max {max_time:.1f}s")


if __name__ == "__main__":
    process_all()
