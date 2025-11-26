"""
Photo Scanner Processing Script
================================
- Splits scanned images containing multiple photos
- Crops out black/dark grey scanner background
- Auto-rotates based on face detection
- GPU accelerated using DirectML (works with NVIDIA RTX 3060, 2080Ti, etc.)

Author: Photo Scanner Assistant
"""

import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import time

# --- CONFIGURATION ---
INPUT_DIR = r"D:\Scans\Raw_Input"
OUTPUT_DIR = r"D:\Scans\Processed_Library"
BG_THRESHOLD = 50          # Threshold for black background detection (lower = darker only)
MIN_PHOTO_AREA = 50000     # Minimum area to be considered a photo (filters out dust/artifacts)
MIN_PHOTO_DIMENSION = 400  # Minimum width/height in pixels for a valid photo (standard prints are large)
MAX_WHITE_RATIO = 0.80     # Maximum ratio of white/near-white pixels (filters white corners)
MIN_COLOR_STD = 20         # Minimum color standard deviation (filters blank/uniform regions)
MIN_ASPECT_RATIO = 0.4     # Minimum aspect ratio (width/height or height/width) - standard photos are ~0.67
MAX_ASPECT_RATIO = 2.5     # Maximum aspect ratio - panoramas excluded, standard prints only
USE_GPU = True             # Enable GPU acceleration via DirectML
NUM_WORKERS = 4            # Parallel processing threads for I/O
DETECTION_CONFIDENCE = 0.3 # Face detection confidence (lower = more faces, more false positives)
MAX_DETECTION_SIZE = 1500  # Max image dimension for face detection (larger = slower but better)

# Standard photo print sizes at 300 DPI (width x height in pixels)
# Photos will be resized to fit the closest matching standard size
STANDARD_PHOTO_SIZES = [
    (1200, 1800),  # 4x6 inches
    (1500, 2100),  # 5x7 inches
    (1800, 2400),  # 6x8 inches
    (2400, 3000),  # 8x10 inches
]
# ---------------------

print("=" * 60)
print("PHOTO SCANNER PROCESSING SCRIPT")
print("=" * 60)

# === GPU / ONNX Runtime Setup ===
ONNX_SESSION = None
ONNX_GPU_AVAILABLE = False
GPU_PROVIDER = None
ort = None

try:
    import onnxruntime as ort

    # Check available providers
    providers = ort.get_available_providers()
    print(f"ONNX Runtime providers: {providers}")

    if 'DmlExecutionProvider' in providers and USE_GPU:
        ONNX_GPU_AVAILABLE = True
        GPU_PROVIDER = 'DmlExecutionProvider'
        print("✓ DirectML GPU acceleration enabled")
    elif 'CUDAExecutionProvider' in providers and USE_GPU:
        ONNX_GPU_AVAILABLE = True
        GPU_PROVIDER = 'CUDAExecutionProvider'
        print("✓ CUDA GPU acceleration enabled")
    else:
        print("⚠ Running on CPU (no GPU provider available)")
except ImportError:
    print("⚠ ONNX Runtime not installed, using OpenCV only")

# --- MODEL PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# YuNet face detector - most accurate, has facial landmarks
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

# Ultra-Light face detector (320 version - GPU accelerated via ONNX)
ULTRAFACE_MODEL_PATH = os.path.join(MODELS_DIR, "version-RFB-320.onnx")
ULTRAFACE_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.onnx"

# SSD ResNet face detector (fallback)
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

# ONNX Runtime GPU Face Detector (Primary - fastest with GPU)
ONNX_FACE_SESSION = None
if ort is not None:
    if download_model(ULTRAFACE_URL, ULTRAFACE_MODEL_PATH, "Ultra-Light Face Detector"):
        try:
            if ONNX_GPU_AVAILABLE and GPU_PROVIDER:
                providers = [GPU_PROVIDER, 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_mem_pattern = True

            ONNX_FACE_SESSION = ort.InferenceSession(
                ULTRAFACE_MODEL_PATH,
                sess_options=sess_options,
                providers=providers
            )
            actual_provider = ONNX_FACE_SESSION.get_providers()[0]
            if 'Dml' in actual_provider or 'CUDA' in actual_provider:
                print(f"  ✓ ONNX Face Detector loaded (GPU: {actual_provider})")
            else:
                print(f"  ✓ ONNX Face Detector loaded ({actual_provider})")
        except Exception as e:
            print(f"  ✗ Could not load ONNX face detector: {e}")

# YuNet via OpenCV (Secondary - most accurate for orientation)
YUNET_DETECTOR = None
if download_model(YUNET_URL, YUNET_MODEL_PATH, "YuNet Face Detector"):
    try:
        YUNET_DETECTOR = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH,
            "",
            (320, 320),
            0.5,  # Score threshold
            0.3,  # NMS threshold
            5000  # Top K
        )
        print("  ✓ YuNet Face Detector loaded")
    except Exception as e:
        print(f"  ✗ Could not load YuNet: {e}")

# SSD ResNet DNN detector (Tertiary)
DNN_FACE_DETECTOR = None
if download_model(DNN_PROTO_URL, DNN_PROTO_PATH, "SSD prototxt"):
    if download_model(DNN_MODEL_URL, DNN_MODEL_PATH, "SSD model"):
        try:
            DNN_FACE_DETECTOR = cv2.dnn.readNetFromCaffe(DNN_PROTO_PATH, DNN_MODEL_PATH)
            print("  ✓ SSD Face Detector loaded")
        except Exception as e:
            print(f"  ✗ Could not load SSD detector: {e}")

# Haar cascades (fast fallback)
FACE_CASCADE = None
FACE_CASCADE_ALT = None
EYE_CASCADE = None
try:
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    FACE_CASCADE_ALT = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    print("  ✓ Haar Cascades loaded")
except:
    print("  ⚠ Could not load Haar Cascades")

print()


# ============================================================
# GEOMETRY UTILITIES
# ============================================================

def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect


def resize_to_standard_size(image):
    """Resize image to the closest standard photo print size.

    Finds the best matching standard size based on aspect ratio and area,
    then resizes the image to fit that size exactly (may crop slightly to match aspect ratio).
    """
    h, w = image.shape[:2]
    img_aspect = w / h  # Current aspect ratio
    img_area = w * h

    best_size = None
    best_score = float('inf')

    for std_w, std_h in STANDARD_PHOTO_SIZES:
        # Try both orientations (portrait and landscape)
        for sw, sh in [(std_w, std_h), (std_h, std_w)]:
            std_aspect = sw / sh
            std_area = sw * sh

            # Score based on aspect ratio difference and area similarity
            aspect_diff = abs(img_aspect - std_aspect)
            area_ratio = min(img_area, std_area) / max(img_area, std_area)

            # Lower score is better: prioritize aspect ratio match, then size match
            score = aspect_diff * 10 + (1 - area_ratio)

            if score < best_score:
                best_score = score
                best_size = (sw, sh)

    if best_size is None:
        return image  # Fallback: return original

    target_w, target_h = best_size
    target_aspect = target_w / target_h

    # Crop to match target aspect ratio, then resize
    if img_aspect > target_aspect:
        # Image is wider than target - crop width
        new_w = int(h * target_aspect)
        offset = (w - new_w) // 2
        cropped = image[:, offset:offset + new_w]
    elif img_aspect < target_aspect:
        # Image is taller than target - crop height
        new_h = int(w / target_aspect)
        offset = (h - new_h) // 2
        cropped = image[offset:offset + new_h, :]
    else:
        cropped = image

    # Resize to exact target size
    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    return resized


def simple_crop(image, contour):
    """Simple bounding box crop - no perspective transform, no zooming.

    Just crops the region containing the photo from the black background.
    Returns the cropped image exactly as it appears, no resizing.
    """
    # Get axis-aligned bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)

    # Ensure we're within image bounds
    img_h, img_w = image.shape[:2]
    x = max(0, x)
    y = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    # Extract the region - pure crop, no transforms
    cropped = image[y:y2, x:x2]

    # Verify we got something
    if cropped.size == 0:
        return None

    return cropped.copy()


def four_point_transform(image, pts):
    """Apply perspective transform to extract and straighten a region.

    Only used if photo is significantly rotated/skewed.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Check if the photo is significantly rotated (needs perspective correction)
    # Calculate angle of the top edge
    angle = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))

    # If angle is small (< 5 degrees), just use simple crop
    if abs(angle) < 5:
        x_coords = [tl[0], tr[0], br[0], bl[0]]
        y_coords = [tl[1], tr[1], br[1], bl[1]]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # Clamp to image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.shape[1], x_max)
        y_max = min(image.shape[0], y_max)

        return image[y_min:y_max, x_min:x_max].copy()

    # For rotated photos, use perspective transform
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def is_valid_photo(image):
    """Check if cropped region is a real photo vs white corner/blank area.

    Returns (is_valid, reason) tuple.
    Standard photo prints have aspect ratios around:
    - 4x6: 1.5
    - 5x7: 1.4
    - 8x10: 1.25
    - Square: 1.0
    """
    h, w = image.shape[:2]

    # Check minimum dimensions - real photos are fairly large when scanned
    if w < MIN_PHOTO_DIMENSION or h < MIN_PHOTO_DIMENSION:
        return False, f"too small ({w}x{h})"

    # Check aspect ratio - standard photos are between 1:1 and about 1:2
    # Reject anything too skinny (likely an edge artifact)
    aspect = min(w, h) / max(w, h)  # This gives 0-1 range, 1=square
    if aspect < MIN_ASPECT_RATIO:
        return False, f"too skinny (aspect={aspect:.2f})"

    long_side = max(w, h) / min(w, h)
    if long_side > MAX_ASPECT_RATIO:
        return False, f"too elongated (ratio={long_side:.2f})"

    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Check for mostly white/bright pixels (white scanner corners)
    white_threshold = 240
    white_pixels = np.sum(gray > white_threshold)
    total_pixels = gray.size
    white_ratio = white_pixels / total_pixels

    if white_ratio > MAX_WHITE_RATIO:
        return False, f"mostly white ({white_ratio:.0%})"

    # Check for mostly black pixels (scanner artifacts)
    black_threshold = 30
    black_pixels = np.sum(gray < black_threshold)
    black_ratio = black_pixels / total_pixels

    if black_ratio > 0.85:
        return False, f"mostly black ({black_ratio:.0%})"

    # Check color variation (real photos have texture/detail)
    std_dev = np.std(gray)
    if std_dev < MIN_COLOR_STD:
        return False, f"too uniform (std={std_dev:.1f})"

    # Check that the image has content in the center (not just edge artifacts)
    center_crop = gray[h//4:3*h//4, w//4:3*w//4]
    if center_crop.size > 0:
        center_std = np.std(center_crop)
        center_mean = np.mean(center_crop)

        # Center should have some variation and not be all white
        if center_std < 10 and center_mean > 230:
            return False, "blank center"

    # Additional check: edges vs center brightness difference
    # White corners often have uniform brightness throughout
    edge_size = min(20, h//10, w//10)
    if edge_size > 5:
        top_edge = gray[:edge_size, :].mean()
        bottom_edge = gray[-edge_size:, :].mean()
        left_edge = gray[:, :edge_size].mean()
        right_edge = gray[:, -edge_size:].mean()
        center_mean = gray[h//3:2*h//3, w//3:2*w//3].mean()

        edge_avg = (top_edge + bottom_edge + left_edge + right_edge) / 4

        # Real photos usually have different edge vs center brightness
        # White corners have uniform high brightness
        if edge_avg > 235 and abs(edge_avg - center_mean) < 15 and center_mean > 220:
            return False, "uniform white region"

    return True, "valid"


# ============================================================
# FACE DETECTION METHODS
# ============================================================

def detect_faces_onnx_gpu(image, conf_threshold=0.5):
    """GPU-accelerated face detection using ONNX Runtime Ultra-Light model."""
    if ONNX_FACE_SESSION is None:
        return []

    h, w = image.shape[:2]
    input_size = (320, 240)  # Ultra-Light 320 model input size

    # Preprocess
    img_resized = cv2.resize(image, input_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = (img_rgb - 127.0) / 128.0
    img_transposed = img_normalized.transpose(2, 0, 1)  # HWC -> CHW
    img_batch = np.expand_dims(img_transposed, axis=0).astype(np.float32)

    # Inference on GPU
    input_name = ONNX_FACE_SESSION.get_inputs()[0].name
    confidences, boxes = ONNX_FACE_SESSION.run(None, {input_name: img_batch})

    results = []
    for i in range(boxes.shape[1]):
        face_conf = confidences[0, i, 1]

        if face_conf > conf_threshold:
            x1 = int(boxes[0, i, 0] * w)
            y1 = int(boxes[0, i, 1] * h)
            x2 = int(boxes[0, i, 2] * w)
            y2 = int(boxes[0, i, 3] * h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            fw, fh = x2 - x1, y2 - y1
            if fw > 10 and fh > 10:
                results.append({
                    'box': (x1, y1, fw, fh),
                    'confidence': float(face_conf),
                    'right_eye': None,
                    'left_eye': None
                })

    return apply_nms(results, 0.3)


def apply_nms(faces, iou_threshold=0.3):
    """Apply Non-Maximum Suppression to face detections."""
    if not faces:
        return []

    faces = sorted(faces, key=lambda f: f['confidence'], reverse=True)
    keep = []

    while faces:
        best = faces.pop(0)
        keep.append(best)

        x1, y1, w1, h1 = best['box']
        remaining = []

        for face in faces:
            x2, y2, w2, h2 = face['box']

            xi1, yi1 = max(x1, x2), max(y1, y2)
            xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)

            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            union_area = w1 * h1 + w2 * h2 - inter_area

            iou = inter_area / union_area if union_area > 0 else 0
            if iou < iou_threshold:
                remaining.append(face)

        faces = remaining

    return keep


def detect_faces_yunet(image, conf_threshold=0.5):
    """YuNet face detector - most accurate, includes facial landmarks."""
    if YUNET_DETECTOR is None:
        return []

    h, w = image.shape[:2]
    YUNET_DETECTOR.setInputSize((w, h))

    _, faces = YUNET_DETECTOR.detect(image)

    if faces is None:
        return []

    results = []
    for face in faces:
        x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        confidence = float(face[14]) if len(face) > 14 else float(face[-1])

        if confidence >= conf_threshold:
            # YuNet provides 5 facial landmarks:
            # [x,y,w,h, right_eye_x, right_eye_y, left_eye_x, left_eye_y,
            #  nose_x, nose_y, mouth_right_x, mouth_right_y, mouth_left_x, mouth_left_y, score]
            right_eye = (float(face[4]), float(face[5])) if len(face) > 5 else None
            left_eye = (float(face[6]), float(face[7])) if len(face) > 7 else None
            nose = (float(face[8]), float(face[9])) if len(face) > 9 else None
            mouth_right = (float(face[10]), float(face[11])) if len(face) > 11 else None
            mouth_left = (float(face[12]), float(face[13])) if len(face) > 13 else None

            results.append({
                'box': (x, y, fw, fh),
                'confidence': confidence,
                'right_eye': right_eye,
                'left_eye': left_eye,
                'nose': nose,
                'mouth_right': mouth_right,
                'mouth_left': mouth_left
            })

    return results


def detect_faces_dnn(image, conf_threshold=0.5):
    """SSD ResNet DNN face detector."""
    if DNN_FACE_DETECTOR is None:
        return []

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    DNN_FACE_DETECTOR.setInput(blob)
    detections = DNN_FACE_DETECTOR.forward()

    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            results.append({
                'box': (x1, y1, x2 - x1, y2 - y1),
                'confidence': float(confidence),
                'right_eye': None,
                'left_eye': None
            })

    return results


def detect_faces_haar(gray):
    """Haar cascade face detector (fast fallback)."""
    if FACE_CASCADE is None:
        return []

    all_faces = []

    for cascade, conf in [(FACE_CASCADE, 0.7), (FACE_CASCADE_ALT, 0.6)]:
        if cascade is None:
            continue
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            all_faces.append({
                'box': (x, y, w, h),
                'confidence': conf,
                'right_eye': None,
                'left_eye': None
            })

    return merge_face_detections(all_faces)


def merge_face_detections(faces, iou_threshold=0.4):
    """Merge overlapping face detections."""
    if not faces:
        return []

    faces = sorted(faces, key=lambda f: f['confidence'], reverse=True)
    merged = []
    used = [False] * len(faces)

    for i, face1 in enumerate(faces):
        if used[i]:
            continue

        used[i] = True
        x1, y1, w1, h1 = face1['box']
        best = face1.copy()

        for j, face2 in enumerate(faces):
            if used[j]:
                continue

            x2, y2, w2, h2 = face2['box']

            xi1, yi1 = max(x1, x2), max(y1, y2)
            xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)

            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            union_area = w1 * h1 + w2 * h2 - inter_area

            if union_area > 0 and inter_area / union_area > iou_threshold:
                used[j] = True
                if face2.get('right_eye') and not best.get('right_eye'):
                    best['right_eye'] = face2['right_eye']
                    best['left_eye'] = face2['left_eye']

        merged.append(best)

    return merged


# ============================================================
# ROTATION DETECTION
# ============================================================

def auto_rotate_by_faces(image):
    """Determine correct rotation by finding orientation with best face detection.

    Strategy: Use multiple face detectors and find the orientation where:
    1. We get the highest total confidence scores
    2. We detect the most faces (face detectors work best on upright faces)

    YuNet is trained on upright faces, so it will have HIGHER confidence
    on correctly oriented faces and LOWER or NO detection on rotated faces.
    """
    h, w = image.shape[:2]
    scale = min(1.0, MAX_DETECTION_SIZE / max(h, w))

    rotations = [
        (0, None),
        (90, cv2.ROTATE_90_CLOCKWISE),
        (180, cv2.ROTATE_180),
        (270, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]

    best_angle = 0
    best_score = -1
    best_face_count = 0

    for angle, flag in rotations:
        # Rotate image
        if flag is None:
            rotated = image
        else:
            rotated = cv2.rotate(image, flag)

        # Scale down for detection
        if scale < 1.0:
            small = cv2.resize(rotated, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = rotated

        # Use HIGHER confidence threshold for rotation detection
        # This makes the detector more selective and orientation-sensitive
        high_conf = max(0.7, DETECTION_CONFIDENCE + 0.4)

        # Detect with YuNet at high confidence
        faces_yunet = detect_faces_yunet(small, high_conf) if YUNET_DETECTOR else []

        # Also try DNN detector as backup
        faces_dnn = detect_faces_dnn(small, high_conf) if DNN_FACE_DETECTOR else []

        # Calculate score: sum of confidence values
        # Higher confidence = face detector is more sure = likely correct orientation
        yunet_score = sum(f['confidence'] for f in faces_yunet)
        dnn_score = sum(f['confidence'] for f in faces_dnn)

        # Combined score with YuNet weighted more (it's more accurate)
        total_score = yunet_score * 2.0 + dnn_score
        total_faces = len(faces_yunet) + len(faces_dnn)

        # Add bonus for more faces detected (correct orientation finds more)
        total_score += total_faces * 0.5

        if total_score > best_score:
            best_score = total_score
            best_angle = angle
            best_face_count = max(len(faces_yunet), len(faces_dnn))

    # If no faces detected at high confidence, try again with lower threshold
    if best_score <= 0:
        for angle, flag in rotations:
            if flag is None:
                rotated = image
            else:
                rotated = cv2.rotate(image, flag)

            if scale < 1.0:
                small = cv2.resize(rotated, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                small = rotated

            faces = detect_faces_yunet(small, DETECTION_CONFIDENCE) if YUNET_DETECTOR else []
            score = sum(f['confidence'] for f in faces)

            if score > best_score:
                best_score = score
                best_angle = angle
                best_face_count = len(faces)

    # Apply the best rotation
    if best_angle == 0:
        return image, best_face_count, 0
    elif best_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), best_face_count, 90
    elif best_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180), best_face_count, 180
    else:  # 270
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), best_face_count, 270


# ============================================================
# IMAGE SAVING
# ============================================================

def safe_save_image(cv2_img, save_path):
    """Save image with auto-contrast enhancement."""
    try:
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img = ImageOps.autocontrast(pil_img, cutoff=1)
        pil_img.save(save_path, quality=95)
        return True
    except Exception as e:
        print(f"    ✗ Error saving {save_path}: {e}")
        return False


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_workflow():
    """Main processing workflow with progress tracking."""
    print("=" * 60)
    print("STARTING PHOTO PROCESSING")
    print("=" * 60)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Find all input files
    img_exts = ('*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif', '*.bmp')
    files = []
    for ext in img_exts:
        files.extend(glob.glob(os.path.join(INPUT_DIR, '**', ext), recursive=True))

    total_files = len(files)
    print(f"Found {total_files} scan files to process.")
    print()

    if total_files == 0:
        print("No files to process!")
        return

    start_time = time.time()
    photos_extracted = 0
    files_processed = 0

    with tqdm(total=total_files, unit="scan", desc="Processing",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:

        for file_path in files:
            filename = os.path.basename(file_path)
            base_name = os.path.splitext(filename)[0]

            # Skip template files
            if "$(NNNN)" in filename:
                pbar.update(1)
                continue

            # Create output folder
            rel_path = os.path.relpath(os.path.dirname(file_path), INPUT_DIR)
            final_out_folder = os.path.join(OUTPUT_DIR, rel_path)
            os.makedirs(final_out_folder, exist_ok=True)

            # Check if already processed
            if os.path.exists(os.path.join(final_out_folder, f"{base_name}_p1.png")):
                pbar.set_postfix_str(f"Skip: {filename[:20]}")
                pbar.update(1)
                continue

            pbar.set_postfix_str(f"{filename[:25]}")

            # Load image
            img = cv2.imread(file_path)
            if img is None:
                pbar.write(f"  ✗ Could not load: {filename}")
                pbar.update(1)
                continue

            # Find photos in scan
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, BG_THRESHOLD, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            photo_count = 0
            skipped_regions = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > MIN_PHOTO_AREA:
                    # Simple crop - just extract the bounding box, no transforms
                    cropped_img = simple_crop(img, cnt)

                    # Validate this is a real photo, not a white corner or artifact
                    is_valid, reason = is_valid_photo(cropped_img)
                    if not is_valid:
                        skipped_regions += 1
                        continue

                    photo_count += 1

                    # Auto-rotate based on faces
                    oriented_img, faces_found, angle = auto_rotate_by_faces(cropped_img)

                    # Resize to standard photo size
                    final_img = resize_to_standard_size(oriented_img)
                    final_h, final_w = final_img.shape[:2]

                    # Log rotation and size info
                    size_str = f"{final_w}x{final_h}"
                    if angle != 0:
                        pbar.write(f"  [{base_name} p{photo_count}] Rotated {angle}° ({faces_found} faces) -> {size_str}")
                    elif faces_found > 0:
                        pbar.write(f"  [{base_name} p{photo_count}] No rotation ({faces_found} faces) -> {size_str}")
                    else:
                        pbar.write(f"  [{base_name} p{photo_count}] -> {size_str}")

                    # Save
                    final_path = os.path.join(final_out_folder, f"{base_name}_p{photo_count}.png")
                    if safe_save_image(final_img, final_path):
                        photos_extracted += 1

            # Log skipped regions if any
            if skipped_regions > 0:
                pbar.write(f"  [{base_name}] Filtered {skipped_regions} white corner(s)/artifact(s)")

            files_processed += 1
            pbar.update(1)

    # Summary
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Files processed:  {files_processed}")
    print(f"Photos extracted: {photos_extracted}")
    print(f"Total time:       {elapsed:.1f} seconds")
    print(f"Average:          {elapsed/max(files_processed,1):.2f} sec/file")
    print()


if __name__ == "__main__":
    process_workflow()
  
