"""
Flask Backend for Road Safety Detection System
===============================================
Serves the frontend UI and processes video/camera feed
"""

from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import time
import base64
import json
import os
import threading
from queue import Queue
from werkzeug.utils import secure_filename

# Import detection functions from live_detect
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==================== CONFIGURATION ====================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect", "yolo11_pothole_MWPD_finetuned", "weights", "best.pt")

# Detection Settings
# Detection Settings
# Detection Settings
# Detection Settings
CONF_THRESHOLD = 0.20           # Lowered for higher recall
IMG_SIZE = 640                  
ROAD_AREA_TOP_RATIO = 0.32      # Slightly lower for more road coverage
MIN_ASPECT_RATIO = 0.3          # Very lenient for irregular potholes
MAX_ASPECT_RATIO = 15.0          
MIN_BOX_AREA_FAR = 50           # Very small distant potholes
MAX_BOX_AREA_FAR = 8000         # Allow larger far detections
MIN_BOX_AREA_NEAR = 200         # Small near potholes
MAX_BOX_AREA_NEAR = 120000      # Very large close-up potholes

# Road surface segmentation parameters
ROAD_SATURATION_MAX = 80        
ROAD_VALUE_MIN = 35             
ROAD_VALUE_MAX = 220            
ROAD_OVERLAP_THRESHOLD = 0.10   # More lenient overlap
EDGE_DENSITY_MAX = 0.30         # More lenient edge density
SHADOW_TEXTURE_THRESHOLD = 35   # v5.2: Relaxed texture check
MIN_EDGE_DENSITY = 0.012        
MIN_PIXEL_SD = 10.0             # Increased from 8.0 for better shadow suppression
LVR_THRESHOLD = 1.15            # More lenient ratio

# Lane Detection Settings
LANE_WIDTH_M = 3.7
OFFSET_LIMIT = 0.7
WARNING_FRAMES = 25
LANE_SMOOTHING = 0.90
MIN_LANE_POINTS = 150           # Daytime minimum
NIGHT_MIN_LANE_POINTS = 40       # Extreme Mode: 40 (was 60)
STALE_FRAME_LIMIT = 30
MIN_BRIGHTNESS = 40
MIN_CONSECUTIVE_FRAMES = 3      # Daytime
NIGHT_MIN_CONSECUTIVE_FRAMES = 2 # Extreme Mode: 2 for flickering lanes

# Aggressive Adaptive Thresholds for Night
NIGHT_WHITE_THRESHOLD = 110      # Extreme low: 110 (down from 160)
NIGHT_YELLOW_S_THRESHOLD = 40    # Extreme low: 40 (down from 80)
SOBEL_THRESHOLD = (20, 100)      # X-gradient thresholds

# Global state
model = None
processing = False
active_cap = None          # Global camera/video capture handle
active_thread = None       # Global processing thread reference
frame_queue = Queue(maxsize=2)  # Smaller queue for less lag
stats = {
    'pothole_detected': False,
    'lane_status': 'Normal',
    'offset': 0.0
}
alerts_enabled = True

# ==================== LOAD MODEL ====================
def load_model():
    global model
    if model is None:
        print("Loading YOLO model...")
        model = YOLO(MODEL_PATH)
        print(f"Model loaded! Classes: {model.names}")
    return model

# ==================== DETECTION FUNCTIONS ====================

def enhance_low_light(img, low_light=False):
    # Aggressive Gamma Correction for extreme darkness
    if low_light:
        gamma = 0.5  # Boost dark areas
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, table)
        
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0 if low_light else 3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

def is_low_light(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    road_area = gray[h//2:, :]
    avg_brightness = np.mean(road_area)
    return avg_brightness < MIN_BRIGHTNESS, avg_brightness

def perspective_transform(img):
    h, w = img.shape[:2]
    src = np.float32([
        [w * 0.45, h * 0.63],
        [w * 0.55, h * 0.63],
        [w * 0.90, h * 0.95],
        [w * 0.10, h * 0.95]
    ])
    dst = np.float32([
        [w * 0.25, 0],
        [w * 0.75, 0],
        [w * 0.75, h],
        [w * 0.25, h]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped, Minv

def lane_binary(img, low_light=False):
    img = enhance_low_light(img, low_light=low_light)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    
    # 1. Color Thresholding
    if low_light:
        # HEADLIGHT MODE: Dynamic Luminosity Thresholding
        # Identify the brightest 5% of the road area as potential lane markings
        brightness_cutoff = np.percentile(l_channel, 95)
        # Ensure we don't pick up mid-grays in very dark scenes
        brightness_cutoff = max(brightness_cutoff, 110) 
        white = cv2.threshold(l_channel, brightness_cutoff, 255, cv2.THRESH_BINARY)[1]
        
        # Desaturated yellow detection (harder at night)
        yellow = cv2.inRange(hls, (15, 30, NIGHT_YELLOW_S_THRESHOLD), (40, 255, 255))
        color_binary = white | yellow
    else:
        white = cv2.inRange(hls, (0, 160, 0), (255, 255, 255))
        yellow = cv2.inRange(hls, (15, 40, 80), (40, 255, 255))
        color_binary = white | yellow
    
    # 2. Sobel Edge Detection
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    max_sobel = np.max(abs_sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / max_sobel) if max_sobel > 0 else abs_sobelx
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= SOBEL_THRESHOLD[0]) & (scaled_sobel <= SOBEL_THRESHOLD[1])] = 255
    
    # Combine
    binary = cv2.bitwise_or(color_binary, sobel_binary) if low_light else color_binary
    binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    h, w = binary.shape
    mask = np.zeros_like(binary)
    # Tighter top for warped perspective to avoid noise
    pts = np.array([[[int(w * 0.12), h], [int(w * 0.88), h],
                     [int(w * 0.65), int(h * 0.45)], [int(w * 0.35), int(h * 0.45)]]], np.int32)
    cv2.fillPoly(mask, pts, 255)
    return cv2.bitwise_and(binary, mask)

def sliding_window(binary, low_light=False):
    try:
        min_points = NIGHT_MIN_LANE_POINTS if low_light else MIN_LANE_POINTS
        histogram = np.sum(binary[binary.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        leftx = np.argmax(histogram[:midpoint])
        rightx = np.argmax(histogram[midpoint:]) + midpoint
        if leftx == 0 or rightx == midpoint:
            return None, None
        nwindows, margin, minpix = 9, 100, 60
        window_h = binary.shape[0] // nwindows
        nonzero = binary.nonzero()
        nz_y, nz_x = np.array(nonzero[0]), np.array(nonzero[1])
        if len(nz_y) < min_points * 2:
            return None, None
        left_inds, right_inds = [], []
        for win in range(nwindows):
            y_low = binary.shape[0] - (win + 1) * window_h
            y_high = binary.shape[0] - win * window_h
            lx_low, lx_high = leftx - margin, leftx + margin
            rx_low, rx_high = rightx - margin, rightx + margin
            good_left = ((nz_y >= y_low) & (nz_y < y_high) & (nz_x >= lx_low) & (nz_x < lx_high)).nonzero()[0]
            good_right = ((nz_y >= y_low) & (nz_y < y_high) & (nz_x >= rx_low) & (nz_x < rx_high)).nonzero()[0]
            left_inds.append(good_left)
            right_inds.append(good_right)
            if len(good_left) > minpix:
                leftx = int(np.mean(nz_x[good_left]))
            if len(good_right) > minpix:
                rightx = int(np.mean(nz_x[good_right]))
        left_inds = np.concatenate(left_inds)
        right_inds = np.concatenate(right_inds)
        left_fit = np.polyfit(nz_y[left_inds], nz_x[left_inds], 2) if len(left_inds) >= min_points else None
        right_fit = np.polyfit(nz_y[right_inds], nz_x[right_inds], 2) if len(right_inds) >= min_points else None
        return left_fit, right_fit
    except:
        return None, None

def validate_lane_fits(left_fit, right_fit, w, h, low_light=False):
    if left_fit is None or right_fit is None:
        return False
    y_eval = h - 1
    left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
    right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
    lane_width = right_x - left_x
    
    # Wider width range for noisy night detections
    min_w = 150 if low_light else 200
    max_w = w * 0.8 if low_light else w * 0.6
    
    if lane_width < min_w or lane_width > max_w:
        return False
    if left_x < -200 or right_x > w + 200: # More permissive off-screen room
        return False
        
    # Relaxation parameters for low light
    curve_limit = 0.008 if low_light else 0.003
    parallel_limit = 0.006 if low_light else 0.002
    
    # Check curvature
    if abs(left_fit[0]) > curve_limit or abs(right_fit[0]) > curve_limit:
        return False
    # Check parallelism
    if abs(left_fit[0] - right_fit[0]) > parallel_limit:
        return False
    return True

def smooth_fit(new_fit, old_fit, alpha=LANE_SMOOTHING):
    if old_fit is None:
        return new_fit
    if new_fit is None:
        return old_fit
    return alpha * old_fit + (1 - alpha) * new_fit

def draw_lane_overlay(frame, left_fit, right_fit, Minv):
    try:
        h, w = frame.shape[:2]
        ploty = np.linspace(0, h - 1, 50)
        leftx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        rightx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        leftx = np.clip(leftx, 0, w - 1)
        rightx = np.clip(rightx, 0, w - 1)
        lane_img = np.zeros_like(frame)
        pts_left = np.array([np.transpose(np.vstack([leftx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(lane_img, np.int32([pts]), (0, 200, 0))
        lane_warped = cv2.warpPerspective(lane_img, Minv, (w, h))
        lane_warped = cv2.GaussianBlur(lane_warped, (5, 5), 0)
        return cv2.addWeighted(frame, 1, lane_warped, 0.35, 0)
    except:
        return frame

def calculate_offset(left_fit, right_fit, w, y_eval=720):
    left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
    right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
    lane_center = (left_x + right_x) / 2
    xm_per_pix = LANE_WIDTH_M / (right_x - left_x) if (right_x - left_x) > 0 else 0.01
    return (w / 2 - lane_center) * xm_per_pix

def detect_road_mask(frame):
    """Create a binary mask of road-like regions."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    road_color_mask = cv2.inRange(
        hsv, (0, 0, ROAD_VALUE_MIN), (180, ROAD_SATURATION_MAX, ROAD_VALUE_MAX)
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = cv2.blur(edges.astype(np.float32), (31, 31)) / 255.0
    smooth_mask = (edge_density < EDGE_DENSITY_MAX).astype(np.uint8) * 255
    road_mask = cv2.bitwise_and(road_color_mask, smooth_mask)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_pts = np.array([[
        [int(w * 0.05), h], [int(w * 0.95), h],
        [int(w * 0.70), int(h * 0.35)], [int(w * 0.30), int(h * 0.35)]
    ]], np.int32)
    cv2.fillPoly(roi_mask, roi_pts, 255)
    road_mask = cv2.bitwise_and(road_mask, roi_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return road_mask

def is_on_road(x1, y1, x2, y2, road_mask):
    """Check if a bounding box overlaps sufficiently with the road mask."""
    h, w = road_mask.shape[:2]
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return False
    bbox_road = road_mask[y1:y2, x1:x2]
    if bbox_road.size == 0:
        return False
    return np.count_nonzero(bbox_road) / bbox_road.size >= ROAD_OVERLAP_THRESHOLD

def is_in_road_roi(cx, cy, h, w):
    """Check if center point is within the narrowed road ROI but wide enough for curves/speed."""
    top_y = h * ROAD_AREA_TOP_RATIO
    if cy < top_y:
        return False
        
    # EXTREME MODE: Centered 90% bottom, 40% top (widened)
    # Left: (0.05*w, h) to (0.30*w, top_y)
    # Right: (0.95*w, h) to (0.70*w, top_y)
    
    left_x = (0.05 * w) + (cy - h) * (0.30 * w - 0.05 * w) / (top_y - h)
    right_x = (0.95 * w) + (cy - h) * (0.70 * w - 0.95 * w) / (top_y - h)
    
    return left_x <= cx <= right_x

def is_valid_pothole(x1, y1, x2, y2, h, w, frame, road_mask=None):
    """Refined filter: Geometry + ROI + Mild Shadow Suppress."""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    top_y = h * ROAD_AREA_TOP_RATIO
    
    # 1. Horizon & ROI Check
    if cy < top_y or not is_in_road_roi(cx, cy, h, w):
        return False
        
    # 2. Geometric Filter
    width, height = x2 - x1, y2 - y1
    if height <= 0 or width <= 0:
        return False
    
    area = width * height
    depth_factor = (cy - top_y) / (h - top_y) 
    dynamic_min_area = MIN_BOX_AREA_FAR + (MIN_BOX_AREA_NEAR - MIN_BOX_AREA_FAR) * (depth_factor ** 2)
    dynamic_max_area = MAX_BOX_AREA_FAR + (MAX_BOX_AREA_NEAR - MAX_BOX_AREA_FAR) * (depth_factor ** 2)
    
    if area < dynamic_min_area or area > dynamic_max_area:
        return False

    aspect_ratio = width / height
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return False
            
    # 3. Mild Shadow/Texture Filter
    # Shadows are smooth, potholes are rough.
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False
        
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Texture Check: Edge density (shadows are too smooth)
    edges = cv2.Canny(gray_roi, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size
    if edge_density < MIN_EDGE_DENSITY:
        return False
        
    # Contrast Check: Pixel SD (shadows have very low internal variance)
    if np.std(gray_roi) < MIN_PIXEL_SD:
        return False
        
    return True


def compute_iou(boxA, boxB):
    """Compute Intersection over Union between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)

# ==================== PROCESSING THREAD ====================
def process_video(source):
    global processing, stats, active_cap
    
    load_model()
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error opening video source: {source}")
        processing = False
        active_cap = None
        return
    
    active_cap = cap  # Store globally so stop_detection can release it
    
    # Camera-specific optimizations
    is_camera = isinstance(source, int)
    if is_camera:
        # Try to minimize buffer at the OS/driver level
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        consecutive_fails = 0
        MAX_CONSECUTIVE_FAILS = 30  
        
        print(f"Camera {source}: connecting and draining buffer...")
        connection_verified = False
        
        # Aggressively drain the buffer (iVCam can have a deep one)
        for i in range(30):
            ret, _ = cap.read()
            if ret:
                connection_verified = True
                if i % 10 == 0:
                    status_msg = f"Connecting... Draining buffer ({i}/30)"
                    print(f"Camera {source}: {status_msg}")
                    # Send status to frontend
                    frame_queue.put({'statusMessage': status_msg})
            else:
                time.sleep(0.05)
        
        if not connection_verified:
            err_msg = "Failed to receive frames. Check iVCam app."
            print(f"Camera {source}: {err_msg}")
            frame_queue.put({'errorMessage': err_msg})
            processing = False
            active_cap = None
            cap.release()
            return
            
        print(f"Camera {source}: connected and buffer drained!")
        frame_queue.put({'statusMessage': 'Connected! Starting detection...'})
    
    prev_left_fit, prev_right_fit = None, None
    stale_counter = 0
    consecutive_valid_frames = 0
    lane_warning_counter = 0
    pothole_total = 0
    confidence_scores = []
    # Robust tracker: each entry is a dict with bbox, hit history, confirmation status
    tracked_potholes = []
    IOU_MATCH_THRESHOLD = 0.2       # More lenient IoU for matching (helps with fast motion)
    # EXTREME MODE: Wider matching for high speed displacement
    DIST_MATCH_BASE = 140           
    DIST_MATCH_SCALE = 240          
    CONFIRM_HITS = 1                # Single detection is enough to confirm (helps fast videos)
    CONFIRM_WINDOW = 6              # Wider window to catch fast-moving potholes
    STALE_FRAMES = 8                # Finalize quickly
    frame_count = 0
    road_mask = None                # Cached road mask
    last_frame_detections = []      # Persist detections across skip frames
    raw_detections = 0
    filtered_detections = 0
    
    while processing and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if is_camera:
                consecutive_fails += 1
                if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                    print("Camera: too many consecutive read failures, stopping.")
                    break
                time.sleep(0.05)  # Brief pause before retry
                continue
            else:
                break  # Video ended — session summary will fire
        
        if is_camera:
            consecutive_fails = 0  # Reset on successful read
        
        frame_count += 1
        h, w = frame.shape[:2]
        output = frame.copy()
        pothole_detected = False
        lane_departure = False
        offset = 0.0
        global alerts_enabled
        
        # Check for low light conditions early (needed for both pothole and lane logic)
        too_dark, brightness = is_low_light(frame)
        
        # Remove stale tracked potholes (left the frame) — count confirmed ones as they exit
        still_active = []
        for p in tracked_potholes:
            if (frame_count - p['last_seen']) >= STALE_FRAMES:
                # Pothole has left the frame — count it if confirmed
                if p['confirmed']:
                    pothole_total += 1
                    confidence_scores.append(p['conf_max'])
            else:
                still_active.append(p)
        tracked_potholes = still_active
        # Road segmentation disabled — not needed with simplified filter
        
        # Pothole detection — skip every 3 frames for speed, persist detections for stable drawing
        if frame_count % 3 == 0:
            current_conf = CONF_THRESHOLD if not too_dark else (CONF_THRESHOLD - 0.05)

            results = model.predict(frame, imgsz=416, conf=current_conf, device='cpu', verbose=False)
            
            raw_detections = 0
            filtered_detections = 0
            last_frame_detections = []  # Reset for this detection frame
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names.get(cls_id, 'unknown')
                    if 'pothole' not in cls_name.lower() and cls_id != 0:
                        continue
                    
                    raw_detections += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                
                    if is_valid_pothole(x1, y1, x2, y2, h, w, frame, road_mask=None):
                        filtered_detections += 1
                        pothole_detected = True
                        det_bbox = [x1, y1, x2, y2]
                        last_frame_detections.append((x1, y1, x2, y2, conf))
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        # --- IoU + adaptive distance matching ---
                        best_idx = -1
                        best_iou = 0.0
                        for i, p in enumerate(tracked_potholes):
                            iou = compute_iou(det_bbox, p['bbox'])
                            if iou > best_iou:
                                best_iou = iou
                                best_idx = i
                        
                        if best_iou < IOU_MATCH_THRESHOLD:
                            best_idx = -1
                            best_dist = float('inf')
                            depth_scale = cy / h
                            dist_thresh = DIST_MATCH_BASE + DIST_MATCH_SCALE * depth_scale
                            for i, p in enumerate(tracked_potholes):
                                pcx = (p['bbox'][0] + p['bbox'][2]) // 2
                                pcy = (p['bbox'][1] + p['bbox'][3]) // 2
                                dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
                                if dist < best_dist:
                                    best_dist = dist
                                    if dist < dist_thresh:
                                        best_idx = i
                        
                        if best_idx >= 0:
                            tp = tracked_potholes[best_idx]
                            tp['bbox'] = det_bbox
                            tp['last_seen'] = frame_count
                            tp['hits'].append(frame_count)
                            tp['hits'] = [f for f in tp['hits'] if frame_count - f < CONFIRM_WINDOW]
                            if conf > tp['conf_max']:
                                tp['conf_max'] = conf
                            if not tp['confirmed'] and len(tp['hits']) >= CONFIRM_HITS:
                                tp['confirmed'] = True
                        else:
                            tracked_potholes.append({
                                'bbox': det_bbox,
                                'first_seen': frame_count,
                                'last_seen': frame_count,
                                'hits': [frame_count],
                                'confirmed': False,
                                'conf_max': conf,
                            })
        else:
            # On skip frames, reuse last detections for stable display
            if last_frame_detections:
                pothole_detected = True
        
        # Draw detections (persisted across skip frames — no blinking)
        for (dx1, dy1, dx2, dy2, dconf) in last_frame_detections:
            cv2.rectangle(output, (dx1, dy1), (dx2, dy2), (0, 0, 255), 3)
            cv2.putText(output, f"POTHOLE {dconf:.2f}", (dx1, dy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        
        # Lane detection
        if too_dark:
            cv2.putText(output, f"LOW LIGHT (Brightness: {brightness:.0f})",
                        (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(output, "ADAPTIVE MODE ACTIVE",
                        (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        try:
            warped, Minv = perspective_transform(frame)
            binary = lane_binary(warped, low_light=too_dark)
            new_left_fit, new_right_fit = sliding_window(binary, low_light=too_dark)
            
            if validate_lane_fits(new_left_fit, new_right_fit, w, h, low_light=too_dark):
                stale_counter = 0
                consecutive_valid_frames += 1

                left_fit = smooth_fit(new_left_fit, prev_left_fit)
                right_fit = smooth_fit(new_right_fit, prev_right_fit)
                prev_left_fit, prev_right_fit = left_fit, right_fit
            elif prev_left_fit is not None:
                stale_counter += 1
                if stale_counter > STALE_FRAME_LIMIT:
                    prev_left_fit, prev_right_fit = None, None
                    left_fit, right_fit = None, None
                    consecutive_valid_frames = 0
                else:
                    left_fit, right_fit = prev_left_fit, prev_right_fit
            else:
                left_fit, right_fit = None, None
                consecutive_valid_frames = 0
            
            # EXTREME MODE: Reduced requirement for flickering night lanes
            min_valid = NIGHT_MIN_CONSECUTIVE_FRAMES if too_dark else MIN_CONSECUTIVE_FRAMES
            if consecutive_valid_frames >= min_valid and left_fit is not None and right_fit is not None:
                output = draw_lane_overlay(output, left_fit, right_fit, Minv)
                offset = calculate_offset(left_fit, right_fit, w)
                if abs(offset) > OFFSET_LIMIT:
                    lane_warning_counter += 1
                else:
                    lane_warning_counter = max(0, lane_warning_counter - 1)
                if lane_warning_counter > WARNING_FRAMES:
                    lane_departure = True
                cv2.putText(output, f"Offset: {offset:.2f}m", (20, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                            (0, 0, 255) if lane_departure else (0, 255, 255), 2)
            else:
                 lane_warning_counter = 0 # Reset warning if lane is lost or unstable

        except:
            pass
        
        # Alerts
        if alerts_enabled:
            if pothole_detected:
                cv2.putText(output, "! POTHOLE ALERT !", (w // 2 - 150, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if lane_departure:
                cv2.putText(output, "! LANE DEPARTURE WARNING !", (w // 2 - 220, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        
        # Show unique pothole count on video
        cv2.putText(output, f"Potholes: {pothole_total}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Debug logging every 30 frames
        if frame_count % 30 == 0:
            print(f"[Frame {frame_count}] Raw: {raw_detections} | Filtered: {filtered_detections} | Tracked: {len(tracked_potholes)} | Confirmed: {pothole_total}")
        
        # Update stats
        stats['pothole_detected'] = pothole_detected
        stats['lane_status'] = 'DEPARTURE' if lane_departure else 'Normal'
        stats['offset'] = offset
        
        # Resize for faster encoding
        display = cv2.resize(output, (854, 480))  # 480p for faster streaming
        
        # Encode frame - balanced quality/speed
        _, buffer = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame_data = base64.b64encode(buffer).decode('utf-8')
        
        # Clear old frames if queue is backing up
        while frame_queue.full():
            try:
                frame_queue.get_nowait()
            except:
                break
        
        # Put new frame
        frame_queue.put({
            'frame': frame_data,
            'pothole_detected': pothole_detected,
            'pothole_count': pothole_total,
            'lane_status': stats['lane_status'],
            'offset': offset
        })
        
        # Minimal delay - let GPU/CPU run at full speed
    
    cap.release()
    active_cap = None
    
    # Count any remaining confirmed tracked potholes still in frame at video end
    for p in tracked_potholes:
        if p['confirmed']:
            pothole_total += 1
            confidence_scores.append(p['conf_max'])
    
    # Session summary stats
    summary = {
        'total_potholes': pothole_total,
        'avg_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
        'min_confidence': min(confidence_scores) if confidence_scores else 0,
        'max_confidence': max(confidence_scores) if confidence_scores else 0,
    }
    stats['session_summary'] = summary
    
    # Send summary as final message
    frame_queue.put({'session_complete': True, 'summary': summary})
    
    print(f"\n{'='*50}")
    print("         DETECTION SESSION SUMMARY")
    print(f"{'='*50}")
    print(f"  Total Potholes Detected : {pothole_total}")
    if confidence_scores:
        print(f"  Average Confidence      : {summary['avg_confidence']:.2%}")
        print(f"  Min Confidence          : {summary['min_confidence']:.2%}")
        print(f"  Max Confidence          : {summary['max_confidence']:.2%}")
    else:
        print(f"  No potholes were detected.")
    print(f"{'='*50}\n")
    
    processing = False

# ==================== ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_detection():
    global processing, active_cap, active_thread
    
    # Force stop any existing processing first (fixes re-open bug)
    if processing:
        processing = False
    
    # Wait for old thread to finish and release camera
    if active_thread and active_thread.is_alive():
        print("Stopping existing detection thread...")
        active_thread.join(timeout=3.0)  # Wait longer
        if active_thread.is_alive():
            print("Warning: Old thread did not terminate in time. Proceeding anyway.")
    
    # Force-release camera if still held (extra safety)
    if active_cap is not None:
        print("Force-releasing locked camera capture...")
        try:
            active_cap.release()
        except Exception as e:
            print(f"Error releasing camera: {e}")
        active_cap = None
    
    # Clear queue completely
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except:
            pass
    
    processing = True
    
    # Reset stats
    stats['pothole_detected'] = False
    stats['lane_status'] = 'Normal'
    stats['offset'] = 0.0
    
    # Determine source
    if 'video' in request.files:
        # Video file upload
        video = request.files['video']
        filename = secure_filename(video.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(filepath)
        source = filepath
    else:
        # Camera mode
        data = request.get_json() or {}
        camera_index = data.get('camera_index', 0)
        source = camera_index
    
    # Start processing thread
    active_thread = threading.Thread(target=process_video, args=(source,), daemon=True)
    active_thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/api/stream')
def stream():
    def generate():
        while processing:
            if not frame_queue.empty():
                data = frame_queue.get()
                yield f"data: {json.dumps(data)}\n\n"
            else:
                time.sleep(0.01)
        # Drain remaining items (including session_complete summary)
        while not frame_queue.empty():
            data = frame_queue.get()
            yield f"data: {json.dumps(data)}\n\n"
        yield f"data: {json.dumps({'status': 'stopped'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    global processing, active_cap
    processing = False
    
    # Force-release camera immediately so it's not locked
    if active_cap is not None:
        try:
            active_cap.release()
        except:
            pass
        active_cap = None
        
    return jsonify({'status': 'stopped'})

@app.route('/api/toggle_alerts', methods=['POST'])
def toggle_alerts():
    global alerts_enabled
    data = request.get_json() or {}
    alerts_enabled = data.get('enabled', True)
    return jsonify({'alerts_enabled': alerts_enabled})

@app.route('/api/status')
def get_status():
    return jsonify({
        'processing': processing,
        'stats': stats
    })

# ==================== MAIN ====================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("Road Safety Detection System - Web Interface")
    print("="*50)
    print("Starting server at http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
