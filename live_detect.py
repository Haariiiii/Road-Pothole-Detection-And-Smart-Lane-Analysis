"""
Road Pothole Detection & Smart Lane Deviation Analysis
======================================================
Final Year Project - Combined Detection System

Features:
1. YOLO-based pothole detection with false positive filtering
2. Lane detection with perspective transform
3. Lane departure warning system
4. Audio/visual alerts for hazards
"""

import cv2
import numpy as np
import time
import winsound
import os
from ultralytics import YOLO




VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_videos", "video1.mp4")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect", "yolo11_pothole_MWPD_finetuned", "weights", "best.pt")


# Detection Settings
CONF_THRESHOLD = 0.20           
IMG_SIZE = 640                  
ROAD_AREA_TOP_RATIO = 0.32      
MIN_BOX_AREA_FAR = 50
MAX_BOX_AREA_FAR = 8000
MIN_BOX_AREA_NEAR = 200
MAX_BOX_AREA_NEAR = 120000

# Road surface segmentation parameters
ROAD_SATURATION_MAX = 80        
ROAD_VALUE_MIN = 35             
ROAD_VALUE_MAX = 220            
ROAD_OVERLAP_THRESHOLD = 0.10   
EDGE_DENSITY_MAX = 0.30         
SHADOW_TEXTURE_THRESHOLD = 35   
MIN_EDGE_DENSITY = 0.012        
MIN_PIXEL_SD = 10.0
LVR_THRESHOLD = 1.15
SVR_LIMIT = 0.7


LANE_WIDTH_M = 3.7              
OFFSET_LIMIT = 0.7             
WARNING_FRAMES = 25             
LANE_SMOOTHING = 0.90           
MIN_LANE_POINTS = 150           # Daytime minimum
NIGHT_MIN_LANE_POINTS = 40       # Extreme Mode: 40 (was 60)
STALE_FRAME_LIMIT = 30          
MIN_BRIGHTNESS = 40            
MIN_CONSECUTIVE_FRAMES = 3      # Daytime
NIGHT_MIN_CONSECUTIVE_FRAMES = 2 # Extreme Mode: 2 for flickering night lanes

# Aggressive Adaptive Thresholds for Night
NIGHT_WHITE_THRESHOLD = 110      
NIGHT_YELLOW_S_THRESHOLD = 40    
SOBEL_THRESHOLD = (20, 100)


POTHOLE_BEEP_FREQ = 1800
POTHOLE_BEEP_DUR = 150
LANE_BEEP_FREQ = 1200
LANE_BEEP_DUR = 200
BEEP_COOLDOWN = 1.0             



print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")

# ==================== LANE DETECTION FUNCTIONS ====================

def enhance_low_light(img, low_light=False):
    """Enhance image for better lane detection in low light."""
    # Aggressive Gamma Correction for extreme darkness
    if low_light:
        gamma = 0.5  # Boost dark areas
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, table)
        
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Higher contrast adjustment at night
    clahe = cv2.createCLAHE(clipLimit=4.0 if low_light else 3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced


def is_low_light(frame):
    """Check if the frame is too dark for lane detection."""
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h = gray.shape[0]
    road_area = gray[h//2:, :]
    avg_brightness = np.mean(road_area)
    return avg_brightness < MIN_BRIGHTNESS, avg_brightness


def perspective_transform(img):
    """Transform road view to bird's eye view for lane detection."""
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
    """Create binary image highlighting lane lines - adaptive for headlight conditions."""
    img = enhance_low_light(img, low_light=low_light)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    
    # 1. Thresholding
    if low_light:
        # HEADLIGHT MODE: Dynamic Luminosity Thresholding
        # Identifying the brightest 5% of pixels in the beam area
        brightness_cutoff = np.percentile(l_channel, 95)
        # Prevent picking up mid-grays in pure dark scenes
        brightness_cutoff = max(brightness_cutoff, 110)
        white = cv2.threshold(l_channel, brightness_cutoff, 255, cv2.THRESH_BINARY)[1]
        
        # Color backup
        yellow = cv2.inRange(hls, (15, 30, NIGHT_YELLOW_S_THRESHOLD), (40, 255, 255))
        color_binary = white | yellow
    else:
        white = cv2.inRange(hls, (0, 180, 0), (255, 255, 255))
        yellow = cv2.inRange(hls, (15, 40, 80), (35, 255, 255))
        color_binary = white | yellow
    
    # 2. Sobel Edge Detection
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    max_sobel = np.max(abs_sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / max_sobel) if max_sobel > 0 else abs_sobelx
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= SOBEL_THRESHOLD[0]) & (scaled_sobel <= SOBEL_THRESHOLD[1])] = 255
    
    # Combine sources
    binary = cv2.bitwise_or(color_binary, sobel_binary) if low_light else color_binary
    binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    h, w = binary.shape
    mask = np.zeros_like(binary)
    # Tighter mask to exclude roadside noise in warped view
    pts = np.array([[
        [int(w * 0.12), h],
        [int(w * 0.88), h],
        [int(w * 0.65), int(h * 0.45)],
        [int(w * 0.35), int(h * 0.45)]
    ]], np.int32)
    cv2.fillPoly(mask, pts, 255)
    binary = cv2.bitwise_and(binary, mask)
    
    return binary


def sliding_window(binary, low_light=False):
    """Detect lane lines using sliding window algorithm."""
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
    except Exception:
        return None, None


def validate_lane_fits(left_fit, right_fit, w, h, low_light=False):
    """Validate lane detection results to avoid wild fluctuations."""
    if left_fit is None or right_fit is None:
        return False
    
    y_eval = h - 1
    left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
    right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
    lane_width = right_x - left_x
    
    # Adaptive width range
    min_w = 150 if low_light else 200
    max_w = w * 0.8 if low_light else w * 0.6
    
    if lane_width < min_w or lane_width > max_w:
        return False
    if left_x < -200 or right_x > w + 200:
        return False
    
    # Relaxation parameters
    curve_limit = 0.008 if low_light else 0.003
    parallel_limit = 0.006 if low_light else 0.001
    
    if abs(left_fit[0]) > curve_limit or abs(right_fit[0]) > curve_limit:
        return False
    if abs(left_fit[0] - right_fit[0]) > parallel_limit:
        return False
        
    return True
    
    return True


def smooth_fit(new_fit, old_fit, alpha=None):
    """Smooth lane line detection between frames."""
    if alpha is None:
        alpha = LANE_SMOOTHING
    if old_fit is None:
        return new_fit
    if new_fit is None:
        return old_fit
    return alpha * old_fit + (1 - alpha) * new_fit


def draw_lane_overlay(frame, left_fit, right_fit, Minv):
    """Draw green lane overlay on frame with improved visualization."""
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
        
        # Fill with gradient green
        cv2.fillPoly(lane_img, np.int32([pts]), (0, 200, 0))
        
        # Draw lane boundary lines for better visibility
        for i in range(len(ploty) - 1):
            pt1_l = (int(leftx[i]), int(ploty[i]))
            pt2_l = (int(leftx[i+1]), int(ploty[i+1]))
            pt1_r = (int(rightx[i]), int(ploty[i]))
            pt2_r = (int(rightx[i+1]), int(ploty[i+1]))
            cv2.line(lane_img, pt1_l, pt2_l, (0, 255, 255), 3)
            cv2.line(lane_img, pt1_r, pt2_r, (0, 255, 255), 3)
        
        # Warp back to original perspective
        lane_warped = cv2.warpPerspective(lane_img, Minv, (w, h))
        
        # Apply Gaussian blur for smoother appearance
        lane_warped = cv2.GaussianBlur(lane_warped, (5, 5), 0)
        
        return cv2.addWeighted(frame, 1, lane_warped, 0.35, 0)
    except Exception:
        return frame


def calculate_offset(left_fit, right_fit, frame_width, y_eval=720):
    """Calculate vehicle offset from lane center in meters."""
    left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
    right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
    
    lane_center = (left_x + right_x) / 2
    vehicle_center = frame_width / 2
    
    # Convert to meters
    xm_per_pix = LANE_WIDTH_M / (right_x - left_x) if (right_x - left_x) > 0 else 0.01
    offset = (vehicle_center - lane_center) * xm_per_pix
    
    return offset


# ==================== ROAD SEGMENTATION FUNCTIONS ====================

def detect_road_mask(frame):
    """
    Create a binary mask of road-like regions using:
    1. HSV color filtering - road is low-saturation gray, vegetation is high-saturation
    2. Edge density filtering - road surface is smooth, trees/foliage have dense edges
    3. Trapezoidal ROI - restrict to the area where road is expected
    """
    h, w = frame.shape[:2]
    
    # Step 1: HSV color filtering
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Road pixels: low saturation (gray/asphalt), moderate brightness
    road_color_mask = cv2.inRange(
        hsv,
        (0, 0, ROAD_VALUE_MIN),
        (180, ROAD_SATURATION_MAX, ROAD_VALUE_MAX)
    )
    
    # Step 2: Edge density filtering
    # Dense edges = foliage/trees; smooth = road surface
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Compute local edge density using a block-based approach
    kernel_size = 31
    edge_density = cv2.blur(edges.astype(np.float32), (kernel_size, kernel_size)) / 255.0
    
    # Road areas have low edge density
    smooth_mask = (edge_density < EDGE_DENSITY_MAX).astype(np.uint8) * 255
    
    # Combine: must be road-colored AND smooth
    road_mask = cv2.bitwise_and(road_color_mask, smooth_mask)
    
    # Step 3: Apply trapezoidal ROI (road area only)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_pts = np.array([[
        [int(w * 0.05), h],               # bottom-left
        [int(w * 0.95), h],               # bottom-right
        [int(w * 0.70), int(h * 0.35)],   # top-right
        [int(w * 0.30), int(h * 0.35)]    # top-left
    ]], np.int32)
    cv2.fillPoly(roi_mask, roi_pts, 255)
    road_mask = cv2.bitwise_and(road_mask, roi_mask)
    
    # Morphological cleanup - fill small gaps and remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    
    return road_mask


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

def is_on_road(x1, y1, x2, y2, road_mask):
    """Check if a detection bounding box overlaps sufficiently with the road mask."""
    h, w = road_mask.shape[:2]
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    bbox_road = road_mask[y1:y2, x1:x2]
    if bbox_road.size == 0:
        return False
    
    return np.count_nonzero(bbox_road) / bbox_road.size >= ROAD_OVERLAP_THRESHOLD


# ==================== POTHOLE FILTERING FUNCTIONS ====================

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
    if aspect_ratio < 0.3 or aspect_ratio > 15.0:
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


# ==================== MAIN DETECTION LOOP ====================

def main():
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {VIDEO_PATH}")
        return
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25
    frame_delay = int(1000 / video_fps)
    
    # State variables
    prev_left_fit = None
    prev_right_fit = None
    lane_warning_counter = 0
    last_pothole_beep = 0
    last_lane_beep = 0
    stale_counter = 0  # Track frames without new detection
    consecutive_valid_frames = 0  # Track stability of detection
    
    # Pothole session stats
    total_potholes = 0
    confidence_scores = []
    tracked_potholes = []  # List of [cx, cy, max_conf, last_seen_frame]
    # EXTREME MODE: Larger tracking context for high speeds
    POTHOLE_DIST_THRESHOLD = 400
    STALE_FRAMES = 60
    frame_count = 0
    road_mask = None # Cached mask
    
    print(f"\n{'='*50}")
    print("Road Pothole Detection & Lane Deviation Analysis")
    print(f"{'='*50}")
    print(f"Video: {VIDEO_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Press ESC to exit")
    print(f"{'='*50}\n")
    
    while cap.isOpened():
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        output = frame.copy()
        pothole_detected = False
        lane_departure = False
        
        # Check for low light conditions early
        too_dark, brightness = is_low_light(frame)
        
        # Remove stale tracked potholes (left the frame)
        tracked_potholes = [p for p in tracked_potholes if (frame_count - p[3]) < STALE_FRAMES]
        
        # ==================== ROAD SEGMENTATION ====================
        road_mask = detect_road_mask(frame)
        
        # ==================== POTHOLE DETECTION ====================
        
        # ==================== POTHOLE DETECTION ====================
        
        # HIGH RECALL Sensitivity
        current_conf = 0.18 if not too_dark else 0.15

        # Skip-frame optimization for road mask
        if frame_count % 10 == 0 or road_mask is None:
            road_mask = detect_road_mask(frame)

        results = model.predict(
            frame,
            imgsz=IMG_SIZE,
            conf=current_conf,
            device='cpu',
            verbose=False
        )
        
        for r in results:
            if r.boxes is None:
                continue
            
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Only process pothole class (class 0)
                if cls_id != 0:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Apply false positive filters (now including shadow texture check)
                if not is_valid_pothole(x1, y1, x2, y2, h, w, frame, road_mask):
                    continue
                
                pothole_detected = True
                
                # Deduplication: check if this pothole is already tracked
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                is_new = True
                for i, p in enumerate(tracked_potholes):
                    dist = ((cx - p[0]) ** 2 + (cy - p[1]) ** 2) ** 0.5
                    if dist < POTHOLE_DIST_THRESHOLD:
                        is_new = False
                        tracked_potholes[i][0] = cx
                        tracked_potholes[i][1] = cy
                        tracked_potholes[i][3] = frame_count
                        if conf > p[2]:
                            tracked_potholes[i][2] = conf
                        break
                
                if is_new:
                    total_potholes += 1
                    tracked_potholes.append([cx, cy, conf, frame_count])
                    confidence_scores.append(conf)
                
                # Draw pothole box
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    output,
                    f"POTHOLE {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
        
        # ==================== LANE DETECTION ====================
        
        # Variables to decide whether to draw lane this frame
        draw_lane = False
        
        if too_dark:
            cv2.putText(
                output,
                f"LOW LIGHT DETECTED (Brightness: {brightness:.0f})",
                (20, h - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2
            )
            cv2.putText(
                output,
                "ADAPTIVE MODE: GAMMA + SOBEL ACTIVE",
                (20, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        try:
            warped, Minv = perspective_transform(frame)
            binary = lane_binary(warped, low_light=too_dark)
            new_left_fit, new_right_fit = sliding_window(binary, low_light=too_dark)
            
            # Only update if detection is valid
            if validate_lane_fits(new_left_fit, new_right_fit, w, h, low_light=too_dark):
                # Got a valid new detection - reset stale counter
                stale_counter = 0
                consecutive_valid_frames += 1
                
                # Smooth detection
                left_fit = smooth_fit(new_left_fit, prev_left_fit)
                right_fit = smooth_fit(new_right_fit, prev_right_fit)
                prev_left_fit, prev_right_fit = left_fit, right_fit
                
            elif prev_left_fit is not None and prev_right_fit is not None:
                # No new detection - increment stale counter
                stale_counter += 1
                
                # If stale for too long, reset to force new detection
                if stale_counter > STALE_FRAME_LIMIT:
                    prev_left_fit, prev_right_fit = None, None
                    left_fit, right_fit = None, None
                    consecutive_valid_frames = 0
                else:
                    # Use previous detection temporarily
                    left_fit, right_fit = prev_left_fit, prev_right_fit
            else:
                left_fit, right_fit = None, None
                consecutive_valid_frames = 0
            
            # EXTREME MODE: Reduced requirement for flickering night lanes
            min_valid = NIGHT_MIN_CONSECUTIVE_FRAMES if too_dark else MIN_CONSECUTIVE_FRAMES
            if consecutive_valid_frames >= min_valid and left_fit is not None and right_fit is not None:
                draw_lane = True
                
                # Draw lane overlay
                output = draw_lane_overlay(output, left_fit, right_fit, Minv)
                
                # Calculate offset
                offset = calculate_offset(left_fit, right_fit, w)
                
                # Check for lane departure
                if abs(offset) > OFFSET_LIMIT:
                    lane_warning_counter += 1
                else:
                    lane_warning_counter = max(0, lane_warning_counter - 1)
                
                if lane_warning_counter > WARNING_FRAMES:
                    lane_departure = True
                
                # Display offset
                offset_color = (0, 255, 255) if not lane_departure else (0, 0, 255)
                cv2.putText(
                    output,
                    f"Offset: {offset:.2f}m",
                    (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    offset_color,
                    2
                )
            else:
                # If not drawing lanes, ensure warning counter resets
                lane_warning_counter = 0
                
        except Exception:
            pass
        
        # ==================== ALERTS ====================
        
        now = time.time()
        
        # Pothole alert
        if pothole_detected:
            cv2.putText(
                output,
                "! POTHOLE ALERT !",
                (w // 2 - 150, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )
            if now - last_pothole_beep > BEEP_COOLDOWN:
                winsound.Beep(POTHOLE_BEEP_FREQ, POTHOLE_BEEP_DUR)
                last_pothole_beep = now
        
        # Lane departure alert
        if lane_departure:
            cv2.putText(
                output,
                "! LANE DEPARTURE WARNING !",
                (w // 2 - 220, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 165, 255),
                3
            )
            if now - last_lane_beep > BEEP_COOLDOWN:
                winsound.Beep(LANE_BEEP_FREQ, LANE_BEEP_DUR)
                last_lane_beep = now
                
        # Status if no lane detected for a while
        if draw_lane is False and not too_dark:
             cv2.putText(
                output,
                "Lane Detection: SEARCHING...",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2
            )
        
        # ==================== DISPLAY ====================
        
        cv2.imshow("Road Safety Detection System", output)
        
        # FPS control
        elapsed = time.time() - start_time
        wait_time = max(1, frame_delay - int(elapsed * 1000))
        
        if cv2.waitKey(wait_time) == 27:  # ESC to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # ==================== SESSION SUMMARY ====================
    print(f"\n{'='*50}")
    print("         DETECTION SESSION SUMMARY")
    print(f"{'='*50}")
    print(f"  Total Potholes Detected : {total_potholes}")
    if confidence_scores:
        avg_conf = sum(confidence_scores) / len(confidence_scores)
        min_conf = min(confidence_scores)
        max_conf = max(confidence_scores)
        print(f"  Average Confidence      : {avg_conf:.2%}")
        print(f"  Min Confidence          : {min_conf:.2%}")
        print(f"  Max Confidence          : {max_conf:.2%}")
    else:
        print(f"  No potholes were detected in this session.")
    print(f"{'='*50}\n")
    
    # Display summary on screen
    summary_img = np.zeros((400, 600, 3), dtype=np.uint8)
    summary_img[:] = (40, 40, 40)  # Dark background
    
    # Title
    cv2.putText(summary_img, "SESSION SUMMARY", (120, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
    cv2.line(summary_img, (50, 70), (550, 70), (0, 200, 255), 2)
    
    # Stats
    cv2.putText(summary_img, f"Total Potholes Detected: {total_potholes}", (50, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    if confidence_scores:
        avg_conf = sum(confidence_scores) / len(confidence_scores)
        min_conf = min(confidence_scores)
        max_conf = max(confidence_scores)
        
        cv2.putText(summary_img, f"Average Confidence: {avg_conf:.2%}", (50, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(summary_img, f"Min Confidence: {min_conf:.2%}", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        cv2.putText(summary_img, f"Max Confidence: {max_conf:.2%}", (50, 285),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
    else:
        cv2.putText(summary_img, "No potholes detected.", (50, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
    
    cv2.putText(summary_img, "Press any key to close", (150, 370),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    cv2.imshow("Detection Session Summary", summary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Detection complete!")


if __name__ == "__main__":
    main()
