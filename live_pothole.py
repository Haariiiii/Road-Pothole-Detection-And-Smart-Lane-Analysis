import cv2
import numpy as np
import time
import winsound
import os
from ultralytics import YOLO

# ================= CONFIG =================
VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_videos", "video2.mp4")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect", "yolo11_pothole_MWPD_finetuned", "weights", "best.pt")
CONF_THRESHOLD = 0.20           # Lowered for higher recall
IMG_SIZE = 640                  

# Beep config
BEEP_FREQ = 1800
BEEP_DUR = 120
BEEP_COOLDOWN = 1.5  # seconds

# Road surface segmentation parameters
ROAD_SATURATION_MAX = 80        
ROAD_VALUE_MIN = 35             
ROAD_VALUE_MAX = 220            
ROAD_OVERLAP_THRESHOLD = 0.10   
EDGE_DENSITY_MAX = 0.30         
ROAD_AREA_TOP_RATIO = 0.32      
SHADOW_TEXTURE_THRESHOLD = 35   
MIN_EDGE_DENSITY = 0.012        
MIN_PIXEL_SD = 10.0
LVR_THRESHOLD = 1.15
SVR_LIMIT = 0.7                 # More lenient
MIN_ASPECT_RATIO = 0.3
MAX_ASPECT_RATIO = 15.0
MIN_BOX_AREA_FAR = 50
MAX_BOX_AREA_FAR = 8000
MIN_BOX_AREA_NEAR = 200
MAX_BOX_AREA_NEAR = 120000

# ================= LOAD MODEL =================
model = YOLO(MODEL_PATH)

# ================= ROAD SEGMENTATION =================

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
        [int(w * 0.70), int(h * 0.32)], [int(w * 0.30), int(h * 0.32)]
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


# ================= VIDEO =================
cap = cv2.VideoCapture(VIDEO_PATH)

# Read original FPS and use it
video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps <= 0:
    video_fps = 25  # fallback
frame_delay = int(1000 / video_fps)

last_beep = 0

# Session stats
total_potholes = 0
confidence_scores = []
tracked_potholes = []  # List of [cx, cy, max_conf, last_seen_frame]
POTHOLE_DIST_THRESHOLD = 200
STALE_FRAMES = 60
frame_count = 0

while cap.isOpened():
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    pothole_detected = False

    # Remove stale tracked potholes
    tracked_potholes = [p for p in tracked_potholes if (frame_count - p[3]) < STALE_FRAMES]

    # ===== ROAD SEGMENTATION =====
    # Skip-frame optimization for expensive road mask
    if frame_count % 10 == 0 or road_mask is None:
        road_mask = detect_road_mask(frame)

    # ===== YOLO INFERENCE =====
    results = model.predict(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        device='cpu',
        verbose=False
    )

    h, w = frame.shape[:2]
    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Assuming pothole class = 0
            if cls_id != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Apply road-surface and shadow filtering
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

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"POTHOLE {conf:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    # ===== BEEP LOGIC =====
    now = time.time()
    if pothole_detected and (now - last_beep > BEEP_COOLDOWN):
        winsound.Beep(BEEP_FREQ, BEEP_DUR)
        last_beep = now

    cv2.imshow("Pothole Detection (Normal Speed)", frame)

    # ===== FPS CONTROL (NORMAL SPEED) =====
    elapsed = time.time() - start_time
    wait_time = max(1, frame_delay - int(elapsed * 1000))

    if cv2.waitKey(wait_time) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

# ================= SESSION SUMMARY =================
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
summary_img[:] = (40, 40, 40)

cv2.putText(summary_img, "SESSION SUMMARY", (120, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
cv2.line(summary_img, (50, 70), (550, 70), (0, 200, 255), 2)

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
