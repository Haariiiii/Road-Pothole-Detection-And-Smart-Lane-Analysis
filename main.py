# main.py

import sys
import subprocess
import os

# -------------------------
# 1. Auto-install missing packages
# -------------------------
required_packages = [
    "torch", "torchvision", "ultralytics", "opencv-python",
    "numpy", "pandas", "matplotlib", "tqdm", "seaborn"
]

for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# -------------------------
# 2. Imports
# -------------------------
import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------
# 3. Paths to models
# -------------------------
# Make sure you have these files in the "models" folder
yolo_model_path = os.path.join("models", "yolo_carla_best.pt")
ufld_model_path = os.path.join("models", "ufld_carla_best.pth")  # placeholder

# -------------------------
# 4. Load YOLO11n model using ultralytics (avoids torch.hub)
# -------------------------
print("Loading YOLO11n CARLA model...")
yolo_model = YOLO(yolo_model_path)
print("YOLO model loaded successfully!")

# -------------------------
# 5. Placeholder lane detection
# -------------------------
# For now, this draws simple green lane lines.
# Later, you can replace with real UFLD outputs
def detect_lanes(frame):
    """
    Detect lanes using simple Canny + Hough transform.
    Returns frame with green lane lines drawn.
    """
    import cv2
    import numpy as np

    # 1. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian blur
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # 3. Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # 4. Mask region of interest (bottom half of image)
    h, w = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, h),
        (w, h),
        (w, int(h*0.6)),
        (0, int(h*0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)

    # 5. Hough lines
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150
    )

    # 6. Draw lines
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(line_image, (x1,y1), (x2,y2), (0,255,0), 5)

    # 7. Overlay lines on original frame
    combo = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return combo


# -------------------------
# 6. Detect objects using YOLO
# -------------------------
def detect_objects(frame):
    results = yolo_model(frame)  # Run inference
    frame = results[0].plot()    # Draw bounding boxes
    return frame

# -------------------------
# 7. Video / Webcam input
# -------------------------
# Replace with 0 for webcam
video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_videos", "video1.mp4")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video {video_path}")
    exit()

# -------------------------
# 8. Main loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Lane detection (green lines)
    frame = detect_lanes(frame)

    # 2. Pothole/vehicle detection (YOLO)
    frame = detect_objects(frame)

    # Display results
    cv2.imshow("Lane + Pothole Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
