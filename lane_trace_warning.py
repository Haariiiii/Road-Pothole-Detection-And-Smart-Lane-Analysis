import cv2
import numpy as np
import os

# ================= SETTINGS =================
VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_videos", "video1.mp4")
LANE_WIDTH_M = 3.7
OFFSET_LIMIT = 0.4          # meters
WARNING_FRAMES = 12         # avoid false alerts
BEEP_FREQ = 1800
BEEP_DUR = 200

cap = cv2.VideoCapture(VIDEO_PATH)

prev_left_fit = None
prev_right_fit = None
warning_counter = 0

# ===========================================

def perspective_transform(img):
    h, w = img.shape[:2]
    src = np.float32([
        [w*0.45, h*0.63],
        [w*0.55, h*0.63],
        [w*0.90, h*0.95],
        [w*0.10, h*0.95]
    ])
    dst = np.float32([
        [w*0.25, 0],
        [w*0.75, 0],
        [w*0.75, h],
        [w*0.25, h]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return cv2.warpPerspective(img, M, (w, h)), Minv

def lane_binary(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    white = cv2.inRange(hls, (0,200,0), (255,255,255))
    yellow = cv2.inRange(hls, (15,30,115), (35,204,255))

    binary = white | yellow
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

def sliding_window(binary):
    histogram = np.sum(binary[binary.shape[0]//2:,:], axis=0)
    midpoint = histogram.shape[0]//2

    leftx = np.argmax(histogram[:midpoint])
    rightx = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    margin = 100
    minpix = 60
    window_h = binary.shape[0]//nwindows

    nonzero = binary.nonzero()
    nz_y = np.array(nonzero[0])
    nz_x = np.array(nonzero[1])

    left_inds, right_inds = [], []

    for win in range(nwindows):
        y_low = binary.shape[0] - (win+1)*window_h
        y_high = binary.shape[0] - win*window_h

        lx_low, lx_high = leftx-margin, leftx+margin
        rx_low, rx_high = rightx-margin, rightx+margin

        good_left = ((nz_y>=y_low)&(nz_y<y_high)&(nz_x>=lx_low)&(nz_x<lx_high)).nonzero()[0]
        good_right = ((nz_y>=y_low)&(nz_y<y_high)&(nz_x>=rx_low)&(nz_x<rx_high)).nonzero()[0]

        left_inds.append(good_left)
        right_inds.append(good_right)

        if len(good_left) > minpix:
            leftx = int(np.mean(nz_x[good_left]))
        if len(good_right) > minpix:
            rightx = int(np.mean(nz_x[good_right]))

    left_inds = np.concatenate(left_inds)
    right_inds = np.concatenate(right_inds)

    left_fit = np.polyfit(nz_y[left_inds], nz_x[left_inds], 2)
    right_fit = np.polyfit(nz_y[right_inds], nz_x[right_inds], 2)

    return left_fit, right_fit

def smooth(new, old, alpha=0.9):
    return new if old is None else alpha*old + (1-alpha)*new

def draw_lane(frame, left_fit, right_fit, Minv):
    ploty = np.linspace(0, frame.shape[0]-1, frame.shape[0])
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    lane = np.zeros_like(frame)
    pts = np.hstack((np.array([np.transpose(np.vstack([leftx, ploty]))]),
                     np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])))
    cv2.fillPoly(lane, np.int32([pts]), (0,255,0))

    return cv2.addWeighted(frame, 1, cv2.warpPerspective(lane, Minv, (frame.shape[1],frame.shape[0])), 0.3, 0)

def vehicle_offset(left_fit, right_fit, w):
    y = 720
    left = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
    right = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
    lane_center = (left + right) / 2
    xm = LANE_WIDTH_M / (right - left)
    return (w/2 - lane_center) * xm

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    warped, Minv = perspective_transform(frame)
    binary = lane_binary(warped)

    try:
        lf, rf = sliding_window(binary)
        lf = smooth(lf, prev_left_fit)
        rf = smooth(rf, prev_right_fit)

        prev_left_fit, prev_right_fit = lf, rf
        output = draw_lane(frame, lf, rf, Minv)

        offset = vehicle_offset(lf, rf, frame.shape[1])

        if abs(offset) > OFFSET_LIMIT:
            warning_counter += 1
        else:
            warning_counter = 0

        if warning_counter > WARNING_FRAMES:
            winsound.Beep(BEEP_FREQ, BEEP_DUR)
            cv2.putText(output, "LANE DEPARTURE WARNING",
                        (40,60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0,0,255), 3)

        cv2.putText(output, f"Offset: {offset:.2f} m",
                    (40,100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255,255,255), 2)

    except:
        output = frame

    cv2.imshow("Lane Trace Assist", output)
    if cv2.waitKey(25) == 27:
        break

cap.release()
cv2.destroyAllWindows()
