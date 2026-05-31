import lib.DobotDllType as dType
import dobotArm
import time
import numpy as np
import cv2
import os
import json

# Useful Global Variables
CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"
}

def find_camera(max_index=6, preferred_index=1):
    available = []
    for idx in range(max_index):
        test = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if test.isOpened():
            ret, frame = test.read()
            if ret and frame is not None:
                available.append(idx)
        test.release()
    if not available:
        return None, None
    selected = preferred_index if preferred_index in available else available[0]
    return cv2.VideoCapture(selected, cv2.CAP_DSHOW), selected


cam, camera_index = find_camera()

if cam is None or not cam.isOpened():
    print("Camera failed to open")
    exit()
print(f"[CAMERA] Using index {camera_index}")
    
#if the program errors for file path problems, copy the relative path to camera_params.npz and paste it here and try again. 
data = np.load("camera_params.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs   = data["dist_coeffs"]

# compute undistort maps once
ret,frame = cam.read()
if not ret:
    print("Camera opened but failed to read a frame")
    exit()
h,w = frame.shape[:2]

new_K, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix,
    dist_coeffs,
    (w,h),
    1
)

map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix,
    dist_coeffs,
    None,
    new_K,
    (w,h),
    cv2.CV_16SC2
)

api = dType.load()


def load_red_ranges():
    default = [
        ([0, 90, 60], [12, 255, 255]),
        ([165, 90, 60], [180, 255, 255]),
    ]
    path = "hsv_ranges.json"
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        colors = data.get("colors", {})
        red = colors.get("red")
        if not red:
            return default
        return [(item[0], item[1]) for item in red if len(item) == 2]
    except Exception:
        return default

# robot coordinates in mm
robot_points = np.array([
    [160, -120], [200, -120], [240, -120], [280, -120],
    [160,  -60], [200,  -60], [240,  -60], [280,  -60],
    [160,    0], [200,    0], [240,    0], [280,    0],
    [160,   60], [200,   60], [240,   60], [280,   60],
    [160,  120], [200,  120], [240,  120], [280,  120],
], dtype=np.float32)


RED_RANGES = load_red_ranges()


def detect_red_center(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in RED_RANGES:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(
                hsv,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8),
            ),
        )

    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours,_ = cv2.findContours(mask,
                                  cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

    contours = [c for c in contours if cv2.contourArea(c) > 80]
    if len(contours) == 0:
        return None, mask

    c = max(contours,key=cv2.contourArea)

    M = cv2.moments(c)

    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])

    return (cx,cy), mask


# ------------------------------------------------
# CALIBRATION
# ------------------------------------------------

def collect_calibration():

    pixel_points = []

    for pt in robot_points:

        x, y = pt

        print("\n----------------------------------")
        print("Moving robot to:", pt)

        # move to pick height
        dobotArm.move_to_xyz(api, x, y, -24)

        print("Press SPACE when robot is in position")
        
        # wait for space
        while True:
            ret, frame = cam.read()
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

            cv2.putText(frame,
                        "Robot at point. Press SPACE.",
                        (30,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        2)

            cv2.imshow("Calibration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # space
                break

        # move robot away so camera can see point
        print("Moving robot away")
        dobotArm.move_to_xyz(api, 200, 0, 80)

        print("Place RED marker where the tip was")
        print("Press SPACE to capture pixel")

        detected = None

        while True:

            ret, frame = cam.read()
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

            center, mask = detect_red_center(frame)

            if center is not None:
                u, v = center
                cv2.circle(frame, (u,v), 6, (0,255,0), -1)
                cv2.putText(frame,
                            "Marker locked - press SPACE",
                            (30,75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,255,0),
                            2)
                detected = center
            else:
                detected = None
                cv2.putText(frame,
                            "No red marker detected",
                            (30,75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,0,255),
                            2)

            cv2.putText(frame,
                        "Place marker. SPACE to save",
                        (30,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        2)

            cv2.imshow("Calibration", frame)
            cv2.imshow("Red Marker Mask", mask)

            key = cv2.waitKey(1) & 0xFF

            if key == 32 and detected is not None:
                print("Saved pixel:", detected)
                pixel_points.append(detected)
                break
            elif key == 32:
                print("SPACE pressed, but no red marker is detected. Move marker into view or improve lighting.")

    return np.array(pixel_points, dtype=np.float32)


def compute_homography(pixel_points):

    H,status = cv2.findHomography(pixel_points,robot_points)

    print("\nHomography Matrix\n")
    print(H)

    np.save("HomographyMatrix.npy",H)

    print("Matrix saved")

    return H


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def run():
    dobotArm.initialize_robot(api)

    pixel_points = collect_calibration()

    if len(pixel_points) < 4:
        print("Not enough points")
        return

    compute_homography(pixel_points)

    cam.release()
    cv2.destroyAllWindows()


# Good Luck!
run()
