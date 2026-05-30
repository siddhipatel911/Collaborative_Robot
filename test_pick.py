"""
test_pick.py — Standalone test: detect a coloured block and move the robot to it.

Opens the camera, shows the feed with the placement zone overlay, waits for a
block to appear and stabilise, prints its pixel and robot coordinates, then moves
the Dobot arm to that position at safe height.

Press Q to quit at any time.
"""

import dobotArm
import lib.DobotDllType as dType
import numpy as np
import cv2
import time

# ── Config ──
Z_SAFE = 40

PZ_X1, PZ_Y1 = 50, 20
PZ_X2, PZ_Y2 = 650, 315

HOLD_FRAMES = 8        # consecutive frames a block must be visible before we act
PICK_PROXIMITY_PX = 30

COLOUR_HSV = {
    "red":   ([(0, 120, 70), (10, 255, 255)], [(170, 120, 70), (180, 255, 255)]),
    "green": ([(40, 80, 70), (80, 255, 255)],),
    "blue":  ([(90, 80, 70), (130, 255, 255)],),
}

COLOUR_BGR = {
    "red":   (0, 0, 255),
    "green": (0, 255, 0),
    "blue":  (255, 0, 0),
}

# ── Helpers ──

def pixel_to_robot(u, v, H):
    p = np.array([u, v, 1])
    xy = H @ p
    xy /= xy[2]
    return xy[0], xy[1]

def detect_coloured_blocks(frame):
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (3, 3), 0), cv2.COLOR_BGR2HSV)
    blocks = []
    for colour, ranges in COLOUR_HSV.items():
        mask = cv2.inRange(hsv, np.array(ranges[0][0]), np.array(ranges[0][1]))
        if len(ranges) > 1:
            mask += cv2.inRange(hsv, np.array(ranges[1][0]), np.array(ranges[1][1]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 400:
                M = cv2.moments(c)
                if M["m00"]:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    blocks.append((cx, cy, colour))
    return blocks

def matches(block, block_list):
    bx, by, bc = block
    for lx, ly, lc in block_list:
        if lc == bc and abs(bx - lx) < PICK_PROXIMITY_PX and abs(by - ly) < PICK_PROXIMITY_PX:
            return True
    return False

# ── Camera setup (before robot init so window appears immediately) ──
api = dType.load()
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[WARN] Camera index 1 failed, trying index 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[FATAL] No camera found")
    exit(1)
print("[CAMERA] Opened")

try:
    H_matrix = np.load("HomographyMatrix.npy")
    data = np.load("camera_params.npz")
    camera_matrix, dist_coeffs = data["camera_matrix"], data["dist_coeffs"]
except FileNotFoundError as e:
    print(f"[FATAL] Missing calibration file: {e}")
    print("       Run calibrateCamera.py and getTransformationMatrix.py first.")
    exit(1)

ret, frame = cap.read()
if not ret:
    print("[FATAL] Cannot read from camera")
    exit(1)

h, w = frame.shape[:2]
new_K, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1)
map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix, dist_coeffs, None, new_K, (w, h), cv2.CV_16SC2
)

# Show camera window immediately (before robot's blocking homing)
cv2.namedWindow("Test Pick", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Test Pick", cv2.WND_PROP_TOPMOST, 1)
frame_show = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
cv2.imshow("Test Pick", frame_show)
cv2.waitKey(1)

# ── Robot setup ──
print("[ROBOT] Connecting and homing...")
dobotArm.initialize_robot(api)
print("[ROBOT] Homing done, opening gripper...")
dobotArm.open_gripper(api)
dobotArm.set_speed(api, 50)

# Move to a safe neutral position so the arm is out of the camera's way
print("[ROBOT] Moving to ready position (180, 0, 40)...")
dobotArm.move_to_xyz(api, 180, 0, Z_SAFE)
print("\n[READY] Place a coloured block in the green zone.")
print("        The robot will move to its location.")
print("        Press Q to quit.\n")

# ── Main loop ──
active_block = None
hold_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    display = frame.copy()

    blocks = detect_coloured_blocks(frame)
    zone_blocks = [(x, y, c) for x, y, c in blocks
                   if PZ_X1 <= x <= PZ_X2 and PZ_Y1 <= y <= PZ_Y2]

    # ── Draw placement zone ──
    cv2.rectangle(display, (PZ_X1, PZ_Y1), (PZ_X2, PZ_Y2), (0, 255, 0), 2)
    cv2.putText(display, "PLACEMENT ZONE", (PZ_X1, PZ_Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ── Draw detected blocks ──
    for bx, by, bc in zone_blocks:
        cv2.circle(display, (bx, by), 8, COLOUR_BGR[bc], -1)
        cv2.circle(display, (bx, by), 8, (255, 255, 255), 1)
        # Show pixel coordinates next to each block
        cv2.putText(display, f"({bx}, {by})", (bx + 12, by + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.putText(display, f"Blocks in zone: {len(zone_blocks)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ── Key input ──
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # ── Debounce: wait for a block to be stable for HOLD_FRAMES ──
    if not zone_blocks:
        active_block = None
        hold_count = 0
        cv2.imshow("Test Pick", display)
        continue

    if active_block is None:
        active_block = zone_blocks[0]
        hold_count = 0
    elif not matches(active_block, zone_blocks):
        active_block = zone_blocks[0]
        hold_count = 0
        cv2.imshow("Test Pick", display)
        continue

    hold_count += 1

    if hold_count >= HOLD_FRAMES:
        px, py, colour = active_block
        rx, ry = pixel_to_robot(px, py, H_matrix)

        print(f"[DETECTED] {colour.upper()} block")
        print(f"  Pixel  : ({px}, {py})")
        print(f"  Robot  : ({rx:.1f}, {ry:.1f}) mm")
        print(f"  Moving robot to ({rx:.1f}, {ry:.1f}, Z={Z_SAFE})...")

        # Add text to the frame before moving (so user sees the coordinate overlay)
        cv2.putText(display, f"TARGET: ({rx:.1f}, {ry:.1f}) mm",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Test Pick", display)
        cv2.waitKey(1)

        # Move the robot to the block's position (blocking call — feed freezes)
        dobotArm.move_to_xyz(api, rx, ry, Z_SAFE)

        print(f"  Arrived at ({rx:.1f}, {ry:.1f}, {Z_SAFE}). Place another block or press Q.\n")

        # Reset so we can detect the next block
        active_block = None
        hold_count = 0
        continue

    cv2.imshow("Test Pick", display)

# ── Cleanup ──
cap.release()
cv2.destroyAllWindows()
dobotArm.move_to_home(api)
print("[DONE] Test ended.")
