"""
test_pick.py — Auto pick blocks detected near the arm.
"""

import dobotArm
import lib.DobotDllType as dType
import cv2
import numpy as np

Z_SAFE = 40
Z_PICK = -25
DROP_X, DROP_Y = 250, 150
HOLD_FRAMES = 8
MISS_LIMIT = 5
MIN_CONTOUR_AREA = 400
PICK_PROXIMITY_PX = 30
MAX_PICKS = 3

# Only pick blocks within this pixel zone around center
PICK_ZONE_RADIUS = 100

COLOUR_HSV = {
    "green":  ([(45, 80, 80),   (80, 255, 255)],),
    "teal":   ([(95, 80, 80),   (120, 255, 255)],),
    "blue":   ([(125, 40, 80),  (150, 255, 255)],),
    "purple": ([(160, 80, 80),  (180, 255, 255)],),
}

COLOUR_BGR = {
    "green":  (0, 255, 0),
    "teal":   (255, 255, 0),
    "blue":   (255, 0, 0),
    "purple": (255, 0, 255),
}

# ── Camera setup ──
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[FATAL] No camera found")
    exit(1)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mask", 320, 240)

def mouse_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(param["frame"], cv2.COLOR_BGR2HSV)
        h, s, v = hsv[y, x]
        b, g, r = param["frame"][y, x]
        print(f"[CLICK] pixel ({x},{y}) — BGR({b},{g},{r})  HSV({h},{s},{v})")

mouse_data = {"frame": None}
cv2.setMouseCallback("Camera", mouse_hsv, mouse_data)

# ── Load calibration for undistortion (helps detection) ──
try:
    data = np.load("camera_params.npz")
    camera_matrix, dist_coeffs = data["camera_matrix"], data["dist_coeffs"]
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1)
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_K, (w, h), cv2.CV_16SC2)
    use_undistort = True
except FileNotFoundError:
    print("[WARN] No camera calibration — using raw feed.")
    use_undistort = False

# ── Robot setup ──
api = dType.load()

print("[ROBOT] Connecting and homing...")
dobotArm.initialize_robot(api)

CENTER_X, CENTER_Y = 180, 0
print(f"[ROBOT] Moving to center ({CENTER_X}, {CENTER_Y}, {Z_SAFE})...")
dobotArm.move_to_xyz(api, CENTER_X, CENTER_Y, Z_SAFE)
print("\n[READY] Place a block under the arm. It will auto-pick.")
print("       Left-click a pixel to see its HSV value.")
print("       Press Q to quit.\n")

# ── Helpers ──

def detect_blocks(frame):
    if use_undistort:
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (3, 3), 0), cv2.COLOR_BGR2HSV)
    blocks = []
    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for colour, ranges in COLOUR_HSV.items():
        for r in ranges:
            lower = np.array(r[0], dtype=np.uint8)
            upper = np.array(r[1], dtype=np.uint8)
            combined_mask = cv2.bitwise_or(combined_mask, cv2.inRange(hsv, lower, upper))
    mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > MIN_CONTOUR_AREA:
            M = cv2.moments(c)
            if M["m00"]:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                blocks.append((cx, cy, colour))
    return blocks, mask

def matches(block, block_list):
    bx, by, bc = block
    for lx, ly, lc in block_list:
        if lc == bc and abs(bx - lx) < PICK_PROXIMITY_PX and abs(by - ly) < PICK_PROXIMITY_PX:
            return True
    return False

# ── Main loop ──
active_block = None
hold_count = 0
miss_count = 0
pick_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    if use_undistort:
        display = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    else:
        display = frame.copy()

    mouse_data["frame"] = display
    blocks, mask = detect_blocks(frame)

    # Only consider blocks near center of frame
    cx, cy = display.shape[1] // 2, display.shape[0] // 2
    near_blocks = [(x, y, c) for x, y, c in blocks
                   if abs(x - cx) < PICK_ZONE_RADIUS and abs(y - cy) < PICK_ZONE_RADIUS]

    # Draw pick zone
    cv2.rectangle(display, (cx - PICK_ZONE_RADIUS, cy - PICK_ZONE_RADIUS),
                  (cx + PICK_ZONE_RADIUS, cy + PICK_ZONE_RADIUS), (0, 255, 255), 2)
    cv2.putText(display, "PICK ZONE", (cx - PICK_ZONE_RADIUS, cy - PICK_ZONE_RADIUS - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    for bx, by, bc in near_blocks:
        cv2.circle(display, (bx, by), 8, COLOUR_BGR[bc], -1)
        cv2.circle(display, (bx, by), 8, (255, 255, 255), 1)

    cv2.putText(display, f"Blocks: {len(near_blocks)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Camera", display)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # ── Debounce ──
    if not near_blocks:
        miss_count += 1
        if miss_count >= MISS_LIMIT:
            active_block = None
            hold_count = 0
        continue
    else:
        miss_count = 0

    if active_block is None:
        active_block = near_blocks[0]
        hold_count = 0
    elif not matches(active_block, near_blocks):
        active_block = near_blocks[0]
        hold_count = 0
        continue

    hold_count += 1

    if hold_count >= HOLD_FRAMES:
        print(f"[PICK] Block detected. Picking at current position...")

        # --- PICK (straight down at current arm position) ---
        dobotArm.move_to_xyz(api, CENTER_X, CENTER_Y, Z_PICK)
        dobotArm.close_gripper(api)
        dobotArm.move_to_xyz(api, CENTER_X, CENTER_Y, Z_SAFE)

        # --- PLACE ---
        dobotArm.move_to_xyz(api, DROP_X, DROP_Y, Z_SAFE)
        dobotArm.move_to_xyz(api, DROP_X, DROP_Y, Z_PICK)
        dobotArm.open_gripper(api)
        dobotArm.move_to_xyz(api, DROP_X, DROP_Y, Z_SAFE)

        pick_count += 1
        print(f"[{pick_count}/{MAX_PICKS}] Done. Back to center...")
        dobotArm.move_to_xyz(api, CENTER_X, CENTER_Y, Z_SAFE)

        if pick_count >= MAX_PICKS:
            print(f"[DONE] All {MAX_PICKS} blocks picked and placed.")
            break

        active_block = None
        hold_count = 0
        miss_count = 0
        continue

# ── Cleanup ──
cap.release()
cv2.destroyAllWindows()
dobotArm.move_to_home(api)
print("Finished.")
