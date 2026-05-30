#This code is a simplified implementation of a collaborative robotics system that detects plates and targets using computer vision, 
#and then commands a Dobot robotic arm to pick and place objects accordingly. The system operates in three phases: scanning for plates, 
#scanning for targets, and executing the pick/place operations. 
#Stability checks are implemented to ensure reliable detection before proceeding to the next phase.

# Note: there are parameters that are useful to the successful operation of the robot arm. Read through the code before running the program.

# How to use: 
# 1. Ensure you have the Dobot robotic arm set up and connected to your computer.
# 2. Place the plates (drop zones) and targets (red blocks) within the camera's
# field of view.
# 3. Run the script. The system will first scan for plates, then targets, and finally execute the pick/place operations based on the detected positions.
# 4. Monitor the console output and the video feed for feedback on the system's status and operations

#Other Useful Codes you can use:
#dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE, rHead): moves the robot to the specified (x, y, z) coordinates with a specified rotation for the end effector (rHead). Z_SAFE is a predefined constant that ensures the robot maintains a safe height to avoid collisions when moving horizontally.

import dobotArm
import lib.DobotDllType as dType
import numpy as np
import cv2
import time
import os  # <-- Moved here for clean importing

"""CONSTANTS"""
Z_SAFE = 40 
Z_PICK = -25 
STABILITY_LIMIT = 60  
PIXEL_TOLERANCE = 10  

machine_state = "scanning plate" 
MOOD_FILE = "mood_output.txt"  # File path linked to your chatbot pipeline

# --- INITIALIZATION FOR CAMERA TRANSFORMATION ---
api = dType.load()
cap = cv2.VideoCapture(0)
H_matrix = np.load("HomographyMatrix.npy")
data = np.load("./camera_params.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs   = data["dist_coeffs"]

# Compute undistort maps once
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read from camera during initialization")
h, w = frame.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1)
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_K, (w,h), cv2.CV_16SC2)

def pixel_to_robot(u, v, H):
    p = np.array([u, v, 1])
    xy = H @ p
    xy /= xy[2]
    return xy[0], xy[1]

def next_state():
    global machine_state
    if machine_state == "scanning plate":
        machine_state = "scanning target"
    elif machine_state == "scanning target":
        machine_state = "pick place"
    elif machine_state == "pick place":
        machine_state = "scanning plate"
    else:
        machine_state = "scanning plate"

# ---------------------------------------------------------
# ROBOT EMOTION PERSONALITY MODIFIER
# ---------------------------------------------------------
def apply_emotion_personality(api, current_x, current_y, current_z, current_r=0):
    """
    Reads the latest mood from the text file and alters the Dobot's 
    speed, acceleration, and performs custom pre-move expressions.
    """
    mood = "neutral"
    if os.path.exists(MOOD_FILE):
        try:
            with open(MOOD_FILE, "r") as f:
                mood = f.read().strip().lower()
        except Exception as e:
            print(f"Error reading mood file: {e}")
            mood = "neutral"

    print(f"Executing motion with Robot Personality profile: [{mood.upper()}]")

    # Reset to standard baseline first
    dType.SetPTPCommonParams(api, 50, 50, isQueued=1)

    if mood == "happy":
        print("Personality: Feeling joyful! Twirling...")
        # Does a little twirl (rotates 185° on joint 1) before picking
        dType.SetPTPCmd(api, dType.PTPMode.PTPJMoveMode, 185, current_y, current_z, current_r, isQueued=1)
        dType.SetPTPCmd(api, dType.PTPMode.PTPJMoveMode, current_x, current_y, current_z, current_r, isQueued=1)
        dType.dSleep(5000)

    elif mood == "sad":
        print("Personality: Feeling low. Moving slowly...")
        # Drop speed drastically
        dType.SetPTPCommonParams(api, 15, 10, isQueued=1)
        # Move halfway down and pause mid-air
        mid_hover_z = current_z + 20
        dobotArm.move_to_xyz(api, current_x, current_y, mid_hover_z)
        time.sleep(1.5)

    elif mood == "angry":
        print("Personality: FRUSTRATED! Snappy movements!")
        # Max out speed profiles
        dType.SetPTPCommonParams(api, 100, 100, isQueued=1)

    elif mood == "tired":
        print("Personality: Exhausted... nodding off.")
        dType.SetPTPCommonParams(api, 20, 15, isQueued=1)
        # Nodding sequence: droop slightly down and back up
        dobotArm.move_to_xyz(api, current_x, current_y, current_z - 15)
        dobotArm.move_to_xyz(api, current_x, current_y, current_z)
        time.sleep(1.0)

    elif mood == "surprised":
        print("Personality: Startled!")
        dType.SetPTPCommonParams(api, 90, 90, isQueued=1)
        # Jump upward quickly
        dobotArm.move_to_xyz(api, current_x, current_y, current_z + 40)
        time.sleep(0.5)

    else: # "neutral"
        dType.SetPTPCommonParams(api, 50, 50, isQueued=1)

    return mood


# ---------------------------------------------------------
# PHASE 1: DETECT Part Drop Zones (Plates)
# ---------------------------------------------------------
def phase_detect_plates():
    print("\n[PHASE 1] Scanning for drop zones. Waiting for stability...")
    stability_counter = 0
    last_count = 0
    
    while True:
        ret, frame = cap.read()
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 150, param1=100, param2=35, minRadius=25, maxRadius=55)

        current_list = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(display_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                rx, ry = pixel_to_robot(i[0], i[1], H_matrix)
                current_list.append((rx, ry))

        if len(current_list) > 0 and len(current_list) == last_count:
            stability_counter += 1
        else:
            stability_counter = 0
            last_count = len(current_list)

        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        cv2.putText(display_frame, f"LOCKING PLATES: {progress}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Detection", display_frame)
        cv2.waitKey(1)

        if stability_counter >= STABILITY_LIMIT:
            print(f"Locked {len(current_list)} plates.")
            return current_list
  

# ---------------------------------------------------------
# PHASE 2: DETECT Red velcros to pick up (Red Blocks)
# ---------------------------------------------------------
def phase_detect_targets():
    print("\n[PHASE 2] Scanning for targets. Waiting for stability...")
    stability_counter = 0
    last_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()
        
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (3,3), 0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0,120,70]), np.array([10,255,255])) + \
               cv2.inRange(hsv, np.array([170,120,70]), np.array([180,255,255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_list = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 800:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    rx, ry = pixel_to_robot(cx, cy, H_matrix)
                    current_list.append((rx, ry))
                    cv2.drawContours(display_frame, [cnt], -1, (0, 255, 0), 2)
                    
        cv2.waitKey(1)

        if len(current_list) != 0:
            if len(current_list) > 0 and len(current_list) == last_count:
                stability_counter += 1
            else:
                stability_counter = 0
                last_count = len(current_list)

        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        color = (0, 255, 0) if progress < 100 else (255, 255, 0)
        
        cv2.putText(display_frame, f"LOCKING TARGETS: {progress}%", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Detection", display_frame)
        
        if stability_counter >= STABILITY_LIMIT:
            print(f"[SUCCESS] Locked {len(current_list)} targets.")
            return current_list


# ---------------------------------------------------------
# PHASE 3: PICK/PLACE LOOP WITH INJECTED PERSONALITY
# ---------------------------------------------------------
def phase_execute_batch(api, pick_list, drop_list):
    cv2.VideoCapture(0)
    time.sleep(0.5)
    
    if len(pick_list) == 0 or len(drop_list) == 0:
        print("missing targets, aborting")
        return False
    
    batch_size = min(len(pick_list), len(drop_list))
    print(f"\n[PHASE 3] Executing batch of {batch_size} operations.")

    for i in range(batch_size):
        pick_x, pick_y = pick_list[i]
        drop_x, drop_y = drop_list[i]

        print(f"Task {i+1}: Moving {pick_x, pick_y} to {drop_x, drop_y}")

        # === INJECTED PERSONALITY CHECK HERE ===
        # Pass the targets coordinates to configure physical parameters before moving
        current_mood = apply_emotion_personality(api, pick_x, pick_y, Z_SAFE)

        # --- PICK SEQUENCE ---
        dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)
        dobotArm.move_to_xyz(api, pick_x, pick_y, Z_PICK)

        # Snappy / Aggressive gripper check for angry state
        if current_mood == "angry":
            dType.SetEndEffectorGripper(api, enableCtrl=1, on=1, isQueued=1)
            time.sleep(0.1) # Snappy clamp action execution buffer
        else:
            dobotArm.close_gripper(api)
            
        dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)

        # --- PLACE SEQUENCE ---
        # Reset back to default neutral pace for placement to ensure drops are uniform and clean
        dType.SetPTPCommonParams(api, 50, 50, isQueued=1)

        dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)
        dobotArm.open_gripper(api)
        try:
            dobotArm.stop_pump(api)
        except Exception:
            pass

        print(f"[TASK {i+1}] Completed")

    print("\nBatch Complete.")
    return True

