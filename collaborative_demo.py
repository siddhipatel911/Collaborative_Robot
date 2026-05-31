"""
collaborative_demo.py — Human places blocks, robot picks & places with pace + mood adaptation.

OVERVIEW
--------
An overhead Orbbec camera watches a rectangular "placement zone" on the table.
When the human places a coloured block in the zone and withdraws their hand,
the Dobot Magician arm picks it up and drops it into the correct per-colour
drop zone (red, green, or blue).

The robot adapts its movement speed based on two factors:
  1. PACE — how fast the human is placing blocks (faster placement → faster robot)
  2. MOOD — the human's facial expression detected via MediaPipe face landmarks
     (happy/focused → full speed, tired → slower, agitated → slowest)

CONTROLS
--------
  Q  — quit
  R  — reset (clear the set of already-picked block IDs so they can be picked again)
  M  — manual test: robot moves to (200, 0, 40) and back

STATE MACHINE
-------------
The main loop runs a simple state machine. Each movement state calls a blocking
move_to_xyz / rotate_end_effector (the DLL call blocks until motion finishes)
then immediately transitions to the next state.

  watching  →  pick_move  →  pick_lower  →  pick_rise  →  pick_rotate
       ↑                                                        │
       │                                                   place_move
       │                                                        │
       │                                                  place_drop
       │                                                        │
       └───────────────────── return_xy  ←───  place_rotate ────┘

FILES REQUIRED
--------------
  HomographyMatrix.npy   — 3×3 homography from camera pixels → robot XY (mm)
  camera_params.npz      — camera intrinsics (camera_matrix, dist_coeffs)
  face_landmarker.task   — MediaPipe face landmark model (for mood detection)
"""

import dobotArm                   # our Dobot control functions
import lib.DobotDllType as dType  # low-level Dobot DLL wrapper (for dType.load())
import numpy as np
import cv2
import time
from collections import deque
import os
import json

try:
    from hand_safety import HandSafety
    HAND_SAFETY_AVAILABLE = True
except Exception as e:
    HandSafety = None
    HAND_SAFETY_AVAILABLE = False
    print(f"[WARN] hand safety unavailable: {e}")


# ══════════════════════════ MEDIAPIPE SETUP ══════════════════════════
# MediaPipe 0.10.x moved to a tasks-based API. We try to import it;
# if it's not installed, mood detection is disabled but everything else works.
try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import face_landmarker as fl
    from mediapipe.tasks.python.vision.core import RunningMode as mp_RunningMode
    from mediapipe.tasks.python.core import base_options as base_opts
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False
    print("[WARN] mediapipe not installed — face mood disabled")

# ══════════════════════════ CONFIGURABLE CONSTANTS ══════════════════════════
# These are the main things you'd tweak to change behaviour.

# -- Heights (mm, robot Z coordinate) --
Z_SAFE = 40     # height for all travel moves (above the blocks)
Z_PICK = -25    # height to descend to when gripping a block (below the table surface
                #   so the gripper fingers surround the block)

# -- Ready position (mm, robot XY) --
# Where the robot waits between pick cycles.
READY_X, READY_Y = 180, 0

# -- BGR colour values for on-screen drawing --
COLOUR_BGR = {
    "red":   (0, 0, 255),
    "green": (0, 255, 0),
    "blue":  (255, 0, 0),
}

# -- Placement zone (pixel coordinates on the camera image) --
# The green rectangle drawn on-screen. Only blocks whose centroid falls
# inside this rectangle are considered for picking.
PZ_X1, PZ_Y1 = 50, 20   # top-left corner
PZ_X2, PZ_Y2 = 650, 315   # bottom-right corner
WINDOW_NAME = "Collaborative Robot Demo"
HSV_CONFIG_FILE = "hsv_ranges.json"

# -- Speed adaptation --
SPEED_WINDOW = 5      # number of recent pick intervals to average
BASE_SPEED   = 50     # default speed (% of max)
MAX_SPEED    = 80     # never go above this
MIN_SPEED    = 25     # never go below this
FAST_PACE_S  = 3.0    # if avg interval < this, robot speeds up
SLOW_PACE_S  = 8.0    # if avg interval > this, robot slows down

HOLD_FRAMES = 8
PICK_PROXIMITY_PX = 30  # blocks within this many pixels are considered the same
PICK_RETRY_COUNT = 2
PICK_RAISE_CHECK = -15  # raise to this Z to perform a quick pickup check
MIN_BLOCK_AREA = 400
MIN_DROP_ZONE_AREA = 1200
DROP_ZONE_SCAN_INTERVAL = 0.5
MOTION_MIN_AREA = 1800
MOTION_WARMUP_FRAMES = 20
MOTION_BLOCK_FRAMES = 2
SKIN_MIN_AREA = 1800
SAFETY_ZONE_MARGIN = 25

# Movement limits keep bad camera calibration from sending impossible moves.
# Recalibrate if a valid table point lands outside this box.
ROBOT_X_RANGE = (120, 320)
ROBOT_Y_RANGE = (-140, 140)
ROBOT_Z_RANGE = (-35, 120)

DEFAULT_COLOUR_HSV = {
    "red": [
        [[0, 120, 70], [10, 255, 255]],
        [[170, 120, 70], [180, 255, 255]],
    ],
    "green": [
        [[40, 80, 70], [80, 255, 255]],
    ],
    "blue": [
        [[90, 80, 70], [130, 255, 255]],
    ],
}

# Kept only so the old MoodAnalyzer helper remains import-safe; main() no longer
# uses face mood to change robot speed.
MOOD_MODIFIERS = {
    "happy": 1.0,
    "focused": 1.0,
    "neutral": 1.0,
    "tired": 1.0,
    "agitated": 1.0,
    "no_face": 1.0,
}


# ══════════════════════════ MOOD ANALYSER ══════════════════════════
class MoodAnalyzer:
    """Detect human mood from a webcam feed using MediaPipe face landmarks.

    Every frame we extract geometric metrics from the face:
      - Eye Aspect Ratio (EAR)  — blink / tiredness
      - Mouth Aspect Ratio (MAR) — smile / happiness
      - Brow distance             — furrowed brow / agitation

    These are fed into a simple rule-based classifier.
    Mood history over ~30 frames prevents flickering.
    """

    # MediaPipe face landmark indices for the features we need.
    LANDMARK_IDS = {
        "left_eye_top":     159, "left_eye_bottom":   145,
        "right_eye_top":    386, "right_eye_bottom":  374,
        "left_eye_left":     33, "left_eye_right":    133,
        "right_eye_left":   362, "right_eye_right":   263,
        "mouth_top":         13, "mouth_bottom":       14,
        "mouth_left":        61, "mouth_right":       291,
        "left_eyebrow_inner": 105,
        "right_eyebrow_inner": 334,
    }

    def __init__(self, history_frames=30):
        """Load the MediaPipe face landmarker model (if available)."""
        self.model = None
        if MEDIAPIPE_AVAILABLE:
            try:
                opts = fl.FaceLandmarkerOptions(
                    base_options=base_opts.BaseOptions(model_asset_path="face_landmarker.task"),
                    running_mode=mp_RunningMode.VIDEO,
                    min_face_detection_confidence=0.6,
                    min_tracking_confidence=0.5,
                    output_face_blendshapes=False,
                )
                self.model = fl.FaceLandmarker.create_from_options(opts)
            except Exception as e:
                print(f"[WARN] Face landmarker load failed: {e}")
        self.history = deque(maxlen=history_frames)
        self.current_mood = "neutral"
        self.current_modifier = 1.0

    # ── helpers ──

    def _as_pt(self, lm, w, h):
        """Convert a normalised landmark [0,1] to pixel coordinates."""
        return np.array([lm.x * w, lm.y * h])

    def _metrics(self, landmarks, w, h):
        """Compute EAR, MAR, and brow distance from face landmarks."""
        ids = self.LANDMARK_IDS
        le_top = self._as_pt(landmarks[ids["left_eye_top"]], w, h)
        le_bot = self._as_pt(landmarks[ids["left_eye_bottom"]], w, h)
        le_l   = self._as_pt(landmarks[ids["left_eye_left"]], w, h)
        le_r   = self._as_pt(landmarks[ids["left_eye_right"]], w, h)

        re_top = self._as_pt(landmarks[ids["right_eye_top"]], w, h)
        re_bot = self._as_pt(landmarks[ids["right_eye_bottom"]], w, h)
        re_l   = self._as_pt(landmarks[ids["right_eye_left"]], w, h)
        re_r   = self._as_pt(landmarks[ids["right_eye_right"]], w, h)

        m_top = self._as_pt(landmarks[ids["mouth_top"]], w, h)
        m_bot = self._as_pt(landmarks[ids["mouth_bottom"]], w, h)
        m_l   = self._as_pt(landmarks[ids["mouth_left"]], w, h)
        m_r   = self._as_pt(landmarks[ids["mouth_right"]], w, h)

        lb = self._as_pt(landmarks[ids["left_eyebrow_inner"]], w, h)
        rb = self._as_pt(landmarks[ids["right_eyebrow_inner"]], w, h)

        # Eye Aspect Ratio (lower → eyes more closed → tired)
        left_ear = (np.linalg.norm(le_top - le_l) + np.linalg.norm(le_bot - le_r)) / (
                    2.0 * np.linalg.norm(le_l - le_r) + 1e-6)
        right_ear = (np.linalg.norm(re_top - re_l) + np.linalg.norm(re_bot - re_r)) / (
                     2.0 * np.linalg.norm(re_l - re_r) + 1e-6)
        ear = (left_ear + right_ear) / 2.0

        # Mouth Aspect Ratio (higher → mouth more open → happy/surprised)
        mouth_width = np.linalg.norm(m_r - m_l)
        mar = np.linalg.norm(m_bot - m_top) / (mouth_width + 1e-6)

        # Brow distance (lower → brows closer together → agitated)
        brow_dist = np.linalg.norm(lb - rb)

        return {"ear": ear, "mar": mar, "brow_dist": brow_dist}

    # ── main entry point ──

    def update(self, frame_rgb, timestamp_ms):
        """Process one camera frame and update the mood estimate."""
        if self.model is None:
            self.current_mood = "no_face"
            self.current_modifier = 1.0
            return self.current_mood

        h, w, _ = frame_rgb.shape
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.model.detect_for_video(mp_img, timestamp_ms)

        if not result.face_landmarks:
            # No face detected — use majority-vote from recent history
            self.history.append("no_face")
            mood = self._majority_mood()
            self.current_mood = mood
            self.current_modifier = MOOD_MODIFIERS.get(mood, 1.0)
            return mood

        m = self._metrics(result.face_landmarks[0], w, h)
        self.history.append(m)
        mood = self._classify(m)
        self.current_mood = mood
        self.current_modifier = MOOD_MODIFIERS.get(mood, 1.0)
        return mood

    # ── classification ──

    def _classify(self, m):
        """Rule-based mood classification from face metrics."""
        if m["mar"] > 0.35:               # big mouth → happy
            return "happy"
        if m["ear"] < 0.18:               # droopy eyes → tired
            return "tired"
        if m["brow_dist"] < 25:           # furrowed brow → agitated
            return "agitated"
        if 0.25 < m["mar"] < 0.35 and m["ear"] > 0.22:   # slight smile, alert → focused
            return "focused"
        return "neutral"

    def _majority_mood(self):
        """When no face is visible, fall back to the most common recent mood."""
        non_face = [x for x in self.history if isinstance(x, str)]
        if len(non_face) > len(self.history) // 2:
            return "no_face"
        metrics = [x for x in self.history if isinstance(x, dict)]
        return self._classify(metrics[-1]) if metrics else "no_face"


# ══════════════════════════ CAMERA SETUP ══════════════════════════
# Open the Orbbec camera (prefer index 1, with tested fallback).
# Then load the pre-computed calibration files and build the undistortion map.

def find_camera(max_index=6, preferred_index=1):
    """Find a readable camera, matching calibrateCamera.py's preview behavior."""
    available = []
    for idx in range(max_index):
        test_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if test_cap.isOpened():
            ret, test_frame = test_cap.read()
            if ret and test_frame is not None:
                available.append(idx)
        test_cap.release()

    if not available:
        return None, None

    selected = preferred_index if preferred_index in available else available[0]
    return cv2.VideoCapture(selected, cv2.CAP_DSHOW), selected

api = dType.load()                                              # load Dobot DLL first
cap, CAMERA_INDEX = find_camera()
if cap is None or not cap.isOpened():
    print("[FATAL] No camera found")
    exit(1)
print(f"[CAMERA] Opened index {CAMERA_INDEX}")

H_matrix = np.load("HomographyMatrix.npy")                      # 3×3 homography: pixel → robot mm
data = np.load("camera_params.npz")                             # camera intrinsics from calibration
camera_matrix, dist_coeffs = data["camera_matrix"], data["dist_coeffs"]

ret, frame = cap.read()
if not ret:
    print("[FATAL] Cannot read from camera")
    exit(1)

h, w = frame.shape[:2]                                          # image dimensions
new_K, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1)
map1, map2 = cv2.initUndistortRectifyMap(                       # precompute remap for speed
    camera_matrix, dist_coeffs, None, new_K, (w, h), cv2.CV_16SC2
)

# Show the first camera frame immediately so the user sees live feed
# even while the robot is homing (which blocks for 10-20 seconds).
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
frame_show = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
cv2.imshow(WINDOW_NAME, frame_show)
cv2.waitKey(1)

# ══════════════════════════ ROBOT SETUP ══════════════════════════
# Now that the camera is live, connect to the robot and home it.
# The user sees a (frozen) camera frame during the homing process.
dobotArm.initialize_robot(api)
dobotArm.open_gripper(api)                                      # start with gripper open


# ══════════════════════════ HELPER FUNCTIONS ══════════════════════════

def pixel_to_robot(u, v, H):
    """Convert a camera pixel coordinate (u, v) to robot XY (mm) using the homography matrix H."""
    p = np.array([u, v, 1])
    xy = H @ p
    xy /= xy[2]                   # de-homogenise (divide by w)
    return xy[0], xy[1]


def normalise_hsv_ranges(raw):
    """Accept the current JSON shape and the older single-colour shape."""
    if isinstance(raw, dict) and "colors" in raw:
        raw = raw["colors"]
    elif isinstance(raw, dict) and "ranges" in raw:
        raw = {"red": raw["ranges"]}

    ranges = {}
    if not isinstance(raw, dict):
        return DEFAULT_COLOUR_HSV.copy()

    for colour, value in raw.items():
        if isinstance(value, dict):
            value = value.get("ranges", [])
        clean = []
        for item in value:
            if len(item) != 2:
                continue
            lower = [int(max(0, min(255, x))) for x in item[0]]
            upper = [int(max(0, min(255, x))) for x in item[1]]
            if len(lower) == 3 and len(upper) == 3:
                clean.append([lower, upper])
        if clean:
            ranges[colour] = clean

    return ranges or DEFAULT_COLOUR_HSV.copy()


def load_colour_hsv(path=HSV_CONFIG_FILE):
    if not os.path.exists(path):
        return DEFAULT_COLOUR_HSV.copy()
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = normalise_hsv_ranges(json.load(f))
        for colour in loaded:
            COLOUR_BGR.setdefault(colour, (255, 255, 255))
        return loaded
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}; using defaults")
        return DEFAULT_COLOUR_HSV.copy()


COLOUR_HSV = load_colour_hsv()


def mask_for_colour(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)),
        )
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def detect_coloured_objects(frame, min_area):
    """Find configured colours and return (cx, cy, colour, area)."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (3, 3), 0), cv2.COLOR_BGR2HSV)
    objects = []
    for colour, ranges in COLOUR_HSV.items():
        mask = mask_for_colour(hsv, ranges)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area > min_area:
                M = cv2.moments(c)
                if M["m00"]:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    objects.append((cx, cy, colour, area))
    return objects


def detect_coloured_blocks(frame):
    """Find pickable coloured blocks in the placement zone."""
    blocks = []
    for cx, cy, colour, _ in detect_coloured_objects(frame, MIN_BLOCK_AREA):
        if PZ_X1 <= cx <= PZ_X2 and PZ_Y1 <= cy <= PZ_Y2:
            blocks.append((cx, cy, colour))
    return blocks


def detect_drop_zones(frame):
    """Detect the largest configured colour target outside the placement zone."""
    zones = {}
    for cx, cy, colour, area in detect_coloured_objects(frame, MIN_DROP_ZONE_AREA):
        if PZ_X1 <= cx <= PZ_X2 and PZ_Y1 <= cy <= PZ_Y2:
            continue
        rx, ry = pixel_to_robot(cx, cy, H_matrix)
        if not is_robot_xy_safe(rx, ry):
            continue
        current = zones.get(colour)
        if current is None or area > current["area"]:
            zones[colour] = {"pixel": (cx, cy), "robot": (rx, ry), "area": area}
    return zones


def is_robot_xy_safe(x, y):
    return ROBOT_X_RANGE[0] <= x <= ROBOT_X_RANGE[1] and ROBOT_Y_RANGE[0] <= y <= ROBOT_Y_RANGE[1]


def is_robot_xyz_safe(x, y, z):
    return is_robot_xy_safe(x, y) and ROBOT_Z_RANGE[0] <= z <= ROBOT_Z_RANGE[1]


def safe_move_to_xyz(api, x, y, z, rHead=0, wait=True):
    if not is_robot_xyz_safe(x, y, z):
        print(f"[BLOCKED] Refusing unsafe move ({x:.1f}, {y:.1f}, {z:.1f})")
        return False
    try:
        result = dobotArm.move_to_xyz(api, x, y, z, rHead, wait)
        if result != 0:
            print(f"[ERROR] Dobot move returned {result}")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] Dobot move failed: {e}")
        return False


def safe_rotate_end_effector(api, angle):
    try:
        result = dobotArm.rotate_end_effector(api, angle)
        if result != 0:
            print(f"[ERROR] Wrist rotation rejected: {angle}")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] Wrist rotation failed: {e}")
        return False


class WorkZoneSafetyMonitor:
    """Blocks when moving or skin-like objects enter the robot work zone."""

    def __init__(self):
        self.background = None
        self.frame_count = 0
        self.motion_hits = 0
        self.last_reason = ""

    def _zone(self, frame):
        h, w = frame.shape[:2]
        x1 = max(0, PZ_X1 - SAFETY_ZONE_MARGIN)
        y1 = max(0, PZ_Y1 - SAFETY_ZONE_MARGIN)
        x2 = min(w, PZ_X2 + SAFETY_ZONE_MARGIN)
        y2 = min(h, PZ_Y2 + SAFETY_ZONE_MARGIN)
        return x1, y1, x2, y2

    def _detect_motion(self, frame):
        x1, y1, x2, y2 = self._zone(frame)
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.background is None:
            self.background = gray.astype("float")
            return False, None

        self.frame_count += 1
        if self.frame_count <= MOTION_WARMUP_FRAMES:
            cv2.accumulateWeighted(gray, self.background, 0.08)
            return False, None

        delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.background))
        _, mask = cv2.threshold(delta, 24, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        moving = [c for c in contours if cv2.contourArea(c) >= MOTION_MIN_AREA]

        if moving:
            self.motion_hits += 1
            largest = max(moving, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            box = (x + x1, y + y1, w, h)
            return self.motion_hits >= MOTION_BLOCK_FRAMES, box

        self.motion_hits = 0
        cv2.accumulateWeighted(gray, self.background, 0.02)
        return False, None

    def _detect_skin_like(self, frame):
        x1, y1, x2, y2 = self._zone(frame)
        roi = frame[y1:y2, x1:x2]
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        skin = [c for c in contours if cv2.contourArea(c) >= SKIN_MIN_AREA]
        if not skin:
            return False, None
        largest = max(skin, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return True, (x + x1, y + y1, w, h)

    def update(self, frame, display):
        moving, motion_box = self._detect_motion(frame)
        skin_like, skin_box = self._detect_skin_like(frame)

        blocked = moving or skin_like
        reasons = []
        if moving:
            reasons.append("motion")
            x, y, w, h = motion_box
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 165, 255), 2)
        if skin_like:
            reasons.append("skin")
            x, y, w, h = skin_box
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)

        self.last_reason = "+".join(reasons)
        return blocked


def warn_if_homography_needs_recalibration():
    corners = [(PZ_X1, PZ_Y1), (PZ_X2, PZ_Y1), (PZ_X1, PZ_Y2), (PZ_X2, PZ_Y2)]
    unsafe = []
    for u, v in corners:
        rx, ry = pixel_to_robot(u, v, H_matrix)
        if not is_robot_xy_safe(rx, ry):
            unsafe.append((u, v, rx, ry))
    if unsafe:
        print("[WARN] Placement zone extends outside the calibrated robot workspace.")
        print("[WARN] Run getTransformationMatrix.py before the demo with the current camera/table setup.")
        for u, v, rx, ry in unsafe:
            print(f"       pixel ({u},{v}) -> robot ({rx:.1f},{ry:.1f})")


def compute_pace_speed(intervals):
    """Map the average block-placement interval to a robot speed percentage.

    Fast placement (< FAST_PACE_S)  → robot speeds up proportionally.
    Slow placement (> SLOW_PACE_S)  → robot slows down proportionally.
    In between → BASE_SPEED.
    """
    if len(intervals) < 1:
        return BASE_SPEED
    avg = sum(intervals) / len(intervals)
    if avg < FAST_PACE_S:
        return min(MAX_SPEED, int(BASE_SPEED * FAST_PACE_S / max(avg, 0.5)))
    elif avg > SLOW_PACE_S:
        return max(MIN_SPEED, int(BASE_SPEED * SLOW_PACE_S / avg))
    return BASE_SPEED


# ══════════════════════════ DRAWING HELPERS ══════════════════════════

def draw_status_panel(display, state, speed, hand_blocked, pace, block_count, colour_counts, drop_zones):
    """Overlay status information on the camera feed."""
    safety = "HAND BLOCK" if hand_blocked else "SAFE"
    safety_color = (0, 0, 255) if hand_blocked else (0, 255, 0)
    cv2.putText(display, f"Speed: {speed}%  Safety: {safety}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, safety_color, 2)
    cv2.putText(display, f"State: {state}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    if pace is not None:
        cv2.putText(display, f"Avg pace: {pace:.1f}s", (10, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 2)

    # Per-colour counts
    y0 = 105
    for i, (c, cnt) in enumerate(colour_counts.items()):
        cv2.putText(display, f"{c}: {cnt}", (10, y0 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR_BGR.get(c, (255, 255, 255)), 2)

    # Placement zone rectangle
    cv2.rectangle(display, (PZ_X1, PZ_Y1), (PZ_X2, PZ_Y2), (0, 255, 0), 2)
    cv2.putText(display, "PLACEMENT ZONE", (PZ_X1, PZ_Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Drop-zone legend (right side of screen)
    lx, ly = 540, 145
    for colour in sorted(COLOUR_HSV):
        zone = drop_zones.get(colour)
        cv2.circle(display, (lx, ly), 8, COLOUR_BGR.get(colour, (255, 255, 255)), -1)
        if zone:
            dx, dy = zone["robot"]
            text = f"{colour} ({dx:.0f},{dy:.0f})"
        else:
            text = f"{colour} zone: searching"
        cv2.putText(display, text, (lx + 14, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOUR_BGR.get(colour, (255, 255, 255)), 2)
        ly += 22

    for colour, zone in drop_zones.items():
        px, py = zone["pixel"]
        cv2.circle(display, (px, py), 14, COLOUR_BGR.get(colour, (255, 255, 255)), 2)
        cv2.putText(display, f"{colour} drop", (px + 12, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOUR_BGR.get(colour, (255, 255, 255)), 2)


def draw_face_expression(display, mood, x=10, y=100, size=35):
    """Draw a simple cartoon face reflecting the detected mood (for visual feedback)."""
    cx, cy = x + size, y + size
    cv2.circle(display, (cx, cy), size, (200, 200, 255), -1)    # face circle
    cv2.circle(display, (cx, cy), size, (100, 100, 200), 2)     # outline

    if mood == "happy":
        # Closed happy eyes, big smile, blush cheeks
        cv2.ellipse(display, (cx - 10, cy - 5), (5, 3), 0, 180, 360, (0, 0, 0), 2)
        cv2.ellipse(display, (cx + 10, cy - 5), (5, 3), 0, 180, 360, (0, 0, 0), 2)
        cv2.ellipse(display, (cx, cy + 8), (8, 6), 0, 0, 180, (0, 0, 0), 2)
        cv2.circle(display, (cx - 12, cy + 12), 4, (200, 100, 150), -1)
        cv2.circle(display, (cx + 12, cy + 12), 4, (200, 100, 150), -1)
    elif mood == "agitated":
        # Angry eyes, furrowed brows, flat mouth
        cv2.ellipse(display, (cx - 10, cy - 6), (4, 4), 0, 180, 360, (0, 0, 0), 2)
        cv2.ellipse(display, (cx + 10, cy - 6), (4, 4), 0, 180, 360, (0, 0, 0), 2)
        cv2.line(display, (cx - 16, cy - 16), (cx - 4, cy - 12), (0, 0, 0), 2)
        cv2.line(display, (cx + 4, cy - 12), (cx + 16, cy - 16), (0, 0, 0), 2)
        cv2.line(display, (cx - 6, cy + 10), (cx + 6, cy + 10), (0, 0, 0), 2)
    elif mood == "tired":
        # Droopy flat eyes, small mouth
        cv2.line(display, (cx - 16, cy - 6), (cx - 4, cy - 6), (0, 0, 0), 2)
        cv2.line(display, (cx + 4, cy - 6), (cx + 16, cy - 6), (0, 0, 0), 2)
        cv2.line(display, (cx - 5, cy + 10), (cx + 5, cy + 10), (0, 0, 0), 2)
    elif mood == "focused":
        # Small dot eyes, straight mouth
        cv2.circle(display, (cx - 10, cy - 6), 3, (0, 0, 0), -1)
        cv2.circle(display, (cx + 10, cy - 6), 3, (0, 0, 0), -1)
        cv2.line(display, (cx - 7, cy + 10), (cx + 7, cy + 10), (0, 0, 0), 2)
    else:  # neutral / no_face
        # Round eyes, flat mouth
        cv2.circle(display, (cx - 10, cy - 6), 4, (0, 0, 0), -1)
        cv2.circle(display, (cx + 10, cy - 6), 4, (0, 0, 0), -1)
        cv2.line(display, (cx - 6, cy + 10), (cx + 6, cy + 10), (0, 0, 0), 2)


# ══════════════════════════ MAIN LOOP ══════════════════════════

def matches(block, block_list):
    """Check if a block (cx, cy, colour) is close enough to any block in the list
    (within PICK_PROXIMITY_PX pixels and same colour) to be considered the same one."""
    bx, by, bc = block
    for lx, ly, lc in block_list:
        if lc == bc and abs(bx - lx) < PICK_PROXIMITY_PX and abs(by - ly) < PICK_PROXIMITY_PX:
            return True
    return False


def main():
    """Entry point — runs the main camera + state-machine loop."""

    safety = None
    if HAND_SAFETY_AVAILABLE:
        try:
            safety = HandSafety()
            print("[HAND SAFETY] Enabled")
        except Exception as e:
            print(f"[WARN] Hand safety disabled: {e}")
    work_zone_safety = WorkZoneSafetyMonitor()
    warn_if_homography_needs_recalibration()

    # Move robot to the ready position (blocks until done)
    dobotArm.set_speed(api, BASE_SPEED)
    safe_move_to_xyz(api, READY_X, READY_Y, Z_SAFE)
    print("[SYSTEM] Ready. Waiting for human to place blocks in the zone...")

    # ── state variables ──
    intervals = deque(maxlen=SPEED_WINDOW)                # rolling window of pick-pace intervals
    colour_counts = {colour: 0 for colour in COLOUR_HSV}  # tally per colour
    block_count = 0                                       # total blocks picked
    state = "watching"                                    # current state-machine state
    target_robot = None                                   # robot XY of current target (mm)
    target_pixel = None                                   # pixel XY of current target
    target_colour = None                                  # colour of current target
    last_pick_time = time.time()                          # when the last block was placed
    picked_ids = set()                                    # set of (px, py, colour) already picked
    hold_count = 0                                        # frames the current block has been stable
    active_block = None                                   # (px, py, colour) we're currently watching
    last_cycle_time = 0                                   # timestamp of last completed pick cycle
    drop_zones = {}
    last_drop_zone_scan = 0

    # ── main loop ──
    while True:
        # ── Read and preprocess camera frame ──
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)   # undistort
        display = frame.copy()
        # Perception
        hand_blocked = False
        safety_reasons = []
        if safety is not None:
            try:
                hand_landmarks, handedness = safety.detect(frame)
                hand_blocked = len(hand_landmarks) > 0
                if hand_blocked:
                    safety_reasons.append("hand")
                    safety.draw_landmarks(display, hand_landmarks, handedness)
            except Exception as e:
                print(f"[WARN] Hand safety detection failed: {e}")
        work_zone_blocked = work_zone_safety.update(frame, display)
        if work_zone_blocked:
            safety_reasons.append(work_zone_safety.last_reason or "work-zone")
        hand_blocked = hand_blocked or work_zone_blocked
        blocks = detect_coloured_blocks(frame)                    # find all coloured blocks
        zone_blocks = [(x, y, c) for x, y, c in blocks
                       if PZ_X1 <= x <= PZ_X2 and PZ_Y1 <= y <= PZ_Y2]   # only those in zone
        unpicked = [b for b in zone_blocks if not matches(b, picked_ids)]  # minus already-picked

        now = time.time()
        if now - last_drop_zone_scan >= DROP_ZONE_SCAN_INTERVAL:
            detected_zones = detect_drop_zones(frame)
            if detected_zones:
                drop_zones.update(detected_zones)
            last_drop_zone_scan = now

        # ── Speed computation ──
        effective_speed = compute_pace_speed(list(intervals))
        dobotArm.set_speed(api, effective_speed)

        # ── Drawing ──
        draw_status_panel(display, state, effective_speed, hand_blocked,
                          sum(intervals) / len(intervals) if intervals else None,
                          block_count, colour_counts, drop_zones)

        # Highlight unpicked blocks with colour circles
        for bx, by, bc in unpicked:
            cv2.circle(display, (bx, by), 8, COLOUR_BGR.get(bc, (255, 255, 255)), -1)
            cv2.circle(display, (bx, by), 8, (255, 255, 255), 1)

        # Grey out already-picked blocks
        for bx, by, bc in zone_blocks:
            if matches((bx, by, bc), picked_ids):
                cv2.circle(display, (bx, by), 6, (100, 100, 100), -1)

        cv2.putText(display, f"Unpicked: {len(unpicked)}  Zone: {len(zone_blocks)}",
                    (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if safety_reasons:
            cv2.putText(display, f"Blocked: {','.join(safety_reasons)}",
                        (10, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # ── Key input ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            picked_ids.clear()
            print("[RESET] Cleared picked-block memory")

        # ── Enforce minimum 2-second gap between pick cycles ──
        # Prevents the robot from immediately re-entering a cycle after
        # returning to ready (gives the human time to place another block).
        if state not in ("watching", "pick_move", "pick_lower"):
            last_cycle_time = now
        if state == "watching" and now - last_cycle_time < 2.0:
            cv2.imshow(WINDOW_NAME, display)
            continue

        # ── Manual test mode (M key) ──
        # Blocking — camera feed freezes during the move.
        if key == ord('m') and state == "watching":
            print("\n[TEST] Moving to test coordinate (200, 0, 40)...")
            if hand_blocked:
                print("[TEST] Blocked by hand safety")
                cv2.imshow(WINDOW_NAME, display)
                continue
            safe_move_to_xyz(api, 200, 0, 40)
            print("[TEST] Returning to ready...")
            safe_move_to_xyz(api, READY_X, READY_Y, Z_SAFE)
            print("[TEST] Done\n")
            cv2.imshow(WINDOW_NAME, display)
            continue

        # ═════════════════════════════════════════════════════════════
        # STATE MACHINE
        #
        # Each movement state calls a blocking move function. The camera
        # feed freezes during the move (the imshow further down shows the
        # frame captured before the move started). When the move finishes,
        # the next loop iteration reads a fresh frame and enters the next
        # state.
        # ═════════════════════════════════════════════════════════════

        # ── WATCHING ──
        # Look for a new block. Once a block is seen for HOLD_FRAMES
        # consecutive frames (debounce), commit to picking it.
        if hand_blocked:
            active_block = None
            hold_count = 0
            cv2.imshow(WINDOW_NAME, display)
            continue

        if state == "watching":
            if not unpicked:
                active_block = None
                hold_count = 0
                cv2.imshow(WINDOW_NAME, display)
                continue

            if active_block is None:
                active_block = unpicked[0]
                hold_count = 0
                print(f"[TRACKING] {active_block[2]} block at px ({active_block[0]}, {active_block[1]})")
            elif not matches(active_block, unpicked):
                # The block we were tracking disappeared — reset
                active_block = None
                hold_count = 0
                cv2.imshow(WINDOW_NAME, display)
                continue

            hold_count += 1

            if hold_count >= HOLD_FRAMES and active_block is not None:
                target_pixel = active_block[:2]
                target_colour = active_block[2]
                if target_colour not in drop_zones:
                    print(f"[WAIT] No {target_colour} drop zone detected yet")
                    active_block = None
                    hold_count = 0
                    cv2.imshow(WINDOW_NAME, display)
                    continue
                rx, ry = pixel_to_robot(target_pixel[0], target_pixel[1], H_matrix)
                if not is_robot_xy_safe(rx, ry):
                    print(f"[BLOCKED] Target maps outside robot workspace: ({rx:.1f}, {ry:.1f})")
                    active_block = None
                    hold_count = 0
                    cv2.imshow(WINDOW_NAME, display)
                    continue
                target_robot = (rx, ry)
                print(f"\n[NEW {target_colour.upper()} BLOCK] "
                      f"px=({target_pixel[0]},{target_pixel[1]}) "
                      f"→ robot ({rx:.1f}, {ry:.1f})")
                state = "pick_move"
                last_cycle_time = time.time()
                print(f"[STATE] → pick_move")

        # ── PICK: Move XY above block ──
        elif state == "pick_move":
            rx, ry = target_robot
            print(f"[PICK MOVE] ({rx:.1f}, {ry:.1f}, Z={Z_SAFE})")
            state = "pick_lower" if safe_move_to_xyz(api, rx, ry, Z_SAFE) else "watching"

        # ── PICK: Lower to Z_PICK, then grip with verification & retries ──
        elif state == "pick_lower":
            rx, ry = target_robot
            print(f"[PICK LOWER] ({rx:.1f}, {ry:.1f}, Z={Z_PICK})")
            if not safe_move_to_xyz(api, rx, ry, Z_PICK):
                state = "watching"
                continue

            # Try closing gripper and verify by raising and re-checking presence visually.
            success = False
            for attempt in range(PICK_RETRY_COUNT + 1):
                print(f"[GRIPPER] Closing... (attempt {attempt+1})")
                dobotArm.close_gripper(api)
                # Raise slightly to clear surface for a quick check
                if not safe_move_to_xyz(api, rx, ry, PICK_RAISE_CHECK):
                    break

                # Minimal visual check: capture a frame and test if color still present near pixel
                ret_chk, chk_frame = cap.read()
                if ret_chk:
                    chk_frame = cv2.remap(chk_frame, map1, map2, cv2.INTER_LINEAR)
                    hsv = cv2.cvtColor(cv2.GaussianBlur(chk_frame, (3, 3), 0), cv2.COLOR_BGR2HSV)
                    x_px, y_px = int(target_pixel[0]), int(target_pixel[1])
                    x0, y0 = max(0, x_px - 8), max(0, y_px - 8)
                    x1, y1 = min(hsv.shape[1]-1, x_px + 8), min(hsv.shape[0]-1, y_px + 8)
                    roi = hsv[y0:y1, x0:x1]
                    ranges = COLOUR_HSV.get(target_colour, ())
                    still_there = False
                    for r in ranges:
                        m = cv2.inRange(roi, np.array(r[0]), np.array(r[1]))
                        if cv2.countNonZero(m) > 50:
                            still_there = True
                            break

                else:
                    still_there = True

                if not still_there:
                    print("[PICK VERIFY] block no longer visible in ROI → assumed picked")
                    success = True
                    break

                # optional: try suction if available on second attempt
                if attempt == 0:
                    try:
                        dobotArm.start_pump(api)
                        print("[PICK] Started pump to assist grip")
                    except Exception:
                        pass

                # if not successful, reopen and try again
                dobotArm.open_gripper(api)
                if not safe_move_to_xyz(api, rx, ry, Z_PICK):
                    break

            if not success:
                print("[ERROR] Failed to pick block after retries — skipping this block")
                # mark as picked to avoid infinite loop and return to watching
                if target_pixel is not None:
                    picked_ids.add((target_pixel[0], target_pixel[1], target_colour))
                state = "watching"
            else:
                print("[GRIPPER] Pick assumed successful")
                state = "pick_rise"

        # ── PICK: Rise back to Z_SAFE (block is now gripped) ──
        elif state == "pick_rise":
            rx, ry = target_robot
            print(f"[PICK RISE] ({rx:.1f}, {ry:.1f}, Z={Z_SAFE})")
            state = "pick_rotate" if safe_move_to_xyz(api, rx, ry, Z_SAFE) else "watching"

        # ── PICK: Rotate the wrist 90° to clear the camera view ──
        elif state == "pick_rotate":
            print(f"[PICK ROTATE] 90°")
            if not safe_rotate_end_effector(api, 90):
                state = "watching"
                continue
            print(f"[PICK] {target_colour} block done")
            state = "place_move"

        # ── PLACE: Move XY to the colour's drop zone ──
        elif state == "place_move":
            if target_colour not in drop_zones:
                print(f"[WAIT] Lost {target_colour} drop zone; waiting for detection")
                cv2.imshow(WINDOW_NAME, display)
                continue
            dx, dy = drop_zones[target_colour]["robot"]
            print(f"[PLACE MOVE] ({dx:.1f}, {dy:.1f}, Z={Z_SAFE})")
            state = "place_drop" if safe_move_to_xyz(api, dx, dy, Z_SAFE) else "watching"

        # ── PLACE: Open gripper (release block), record stats ──
        elif state == "place_drop":
            dx, dy = drop_zones[target_colour]["robot"]
            dobotArm.open_gripper(api)
            dobotArm.stop_pump(api)
            print(f"[DROP] released at ({dx:.1f}, {dy:.1f})")

            # Record the time interval since the last pick
            now = time.time()
            if block_count > 0:
                intervals.append(now - last_pick_time)
            last_pick_time = now
            block_count += 1
            colour_counts[target_colour] += 1

            # Mark this block as picked (by pixel coordinate + colour)
            if target_pixel is not None:
                picked_ids.add((target_pixel[0], target_pixel[1], target_colour))
                active_block = None
                hold_count = 0
            print(f"[PLACE] #{colour_counts[target_colour]} {target_colour} delivered")
            state = "place_rotate"
            print(f"[STATE] → place_rotate")

        # ── PLACE: Rotate wrist back to 0° (neutral) ──
        elif state == "place_rotate":
            print(f"[PLACE ROTATE] 0°")
            state = "return_xy" if safe_rotate_end_effector(api, 0) else "watching"

        # ── RETURN: Move back to the ready position ──
        elif state == "return_xy":
            print(f"[RETURN] ({READY_X}, {READY_Y}, Z={Z_SAFE})")
            safe_move_to_xyz(api, READY_X, READY_Y, Z_SAFE)
            pace = sum(intervals) / len(intervals) if intervals else None
            print(f"[READY] Pace: {pace:.1f}s" if pace else "[READY]")
            print()
            state = "watching"

        # ── Show the annotated camera frame ──
        cv2.imshow(WINDOW_NAME, display)

        # Export a JPEG for the UI server to show (atomic write)
        try:
            ui_dir = os.path.join(os.path.dirname(__file__), 'ui')
            os.makedirs(ui_dir, exist_ok=True)
            tmp_path = os.path.join(ui_dir, 'latest.jpg.tmp')
            out_path = os.path.join(ui_dir, 'latest.jpg')
            ret_jpg, jpg = cv2.imencode('.jpg', display, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret_jpg:
                with open(tmp_path, 'wb') as f:
                    f.write(jpg.tobytes())
                try:
                    os.replace(tmp_path, out_path)
                except Exception:
                    with open(out_path, 'wb') as f:
                        f.write(jpg.tobytes())
        except Exception:
            pass

    # ── Cleanup on quit ──
    cap.release()
    cv2.destroyAllWindows()
    dobotArm.move_to_home(api)
    print("[SYSTEM] Demo ended.")


if __name__ == "__main__":
    main()
