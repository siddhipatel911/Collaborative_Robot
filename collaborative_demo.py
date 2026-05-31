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
from pathlib import Path

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
OBJECT_TEMPLATE_DIR = "object_templates"
OBJECT_ROUTE_FILE = "object_routes.json"
OBJECT_DROP_ZONE_FILE = "object_drop_zones.json"
USE_TEMPLATE_OBJECT_DETECTION = False
GENERIC_OBJECT_NAME = "object"
GENERIC_DROP_KEY = "generic"
GENERIC_DEBUG_OBJECT_MASK = False

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
TEMPLATE_SCAN_INTERVAL = 0.4
TEMPLATE_MATCH_THRESHOLD = 0.36
TEMPLATE_MIN_SIZE = 16
TEMPLATE_MAX_SIZE = 120
TEMPLATE_WIDTHS = (18, 24, 32, 42, 56, 72, 96, 120)
TEMPLATE_ROTATIONS = (-30, -15, 0, 15, 30, 45, 90)
GENERIC_WARMUP_FRAMES = 25
GENERIC_MIN_AREA = 450
GENERIC_MIN_BOX_WIDTH = 18
GENERIC_MIN_BOX_HEIGHT = 18
GENERIC_MAX_ASPECT_RATIO = 4.8
GENERIC_MIN_FILL_RATIO = 0.24
GENERIC_BORDER_MARGIN = 8
GENERIC_GRAY_DIFF_THRESHOLD = 18
GENERIC_COLOR_DIFF_THRESHOLD = 18
PICK_X_OFFSET_MM = 0.0
PICK_Y_OFFSET_MM = 0.0
MAX_OBJECTS_PER_FRAME = 6
MOTION_MIN_AREA = 6500
MOTION_WARMUP_FRAMES = 20
MOTION_BLOCK_FRAMES = 6
MOTION_MIN_BOX_WIDTH = 45
MOTION_MIN_BOX_HEIGHT = 45
MOTION_MIN_FILL_RATIO = 0.18
SKIN_MIN_AREA = 6500
SKIN_MIN_BOX_WIDTH = 45
SKIN_MIN_BOX_HEIGHT = 45
SKIN_MIN_FILL_RATIO = 0.25
SAFETY_ZONE_MARGIN = 0
ENABLE_MOTION_SAFETY = False

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

DEFAULT_OBJECT_DROP_ZONES = {
    "generic": [260, 0],
    "redstrip": [260, -120],
    "pinkblock": [260, -80],
    "purpleblock": [260, -40],
    "greyblock": [260, 0],
    "chargerblock": [260, 40],
    "ledbox": [260, 80],
    "lipbalm": [260, 120],
    "red": [280, -80],
    "green": [280, 0],
    "blue": [280, 80],
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


def robot_to_pixel(x, y, H):
    p = np.array([x, y, 1.0])
    inv_h = np.linalg.inv(H)
    uv = inv_h @ p
    uv /= uv[2]
    return int(round(uv[0])), int(round(uv[1]))


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


def load_object_routes(path=OBJECT_ROUTE_FILE):
    defaults = {}
    if not os.path.exists(path):
        return defaults
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            defaults.update({str(k): str(v) for k, v in loaded.items()})
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}; using default object routes")
    return defaults


OBJECT_ROUTES = load_object_routes()


def load_object_drop_zones(path=OBJECT_DROP_ZONE_FILE):
    zones = DEFAULT_OBJECT_DROP_ZONES.copy()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                for name, xy in loaded.items():
                    if isinstance(xy, (list, tuple)) and len(xy) == 2:
                        zones[str(name).lower()] = [float(xy[0]), float(xy[1])]
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}; using default drop coordinates")

    safe_zones = {}
    for name, xy in zones.items():
        x, y = xy
        if is_robot_xy_safe(x, y):
            safe_zones[name] = {
                "pixel": None,
                "robot": (x, y),
                "area": 0,
                "score": 1.0,
                "kind": "fixed",
            }
        else:
            print(f"[WARN] Drop zone for {name} is outside workspace and was ignored: ({x}, {y})")
    return safe_zones


class TemplateObjectDetector:
    def __init__(self, template_dir=OBJECT_TEMPLATE_DIR):
        self.templates = []
        self._load_templates(template_dir)

    def _load_templates(self, template_dir):
        root = Path(template_dir)
        if not USE_TEMPLATE_OBJECT_DETECTION:
            print("[OBJECTS] Template detection disabled; using generic object pickup")
            return
        if not root.exists():
            print(f"[OBJECTS] Template folder not found: {template_dir}")
            return
        for path in sorted(root.iterdir()):
            if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            image = cv2.imread(str(path))
            if image is None:
                continue
            name = path.stem.lower()
            image = self._crop_template_object(image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            variants = []
            for angle in TEMPLATE_ROTATIONS:
                rotated = self._rotate_bound(gray, angle)
                edges = cv2.Canny(rotated, 40, 130)
                for target_w in TEMPLATE_WIDTHS:
                    scale = target_w / max(1, edges.shape[1])
                    w = int(edges.shape[1] * scale)
                    h = int(edges.shape[0] * scale)
                    if w < TEMPLATE_MIN_SIZE or h < TEMPLATE_MIN_SIZE:
                        continue
                    if w > TEMPLATE_MAX_SIZE or h > TEMPLATE_MAX_SIZE:
                        continue
                    variants.append(cv2.resize(edges, (w, h), interpolation=cv2.INTER_AREA))
            if variants:
                self.templates.append({"name": name, "variants": variants})
                print(f"[OBJECTS] {name}: {len(variants)} variant(s), crop={image.shape[1]}x{image.shape[0]}")
        print(f"[OBJECTS] Loaded {len(self.templates)} object template(s)")

    def _crop_template_object(self, image):
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        border = np.concatenate([
            image[:max(3, h // 20), :, :].reshape(-1, 3),
            image[-max(3, h // 20):, :, :].reshape(-1, 3),
            image[:, :max(3, w // 20), :].reshape(-1, 3),
            image[:, -max(3, w // 20):, :].reshape(-1, 3),
        ])
        bg = np.median(border, axis=0).astype(np.float32)
        dist = np.linalg.norm(image.astype(np.float32) - bg, axis=2)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[(dist > 35) | (sat > 70) | (val < 90)] = 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        cx_img, cy_img = w / 2, h / 2
        def score(c):
            area = cv2.contourArea(c)
            x, y, bw, bh = cv2.boundingRect(c)
            cx, cy = x + bw / 2, y + bh / 2
            center_penalty = np.hypot(cx - cx_img, cy - cy_img)
            return area - center_penalty * 2

        c = max(contours, key=score)
        x, y, bw, bh = cv2.boundingRect(c)
        pad = int(max(bw, bh) * 0.18)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y + bh + pad)
        if (x1 - x0) < 20 or (y1 - y0) < 20:
            return image
        return image[y0:y1, x0:x1]

    def _rotate_bound(self, image, angle):
        if angle == 0:
            return image
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(image, matrix, (new_w, new_h), borderValue=255)

    def detect(self, frame, region=None):
        if not self.templates:
            return []
        if region is None:
            region = (PZ_X1, PZ_Y1, PZ_X2, PZ_Y2)
        x1, y1, x2, y2 = region
        zone = frame[y1:y2, x1:x2]
        if zone.size == 0:
            return []
        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 50, 150)

        matches_found = []
        best_debug = []
        for template in self.templates:
            best = None
            for tmpl in template["variants"]:
                th, tw = tmpl.shape[:2]
                if th >= edges.shape[0] or tw >= edges.shape[1]:
                    continue
                result = cv2.matchTemplate(edges, tmpl, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(result)
                if best is None or score > best["score"]:
                    best = {"score": score, "loc": loc, "size": (tw, th)}
            if best is not None:
                best_debug.append((template["name"], best["score"]))
            if best and best["score"] >= TEMPLATE_MATCH_THRESHOLD:
                x, y = best["loc"]
                tw, th = best["size"]
                cx = x1 + x + tw // 2
                cy = y1 + y + th // 2
                route = OBJECT_ROUTES.get(template["name"], template["name"])
                matches_found.append((cx, cy, template["name"], best["score"], route, (x1 + x, y1 + y, tw, th)))

        matches_found.sort(key=lambda item: item[3], reverse=True)
        if not matches_found and best_debug:
            debug = ", ".join(f"{name}:{score:.2f}" for name, score in sorted(best_debug, key=lambda x: x[1], reverse=True)[:3])
            print(f"[OBJECTS] No template above {TEMPLATE_MATCH_THRESHOLD:.2f}. Best: {debug}")
        return self._suppress_overlaps(matches_found)

    def _suppress_overlaps(self, detections):
        kept = []
        for det in detections:
            x, y, _, _, _, _ = det
            if all(abs(x - k[0]) > PICK_PROXIMITY_PX or abs(y - k[1]) > PICK_PROXIMITY_PX for k in kept):
                kept.append(det)
        return kept


template_detector = TemplateObjectDetector() if USE_TEMPLATE_OBJECT_DETECTION else None


class GenericObjectDetector:
    """Detect a solid newly placed object in the placement zone."""

    def __init__(self):
        self.background_gray = None
        self.background_lab = None
        self.frames = 0

    def detect(self, frame):
        zone = frame[PZ_Y1:PZ_Y2, PZ_X1:PZ_X2]
        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        lab = cv2.cvtColor(zone, cv2.COLOR_BGR2LAB)
        lab = cv2.GaussianBlur(lab, (11, 11), 0)

        if self.background_gray is None:
            self.background_gray = gray.astype("float")
            self.background_lab = lab.astype("float")
            return []

        self.frames += 1
        if self.frames <= GENERIC_WARMUP_FRAMES:
            cv2.accumulateWeighted(gray, self.background_gray, 0.08)
            cv2.accumulateWeighted(lab, self.background_lab, 0.08)
            return []

        gray_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.background_gray))
        lab_delta = lab.astype(np.float32) - cv2.convertScaleAbs(self.background_lab).astype(np.float32)
        color_delta = np.linalg.norm(lab_delta, axis=2).astype(np.uint8)
        gray_mask = cv2.threshold(gray_delta, GENERIC_GRAY_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        color_mask = cv2.threshold(color_delta, GENERIC_COLOR_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.bitwise_or(gray_mask, color_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < GENERIC_MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < GENERIC_MIN_BOX_WIDTH or h < GENERIC_MIN_BOX_HEIGHT:
                continue
            if (
                x <= GENERIC_BORDER_MARGIN or y <= GENERIC_BORDER_MARGIN or
                x + w >= mask.shape[1] - GENERIC_BORDER_MARGIN or
                y + h >= mask.shape[0] - GENERIC_BORDER_MARGIN
            ):
                continue
            aspect = max(w / max(1, h), h / max(1, w))
            if aspect > GENERIC_MAX_ASPECT_RATIO:
                continue
            fill_ratio = area / max(1, w * h)
            if fill_ratio < GENERIC_MIN_FILL_RATIO:
                continue
            M = cv2.moments(c)
            if M["m00"]:
                cx = PZ_X1 + int(M["m10"] / M["m00"])
                cy = PZ_Y1 + int(M["m01"] / M["m00"])
            else:
                cx = PZ_X1 + x + w // 2
                cy = PZ_Y1 + y + h // 2
            candidates.append((cx, cy, GENERIC_OBJECT_NAME, GENERIC_DROP_KEY, area, (PZ_X1 + x, PZ_Y1 + y, w, h)))

        if GENERIC_DEBUG_OBJECT_MASK:
            cv2.imshow("Generic Object Mask", mask)
        if not candidates:
            cv2.accumulateWeighted(gray, self.background_gray, 0.03)
            cv2.accumulateWeighted(lab, self.background_lab, 0.03)
            return []

        candidates.sort(key=lambda item: item[4], reverse=True)
        return candidates[:MAX_OBJECTS_PER_FRAME]

    def reset(self):
        self.background_gray = None
        self.background_lab = None
        self.frames = 0

    def forget_region(self, box):
        # Force a short relearn after a pick/place cycle so removed objects do
        # not remain as "negative" background-difference detections.
        self.reset()


generic_detector = GenericObjectDetector()


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


def detect_pickable_objects(frame):
    """Detect one newly placed object, without shape/template sorting."""
    if not USE_TEMPLATE_OBJECT_DETECTION:
        return generic_detector.detect(frame)

    """Use templates first, then colour fallback for older tag-style parts."""
    objects = []
    for cx, cy, name, score, route, box in template_detector.detect(frame):
        objects.append((cx, cy, name, route, score, box))

    for cx, cy, colour in detect_coloured_blocks(frame):
        if any(abs(cx - obj[0]) < PICK_PROXIMITY_PX and abs(cy - obj[1]) < PICK_PROXIMITY_PX for obj in objects):
            continue
        objects.append((cx, cy, colour, colour, 1.0, None))

    return objects


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
        moving = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < MOTION_MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(c)
            fill_ratio = area / max(1, w * h)
            if w < MOTION_MIN_BOX_WIDTH or h < MOTION_MIN_BOX_HEIGHT:
                continue
            if fill_ratio < MOTION_MIN_FILL_RATIO:
                continue
            moving.append(c)

        if moving:
            self.motion_hits += 1
            largest = max(moving, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            box = (x + x1, y + y1, w, h)
            return self.motion_hits >= MOTION_BLOCK_FRAMES, box

        self.motion_hits = 0
        cv2.accumulateWeighted(gray, self.background, 0.04)
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
        skin = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < SKIN_MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(c)
            fill_ratio = area / max(1, w * h)
            if w < SKIN_MIN_BOX_WIDTH or h < SKIN_MIN_BOX_HEIGHT:
                continue
            if fill_ratio < SKIN_MIN_FILL_RATIO:
                continue
            skin.append(c)
        if not skin:
            return False, None
        largest = max(skin, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return True, (x + x1, y + y1, w, h)

    def update(self, frame, display):
        moving, motion_box = self._detect_motion(frame)
        skin_like, skin_box = self._detect_skin_like(frame)

        # Skin colour alone is too close to the wood table. It is only visual
        # context now; blocking comes from real work-zone motion or MediaPipe.
        blocked = moving
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

    cv2.putText(display, f"Picked: {block_count}", (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Placement zone rectangle
    cv2.rectangle(display, (PZ_X1, PZ_Y1), (PZ_X2, PZ_Y2), (0, 255, 0), 2)
    cv2.putText(display, "PLACEMENT ZONE", (PZ_X1, PZ_Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # One fixed drop location for the generic object flow.
    visible_drop_zones = {
        GENERIC_DROP_KEY: drop_zones[GENERIC_DROP_KEY]
    } if GENERIC_DROP_KEY in drop_zones else {}
    for colour, zone in visible_drop_zones.items():
        dx, dy = zone["robot"]
        try:
            drop_px, drop_py = robot_to_pixel(dx, dy, H_matrix)
            if 0 <= drop_px < display.shape[1] and 0 <= drop_py < display.shape[0]:
                cv2.drawMarker(display, (drop_px, drop_py), (255, 255, 255),
                               markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2)
                cv2.circle(display, (drop_px, drop_py), 14, (255, 255, 255), 2)
                cv2.putText(display, "DROP", (drop_px + 14, drop_py - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        except Exception:
            pass
        pixel = zone.get("pixel")
        if pixel is None:
            continue
        px, py = pixel
        cv2.circle(display, (px, py), 14, (255, 255, 255), 2)
        cv2.putText(display, "DROP", (px + 12, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)


def draw_detected_objects(display, objects, active=None):
    items = []
    if active is not None:
        items = [active]
    elif objects:
        items = objects[:1]

    for cx, cy, name, route, score, box in items:
        color = COLOUR_BGR.get(route, (255, 255, 255))
        if box:
            x, y, w, h = box
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        cv2.circle(display, (cx, cy), 8, color, -1)
        cv2.putText(display, "TARGET", (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)


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
    bx, by, bc = block[:3]
    for item in block_list:
        lx, ly, lc = item[:3]
        if lc == bc and abs(bx - lx) < PICK_PROXIMITY_PX and abs(by - ly) < PICK_PROXIMITY_PX:
            return True
    return False


def landmarks_center_in_work_zone(landmarks, frame_shape):
    if not landmarks:
        return False
    h, w = frame_shape[:2]
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    return PZ_X1 <= cx <= PZ_X2 and PZ_Y1 <= cy <= PZ_Y2


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
    target_route = None                                   # detected destination zone key
    target_box = None                                     # camera box of current target
    last_pick_time = time.time()                          # when the last block was placed
    picked_ids = set()                                    # set of (px, py, colour) already picked
    hold_count = 0                                        # frames the current block has been stable
    active_block = None                                   # (px, py, colour) we're currently watching
    last_cycle_time = 0                                   # timestamp of last completed pick cycle
    drop_zones = load_object_drop_zones()
    print(f"[DROP ZONES] Loaded {len(drop_zones)} fixed drop coordinate(s)")

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
                filtered_landmarks = []
                filtered_handedness = []
                for idx, landmarks in enumerate(hand_landmarks):
                    if landmarks_center_in_work_zone(landmarks, frame.shape):
                        filtered_landmarks.append(landmarks)
                        if idx < len(handedness):
                            filtered_handedness.append(handedness[idx])
                hand_blocked = len(filtered_landmarks) > 0
                if hand_blocked:
                    safety_reasons.append("hand")
                    safety.draw_landmarks(display, filtered_landmarks, filtered_handedness)
            except Exception as e:
                print(f"[WARN] Hand safety detection failed: {e}")
        work_zone_blocked = False
        if ENABLE_MOTION_SAFETY:
            work_zone_blocked = work_zone_safety.update(frame, display)
        if work_zone_blocked:
            safety_reasons.append(work_zone_safety.last_reason or "work-zone")
        hand_blocked = hand_blocked or work_zone_blocked
        zone_objects = detect_pickable_objects(frame)
        unpicked = [obj for obj in zone_objects if not matches(obj, picked_ids)]  # minus already-picked

        now = time.time()

        # ── Speed computation ──
        effective_speed = compute_pace_speed(list(intervals))
        dobotArm.set_speed(api, effective_speed)

        # ── Drawing ──
        draw_status_panel(display, state, effective_speed, hand_blocked,
                          sum(intervals) / len(intervals) if intervals else None,
                          block_count, colour_counts, drop_zones)

        # Highlight only the active/next target to keep the preview readable.
        draw_detected_objects(display, unpicked, active_block)
        if not unpicked and generic_detector.frames < GENERIC_WARMUP_FRAMES:
            cv2.putText(display, "Learning empty zone - keep clear",
                        (PZ_X1 + 20, PZ_Y1 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        if safety_reasons:
            cv2.putText(display, f"Blocked: {','.join(safety_reasons)}",
                        (10, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # ── Key input ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            picked_ids.clear()
            generic_detector.reset()
            active_block = None
            hold_count = 0
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
                print(f"[TRACKING] {active_block[2]} at px ({active_block[0]}, {active_block[1]})")
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
                target_route = active_block[3]
                target_box = active_block[5] if len(active_block) > 5 else None
                if target_route not in drop_zones:
                    print(f"[WAIT] No fixed drop coordinate for {target_route}")
                    active_block = None
                    hold_count = 0
                    cv2.imshow(WINDOW_NAME, display)
                    continue
                rx, ry = pixel_to_robot(target_pixel[0], target_pixel[1], H_matrix)
                rx += PICK_X_OFFSET_MM
                ry += PICK_Y_OFFSET_MM
                if not is_robot_xy_safe(rx, ry):
                    print(f"[BLOCKED] Target maps outside robot workspace: ({rx:.1f}, {ry:.1f})")
                    active_block = None
                    hold_count = 0
                    cv2.imshow(WINDOW_NAME, display)
                    continue
                target_robot = (rx, ry)
                print(f"\n[NEW {target_colour.upper()}] "
                      f"px=({target_pixel[0]},{target_pixel[1]}) "
                      f"-> {target_route} zone -> robot ({rx:.1f}, {ry:.1f})")
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

            print("[GRIPPER] Closing...")
            dobotArm.close_gripper(api)
            state = "place_move" if safe_move_to_xyz(api, rx, ry, Z_SAFE) else "watching"
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
            if target_route not in drop_zones:
                print(f"[WAIT] Lost {target_route} drop zone; waiting for detection")
                cv2.imshow(WINDOW_NAME, display)
                continue
            dx, dy = drop_zones[target_route]["robot"]
            print(f"[PLACE MOVE] ({dx:.1f}, {dy:.1f}, Z={Z_SAFE})")
            state = "place_drop" if safe_move_to_xyz(api, dx, dy, Z_SAFE) else "watching"

        # ── PLACE: Open gripper (release block), record stats ──
        elif state == "place_drop":
            dx, dy = drop_zones[target_route]["robot"]
            print(f"[PLACE LOWER] ({dx:.1f}, {dy:.1f}, Z={Z_PICK})")
            if not safe_move_to_xyz(api, dx, dy, Z_PICK):
                state = "watching"
                continue

            dobotArm.open_gripper(api)
            dobotArm.stop_pump(api)
            print(f"[DROP] released at ({dx:.1f}, {dy:.1f})")
            if not safe_move_to_xyz(api, dx, dy, Z_SAFE):
                state = "watching"
                continue

            # Record the time interval since the last pick
            now = time.time()
            if block_count > 0:
                intervals.append(now - last_pick_time)
            last_pick_time = now
            block_count += 1
            colour_counts.setdefault(target_route, 0)
            colour_counts[target_route] += 1

            # Mark this block as picked (by pixel coordinate + colour)
            if target_pixel is not None:
                picked_ids.add((target_pixel[0], target_pixel[1], target_colour))
                active_block = None
                hold_count = 0
            print(f"[PLACE] #{colour_counts[target_route]} {target_colour} delivered to {target_route}")
            state = "return_xy"
            print("[STATE] -> return_xy")

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
            active_block = None
            target_robot = None
            target_pixel = None
            target_colour = None
            target_route = None
            target_box = None
            hold_count = 0
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
