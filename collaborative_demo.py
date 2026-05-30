"""
collaborative_demo.py — Human places blocks, robot picks & places with pace + mood adaptation.

The robot watches a placement zone via overhead camera, detects new coloured blocks
after the human's hand leaves, picks them up, and sorts them into per-colour drop zones.
It adapts its speed to match both the human's placement pace and their
facial expression mood (detected via MediaPipe).
t
Controls:
  Q  — quit
  R  — reset (clear picked-block memory)
  M  — manual test: move robot to test coordinate (200,0,40) and back
"""

import dobotArm
import lib.DobotDllType as dType
import numpy as np
import cv2
import time
from collections import deque

# ────────────────────── MediaPipe (tasks API) ──────────────────────
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

# ────────────────────── CONFIGURABLE CONSTANTS ──────────────────────
Z_SAFE = 40
Z_PICK = -25

READY_X, READY_Y = 180, 0

DROP_RED = (260, 80)
DROP_GREEN = (260, 40)
DROP_BLUE = (260, 0)

DROP_ZONES = {"red": DROP_RED, "green": DROP_GREEN, "blue": DROP_BLUE}

COLOUR_BGR = {
    "red":   (0, 0, 255),
    "green": (0, 255, 0),
    "blue":  (255, 0, 0),
}

PZ_X1, PZ_Y1 = 150, 150 # top-left corner of the green box
PZ_X2, PZ_Y2 = 500, 500 # bottom-right corner
WINDOW_NAME = "Collaborative Robot Demo"

SPEED_WINDOW = 5
BASE_SPEED = 50
MAX_SPEED = 80
MIN_SPEED = 25
FAST_PACE_S = 3.0
SLOW_PACE_S = 8.0

MOOD_MODIFIERS = {
    "happy":    1.0,
    "focused":  1.0,
    "neutral":  0.9,
    "tired":    0.7,
    "agitated": 0.5,
    "no_face":  1.0,
}

HOLD_FRAMES = 8
PICK_PROXIMITY_PX = 30  # blocks within this many pixels are considered the same

# ────────────────────── MEDIAPIPE ANALYSERS ──────────────────────
class MoodAnalyzer:
    LANDMARK_IDS = {
        "left_eye_top": 159, "left_eye_bottom": 145,
        "right_eye_top": 386, "right_eye_bottom": 374,
        "left_eye_left": 33, "left_eye_right": 133,
        "right_eye_left": 362, "right_eye_right": 263,
        "mouth_top": 13, "mouth_bottom": 14,
        "mouth_left": 61, "mouth_right": 291,
        "left_eyebrow_inner": 105,
        "right_eyebrow_inner": 334,
    }

    def __init__(self, history_frames=30):
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

    def _as_pt(self, lm, w, h):
        return np.array([lm.x * w, lm.y * h])

    def _metrics(self, landmarks, w, h):
        ids = self.LANDMARK_IDS
        le_top = self._as_pt(landmarks[ids["left_eye_top"]], w, h)
        le_bot = self._as_pt(landmarks[ids["left_eye_bottom"]], w, h)
        le_l = self._as_pt(landmarks[ids["left_eye_left"]], w, h)
        le_r = self._as_pt(landmarks[ids["left_eye_right"]], w, h)
        re_top = self._as_pt(landmarks[ids["right_eye_top"]], w, h)
        re_bot = self._as_pt(landmarks[ids["right_eye_bottom"]], w, h)
        re_l = self._as_pt(landmarks[ids["right_eye_left"]], w, h)
        re_r = self._as_pt(landmarks[ids["right_eye_right"]], w, h)
        m_top = self._as_pt(landmarks[ids["mouth_top"]], w, h)
        m_bot = self._as_pt(landmarks[ids["mouth_bottom"]], w, h)
        m_l = self._as_pt(landmarks[ids["mouth_left"]], w, h)
        m_r = self._as_pt(landmarks[ids["mouth_right"]], w, h)
        lb = self._as_pt(landmarks[ids["left_eyebrow_inner"]], w, h)
        rb = self._as_pt(landmarks[ids["right_eyebrow_inner"]], w, h)

        left_ear = (np.linalg.norm(le_top - le_l) + np.linalg.norm(le_bot - le_r)) / (
                    2.0 * np.linalg.norm(le_l - le_r) + 1e-6)
        right_ear = (np.linalg.norm(re_top - re_l) + np.linalg.norm(re_bot - re_r)) / (
                     2.0 * np.linalg.norm(re_l - re_r) + 1e-6)
        ear = (left_ear + right_ear) / 2.0
        mouth_width = np.linalg.norm(m_r - m_l)
        mar = np.linalg.norm(m_bot - m_top) / (mouth_width + 1e-6)
        brow_dist = np.linalg.norm(lb - rb)
        return {"ear": ear, "mar": mar, "brow_dist": brow_dist}

    def update(self, frame_rgb, timestamp_ms):
        if self.model is None:
            self.current_mood = "no_face"
            self.current_modifier = 1.0
            return self.current_mood

        h, w, _ = frame_rgb.shape
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.model.detect_for_video(mp_img, timestamp_ms)

        if not result.face_landmarks:
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

    def _classify(self, m):
        if m["mar"] > 0.35:
            return "happy"
        if m["ear"] < 0.18:
            return "tired"
        if m["brow_dist"] < 25:
            return "agitated"
        if 0.25 < m["mar"] < 0.35 and m["ear"] > 0.22:
            return "focused"
        return "neutral"

    def _majority_mood(self):
        non_face = [x for x in self.history if isinstance(x, str)]
        if len(non_face) > len(self.history) // 2:
            return "no_face"
        metrics = [x for x in self.history if isinstance(x, dict)]
        return self._classify(metrics[-1]) if metrics else "no_face"


# ────────────────────── CAMERA SETUP ──────────────────────
api = dType.load()
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[WARN] Camera index 1 failed, trying index 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[FATAL] No camera found")
    exit(1)
print(f"[CAMERA] Opened")

H_matrix = np.load("HomographyMatrix.npy")
data = np.load("camera_params.npz")
camera_matrix, dist_coeffs = data["camera_matrix"], data["dist_coeffs"]

ret, frame = cap.read()
if not ret:
    print("[FATAL] Cannot read from camera")
    exit(1)

h, w = frame.shape[:2]
new_K, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1)
map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix, dist_coeffs, None, new_K, (w, h), cv2.CV_16SC2
)

# Show camera preview before blocking robot init
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
frame_show = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
cv2.imshow(WINDOW_NAME, frame_show)
cv2.waitKey(1)

# ────────────────────── ROBOT SETUP ──────────────────────
dobotArm.initialize_robot(api)
dobotArm.open_gripper(api)

# ────────────────────── HELPERS ──────────────────────
def pixel_to_robot(u, v, H):
    p = np.array([u, v, 1])
    xy = H @ p
    xy /= xy[2]
    return xy[0], xy[1]


COLOUR_HSV = {
    "red":   ([(0, 120, 70), (10, 255, 255)], [(170, 120, 70), (180, 255, 255)]),
    "green": ([(40, 80, 70), (80, 255, 255)],),
    "blue":  ([(90, 80, 70), (130, 255, 255)],),
}


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


def compute_pace_speed(intervals):
    if len(intervals) < 1:
        return BASE_SPEED
    avg = sum(intervals) / len(intervals)
    if avg < FAST_PACE_S:
        return min(MAX_SPEED, int(BASE_SPEED * FAST_PACE_S / max(avg, 0.5)))
    elif avg > SLOW_PACE_S:
        return max(MIN_SPEED, int(BASE_SPEED * SLOW_PACE_S / avg))
    return BASE_SPEED


# ────────────────────── DRAWING HELPERS ──────────────────────
def draw_status_panel(display, state, speed, mood, pace, block_count, colour_counts):
    cv2.putText(display, f"Speed: {speed}%  Mood: {mood}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(display, f"State: {state}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    if pace is not None:
        cv2.putText(display, f"Avg pace: {pace:.1f}s", (10, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 2)

    y0 = 105
    for i, (c, cnt) in enumerate(colour_counts.items()):
        cv2.putText(display, f"{c}: {cnt}", (10, y0 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR_BGR.get(c, (255, 255, 255)), 2)

    cv2.rectangle(display, (PZ_X1, PZ_Y1), (PZ_X2, PZ_Y2), (0, 255, 0), 2)
    cv2.putText(display, "PLACEMENT ZONE", (PZ_X1, PZ_Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    lx, ly = 540, 145
    for colour, (dx, dy) in DROP_ZONES.items():
        cv2.circle(display, (lx, ly), 8, COLOUR_BGR[colour], -1)
        cv2.putText(display, f"{colour} ({dx},{dy})", (lx + 14, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOUR_BGR[colour], 2)
        ly += 22


def draw_face_expression(display, mood, x=10, y=100, size=35):
    cx, cy = x + size, y + size
    cv2.circle(display, (cx, cy), size, (200, 200, 255), -1)
    cv2.circle(display, (cx, cy), size, (100, 100, 200), 2)

    if mood == "happy":
        cv2.ellipse(display, (cx - 10, cy - 5), (5, 3), 0, 180, 360, (0, 0, 0), 2)
        cv2.ellipse(display, (cx + 10, cy - 5), (5, 3), 0, 180, 360, (0, 0, 0), 2)
        cv2.ellipse(display, (cx, cy + 8), (8, 6), 0, 0, 180, (0, 0, 0), 2)
        cv2.circle(display, (cx - 12, cy + 12), 4, (200, 100, 150), -1)
        cv2.circle(display, (cx + 12, cy + 12), 4, (200, 100, 150), -1)
    elif mood == "agitated":
        cv2.ellipse(display, (cx - 10, cy - 6), (4, 4), 0, 180, 360, (0, 0, 0), 2)
        cv2.ellipse(display, (cx + 10, cy - 6), (4, 4), 0, 180, 360, (0, 0, 0), 2)
        cv2.line(display, (cx - 16, cy - 16), (cx - 4, cy - 12), (0, 0, 0), 2)
        cv2.line(display, (cx + 4, cy - 12), (cx + 16, cy - 16), (0, 0, 0), 2)
        cv2.line(display, (cx - 6, cy + 10), (cx + 6, cy + 10), (0, 0, 0), 2)
    elif mood == "tired":
        cv2.line(display, (cx - 16, cy - 6), (cx - 4, cy - 6), (0, 0, 0), 2)
        cv2.line(display, (cx + 4, cy - 6), (cx + 16, cy - 6), (0, 0, 0), 2)
        cv2.line(display, (cx - 5, cy + 10), (cx + 5, cy + 10), (0, 0, 0), 2)
    elif mood == "focused":
        cv2.circle(display, (cx - 10, cy - 6), 3, (0, 0, 0), -1)
        cv2.circle(display, (cx + 10, cy - 6), 3, (0, 0, 0), -1)
        cv2.line(display, (cx - 7, cy + 10), (cx + 7, cy + 10), (0, 0, 0), 2)
    else:
        cv2.circle(display, (cx - 10, cy - 6), 4, (0, 0, 0), -1)
        cv2.circle(display, (cx + 10, cy - 6), 4, (0, 0, 0), -1)
        cv2.line(display, (cx - 6, cy + 10), (cx + 6, cy + 10), (0, 0, 0), 2)


# ────────────────────── MAIN LOOP ──────────────────────
def matches(block, block_list):
    bx, by, bc = block
    for lx, ly, lc in block_list:
        if lc == bc and abs(bx - lx) < PICK_PROXIMITY_PX and abs(by - ly) < PICK_PROXIMITY_PX:
            return True
    return False



def main():
    mood_analyzer = MoodAnalyzer()
    frame_ts = int(time.perf_counter() * 1000)

    dobotArm.set_speed(api, BASE_SPEED)
    dobotArm.move_to_xyz(api, READY_X, READY_Y, Z_SAFE)
    print("[SYSTEM] Ready. Waiting for human to place blocks in the zone...")

    intervals = deque(maxlen=SPEED_WINDOW)
    colour_counts = {"red": 0, "green": 0, "blue": 0}
    block_count = 0
    state = "watching"
    target_robot = None
    target_pixel = None
    target_colour = None
    last_pick_time = time.time()
    picked_ids = set()
    hold_count = 0
    active_block = None
    last_cycle_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_ts += 33

        # ── Perception ──
        mood = mood_analyzer.update(frame_rgb, frame_ts)
        blocks = detect_coloured_blocks(frame)
        zone_blocks = [(x, y, c) for x, y, c in blocks
                       if PZ_X1 <= x <= PZ_X2 and PZ_Y1 <= y <= PZ_Y2]
        unpicked = [b for b in zone_blocks if not matches(b, picked_ids)]

        # ── Speed computation ──
        pace_speed = compute_pace_speed(list(intervals))
        modifier = mood_analyzer.current_modifier
        effective_speed = max(MIN_SPEED, min(MAX_SPEED, int(pace_speed * modifier)))
        dobotArm.set_speed(api, effective_speed)

        # ── Drawing ──
        draw_status_panel(display, state, effective_speed, mood,
                          sum(intervals) / len(intervals) if intervals else None,
                          block_count, colour_counts)
        draw_face_expression(display, mood)

        for bx, by, bc in unpicked:
            cv2.circle(display, (bx, by), 8, COLOUR_BGR[bc], -1)
            cv2.circle(display, (bx, by), 8, (255, 255, 255), 1)

        for bx, by, bc in zone_blocks:
            if matches((bx, by, bc), picked_ids):
                cv2.circle(display, (bx, by), 6, (100, 100, 100), -1)

        cv2.putText(display, f"Unpicked: {len(unpicked)}  Zone: {len(zone_blocks)}",
                    (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ── Key input ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            picked_ids.clear()
            print("[RESET] Cleared picked-block memory")

        # ── Enforce minimum 2s between cycles ──
        now = time.time()
        if state not in ("watching", "pick_move", "pick_lower"):
            last_cycle_time = now
        if state == "watching" and now - last_cycle_time < 2.0:
            cv2.imshow(WINDOW_NAME, display)
            continue

        # ── Key: manual test mode (blocking — intentionally freezes feed) ──
        if key == ord('m') and state == "watching":
            print("\n[TEST] Moving to test coordinate (200, 0, 40)...")
            dobotArm.move_to_xyz(api, 200, 0, 40)
            print("[TEST] Returning to ready...")
            dobotArm.move_to_xyz(api, READY_X, READY_Y, Z_SAFE)
            print("[TEST] Done\n")
            cv2.imshow(WINDOW_NAME, display)
            continue

        # ── State machine ──
        #   Movement uses blocking move_to_xyz/rotate_end_effector calls.
        #   Each state makes one blocking call then transitions immediately.
        #   Non-movement states (watching, place_drop) are instant.
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
                active_block = None
                hold_count = 0
                cv2.imshow(WINDOW_NAME, display)
                continue

            hold_count += 1

            if hold_count >= HOLD_FRAMES and active_block is not None:
                target_pixel = active_block[:2]
                target_colour = active_block[2]
                rx, ry = pixel_to_robot(target_pixel[0], target_pixel[1], H_matrix)
                target_robot = (rx, ry)
                print(f"\n[NEW {target_colour.upper()} BLOCK] px=({target_pixel[0]},{target_pixel[1]}) "
                      f"→ robot ({rx:.1f}, {ry:.1f})")
                state = "pick_move"
                last_cycle_time = time.time()
                print(f"[STATE] → pick_move")

        # ── PICK: Move XY above block ──
        elif state == "pick_move":
            rx, ry = target_robot
            print(f"[PICK MOVE] ({rx:.1f}, {ry:.1f}, Z={Z_SAFE})")
            dobotArm.move_to_xyz(api, rx, ry, Z_SAFE)
            state = "pick_lower"

        # ── PICK: Lower to Z_PICK, then grip ──
        elif state == "pick_lower":
            rx, ry = target_robot
            print(f"[PICK LOWER] ({rx:.1f}, {ry:.1f}, Z={Z_PICK})")
            dobotArm.move_to_xyz(api, rx, ry, Z_PICK)
            print("[GRIPPER] Closing...")
            dobotArm.close_gripper(api)
            print("[GRIPPER] Closed")
            state = "pick_rise"

        # ── PICK: Rise to Z_SAFE ──
        elif state == "pick_rise":
            rx, ry = target_robot
            print(f"[PICK RISE] ({rx:.1f}, {ry:.1f}, Z={Z_SAFE})")
            dobotArm.move_to_xyz(api, rx, ry, Z_SAFE)
            state = "pick_rotate"

        # ── PICK: Rotate 90° ──
        elif state == "pick_rotate":
            print(f"[PICK ROTATE] 90°")
            dobotArm.rotate_end_effector(api, 90)
            print(f"[PICK] {target_colour} block done")
            state = "place_move"

        # ── PLACE: Move XY to drop zone ──
        elif state == "place_move":
            dx, dy = DROP_ZONES[target_colour]
            print(f"[PLACE MOVE] ({dx:.1f}, {dy:.1f}, Z={Z_SAFE})")
            dobotArm.move_to_xyz(api, dx, dy, Z_SAFE)
            state = "place_drop"

        # ── PLACE: Open gripper + record (instant, no movement) ──
        elif state == "place_drop":
            dx, dy = DROP_ZONES[target_colour]
            dobotArm.open_gripper(api)
            dobotArm.stop_pump(api)
            print(f"[DROP] released at ({dx:.1f}, {dy:.1f})")

            now = time.time()
            if block_count > 0:
                intervals.append(now - last_pick_time)
            last_pick_time = now
            block_count += 1
            colour_counts[target_colour] += 1

            if target_pixel is not None:
                picked_ids.add((target_pixel[0], target_pixel[1], target_colour))
                active_block = None
                hold_count = 0
            print(f"[PLACE] #{colour_counts[target_colour]} {target_colour} delivered")
            state = "place_rotate"
            print(f"[STATE] → place_rotate")

        # ── PLACE: Rotate 0° back ──
        elif state == "place_rotate":
            print(f"[PLACE ROTATE] 0°")
            dobotArm.rotate_end_effector(api, 0)
            state = "return_xy"

        # ── RETURN: Move to ready ──
        elif state == "return_xy":
            print(f"[RETURN] ({READY_X}, {READY_Y}, Z={Z_SAFE})")
            dobotArm.move_to_xyz(api, READY_X, READY_Y, Z_SAFE)
            pace = sum(intervals) / len(intervals) if intervals else None
            print(f"[READY] Pace: {pace:.1f}s" if pace else "[READY]")
            print()
            state = "watching"

        cv2.imshow(WINDOW_NAME, display)

    cap.release()
    cv2.destroyAllWindows()
    dobotArm.move_to_home(api)
    print("[SYSTEM] Demo ended.")


if __name__ == "__main__":
    main()
