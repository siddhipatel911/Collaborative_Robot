# Project Kizuna (絆) — Implementation Outline for Claude

## Context
You are building a collaborative robotic arm demo for a Toyota manufacturing hackathon using:
- **Dobot Magician** arm (controlled via `dobotArm.py` / `DobotDllType.py`)
- **Orbbec camera** overhead (OpenCV)
- Existing code: `pickCVBlock.py` (state machine: scan plates → scan targets → pick/place)
- Existing calibration: `HomographyMatrix.npy`, `camera_params.npz`

The core idea: reframe the robot from a pick-and-place tool into a **personality-driven collaborative partner** with emotional expression, handoff-based interaction, mood awareness, and narrative-driven safety.

---

## Architecture Overview

```
MAIN LOOP
├── init_robot()
├── init_camera()
├── PersonalityEngine    (face rendering + sound + movement qualities)
├── SafetySystem         (hand detection + retreat + resume protocol)
├── MoodDetector         (MediaPipe face expression analysis)
│
├── ACT 1: Wake & Greet
├── ACT 2: Workspace Scan
├── ACT 3: Handoff Cycles (repeat for N parts)
│   ├── detect_targets()
│   ├── pick_part()
│   ├── move_to_handoff_zone()
│   ├── wait_for_human_take()
│   ├── celebrate()
│   └── [safety interrupt can fire at any point]
├── ACT 4: Celebration & Score
└── ACT 5: Goodbye
```

---

## Files to Create / Modify

### 1. NEW: `personality.py` — Personality Engine
The robot's emotional state machine and expression system.

```python
class Emotion(Enum):
    NEUTRAL, HAPPY, FOCUSED, CAUTIOUS, CONFUSED, APOLOGETIC, EXCITED, SLEEPY

class PersonalityEngine:
    def __init__(self):
        self.emotion = Emotion.NEUTRAL
        self.last_emotion = None

    def set_emotion(self, emotion):
        # transitions emotion, triggers face + sound + movement update

    # Movement quality modifiers (return speed/accel ratios)
    def get_speed_factor(self) -> float
    def get_accel_factor(self) -> float
    # Examples: HAPPY -> 1.0, CAUTIOUS -> 0.4, CONFUSED -> 0.6 jerky

    def get_joint_damping(self) -> float
```

#### Face Rendering (on the OpenCV display frame)
Draw a simple expressive face on the camera feed overlay:
- **Eyes**: Two circles. Size changes with emotion (wide for scared/surprised, narrow for focused/happy)
- **Mouth**: Arc (smile, frown, flat, O-shape)
- **Eyebrows**: Lines that angle with emotion
- **Blush**: Pink circles for happy/embarrassed
- Use `cv2.ellipse()`, `cv2.circle()`, `cv2.line()` — no external library needed

Suggested positions: top-right corner of the frame, ~120x80px box

Expressions to implement:
- `NEUTRAL`: flat mouth, normal eyes
- `HAPPY`: ^_^ smile, curved eyes, blush
- `FOCUSED`: >_> narrowed eyes, slight frown
- `CAUTIOUS`: O_O wide eyes, straight mouth
- `CONFUSED`: sideways mouth, uneven eyebrows @_@
- `APOLOGETIC`: >_< squinted eyes, wavy mouth
- `EXCITED`: big O_O eyes, big smile, eyebrows raised
- `SLEEPY`: -- closed eyes, small mouth

#### Sound System (using `winsound` or `pygame`)
Map of beep sequences to events:
- Greeting: rising tones (low→mid→high)
- Part found: short happy ding-ding
- Handoff ready: gentle two-tone chime
- Safety pause: rapid three-beep alert
- Safety resume: ascending triplet
- Confusion: descending wah-wah
- Celebration: victory jingle (5+ notes)
- Goodbye: descending tones (high→mid→low)

Use `winsound.Beep(freq, duration)` for simplicity (Windows-only). If cross-platform needed, generate sine waves via `numpy` + `sounddevice`, or use `pygame.mixer`.

### 2. NEW: `safety_system.py` — Advanced Safety with Intent Communication

```python
class SafetySystem:
    def __init__(self, personality_engine):
        self.hand_detected = False
        self.hand_roi = None  # bounding box
        self.personality = personality_engine
        self.paused = False
        self.consecutive_hand_frames = 0

    def update(self, frame):
        # Run hand detection on frame
        # Update hand_detected flag and ROI
        # Manage pause state transitions

    def is_safe_to_move(self) -> bool
    def get_retreat_position(self) -> tuple  # (x, y, z) safe pose
    def wait_for_resume_gesture(self, api, cap)  # blocking: waits for thumbs-up

    def draw_safety_overlay(self, frame):
        # Draw safe zone boundaries on frame
        # Draw danger zone if hand detected
        # Draw status text
```

#### Hand Detection (two approaches, implement whichever is easier):
**Approach A — MediaPipe Hands** (recommended, robust):
```python
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
if results.multi_hand_landmarks:
    # hand detected, get bounding box from landmarks
```

**Approach B — HSV Skin Detection** (zero dependencies):
- Convert to HSV, threshold skin-colored pixels
- Find largest contour, if area > threshold → hand detected
- Less robust but works for a demo

#### Safety Zones (draw on camera feed):
- **GREEN zone**: outer area → robot moves at full speed
- **YELLOW zone**: middle → robot slows to 40%
- **RED zone**: inner → robot pauses
- Draw these as semi-transparent rectangles on the camera frame

#### Gesture Recognition (for resume):
- **Thumbs-up gesture**: detect using MediaPipe hand landmarks:
  - All fingers closed except thumb extended
  - Landmark angles: thumb tip far from index MCP, other finger tips close to palm
- Can also use a simple "hand visible = pause, hand gone = resume" approach

### 3. NEW: `mood_detector.py` — Facial Expression Analysis

```python
class MoodDetector:
    def __init__(self):
        self.face_mesh = None  # MediaPipe FaceMesh
        self.current_mood = "neutral"
        self.mood_history = []  # last N frames for smoothing

    def analyze(self, frame) -> str:
        # Returns: "focused", "tired", "happy", "distracted", "neutral", "no_face"

    def get_smooth_mood(self) -> str
        # Returns majority mood from last 30 frames
```

Implementation with MediaPipe FaceMesh:
```python
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
results = face_mesh.process(rgb_frame)
if results.multi_face_landmarks:
    landmarks = results.multi_face_landmarks[0].landmark
    # Calculate:
    # - Eye Aspect Ratio (EAR): how open/closed eyes are -> tired if low for many frames
    # - Mouth aspect ratio: smiling vs neutral
    # - Head pose: looking at camera (engaged) vs away (distracted)
    # - Eyebrow position: raised (surprised/focused) vs neutral
```

Simple heuristic approach (no ML):
- Track face bounding box position (looking at workspace vs away)
- Use template matching for smile detection
- Measure eye-blink rate for fatigue

### 4. MODIFY: `pickCVBlock.py` — Complete Rewrite of State Machine

Remove the old simple 3-phase state machine. Replace with a **narrative act-based controller**.

```python
# === CONSTANTS ===
HANDOFF_X = 250   # robot coordinate for handoff
HANDOFF_Y = 0
HANDOFF_Z = 30    # comfortable height for human to reach

TEAM_SCORE = 0

# === MAIN ===
def main():
    api = dType.load()
    cap = cv2.VideoCapture(0)
    # ... calibration loading ...

    personality = PersonalityEngine()
    safety = SafetySystem(personality)
    mood = MoodDetector()
    H_matrix = np.load("HomographyMatrix.npy")

    dobotArm.initialize_robot(api)

    # ACT 1: WAKE & GREET
    personality.set_emotion(Emotion.HAPPY)
    play_sound("greeting")
    draw_face(frame, emotion=HAPPY)
    dobotArm.move_to_home(api)  # "stretch"
    # Small "wave" motion: rotate end effector back and forth
    dobotArm.rotate_end_effector(api, -20)
    dobotArm.rotate_end_effector(api, 20)
    dobotArm.rotate_end_effector(api, 0)
    show_text(frame, "Hello! Ready to work together?")
    wait_for_spacebar_or_gesture()

    # ACT 2: WORKSPACE SCAN
    personality.set_emotion(Emotion.FOCUSED)
    play_sound("scanning")
    show_text(frame, "Scanning workspace...")
    drop_zones = phase_detect_plates()  # reuse existing function
    pick_targets = phase_detect_targets()
    personality.set_emotion(Emotion.HAPPY)
    show_text(frame, f"I see {len(pick_targets)} parts to sort!")
    play_sound("part_found")

    # ACT 3: HANDOFF CYCLES
    for i in range(min(len(pick_targets), len(drop_zones))):
        # Check mood at start of cycle
        human_mood = mood.get_smooth_mood()
        if human_mood == "tired":
            personality.set_emotion(Emotion.SLEEPY)
            # Robot takes over more — picks 2 parts before one handoff
            # Or slows down movement

        pick_x, pick_y = pick_targets[i]

        # --- PICK ---
        personality.set_emotion(Emotion.FOCUSED)

        # Before moving: show intent
        draw_path_overlay(frame, (pick_x, pick_y))
        show_text(frame, "Going to pick part...")
        play_sound("intent_chime")

        # SAFETY CHECK before pick
        if safety.hand_detected:
            handle_safety_interrupt(api, safety, personality, frame)
        else:
            dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)
            # mid-move safety check (continuous)
            if safety.hand_detected:
                handle_safety_interrupt(api, safety, personality, frame)
                continue  # restart this cycle

            dobotArm.move_to_xyz(api, pick_x, pick_y, Z_PICK)
            dobotArm.close_gripper(api)
            dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)

        # --- HANDOFF (not drop-off) ---
        personality.set_emotion(Emotion.CAUTIOUS)
        show_text(frame, "Bringing part to you...")
        play_sound("moving_carefully")

        dobotArm.move_to_xyz(api, HANDOFF_X, HANDOFF_Y, HANDOFF_Z)

        personality.set_emotion(Emotion.HAPPY)
        draw_face(frame, HAPPY)
        show_text(frame, "Here you go! Take the part.")
        play_sound("handoff_ready")

        # Wait for human to take the part with safety monitoring
        wait_for_human_take(api, dobotArm, safety, frame, timeout=10.0)
        # Detection: gripper force change OR watch for hand entering gripper area OR just time + spacebar

        # Human took it -> celebrate
        personality.set_emotion(Emotion.EXCITED)
        play_sound("celebration")
        dobotArm.open_gripper(api)
        # Happy wiggle: small rapid joint oscillations
        dobotArm.move_joint_angles(api, 0, 30, 30)
        dobotArm.move_joint_angles(api, 0, -30, 30)
        dobotArm.move_joint_angles(api, 0, 0, 0)

        TEAM_SCORE += 10
        show_text(frame, f"Team Score: {TEAM_SCORE}")
        wait(0.5)

    # ACT 4: CELEBRATION
    personality.set_emotion(Emotion.EXCITED)
    play_sound("victory_jingle")
    # Celebration animation: rapid small moves
    for _ in range(3):
        dobotArm.move_to_xyz(api, 200, 50, 50)
        dobotArm.move_to_xyz(api, 200, -50, 50)
    show_text(frame, f"ALL DONE! FINAL SCORE: {TEAM_SCORE}/100")
    draw_confetti(frame)  # draw colored dots on screen
    dobotArm.move_to_home(api)

    # ACT 5: GOODBYE
    personality.set_emotion(Emotion.HAPPY)
    play_sound("goodbye")
    draw_face(frame, HAPPY, winking=True)
    show_text(frame, "Great teamwork today!")
    wait(2)

    cap.release()
    cv2.destroyAllWindows()
```

#### Helper: `wait_for_human_take()`
```python
def wait_for_human_take(api, dobotArm, safety, frame, timeout=10.0):
    """Robot presents part, waits for human to take it."""
    start = time.time()
    while time.time() - start < timeout:
        ret, frame = cap.read()
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        # Method 1: check gripper sensor feedback if available
        # Method 2: camera-based detection of hand approaching gripper
        # Method 3: simple keypress (spacebar = taken)
        # Method 4: combined

        safety.update(frame)
        personality.draw_face(frame)
        cv2.imshow("Kizuna - Collaborative Assembly", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # human pressed space = "I took it"
            return True
        if key == ord('q'):
            return False

    # Timeout: robot puts part in drop zone as fallback
    personality.set_emotion(Emotion.CONFUSED)
    play_sound("confused")
    return False
```

#### Helper: `handle_safety_interrupt()`
```python
def handle_safety_interrupt(api, safety, personality, frame):
    """Robot detects hand, retreats, communicates, waits for all-clear."""
    personality.set_emotion(Emotion.APOLOGETIC)
    play_sound("safety_alert")
    show_text(frame, "HAND DETECTED! Retreating safely.")

    # Retreat to safe position
    retreat_x, retreat_y, retreat_z = safety.get_retreat_position()
    dobotArm.move_to_xyz(api, retreat_x, retreat_y, retreat_z)

    show_text(frame, "Please move your hand away, then give a thumbs-up.")
    draw_face(frame, CAUTIOUS)

    # Wait for hand to leave + thumbs-up gesture
    while True:
        ret, frame = cap.read()
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        safety.update(frame)
        personality.draw_face(frame)
        cv2.imshow("Kizuna - Collaborative Assembly", frame)
        cv2.waitKey(1)

        if not safety.hand_detected:
            # Hand left, do a quick check for thumbs-up
            show_text(frame, "Hand clear! Thumbs-up to resume?")
            if check_thumbs_up(frame):  # MediaPipe gesture detection
                personality.set_emotion(Emotion.HAPPY)
                play_sound("safety_resume")
                show_text(frame, "Resuming! Thanks.")
                wait(0.5)
                return  # resume operation
```

### 5. MODIFY: `dobotArm.py` — Add Movement Quality Functions
Add functions to support expressive movement:

```python
def move_with_emotion(api, x, y, z, emotion, rHead=0):
    """Move with speed/accel modulated by emotion."""
    speed_factor = get_speed_factor(emotion)
    accel_factor = get_accel_factor(emotion)
    # Temporarily set PTPCommonParams with modified ratios
    # Then execute normal move

def celebratory_wiggle(api):
    """Small rapid joint oscillations."""
    for _ in range(3):
        move_joint_angles(api, 0, 20, 20)
        move_joint_angles(api, 0, -20, 20)
    move_joint_angles(api, 0, 0, 0)

def wave(api):
    """Wave end effector side to side."""
    for angle in [-15, 15, -15, 15, 0]:
        rotate_end_effector(api, angle)
        dType.dSleep(200)

def nod(api):
    """Small vertical bob."""
    pose = dType.GetPose(api)
    for _ in range(2):
        move_to_xyz(api, pose[0], pose[1], pose[2] + 10)
        move_to_xyz(api, pose[0], pose[1], pose[2])
```

### 6. NEW: `utils_kizuna.py` — Drawing Helpers

```python
def draw_face(frame, emotion, x=50, y=50, size=40):
    """Draw animated robot face on frame."""
    # x, y = top-left of face bounding box
    cx, cy = x + size, y + size  # face center

    # Face circle
    cv2.circle(frame, (cx, cy), size, (200, 200, 255), -1)  # light blue face
    cv2.circle(frame, (cx, cy), size, (100, 100, 200), 2)   # outline

    # Eyes depend on emotion
    if emotion in [CAUTIOUS, EXCITED]:
        # Wide eyes
        cv2.circle(frame, (cx - 12, cy - 8), 6, (0, 0, 0), -1)
        cv2.circle(frame, (cx + 12, cy - 8), 6, (0, 0, 0), -1)
    elif emotion == SLEEPY:
        # Closed eyes (lines)
        cv2.line(frame, (cx - 18, cy - 8), (cx - 6, cy - 8), (0, 0, 0), 2)
        cv2.line(frame, (cx + 6, cy - 8), (cx + 18, cy - 8), (0, 0, 0), 2)
    elif emotion == HAPPY:
        # Happy closed eyes ^_^
        cv2.ellipse(frame, (cx - 12, cy - 6), (6, 3), 0, 180, 360, (0, 0, 0), 2)
        cv2.ellipse(frame, (cx + 12, cy - 6), (6, 3), 0, 180, 360, (0, 0, 0), 2)
    else:
        # Normal eyes
        cv2.circle(frame, (cx - 12, cy - 8), 4, (0, 0, 0), -1)
        cv2.circle(frame, (cx + 12, cy - 8), 4, (0, 0, 0), -1)

    # Mouth
    if emotion == HAPPY or emotion == EXCITED:
        cv2.ellipse(frame, (cx, cy + 10), (10, 8), 0, 0, 180, (0, 0, 0), 2)
    elif emotion == APOLOGETIC or emotion == CONFUSED:
        cv2.ellipse(frame, (cx, cy + 10), (10, 8), 0, 180, 360, (0, 0, 0), 2)
    elif emotion == CAUTIOUS:
        cv2.ellipse(frame, (cx, cy + 10), (8, 5), 0, 0, 180, (0, 0, 0), 2)  # small o mouth
    elif emotion == SLEEPY:
        cv2.line(frame, (cx - 6, cy + 10), (cx + 6, cy + 10), (0, 0, 0), 2)
    else:
        cv2.line(frame, (cx - 8, cy + 10), (cx + 8, cy + 10), (0, 0, 0), 2)

    return frame


def draw_intent_overlay(frame, target_x, target_y, H_matrix, robot_pos):
    """Draw a 'proposed path' line from current position to target."""
    # Convert robot target to pixel coords
    # Draw dotted line from robot current pixel pos to target pixel
    # Draw circle at destination
    pass


def draw_safety_zones(frame):
    """Draw green/yellow/red safety zone indicators."""
    h, w = frame.shape[:2]
    # Red inner zone (20% from center)
    overlay = frame.copy()
    cv2.rectangle(overlay, (w//2 - w//6, h//2 - h//6),
                  (w//2 + w//6, h//2 + h//6), (0, 0, 200), -1)
    # Yellow middle zone (40%)
    cv2.rectangle(overlay, (w//2 - w//3, h//2 - h//3),
                  (w//2 + w//3, h//2 + h//3), (0, 200, 200), -1)
    # Blend with transparency
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)


def draw_confetti(frame, num_pieces=50):
    """Draw random colored dots for celebration."""
    import random
    h, w = frame.shape[:2]
    for _ in range(num_pieces):
        x = random.randint(0, w)
        y = random.randint(0, h)
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        cv2.circle(frame, (x, y), random.randint(2, 6), color, -1)
    return frame
```

---

## New Dependencies Needed

Add to a `requirements.txt`:
```
opencv-python
numpy
mediapipe          # for hand detection + face mesh
winsound           # (built-in on Windows) for beeps
pygame             # (alternative, for better audio)
```

---

## File Structure After Implementation

```
Collaborative_Robot/
├── pickCVBlock.py              # REWRITTEN — main Kizuna controller
├── personality.py              # NEW — emotion engine, face, sound
├── safety_system.py            # NEW — hand detection, zones, retreat
├── mood_detector.py            # NEW — facial expression analysis
├── utils_kizuna.py             # NEW — drawing helpers
├── dobotArm.py                 # MODIFIED — add expressive movements
├── calibrateCamera.py          # UNCHANGED
├── getTransformationMatrix.py  # UNCHANGED
├── testDobot.py                # UNCHANGED
├── lib/                        # UNCHANGED
├── camera_params.npz
├── HomographyMatrix.npy
└── KIZUNA_IMPLEMENTATION.md    # this file
```

---

## Implementation Order (recommended)

| Step | What | Time |
|---|---|---|
| 1 | `utils_kizuna.py` — face drawing, confetti, overlays | 30min |
| 2 | `personality.py` — emotion enum, engine, sound map | 30min |
| 3 | `safety_system.py` — MediaPipe hand detection + zones | 1hr |
| 4 | `mood_detector.py` — MediaPipe FaceMesh expression | 1hr |
| 5 | `dobotArm.py` — add expressive movement functions | 20min |
| 6 | `pickCVBlock.py` — rewrite as Kizuna main loop with acts | 2hr |
| 7 | Integration testing + tune parameters | 1hr |
| 8 | Rehearse demo narrative | 30min |
| **Total** | | **~7hr** |

---

## Key Design Principles for Claude

1. **Everything is non-blocking where possible** — the face updates every frame, safety checks run every frame, mood is polled every frame. Only robot movement is blocking (while loop waiting for command completion).

2. **The display frame is the central UI** — everything (camera feed, face, overlays, score, text) renders onto one OpenCV window. No separate GUIs.

3. **Robot safety is always the priority** — the safety check runs inside EVERY movement function and can interrupt at any point. The retreat position should be a fixed safe coordinate outside the human's reach.

4. **Fail gracefully** — if MediaPipe fails to install, fall back to simpler methods (HSV skin detection, no face expression analysis). The demo should still work without ML dependencies.

5. **The narrative sells the demo** — every robot action should be accompanied by on-screen text explaining what's happening. Judges need to understand the story even without narration.

6. **Reuse existing code as much as possible** — `phase_detect_plates()`, `phase_detect_targets()`, `pixel_to_robot()`, and all dobotArm functions should be called as-is. Only add new layers on top.
