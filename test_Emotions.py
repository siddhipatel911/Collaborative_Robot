import cv2
import mediapipe as mp
import numpy as np
import collections
import dobotArm
import lib.DobotDllType as dType

# ---- robot setup ----
api = dType.load()
dobotArm.initialize_robot(api)
dobotArm.open_gripper(api)
dobotArm.stop_pump(api)

# ---- face detector setup ----
facelib   = mp.solutions.face_mesh
detector  = facelib.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
drawer    = mp.solutions.drawing_utils
drawer_styles = mp.solutions.drawing_styles

# ---- face dot numbers we need ----
leye   = [33, 133, 160, 144, 158, 153]
reye   = [362, 263, 387, 373, 385, 380]
brow   = [70, 63, 105, 66, 107]
browref = [27, 23, 28, 56, 190]
mleft  = 61
mright = 291
mtop   = 13

# ---- buffer so emotion doesnt flicker ----
LOCK_AFTER    = 20
curr_emotion  = "neutral"
hold_count    = 0
final_emotion = "neutral"

# ---- smooth eye value over last 15 frames ----
eye_smooth = collections.deque(maxlen=15)

# ---- colours for emotion label on screen ----
colors = {
    "happy":     (0, 215, 255),
    "sad":       (200, 80, 50),
    "angry":     (0, 0, 220),
    "tired":     (130, 60, 180),
    "surprised": (0, 180, 255),
    "neutral":   (160, 160, 160),
}

# ---- mediapipe coords to pixels ----
def to_px(dots, n, w, h):
    return int(dots[n].x * w), int(dots[n].y * h)

# ---- how open is the eye ----
def eye_open(dots, eye_dots, w, h):
    p = [to_px(dots, i, w, h) for i in eye_dots]
    width  = np.linalg.norm(np.array(p[0]) - np.array(p[1]))
    height = (np.linalg.norm(np.array(p[2]) - np.array(p[3])) +
              np.linalg.norm(np.array(p[4]) - np.array(p[5]))) / 2
    return height / width if width > 1 else 0.0

# ---- robot reactions per emotion ----
def robot_do(emotion):

    if emotion == "happy":
        # does a little twirl (rotates joint 1 by 185 degrees then comes back)
        print("robot: happy twirl!")
        pose = dType.GetPose(api)
        dType.SetPTPCommonParams(api, 80, 80, isQueued=1)
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJANGLEMode,
                        pose[4] + 185, pose[5], pose[6], pose[7], isQueued=0)
        dType.dSleep(1000)
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJANGLEMode,
                        pose[4], pose[5], pose[6], pose[7], isQueued=0)
        dType.dSleep(1000)

    elif emotion == "sad":
        # moves slow, droops down mid air and pauses like its sad
        print("robot: sad droop...")
        pose = dType.GetPose(api)
        dType.SetPTPCommonParams(api, 20, 20, isQueued=1)
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode,
                        pose[0], pose[1], pose[2] - 20, pose[3], isQueued=0)
        dType.dSleep(1500)
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode,
                        pose[0], pose[1], pose[2], pose[3], isQueued=0)
        dType.dSleep(1000)

    elif emotion == "angry":
        # snappy fast movements, slams gripper shut
        print("robot: ANGRY!!")
        dType.SetPTPCommonParams(api, 90, 90, isQueued=1)
        dType.SetEndEffectorGripper(api, 1, 1, 0)  # slam shut
        dType.dSleep(200)
        dType.SetEndEffectorGripper(api, 1, 0, 0)  # open back up
        dType.dSleep(200)

    elif emotion == "tired":
        # slow nod - dips down then back up like nodding off
        print("robot: tired nod...")
        pose = dType.GetPose(api)
        dType.SetPTPCommonParams(api, 15, 15, isQueued=1)
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode,
                        pose[0], pose[1], pose[2] - 15, pose[3], isQueued=0)
        dType.dSleep(800)
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode,
                        pose[0], pose[1], pose[2], pose[3], isQueued=0)
        dType.dSleep(800)

    elif emotion == "surprised":
        # quick jump upward
        print("robot: SURPRISED jump!")
        pose = dType.GetPose(api)
        dType.SetPTPCommonParams(api, 90, 90, isQueued=1)
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode,
                        pose[0], pose[1], pose[2] + 30, pose[3], isQueued=0)
        dType.dSleep(400)
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode,
                        pose[0], pose[1], pose[2], pose[3], isQueued=0)
        dType.dSleep(400)

    elif emotion == "neutral":
        # nothing special just normal speed
        print("robot: neutral, normal operation")
        dType.SetPTPCommonParams(api, 50, 50, isQueued=1)

# ---- open camera ----
cam = cv2.VideoCapture(0)
print("camera open, press Q to quit")

while cam.isOpened():
    ok, frame = cam.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = detector.process(rgb)

    eye_val   = 0.0
    brow_val  = 0.0
    mouth_val = 0.0
    guess     = "neutral"

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark

        # draw face mesh
        drawer.draw_landmarks(
            frame,
            res.multi_face_landmarks[0],
            facelib.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawer_styles.get_default_face_mesh_tesselation_style()
        )

        # blue dots on eyes
        for i in leye + reye:
            cv2.circle(frame, to_px(lm, i, fw, fh), 3, (255, 100, 0), -1)

        # green dots on brow
        for i in brow:
            cv2.circle(frame, to_px(lm, i, fw, fh), 3, (0, 220, 80), -1)

        # yellow dots on mouth
        for i in [mleft, mright, mtop]:
            cv2.circle(frame, to_px(lm, i, fw, fh), 4, (0, 220, 220), -1)

        # mouth triangle lines
        cv2.line(frame, to_px(lm, mleft, fw, fh),  to_px(lm, mright, fw, fh), (0, 220, 220), 1)
        cv2.line(frame, to_px(lm, mleft, fw, fh),  to_px(lm, mtop,   fw, fh), (0, 220, 220), 1)
        cv2.line(frame, to_px(lm, mright, fw, fh), to_px(lm, mtop,   fw, fh), (0, 220, 220), 1)

        # eye openness measurement
        eye_val = (eye_open(lm, leye, fw, fh) + eye_open(lm, reye, fw, fh)) / 2
        eye_smooth.append(eye_val)
        smooth = float(np.mean(eye_smooth))

        # brow raise measurement
        brow_val = float(np.mean([lm[i].y for i in browref]) - np.mean([lm[i].y for i in brow]))

        # mouth curve measurement
        mouth_val = float(lm[mtop].y - (lm[mleft].y + lm[mright].y) / 2)

        # emotion decision
        if brow_val > 0.06 and smooth > 0.30:
            guess = "surprised"
        elif smooth < 0.18:
            guess = "tired"
        elif mouth_val < -0.015:
            guess = "happy"
        elif mouth_val > 0.010 and brow_val < 0.02:
            guess = "angry"
        elif mouth_val > 0.005:
            guess = "sad"
        else:
            guess = "neutral"

    # buffer
    if guess == curr_emotion:
        hold_count = min(hold_count + 1, LOCK_AFTER)
    else:
        curr_emotion = guess
        hold_count = 0

    if hold_count >= LOCK_AFTER:
        if final_emotion != curr_emotion:
            final_emotion = curr_emotion
            robot_do(final_emotion)  # trigger robot reaction

    # draw HUD
    cv2.rectangle(frame, (0, 0), (260, 225), (20, 20, 20), -1)

    cv2.putText(frame, f"eye open:   {eye_val:.2f}",   (10, 28),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100,   0), 1)
    cv2.putText(frame, f"brow raise: {brow_val:.2f}",  (10, 52),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,   220,  80), 1)
    cv2.putText(frame, f"mouth:      {mouth_val:.2f}", (10, 76),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,   220, 220), 1)
    cv2.putText(frame, f"seeing: {guess}",             (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # buffer bar
    fill = int((hold_count / LOCK_AFTER) * 200)
    cv2.rectangle(frame, (10, 120), (210, 135), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, 120), (10 + fill, 135), colors.get(curr_emotion, (160,160,160)), -1)
    cv2.rectangle(frame, (10, 120), (210, 135), (120, 120, 120), 1)
    cv2.putText(frame, f"buffer: {hold_count}/{LOCK_AFTER}", (10, 153), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160,160,160), 1)

    cv2.putText(frame, f"EMOTION: {final_emotion.upper()}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors.get(final_emotion,(160,160,160)), 2)

    cv2.imshow("emotion test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()