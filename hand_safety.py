"""
hand_safety.py — Hand detection safety monitor.

Detects hands in the camera feed. When a hand is visible, robot movement
should be blocked. Provides both a standalone demo mode and a reusable class.

Standalone usage:
    python hand_safety.py          # shows camera feed with hand detection overlay
    python hand_safety.py --image path.jpg  # test on a static image

Import usage:
    from hand_safety import HandSafety
    safety = HandSafety()
    if safety.hand_detected(frame):
        print("Hand in frame — blocking robot movement")
"""

import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import mediapipe as mp
import sys
import os
import time

MODEL_PATH = "hand_landmarker.task"
_HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (0, 9), (9, 10), (10, 11), (11, 12),      # middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (5, 9), (9, 13), (13, 17),                # palm
])


class HandSafety:
    def __init__(self, model_path=MODEL_PATH, min_hand_detection_confidence=0.25):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found: {model_path}\n"
                f"Download from: https://storage.googleapis.com/mediapipe-models/"
                f"hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )
        options_kwargs = {
            "base_options": BaseOptions(model_asset_path=model_path),
            "running_mode": vision.RunningMode.VIDEO,
            "min_hand_detection_confidence": min_hand_detection_confidence,
            "num_hands": 2,
        }
        for key, value in {
            "min_hand_presence_confidence": 0.25,
            "min_tracking_confidence": 0.25,
        }.items():
            try:
                vision.HandLandmarkerOptions(**{**options_kwargs, key: value})
                options_kwargs[key] = value
            except TypeError:
                pass
        options = vision.HandLandmarkerOptions(**options_kwargs)
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._mp_image_format = mp.ImageFormat.SRGB
        self._last_timestamp_ms = 0

    def detect(self, frame):
        """
        Run hand detection on a BGR frame.
        Returns a list of detected hands, each containing:
          - hand_landmarks: list of 21 (x, y, z) landmarks
          - handedness: list of (category_name, score)
        Returns empty list if no hands.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=self._mp_image_format, data=rgb)
        timestamp_ms = int(time.perf_counter() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        result = self._detector.detect_for_video(image, timestamp_ms)
        return result.hand_landmarks, result.handedness

    def hand_detected(self, frame):
        """Returns True if at least one hand is visible in the frame."""
        landmarks, _ = self.detect(frame)
        return len(landmarks) > 0

    def draw_landmarks(self, frame, hand_landmarks_list, handedness_list=None):
        """Draw hand landmarks and connections on the frame (in-place)."""
        h, w = frame.shape[:2]
        for i, landmarks in enumerate(hand_landmarks_list):
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            for start_idx, end_idx in _HAND_CONNECTIONS:
                if start_idx < len(pts) and end_idx < len(pts):
                    cv2.line(frame, pts[start_idx], pts[end_idx], (0, 255, 0), 2)

            for j, pt in enumerate(pts):
                cv2.circle(frame, pt, 4, (255, 0, 0), -1)

            label = "Hand"
            if handedness_list and i < len(handedness_list):
                label = handedness_list[i][0].category_name
            cx = int(np.mean([p[0] for p in pts]))
            cy = int(np.mean([p[1] for p in pts]))
            cv2.putText(frame, label, (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def close(self):
        self._detector.close()


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--image":
        # Static image mode
        img = cv2.imread(sys.argv[2])
        if img is None:
            print(f"Failed to load {sys.argv[2]}")
            sys.exit(1)
        safety = HandSafety()
        landmarks, handedness = safety.detect(img)
        safety.draw_landmarks(img, landmarks, handedness)
        print(f"Hands detected: {len(landmarks)}")
        for h in handedness:
            print(f"  {h[0].category_name} ({h[0].score:.2f})")
        cv2.imshow("Hand Safety", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        safety.close()
        return

    # Live camera mode — external USB camera (index 1) first, fallback to laptop
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No camera found")
        sys.exit(1)

    safety = HandSafety()
    print("[HAND SAFETY] Live detection. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        landmarks, handedness = safety.detect(frame)
        hand_count = len(landmarks)

        if hand_count > 0:
            safety.draw_landmarks(frame, landmarks, handedness)

        cv2.rectangle(frame, (0, 0), (300, 50), (0, 0, 0), -1)
        color = (0, 0, 255) if hand_count > 0 else (0, 255, 0)
        status = f"HANDS: {hand_count}  BLOCKING" if hand_count > 0 else "HANDS: 0  SAFE"
        cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Hand Safety", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    safety.close()
    print("[DONE] Hand safety stopped.")


if __name__ == "__main__":
    main()
