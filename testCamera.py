import cv2
import numpy as np
import os
import time
import argparse

"""Camera Test Script for Dobot Vision Setup

Tests:
  - Camera connectivity across indices 0-3
  - Frame capture rate and resolution
  - Calibration data loading (if available)
  - Undistortion preview
  - Grid overlay for positioning reference

Controls:
  Q    - Quit
  U    - Toggle undistortion
  G    - Toggle grid overlay
  S    - Save snapshot
  1-4  - Switch camera index
"""

CALIBRATION_FILE = "camera_params.npz"

def list_cameras(max_index=4):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available.append(i)
        cap.release()
    return available

def find_camera(max_index=4):
    available = list_cameras(max_index)
    return available[0] if available else None

def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        return None, None, None, None
    data = np.load(CALIBRATION_FILE)
    cm = data["camera_matrix"]
    dc = data["dist_coeffs"]
    rms = float(data["rms_error"].item())
    return cm, dc, rms, data["image_size"]

def build_undistort_maps(camera_matrix, dist_coeffs, w, h):
    new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1)
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_K, (w, h), cv2.CV_16SC2
    )
    return map1, map2, roi

def draw_grid(frame, spacing=50, color=(100, 100, 100, 100)):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    for x in range(0, w, spacing):
        cv2.line(overlay, (x, 0), (x, h), color, 1)
    for y in range(0, h, spacing):
        cv2.line(overlay, (0, y), (w, y), color, 1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    return frame

def main():
    parser = argparse.ArgumentParser(description="Test camera for Dobot vision setup")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (default: auto-detect)")
    args = parser.parse_args()

    available = list_cameras(6)
    if not available:
        print("No cameras found on indices 0-5.")
        return

    cam_idx = args.camera if args.camera is not None else available[0]
    if cam_idx not in available:
        print(f"Camera {cam_idx} not available. Available indices: {available}")
        return

    print(f"Available camera indices: {available}")
    print(f"Using camera {cam_idx}")

    camera_matrix, dist_coeffs, rms, img_size = load_calibration()
    if camera_matrix is not None:
        print(f"Calibration loaded | RMS error: {rms:.4f}")
    else:
        print("No calibration data found. Run calibrateCamera.py --calibrate first.")

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Failed to open camera index {cam_idx}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Resolution: {w}x{h} | Reported FPS: {fps:.1f}")

    map1 = map2 = roi = None
    use_undistort = camera_matrix is not None
    if use_undistort:
        map1, map2, roi = build_undistort_maps(camera_matrix, dist_coeffs, w, h)

    show_grid = False
    frame_count = 0
    fps_start = time.perf_counter()
    fps_display = 0

    print("\nControls: [U]ndistort toggle  [G]rid toggle  [S]napshot  [1-4] Camera  [Q]uit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.perf_counter() - fps_start
            fps_display = frame_count / elapsed

        original = frame.copy()

        if use_undistort and map1 is not None:
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        if show_grid:
            frame = draw_grid(frame)

        info_lines = [
            f"Camera {cam_idx} | {w}x{h} | FPS: {fps_display:.1f}",
            f"[U]ndistort: {'ON' if use_undistort else 'OFF' if camera_matrix is not None else 'N/A'}  [G]rid: {'ON' if show_grid else 'OFF'}",
            "Q=quit  S=snapshot  1-4=switch camera",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.imshow("Dobot Camera Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('u') and camera_matrix is not None:
            use_undistort = not use_undistort
            s = "ON" if use_undistort else "OFF"
            print(f"Undistortion: {s}")
        elif key == ord('g'):
            show_grid = not show_grid
            s = "ON" if show_grid else "OFF"
            print(f"Grid overlay: {s}")
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"snapshot_{ts}.png", original)
            print(f"Snapshot saved: snapshot_{ts}.png")
        elif ord('1') <= key <= ord('4'):
            new_idx = key - ord('1')
            cap.release()
            cap = cv2.VideoCapture(new_idx)
            if cap.isOpened():
                cam_idx = new_idx
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Switched to camera {cam_idx} | {w}x{h}")
            else:
                print(f"Camera {new_idx} not available")
                cap = cv2.VideoCapture(cam_idx)

    cap.release()
    cv2.destroyAllWindows()
    total_time = time.perf_counter() - fps_start
    print(f"\nCaptured {frame_count} frames in {total_time:.1f}s ({frame_count/total_time:.1f} avg FPS)")

if __name__ == "__main__":
    main()
