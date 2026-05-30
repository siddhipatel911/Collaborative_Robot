from flask import Flask, Response, send_from_directory
import cv2
import threading
import os
import sys
import time

app = Flask(__name__, static_folder='.', template_folder='.')


def find_camera(max_index=4):
    """Auto-detect cameras, prefer index 1 (external USB)."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available.append(i)
        cap.release()
    cam_idx = 1 if 1 in available else (available[0] if available else None)
    if cam_idx is None:
        print('[STREAM] No camera found')
        return None
    print(f'[STREAM] Cameras: {available}  Using: {cam_idx}')
    return cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/styles.css')
def styles():
    return send_from_directory('.', 'styles.css')

def mjpeg_generator():
    ui_latest = os.path.join(os.path.dirname(__file__), 'latest.jpg')
    cap = find_camera()
    try:
        while True:
            if os.path.exists(ui_latest):
                try:
                    with open(ui_latest, 'rb') as f:
                        img = f.read()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
                    time.sleep(0.05)
                    continue
                except Exception:
                    pass

            if cap is None or not cap.isOpened():
                cap = find_camera()
                if cap is None:
                    print('[STREAM] Camera not available')
                    time.sleep(0.5)
                    continue

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            ret2, jpg = cv2.imencode('.jpg', frame)
            if not ret2:
                continue
            frame_bytes = jpg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        if cap is not None and cap.isOpened():
            cap.release()

@app.route('/stream')
def stream():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
