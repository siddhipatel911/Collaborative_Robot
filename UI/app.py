from flask import Flask, Response, send_from_directory
import cv2
import threading
import os
import sys
import time
import platform
from flask import request, jsonify

# Optional robot control trigger
try:
    import mood_movement
    import dobotArm as _dobot_arm
    DOBOT_AVAILABLE = hasattr(_dobot_arm, 'api') and _dobot_arm.api is not None
except Exception:
    mood_movement = None
    DOBOT_AVAILABLE = False

app = Flask(__name__, static_folder='.', template_folder='.')


def find_camera(max_index=4):
    """Auto-detect cameras, prefer index 1 (external USB).
    Avoid using the Windows-only CAP_DSHOW flag on macOS/Linux.
    """
    available = []
    use_dshow = platform.system() == "Windows" and hasattr(cv2, 'CAP_DSHOW')
    for i in range(max_index):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if use_dshow else cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available.append(i)
            cap.release()
        except Exception:
            try:
                cap.release()
            except Exception:
                pass
    cam_idx = 1 if 1 in available else (available[0] if available else None)
    if cam_idx is None:
        print('[STREAM] No camera found')
        return None
    print(f'[STREAM] Cameras: {available}  Using: {cam_idx}')
    return cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW) if use_dshow else cv2.VideoCapture(cam_idx)


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


@app.route('/mood.json')
def mood_json():
    mood_file = os.path.join(os.path.dirname(__file__), 'mood.json')
    if os.path.exists(mood_file):
        return send_from_directory(os.path.dirname(__file__), 'mood.json')
    return Response('{"mood":"unknown"}', mimetype='application/json')


@app.route('/mood.log')
def mood_log():
    mood_file = os.path.join(os.path.dirname(__file__), 'mood.log')
    if os.path.exists(mood_file):
        return send_from_directory(os.path.dirname(__file__), 'mood.log')
    # Return empty log if missing to avoid browser 404 noise
    return Response('', mimetype='text/plain')


@app.route('/assets/<path:filename>')
def assets(filename):
    # assets are stored one level up from the ui folder
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets'))
    if os.path.exists(os.path.join(assets_dir, filename)):
        return send_from_directory(assets_dir, filename)
    return Response('Not found', status=404)


@app.route('/trigger_mood', methods=['POST'])
def trigger_mood():
    body = request.get_json(silent=True) or {}
    mood = body.get('mood') or body.get('emotion') or 'neutral'
    if mood_movement is None:
        return jsonify({'ok': False, 'error': 'mood movement module not available'}), 503
    try:
        api = getattr(_dobot_arm, 'api', None)
        mood_movement.perform_mood_movement(api, mood)
        return jsonify({'ok': True, 'mood': mood})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
