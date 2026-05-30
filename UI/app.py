from flask import Flask, Response, send_from_directory
import cv2
import threading
import os
import time

app = Flask(__name__, static_folder='.', template_folder='.')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/styles.css')
def styles():
    return send_from_directory('.', 'styles.css')

def mjpeg_generator(device=0):
    ui_latest = os.path.join(os.path.dirname(__file__), 'latest.jpg')
    cap = cv2.VideoCapture(device)
    try:
        while True:
            # If collaborative_demo is writing latest.jpg, serve that image repeatedly
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

            # Fallback to live camera capture
            if not cap.isOpened():
                cap.open(device)
                if not cap.isOpened():
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
        if cap.isOpened():
            cap.release()

@app.route('/stream')
def stream():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run on localhost:5000
    app.run(host='0.0.0.0', port=5000, threaded=True)
