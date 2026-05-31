from flask import Flask, Response, send_from_directory
import os
import time

app = Flask(__name__, static_folder='.', template_folder='.')


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/styles.css')
def styles():
    return send_from_directory('.', 'styles.css')

def mjpeg_generator():
    ui_latest = os.path.join(os.path.dirname(__file__), 'latest.jpg')
    while True:
        img = None
        if os.path.exists(ui_latest):
            try:
                with open(ui_latest, 'rb') as f:
                    img = f.read()
            except Exception:
                pass
        if img is None:
            img = _make_placeholder()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
        time.sleep(0.05)


@app.route('/stream')
def stream():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/frame')
def frame():
    ui_latest = os.path.join(os.path.dirname(__file__), 'latest.jpg')
    if os.path.exists(ui_latest):
        try:
            with open(ui_latest, 'rb') as f:
                img = f.read()
            if len(img) > 100:
                return Response(img, mimetype='image/jpeg',
                                headers={'Cache-Control': 'no-cache, no-store, must-revalidate',
                                         'Pragma': 'no-cache', 'Expires': '0'})
        except Exception:
            pass
    return Response(_make_placeholder(), mimetype='image/jpeg',
                    headers={'Cache-Control': 'no-cache, no-store, must-revalidate',
                             'Pragma': 'no-cache', 'Expires': '0'})


_PLACEHOLDER = None

def _make_placeholder():
    global _PLACEHOLDER
    if _PLACEHOLDER is not None:
        return _PLACEHOLDER
    try:
        import numpy as np
        import cv2
        img = np.zeros((360, 640, 3), dtype=np.uint8) + 40
        cv2.putText(img, "Waiting for camera...", (120, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)
        cv2.putText(img, "Run collaborative_demo.py", (145, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
        _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        _PLACEHOLDER = buf.tobytes()
    except Exception:
        _PLACEHOLDER = (
            b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
            b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n'
            b'\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d'
            b'\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00'
            b'\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01'
            b'\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02'
            b'\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03'
            b'\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x00\x00\x01\x02\x03\x00\x04'
            b'\x11\x05\x06!1A\x07Q\x13\x162\x81\x91\xa1\x08\x14B\xd1\xc1"2\x82\x92'
            b'\xa2\x17\x18\x19\n#$\x83\x93\xb1\xe1\xf0\xff\xc4\x00!\x01\x01\x01\x01'
            b'\x01\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03'
            b'\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\x02'
            b'\x10\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00'
            b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xd9'
        )
    return _PLACEHOLDER


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
    return Response('', mimetype='text/plain')


@app.route('/assets/<path:filename>')
def assets(filename):
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets'))
    if os.path.exists(os.path.join(assets_dir, filename)):
        return send_from_directory(assets_dir, filename)
    return Response('Not found', status=404)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
