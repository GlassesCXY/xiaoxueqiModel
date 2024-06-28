from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import base64
import websocket

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return render_template('index.html')

def get_DroidCam_url(ip, port=4747, res='480p'):
    res_dict = {
        '240p': '320x240',
        '480p': '640x480',
        '720p': '1280x720',
        '1080p': '1920x1080',
    }
    url = f'http://{ip}:{port}/mjpegfeed?{res_dict[res]}'
    return url

def generate_frames():
    cap = cv2.VideoCapture(get_DroidCam_url('192.168.43.1', 4747, '720p'))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame = base64.b64encode(buffer).decode('utf-8')
        yield frame
    cap.release()

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('video_feed')
def video_feed():
    for frame in generate_frames():
        socketio.emit('video_frame', {'frame': frame})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
