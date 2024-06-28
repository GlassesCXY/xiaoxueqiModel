import asyncio
import cv2
import websockets
import numpy as np

def get_DroidCam_url(ip, port=4747, res='480p'):
    res_dict = {
        '240p': '320x240',
        '480p': '640x480',
        '720p': '1280x720',
        '1080p': '1920x1080',
    }
    url = f'http://{ip}:{port}/mjpegfeed?{res_dict[res]}'
    return url

async def camera_stream(websocket, path):
    print(1)
    cap = cv2.VideoCapture(get_DroidCam_url('192.168.43.1', 4747, '720p'))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            buffer = cv2.imencode('.jpg', frame)[1]
            await websocket.send(buffer.tobytes())
    finally:
        cap.release()


start_server = websockets.serve(camera_stream, "0.0.0.0", 8080)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
