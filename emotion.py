import asyncio
from datetime import datetime

import cv2
import time
import threading
import queue

import websockets
from deepface import DeepFace


import pymysql

negative_emotions = ['angry', 'disgust', 'fear', 'sad']


# 配置MySQL数据库
db = pymysql.connect(
    host='43.138.118.5',
    port=3306,
    user='root',
    password='Woshishabi@233',
    database='xxq'
)

cursor = db.cursor()

from minio import Minio

minio_client = Minio(
    '43.138.118.5:9000',  # MinIO服务器地址
    access_key='nI20OQaJKUCJFBPYREQT',
    secret_key='viJ7ZoA3wCkNbkl7WQeKLe5O9mretoEFSoXvLKsn',
    secure=False
)

bucket_name = "xxq"

def get_DroidCam_url(ip, port=4747, res='480p'):
    res_dict = {
        '240p': '320x240',
        '480p': '640x480',
        '720p': '1280x720',
        '1080p': '1920x1080',
    }
    url = f'http://{ip}:{port}/mjpegfeed?{res_dict[res]}'
    return url


def detect_and_analyze_faces(frame, backend='opencv'):
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        analysis = DeepFace.analyze(img_path=rgb_frame, actions=['emotion'], detector_backend=backend, enforce_detection=False)
        results = []
        for face in analysis:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            emotion = face['dominant_emotion']
            results.append((x, y, w, h, emotion))
        return results
    except Exception as e:
        return []


def process_frame_queue(frame_queue, result, lock, stop_event):
    while not stop_event.is_set():
        try:
            item = frame_queue.get(timeout=1)
            frame_count, frame = item
            face_data = detect_and_analyze_faces(frame)

            # 使用锁更新结果
            with lock:
                result['value'] = face_data

            frame_queue.task_done()
        except queue.Empty:
            continue


def manage_frame_queue(frame_queue, mod_value, stop_event):
    while not stop_event.is_set():
        time.sleep(1)  # 每秒检查一次队列
        if not frame_queue.empty():
            first_frame_count = frame_queue.queue[0][0]
            current_frame_count = frame_queue.queue[-1][0]
            if (current_frame_count - first_frame_count) % mod_value > 10:
                with frame_queue.mutex:
                    frame_queue.queue.clear()


def save_emo_data(emo_queue, db_lock, stop_event):
    last_save_time = 0  # 上次保存时间
    while not stop_event.is_set():
        try:
            emo_date, emo_frame, emotion = emo_queue.get(timeout=1)

            current_time = time.time()
            if current_time - last_save_time >= 60:
                # 保存摔倒信息到数据库并获取插入后的fid
                with db_lock:
                    cursor.execute("INSERT INTO emotion(type, emotion_date) VALUES (%s, %s)", (emotion, emo_date))
                    db.commit()
                    eid = cursor.lastrowid

                # 使用fid作为图像名称
                image_name = f'{eid}.jpg'
                image_path = f'emo/{image_name}'

                # 保存图像到本地
                cv2.imwrite(image_path, emo_frame)

                # 上传图像到MinIO
                minio_client.fput_object(bucket_name, f'emo/{image_name}', image_path, content_type='image/')

                last_save_time = current_time  # 更新上次保存时间

                print('Saved emo data')
            emo_queue.task_done()
        except queue.Empty:
            continue


# 创建一个队列用来存储图像帧
frame_queue = queue.Queue()
emo_queue = queue.Queue()


# 用于存储结果的字典和锁
result = {'value': []}
lock = threading.Lock()
db_lock = threading.Lock()
# 用于停止线程的事件
stop_event = threading.Event()

# 启动处理线程
processing_thread = threading.Thread(target=process_frame_queue, args=(frame_queue, result, lock, stop_event))
processing_thread.daemon = True  # 设置为后台线程
processing_thread.start()

# 启动管理队列的线程
mod_value = 2 ** 32 - 1
queue_management_thread = threading.Thread(target=manage_frame_queue, args=(frame_queue, mod_value, stop_event))
queue_management_thread.daemon = True  # 设置为后台线程
queue_management_thread.start()

# 启动保存数据线程
saving_thread = threading.Thread(target=save_emo_data, args=(emo_queue, db_lock, stop_event))
saving_thread.daemon = True  # 设置为后台线程
saving_thread.start()


async def camera_stream(websocket, path):
    # DroidCam 显示的IP地址、端口号和相机分辨率（可选 240p,480p,720p,1080p）
    cap = cv2.VideoCapture(get_DroidCam_url('192.168.43.1', 4747, res='480p'))
    # cap = cv2.VideoCapture(0)
    # 初始化帧计数器和时间
    frame_count = 0
    capture_count = 0
    start_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if ret:
                # 显示图像
                frame_count = (frame_count + 1) % mod_value

                # 每3帧将一帧和帧序号放到队列中
                if frame_count % 30 == 0:
                    frame_queue.put((frame_count, frame))
                    capture_count += 1

                # 读取处理结果并画框
                with lock:
                    face_data = result['value']
                for (x, y, w, h, emotion) in face_data:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    if emotion in negative_emotions:
                        emo_date = datetime.now()
                        emo_queue.put((emo_date, frame, emotion))

                cv2.imshow('Real-time Emotion Analysis', frame)
                buffer = cv2.imencode('.jpg', frame)[1]
                await websocket.send(buffer.tobytes())
            # 每秒钟输出当前帧率
            if time.time() - start_time >= 1:
                print(f'FPS: {capture_count}')
                capture_count = 0
                start_time = time.time()

            key = cv2.waitKey(1)
            # 按q退出程序
            if key == ord('q'):
                break


    finally:
        cap.release()

start_server = websockets.serve(camera_stream, "0.0.0.0", 8080)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

# 销毁所有的窗口
cv2.destroyAllWindows()

# 通知线程结束
stop_event.set()

# 结束处理线程
frame_queue.put(None)
processing_thread.join()
queue_management_thread.join()
