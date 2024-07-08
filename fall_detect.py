import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import time
import asyncio
import websockets

# 初始化MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 打开视频文件或摄像头
# cap = cv2.VideoCapture('queda.mp4')  # 使用视频文件

def send_email_to_all(users):
    smtp_server = 'smtp.qq.com'  # 替换为您的SMTP服务器
    smtp_port = 587  # 替换为您的SMTP端口
    sender_email = '1261751931@qq.com'  # 替换为您的发送邮箱
    sender_password = 'zzjicxwvvtbvfhfd'  # 替换为您的邮箱密码

    subject = 'Fall Detected'
    body = 'A fall has been detected. Please take appropriate action.'

    for user_email in users:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = user_email[0]
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            text = msg.as_string()
            server.sendmail(sender_email, user_email[0], text)
            server.quit()
            print(f'Email sent to {user_email}')
        except Exception as e:
            print(f'Failed to send email to {user_email}: {e}')

def get_DroidCam_url(ip, port=4747, res='480p'):
    res_dict = {
        '240p': '320x240',
        '480p': '640x480',
        '720p': '1280x720',
        '1080p': '1920x1080',
    }
    url = f'http://{ip}:{port}/video?{res_dict[res]}'
    return url


import pymysql

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

# 创建一个队列用来存储图像帧
frame_queue = queue.Queue()

# 用于存储结果的字典和锁
result = {'value': None}
lock = threading.Lock()

# 用于停止线程的事件
stop_event = threading.Event()


def detect_fall(landmarks):
    if landmarks:
        # 获取头部和脚部的y坐标
        head_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
        head_x = landmarks[mp_pose.PoseLandmark.NOSE.value].x
        left_foot_x = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x
        right_foot_x = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x

        left_foot_y = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y
        right_foot_y = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y

        foot_y = max(left_foot_y, right_foot_y)
        foot_x = (left_foot_x + right_foot_x) / 2

        # 计算头部和脚部之间的斜率
        slope = abs((head_y - foot_y) / (head_x - foot_x)) if (head_x != foot_x) else float('inf')

        # 跌倒检测逻辑：头部的y坐标高于脚部的y坐标或斜率小于1
        if head_y > foot_y or slope < 1:
            return True
    return False


def process_frame_queue(frame_queue, result, lock, stop_event):
    while not stop_event.is_set():
        try:
            item = frame_queue.get(timeout=1)
            frame_count, frame = item

            # 转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 进行姿态估计
            results = pose.process(frame_rgb)

            # 检测跌倒
            fall_detected = False
            if results.pose_landmarks:
                fall_detected = detect_fall(results.pose_landmarks.landmark)

            # 使用锁更新结果
            with lock:
                result['value'] = (frame, results.pose_landmarks, fall_detected)

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


def save_fall_data(fall_queue, db_lock, stop_event):
    last_save_time = 0  # 上次保存时间
    while not stop_event.is_set():
        try:
            fall_date, fall_frame = fall_queue.get(timeout=1)

            current_time = time.time()
            if current_time - last_save_time >= 60:
                # 保存摔倒信息到数据库并获取插入后的fid
                with db_lock:
                    cursor.execute("INSERT INTO fall (fall_date) VALUES (%s)", (fall_date,))
                    db.commit()
                    fid = cursor.lastrowid

                # 使用fid作为图像名称
                image_name = f'{fid}.jpg'
                image_path = f'fall/{image_name}'

                # 保存图像到本地
                cv2.imwrite(image_path, fall_frame)

                # 上传图像到MinIO
                minio_client.fput_object(bucket_name, f'fall/{image_name}', image_path, content_type='image/')

                last_save_time = current_time  # 更新上次保存时间

                print('Saved fall data')

                cursor.execute("SELECT email FROM users")
                users = cursor.fetchall()
                send_email_to_all(users)

            fall_queue.task_done()
        except queue.Empty:
            continue


fall_queue = queue.Queue()
db_lock = threading.Lock()


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
saving_thread = threading.Thread(target=save_fall_data, args=(fall_queue, db_lock, stop_event))
saving_thread.daemon = True  # 设置为后台线程
saving_thread.start()


async def camera_stream(websocket, path):
    # cap = cv2.VideoCapture(get_DroidCam_url('192.168.43.3', 4747, res='480p'))

    cap = cv2.VideoCapture(0)  # 使用摄像头
    # 初始化帧计数器和时间
    frame_count = 0
    capture_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 显示图像
            frame_count += 1

            # 每3帧将一帧和帧序号放到队列中
            if frame_count % 3 == 0:
                frame_queue.put((frame_count, frame))
                capture_count += 1

            # 读取处理结果并画框
            with lock:
                processed_frame, landmarks, fall_detected = result['value'] if result['value'] is not None else (
                    frame, None, False)

            if landmarks:
                mp.solutions.drawing_utils.draw_landmarks(processed_frame, landmarks, mp_pose.POSE_CONNECTIONS)
                if fall_detected:
                    cv2.putText(processed_frame, "Fall Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    fall_date = datetime.now()
                    fall_queue.put((fall_date, processed_frame))

            # 显示结果
            cv2.imshow('Frame', processed_frame)

            # 将帧发送到WebSocket客户端
            _, buffer = cv2.imencode('.jpg', processed_frame)
            await websocket.send(buffer.tobytes())

            # 每秒钟输出当前帧率
            if time.time() - start_time >= 1:
                print(f'FPS: {capture_count}')
                capture_count = 0
                start_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()


# 启动WebSocket服务器
start_server = websockets.serve(camera_stream, "0.0.0.0", 8081)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

cv2.destroyAllWindows()
pose.close()

# 通知线程结束
stop_event.set()

# 结束处理线程
frame_queue.put(None)
processing_thread.join()
