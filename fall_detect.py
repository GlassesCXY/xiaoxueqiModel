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


# 启动处理线程
processing_thread = threading.Thread(target=process_frame_queue, args=(frame_queue, result, lock, stop_event))
processing_thread.daemon = True  # 设置为后台线程
processing_thread.start()


async def camera_stream(websocket, path):
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
start_server = websockets.serve(camera_stream, "0.0.0.0", 8080)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

cv2.destroyAllWindows()
pose.close()

# 通知线程结束
stop_event.set()

# 结束处理线程
frame_queue.put(None)
processing_thread.join()
