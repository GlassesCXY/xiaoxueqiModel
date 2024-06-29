import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import queue
import asyncio
import websockets

# 初始化MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def get_DroidCam_url(ip, port=4747, res='480p'):
    res_dict = {
        '240p': '320x240',
        '480p': '640x480',
        '720p': '1280x720',
        '1080p': '1920x1080',
    }
    url = f'http://{ip}:{port}/mjpegfeed?{res_dict[res]}'
    return url


def detect_fall(landmarks, image_width, image_height):
    if landmarks:
        try:
            # 获取头部和脚部的坐标
            head_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y * image_height
            head_x = landmarks[mp_pose.PoseLandmark.NOSE.value].x * image_width
            left_foot_y = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y * image_height
            left_foot_x = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x * image_width
            right_foot_y = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y * image_height
            right_foot_x = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x * image_width
        except IndexError:
            # 当关键点缺失时返回False
            return False

        foot_y = max(left_foot_y, right_foot_y)
        foot_x = (left_foot_x + right_foot_x) / 2

        # 计算头部和脚部之间的斜率
        slope = abs((head_y - foot_y) / (head_x - foot_x)) if (head_x != foot_x) else float('inf')

        # 跌倒检测逻辑：斜率
        if head_y > foot_y or slope >= 1:
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
            fall_detected = detect_fall(results.pose_landmarks.landmark) if results.pose_landmarks else False

            # 使用锁更新结果
            with lock:
                result['value'] = (fall_detected, results.pose_landmarks)

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


# 创建一个队列用来存储图像帧
frame_queue = queue.Queue()

# 用于存储结果的字典和锁
result = {'value': (False, None)}
lock = threading.Lock()

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


async def camera_stream(websocket, path):
    cap = cv2.VideoCapture(get_DroidCam_url('192.168.43.1', 4747, '720p'))
    # 初始化帧计数器和时间
    frame_count = 0
    capture_count = 0
    start_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if ret:
                # 显示图像
                frame_count = (frame_count + 1) % mod_value

                # 每3帧将一帧和帧序号放到队列中
                if frame_count % 3 == 0:
                    frame_queue.put((frame_count, frame))
                    capture_count += 1

                # 读取处理结果并显示
                with lock:
                    fall_detected, pose_landmarks = result['value']
                if pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)
                if fall_detected:
                    cv2.putText(frame, "Fall Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)

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
pose.close()

# 通知线程结束
stop_event.set()

# 结束处理线程
frame_queue.put(None)
processing_thread.join()
queue_management_thread.join()
