import cv2
import time
import threading
import queue
import asyncio
import websockets
from model import Model
import nest_asyncio

# 应用 nest_asyncio
nest_asyncio.apply()

# 初始化场景识别模型
model = Model()

def get_DroidCam_url(ip, port=4747, res='480p'):
    res_dict = {
        '240p': '320x240',
        '480p': '640x480',
        '720p': '1280x720',
        '1080p': '1920x1080',
    }
    url = f'http://{ip}:{port}/mjpegfeed?{res_dict[res]}'
    return url

def recognize_scene(frame):
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label = model.predict(image=rgb_frame)['label']
        return label
    except Exception as e:
        return "Unknown"

def process_frame_queue(frame_queue, result, lock, stop_event):
    while not stop_event.is_set():
        try:
            item = frame_queue.get(timeout=1)
            frame_count, frame = item
            scene_label = recognize_scene(frame)

            # 使用锁更新结果
            with lock:
                result['value'] = scene_label

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
result = {'value': ""}
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
    cap = cv2.VideoCapture(0)
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
                if frame_count % 10 == 0:
                    frame_queue.put((frame_count, frame))
                    capture_count += 1

                # 读取处理结果并显示场景标签
                with lock:
                    scene_label = result['value']
                cv2.putText(frame, scene_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow('image', frame)
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

start_server = websockets.serve(camera_stream, "0.0.0.0", 8083)

# 运行 WebSocket 服务器
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
