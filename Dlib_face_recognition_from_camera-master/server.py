import dlib
import numpy as np
import cv2
import os
import shutil
import logging
from deepface import DeepFace
from flask import Flask, request, jsonify
import subprocess
import shutil
from flask_cors import CORS
# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()


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

class Face_Register:
    def __init__(self):
        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.font = cv2.FONT_ITALIC
        self.existing_faces_cnt = 0  # 已录入的人脸计数器 / cnt for counting saved faces
        self.ss_cnt = 0  # 录入 personX 人脸时图片计数器 / cnt for screen shots

    # 新建保存人脸图像文件和数据 CSV 文件夹 / Mkdir for saving photos and csv
    def pre_work_mkdir(self):
        # 新建文件夹 / Create folders to save face images and csv
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)

    # 删除之前存的人脸数据文件夹 / Delete old face folders
    def pre_work_del_old_face_folders(self):
        # 删除之前存的人脸数据文件夹, 删除 "/data_faces_from_camera/person_x/"...
        folders_rd = os.listdir(self.path_photos_from_camera)
        for i in range(len(folders_rd)):
            shutil.rmtree(self.path_photos_from_camera+folders_rd[i])
        if os.path.isfile("data/features_all.csv"):
            os.remove("data/features_all.csv")

    # 如果有之前录入的人脸, 在之前 person_x 的序号按照 person_x+1 开始录入 / Start from person_x+1
    def check_existing_faces_cnt(self):
        if os.listdir("data/data_faces_from_camera/"):
            # 获取已录入的最后一个人脸序号 / Get the order of latest person
            person_list = os.listdir("data/data_faces_from_camera/")
            self.existing_faces_cnt = len(person_list)

        # 如果第一次存储或者没有之前录入的人脸, 按照 person_1 开始录入 / Start from person_1
        else:
            self.existing_faces_cnt = 0

    def process_image(self, image, person_name):
        self.pre_work_mkdir()
        self.check_existing_faces_cnt()

        # 读取图片
        img_rd = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        if img_rd is None:
            logging.error("无法读取图片 / Unable to read image")
            return False, "无法读取图片"

        faces = self.detect_and_draw_faces(img_rd)

        # 将检测结果转化为dlib的矩形对象
        faces_dlib = [dlib.rectangle(left, top, left + width, top + height) for (left, top, width, height) in faces]

        valid_faces = []

        # 检查人脸是否符合规范
        for k, d in enumerate(faces_dlib):
            # 计算矩形框大小 / Compute the size of rectangle box
            height = (d.bottom() - d.top())
            width = (d.right() - d.left())
            hh = int(height/2)
            ww = int(width/2)

            valid_faces.append(d)

        # 如果有符合规范的人脸，创建文件夹并保存
        if len(valid_faces) > 0:
            self.existing_faces_cnt += 1
            current_face_dir = self.path_photos_from_camera + person_name
            os.makedirs(current_face_dir)
            logging.info("\n%-40s %s", "新建的人脸文件夹 / Create folders:", current_face_dir)

            self.ss_cnt = 0  # 将人脸计数器清零 / Clear the cnt of screen shots

            for d in valid_faces:
                height = (d.bottom() - d.top())
                width = (d.right() - d.left())
                hh = int(height/2)
                ww = int(width/2)

                img_blank = np.zeros((int(height*2), width*2, 3), np.uint8)

                self.ss_cnt += 1
                for ii in range(height*2):
                    for jj in range(width*2):
                        if 0<=d.top()-hh + ii<img_rd.shape[0] and 0<=d.left()-ww + jj<img_rd.shape[1]:
                            img_blank[ii][jj] = img_rd[d.top()-hh + ii][d.left()-ww + jj]
                cv2.imwrite(current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg", img_blank)
                logging.info("%-40s %s/img_face_%s.jpg", "写入本地 / Save into：", str(current_face_dir), str(self.ss_cnt))
            return True, "人脸录入成功"
        else:
            logging.warning("未检测到符合规范的人脸 / No valid face detected")
            return False, "未检测到符合规范的人脸"

    @staticmethod
    def detect_and_draw_faces(frame, backend='opencv'):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = DeepFace.extract_faces(img_path=rgb_frame, detector_backend=backend, enforce_detection=False)
            face_coords = []
            for face in faces:
                x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
                face_coords.append((x, y, w, h))
            return face_coords
        except Exception as e:
            logging.error(f"Error in face detection: {str(e)}")
            return []

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"status": "failed", "message": "No file part"})
    file = request.files['file']
    age = request.form['age']
    name = request.form['name']
    gender = request.form['gender']
    if file.filename == '':
        return jsonify({"status": "failed", "message": "No selected file"})
    if file:
        face_register = Face_Register()
        success, message = face_register.process_image(file, name)
        if success:
            cursor.execute("INSERT INTO old (name, age, gender) VALUES (%s, %s, %s)", (name, age, gender))
            db.commit()
            # 更新CSV
            try:
                subprocess.run(["python", "features_extraction_to_csv.py"], check=True)
                return jsonify({"status": "success", "message": message})
            except subprocess.CalledProcessError as e:
                return jsonify({"status": "failed", "message": f"Error updating CSV: {str(e)}"})
        else:
            return jsonify({"status": "failed", "message": message})

@app.route('/delete', methods=['POST'])
def delete_face():
    oid = request.args['oid']
    cursor.execute("SELECT name FROM old WHERE oid = %s", (oid,))
    res = cursor.fetchone()
    name = res[0]
    face_register = Face_Register()
    shutil.rmtree(face_register.path_photos_from_camera+name)
    cursor.execute("DELETE FROM old WHERE oid = %s", (oid,))
    db.commit()
    # 更新CSV
    try:
        subprocess.run(["python", "features_extraction_to_csv.py"], check=True)
        return jsonify({"status": "success", "message": 'delete success'})
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "failed", "message": f"Error updating CSV: {str(e)}"})


@app.route('/getfaces', methods=['GET'])
def get_faces():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))

    offset = (page - 1) * per_page

    cursor.execute("SELECT oid, name, age, gender FROM old LIMIT %s OFFSET %s", (per_page, offset))
    rows = cursor.fetchall()

    cursor.execute("SELECT COUNT(*) FROM old")
    total = cursor.fetchone()[0]

    # 构建返回结果
    result = {
        "total": total,
        "data": [{"oid": row[0], "name": row[1], "age": row[2], "gender": row[3]} for row in rows]
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)