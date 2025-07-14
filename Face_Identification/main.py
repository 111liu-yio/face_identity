from PyQt5.QtWidgets import QApplication, QMainWindow
from ui_resource.FaceAtten import Ui_Form
import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import os
from PIL import Image

class LoginMainWindow(QMainWindow, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        print("hhhhh")
        self.cap = cv2.VideoCapture('./1.mp4')
        flag, image = self.cap.read()
        print(flag)
        # 加载训练数据集文件
        self.recogizer = cv2.face.LBPHFaceRecognizer_create()
        self.recogizer.read('./triner/triner.yml')
        self.names = []
        print("hhhhhhhhhhhhhhhhhhhh")
        self.timer = QTimer()
        self.init_timer()
        self.warningtime = 0
        self.name()
        self.closeEvent = self.before_close
        # self.getImageAndLabels('./detection')

    def init_timer(self):
        print("aaaaaaaaaaaaaaaaaaaaaaaaa")
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.display)
        self.timer.start()
        # 循环定时器（每秒执行一次）

        # self.timer.timeout.connect(self.display)
        # self.timer.start(100)

    def getImageAndLabels(self, path):
        # 存储人脸数据
        facesSamples = []
        # 存储姓名数据
        ids = []
        faces = []
        # 存储图片信息
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        # 加载分类器
        face_detector = cv2.CascadeClassifier(
            'D:/python3.9.10/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
        for imagePath in imagePaths:
            # Image.open().convert()是用于打开图像并转换其色彩模式的重要方法
            # Image.open()：打开并读取图像文件，返回Image对象。
            # .convert(mode=None)：将图像转换为指定的色彩模式，返回新的Image对象
            # 打开图片,灰度化 PIL有九种不同的模式: 1,L,P,RGB,RGBA,CMYK,YCbCr,I,F
            """
            模式	        含义	                        应用场景
            'L'	    灰度图（8 位像素，0-255）	        图像处理、OCR、简单视觉任务
            'RGB'	真彩色（3 通道，红、绿、蓝各 8 位）	彩色图像显示、网页图片
            'RGBA'	带透明通道的 RGB（4 通道）	        透明背景图像（如 PNG）
            'CMYK'	印刷四色模式（青、品红、黄、黑）	    印刷行业图像
            '1'	    二值图（1 位像素，0 或 1）	        简单线条图、黑白印章
            'P'	    调色板模式（256 色）	            GIF 动画、节省存储空间
            """
            PIL_img = Image.open(imagePath).convert('L')
            # 将图像转换为数组,以黑白深浅
            img_numpy = np.array(PIL_img, 'uint8')
            # 获取图片人脸特征
            faces = face_detector.detectMultiScale(img_numpy)
            # 获取每张图片的id和姓名
            id = int(os.path.split(imagePath)[1].split('.')[0])
            # 预防无面容图片
            if len(faces) == 0:
                return
            for x, y, w, h in faces:
                ids.append(id)
                facesSamples.append(img_numpy[y:y + h, x:x + h])

        # 打印脸部特征和id
        print('id:', id)
        print('fs:', facesSamples)
        if len(faces) > 0:
            # 加载识别器
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            # 训练
            recognizer.train(faces, np.array(ids))
            # 保存文件
            recognizer.write("./triner/triner.yml")

    def before_close(self):
        print("oooooo")
        self.cap.release()

    # 准备识别的图片
    def face_detect_demo(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度
        face_detector = cv2.CascadeClassifier(
            'D:/python3.9.10/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
        face = face_detector.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))
        # face=face_detector.detectMultiScale(gray)

        if len(face) == 0:
            self.widget_2.hide()
            return
        for x, y, w, h in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
            cv2.circle(img, center=(x + w // 2, y + h // 2), radius=w // 2, color=(0, 255, 0), thickness=1)
            # face_recognition
            ids, confidence = self.recogizer.predict(gray[y:y + h, x:x + w])
            print('标签id:', ids, '置信评分：', confidence)
            if confidence > 75:
                self.widget_2.hide()
                self.warningtime += 1
                if self.warningtime > 100:
                    # warning()
                    self.warningtime = 0
                cv2.putText(img, 'unkonw', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
            else:
                self.widget_2.show()
                cv2.putText(img, str(ids) + "-" + str(self.names[ids - 1]), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 255, 0), 1)

        # cv2.imshow('result', img)
        # return img

        # print('bug:',ids)

    def name(self):
        path = './detection/'
        # names = []
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        for imagePath in imagePaths:
            name = str(os.path.split(imagePath)[1].split('.', 2)[1])
            self.names.append(name)

    def display(self):
        flag, image = self.cap.read()
        self.face_detect_demo(image)
        print(image.shape)
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)  # 16位转8位
        cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qImg = QImage(cv_image.data,
                      cv_image.shape[1],
                      cv_image.shape[0],
                      cv_image.strides[0],
                      QImage.Format_RGB888)
        pixmap = QPixmap(qImg)
        self.VideoLabel.setPixmap(pixmap)  # 加载 PyQt 图像


app = QApplication([])
loginw = LoginMainWindow()
loginw.show()
app.exec_()
