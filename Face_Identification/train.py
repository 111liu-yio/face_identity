import os
import cv2
from PIL import Image
import numpy as np


def getImageAndLabels(path):
    # 存储人脸数据
    facesSamples = []
    # 存储姓名数据
    ids = []
    # 存储图片信息
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # 加载分类器
    face_detector = cv2.CascadeClassifier('D:/python3.9.10/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
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
        for x, y, w, h in faces:
            ids.append(id)
            facesSamples.append(img_numpy[y:y + h, x:x + h])
    # 打印脸部特征和id
    print('id:', id)
    print('fs:', facesSamples)
    return facesSamples, ids


if __name__ == '__main__':
    # 图片路径
    path = './detection/'
    faces, ids = getImageAndLabels(path)
    # 加载识别器
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # 训练
    recognizer.train(faces, np.array(ids))
    # 保存文件
    recognizer.write("./triner/triner.yml")
    print("oookkk")
