# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QTimer
from PIL import Image
from PyQt5.QtGui import QPixmap, QImage
import cv2
from datetime import datetime
import os
import numpy as np
import pymysql
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox
from ui_resource.Camera import CameraManager
from ui_resource.DB import SQLPool


class Ui_Manager(QDialog):
    def __init__(self):
        super().__init__()
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(self.project_root)  # 切换工作目录
        self.setupUi(self)
        self.captured_frame = None
        self.classifier_path = 'D:/python3.9.10/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml'
        self.train_path = rf'{self.project_root}/train_model/face_train.yml'
        self.jpgs_save_path = rf'{self.project_root}/detection'
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.connection = SQLPool()
        self.camera = CameraManager()
        # 定时器和摄像头
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.ini_fun()

    def setupUi(self, QDialog):
        QDialog.setObjectName("QDialog")
        QDialog.setFixedSize(500, 600)  # 适当增大窗口尺寸

        # ================ 背景图设置（最底层） ================
        self.background_label = QtWidgets.QLabel(QDialog)
        self.background_label.setGeometry(QtCore.QRect(0, 0, 500, 600))
        self.background_label.setScaledContents(True)
        # 替换为实际背景图路径，若不存在则显示雅白色
        bg_pixmap = QtGui.QPixmap("t01.jpg")  # 可替换为你的背景图路径
        if bg_pixmap.isNull():
            self.background_label.setStyleSheet("background-color: rgba(250, 249, 245, 0.8);")  # 提高背景不透明度
        else:
            self.background_label.setPixmap(bg_pixmap)
            self.background_label.setStyleSheet("opacity: 0.8;")  # 背景图稍清晰
        self.background_label.lower()  # 确保背景图在最底层

        # ================ 样式表设置（降低透明度，确保label清晰） ================
        QDialog.setStyleSheet("""
            QDialog {
                background-color: rgba(250, 249, 245, 0.8);  /* 提高整体不透明度 */
            }
            /* 标签页容器样式 */
            QTabWidget::pane {
                border: 1px solid rgba(221, 221, 221, 0.85);  /* 边框更清晰 */
                background-color: rgba(250, 249, 245, 0.8);  /* 稍提高不透明度 */
                border-radius: 5px;
                margin: 5px;
            }
            /* 标签样式 */
            QTabBar::tab {
                background-color: rgba(232, 231, 226, 0.85);  /* 标签更清晰 */
                color: rgba(85, 85, 85, 0.95);  /* 标签文字显著提高清晰度 */
                padding: 11px 26px;
                height: 15px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 2px;
                margin-left: 5px; 
                font: 9pt "微软雅黑";
            }
            /* 选中标签样式 */
            QTabBar::tab:selected {
                background-color: rgba(250, 249, 245, 0.9);  /* 选中标签更清晰 */
                color: rgba(51, 51, 51, 0.95);  /* 文字接近完全不透明 */
                font-weight: bold;
                border: 1px solid rgba(221, 221, 221, 0.85);
                border-bottom-color: rgba(250, 249, 245, 0.9);
            }
            /* 悬停标签样式 */
            QTabBar::tab:hover:!selected {
                background-color: rgba(220, 219, 214, 0.85);  /* 悬停标签更清晰 */
            }
            /* 按钮样式 */
            QPushButton {
                background-color: #faf9f5;
                color: rgba(85, 85, 85, 0.9);  /* 按钮文字更清晰 */
                border: 1px solid rgba(221, 221, 221, 0.85);
                border-radius: 6px;
                font: 10pt "微软雅黑";
                padding: 8px;
                transition: all 0.3s ease;
            }
            QPushButton:hover {
                background-color: rgba(232, 231, 226, 0.85);
                transform: translateY(-2px);
            }
            QPushButton:pressed {
                background-color: rgba(220, 219, 214, 0.85);
                transform: translateY(0);
            }
            /* 输入框样式 */
            QLineEdit {
                background-color: #faf9f5;  /* 输入框更清晰 */
                border: 1px solid rgba(221, 221, 221, 0.85);
                border-radius: 4px;
                padding: 8px;
                font: 10pt "微软雅黑";
                color: rgba(51, 51, 51, 0.95);  /* 输入文字清晰 */
            }
            QLineEdit:focus {
                border-color: #faf9f5;
                outline: none;
                background-color: #faf9f5;
            }
            /* 标签样式（核心：提高文字和背景清晰度） */
            QLabel {
                color: rgba(85, 85, 85, 0.95);  /* 标签文字接近完全不透明 */
                background-color: transparent;
            }
            /* 文本编辑区样式 */
            QTextEdit {
                background-color: #faf9f5;
                border: 1px solid rgba(221, 221, 221, 0.85);
                border-radius: 4px;
                font: 10pt "微软雅黑";
                color: rgba(51, 51, 51, 0.95);  /* 文本清晰 */
                padding: 5px;
            }
        """)

        self.tabWidget = QtWidgets.QTabWidget(QDialog)  # 整个标签页大小位置
        self.tabWidget.setGeometry(QtCore.QRect(10, 10, 480, 590))
        self.tabWidget.setObjectName("tabWidget")

        # 录入标签页
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")

        # 视频区域（保持清晰）
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setGeometry(QtCore.QRect(10, 10, 450, 300))  # 虚线框的位置大小
        self.label.setStyleSheet("""
            background-color: #faf9f5; 
            border: 2px dashed rgba(221, 221, 221, 0.95);  /* 边框清晰可见 */
            border-radius: 5px;
        """)
        self.label.setText("")
        self.label.setObjectName("label")

        self.pushButton = QtWidgets.QPushButton(self.tab)
        self.pushButton.setGeometry(QtCore.QRect(30, 450, 125, 50))
        self.pushButton.setObjectName("pushButton")

        self.pushButton_2 = QtWidgets.QPushButton(self.tab)
        self.pushButton_2.setGeometry(QtCore.QRect(175, 450, 125, 50))
        self.pushButton_2.setObjectName("pushButton_2")

        self.pushButton_6 = QtWidgets.QPushButton(self.tab)
        self.pushButton_6.setGeometry(QtCore.QRect(320, 450, 125, 50))
        self.pushButton_6.setObjectName("pushButton_6")

        self.widget_7 = QtWidgets.QWidget(self.tab)  # 部门+输入
        self.widget_7.setGeometry(QtCore.QRect(32, 385, 410, 40))
        self.widget_7.setObjectName("widget_7")
        self.widget_7.setStyleSheet("background-color: transparent;")

        self.label_8 = QtWidgets.QLabel(self.widget_7)
        self.label_8.setGeometry(QtCore.QRect(0, 0, 80, 40))
        self.label_8.setStyleSheet(
            "background-color: rgba(220, 219, 214, 0.85); color: rgba(85, 85, 85, 0.95); font: 10pt '微软雅黑';")
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")

        self.lineEdit_7 = QtWidgets.QLineEdit(self.widget_7)
        self.lineEdit_7.setGeometry(QtCore.QRect(90, 0, 320, 40))
        self.lineEdit_7.setObjectName("lineEdit_7")

        self.widget_8 = QtWidgets.QWidget(self.tab)  # 姓名+输入
        self.widget_8.setGeometry(QtCore.QRect(32, 335, 410, 40))
        self.widget_8.setObjectName("widget_8")
        self.widget_8.setStyleSheet("background-color: transparent;")

        self.label_9 = QtWidgets.QLabel(self.widget_8)
        self.label_9.setGeometry(QtCore.QRect(0, 0, 80, 40))
        self.label_9.setStyleSheet(
            "background-color: rgba(220, 219, 214, 0.85); color: rgba(85, 85, 85, 0.95); font: 10pt '微软雅黑';")
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")

        self.lineEdit_8 = QtWidgets.QLineEdit(self.widget_8)
        self.lineEdit_8.setGeometry(QtCore.QRect(90, 0, 320, 40))
        self.lineEdit_8.setObjectName("lineEdit_8")

        self.tabWidget.addTab(self.tab, "")

        # 修改标签页
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")

        self.widget_3 = QtWidgets.QWidget(self.tab_2)
        self.widget_3.setGeometry(QtCore.QRect(10, 20, 410, 40))
        self.widget_3.setObjectName("widget_3")
        self.widget_3.setStyleSheet("background-color: transparent;")

        self.label_4 = QtWidgets.QLabel(self.widget_3)
        self.label_4.setGeometry(QtCore.QRect(0, 0, 80, 40))
        self.label_4.setStyleSheet(
            "background-color: rgba(220, 219, 214, 0.85); color: rgba(85, 85, 85, 0.95); font: 10pt '微软雅黑';")
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")

        self.lineEdit_3 = QtWidgets.QLineEdit(self.widget_3)
        self.lineEdit_3.setGeometry(QtCore.QRect(90, 0, 320, 40))
        self.lineEdit_3.setObjectName("lineEdit_3")

        self.pushButton_3 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 180, 410, 50))
        self.pushButton_3.setObjectName("pushButton_3")

        self.widget_4 = QtWidgets.QWidget(self.tab_2)
        self.widget_4.setGeometry(QtCore.QRect(10, 75, 410, 40))
        self.widget_4.setObjectName("widget_4")
        self.widget_4.setStyleSheet("background-color: transparent;")

        self.label_5 = QtWidgets.QLabel(self.widget_4)
        self.label_5.setGeometry(QtCore.QRect(0, 0, 80, 40))
        self.label_5.setStyleSheet(
            "background-color: rgba(220, 219, 214, 0.85); color: rgba(85, 85, 85, 0.95); font: 10pt '微软雅黑';")
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")

        self.lineEdit_4 = QtWidgets.QLineEdit(self.widget_4)
        self.lineEdit_4.setGeometry(QtCore.QRect(90, 0, 320, 40))
        self.lineEdit_4.setObjectName("lineEdit_4")

        self.widget_5 = QtWidgets.QWidget(self.tab_2)
        self.widget_5.setGeometry(QtCore.QRect(10, 130, 410, 40))
        self.widget_5.setObjectName("widget_5")
        self.widget_5.setStyleSheet("background-color: transparent;")

        self.label_6 = QtWidgets.QLabel(self.widget_5)
        self.label_6.setGeometry(QtCore.QRect(0, 0, 80, 40))
        self.label_6.setStyleSheet(
            "background-color: rgba(220, 219, 214, 0.85); color: rgba(85, 85, 85, 0.95); font: 10pt '微软雅黑';")
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")

        self.lineEdit_5 = QtWidgets.QLineEdit(self.widget_5)
        self.lineEdit_5.setGeometry(QtCore.QRect(90, 0, 320, 40))
        self.lineEdit_5.setObjectName("lineEdit_5")

        self.tabWidget.addTab(self.tab_2, "")

        # 查询标签页
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")

        self.widget_6 = QtWidgets.QWidget(self.tab_3)
        self.widget_6.setGeometry(QtCore.QRect(10, 20, 310, 40))
        self.widget_6.setObjectName("widget_6")
        self.widget_6.setStyleSheet("background-color: transparent;")

        self.label_7 = QtWidgets.QLabel(self.widget_6)
        self.label_7.setGeometry(QtCore.QRect(0, 0, 90, 40))
        self.label_7.setStyleSheet(
            "background-color: rgba(220, 219, 214, 0.85); color: rgba(85, 85, 85, 0.95); font: 10pt '微软雅黑';")
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")

        self.lineEdit_6 = QtWidgets.QLineEdit(self.widget_6)
        self.lineEdit_6.setGeometry(QtCore.QRect(110, 0, 200, 40))
        self.lineEdit_6.setObjectName("lineEdit_6")

        self.pushButton_4 = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_4.setGeometry(QtCore.QRect(330, 20, 90, 40))
        self.pushButton_4.setObjectName("pushButton_4")

        self.textEdit = QtWidgets.QTextEdit(self.tab_3)
        self.textEdit.setGeometry(QtCore.QRect(10, 80, 420, 440))
        self.textEdit.setObjectName("textEdit")

        self.tabWidget.addTab(self.tab_3, "")

        # 删除标签页
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")

        self.widget_9 = QtWidgets.QWidget(self.tab_4)
        self.widget_9.setGeometry(QtCore.QRect(10, 20, 310, 40))
        self.widget_9.setObjectName("widget_9")
        self.widget_9.setStyleSheet("background-color: transparent;")

        self.label_10 = QtWidgets.QLabel(self.widget_9)
        self.label_10.setGeometry(QtCore.QRect(0, 0, 100, 40))
        self.label_10.setStyleSheet(
            "background-color: rgba(220, 219, 214, 0.85); color: rgba(85, 85, 85, 0.95); font: 10pt '微软雅黑';")
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")

        self.lineEdit_9 = QtWidgets.QLineEdit(self.widget_9)
        self.lineEdit_9.setGeometry(QtCore.QRect(110, 0, 200, 40))
        self.lineEdit_9.setObjectName("lineEdit_9")

        self.pushButton_5 = QtWidgets.QPushButton(self.tab_4)
        self.pushButton_5.setGeometry(QtCore.QRect(330, 20, 90, 40))
        self.pushButton_5.setObjectName("pushButton_5")

        self.tabWidget.addTab(self.tab_4, "")

        # 设置文本
        self.retranslateUi(QDialog)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(QDialog)

    def retranslateUi(self, QDialog):
        _translate = QtCore.QCoreApplication.translate
        QDialog.setWindowTitle(_translate("QDialog", "数据管理"))
        self.pushButton.setText(_translate("QDialog", "打开摄像头"))
        self.pushButton_2.setText(_translate("QDialog", "拍照"))
        self.pushButton_6.setText(_translate("QDialog", "录入"))
        self.label_8.setText(_translate("QDialog", "部门"))
        self.lineEdit_8.setPlaceholderText(_translate("QDialog", "请输入姓名"))
        self.label_9.setText(_translate("QDialog", "姓名"))
        self.lineEdit_7.setPlaceholderText(_translate("QDialog", "请输入部门"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("QDialog", "录入"))
        self.label_4.setText(_translate("QDialog", "工号"))
        self.lineEdit_3.setPlaceholderText(_translate("QDialog", "请输入工号"))
        self.pushButton_3.setText(_translate("QDialog", "确认修改"))
        self.label_5.setText(_translate("QDialog", "姓名"))
        self.lineEdit_4.setPlaceholderText(_translate("QDialog", "请输入姓名"))
        self.label_6.setText(_translate("QDialog", "部门"))
        self.lineEdit_5.setPlaceholderText(_translate("QDialog", "请输入部门"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("QDialog", "修改"))
        self.label_7.setText(_translate("QDialog", "查询考勤"))
        self.lineEdit_6.setPlaceholderText(_translate("QDialog", "请输入工号"))
        self.pushButton_4.setText(_translate("QDialog", "查询"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("QDialog", "查询"))
        self.label_10.setText(_translate("QDialog", "删除"))
        self.lineEdit_9.setPlaceholderText(_translate("QDialog", "请输入工号"))
        self.pushButton_5.setText(_translate("QDialog", "删除"))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("QDialog", "删除"))

    # 弹窗显示
    def show_warning_dialog(self, title, message):
        """显示警告弹窗"""
        QMessageBox.warning(self, title, message, QMessageBox.Ok)

    def start_camera(self):
        """在主窗口中开启摄像头"""
        if self.camera.open():
            # 100ms
            self.timer.start(100)
        else:
            self.show_warning_dialog("open fail", "打开摄像头失败!")

    def ini_fun(self):
        # manager ui连接按钮事件
        self.pushButton.clicked.connect(self.open_camera)  # 打开/关闭摄像头
        self.pushButton_2.clicked.connect(self.take_photo)  # 拍照
        self.pushButton_6.clicked.connect(self.enter_database)  # 录入
        self.pushButton_4.clicked.connect(self.query_by_id)  # 查询绑定点击事件
        self.pushButton_3.clicked.connect(self.update_user)  # 修改绑定点击事件
        self.pushButton_5.clicked.connect(self.delete_employee)  # 删除绑定按钮事件

    # 打开摄像头函数
    def open_camera(self):
        """更新视频帧到 QLabel 上"""
        self.start_camera()

    def update_frame(self):
        if self.camera.is_opened():
            ret, frame = self.camera.read_frame()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image).scaled(self.label.width(), self.label.height())
                self.label.setPixmap(pixmap)

    # 拍照 按钮函数
    def take_photo(self):
        """拍照并保存图像到内存，同时显示在界面上"""
        if self.camera.is_opened():
            ret, frame = self.camera.read_frame()
            if ret:
                self.captured_frame = frame
                self.update_to_label(frame)  # 关闭视频流,将拍照的那张图片显示到 label
                print("照片已捕获, 更新label成功")

    def update_to_label(self, image):
        """显示捕获的照片"""
        try:
            # 关闭定时器,停止显示视频流
            self.timer.stop()
            self.camera.release()
            # 转换jpg格式为QPixmap
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(pixmap)
            return True
        except Exception as e:
            self.show_warning_dialog("Error", "拍照图片更新失败!")
            print(rf'[ERROR] {e}')
            return False

    def getImageAndLabels(self, imagePath):
        # 存储人脸数据
        faces_samples = []
        # 存储id数据
        ids = []
        # 加载分类器
        face_detector = cv2.CascadeClassifier(self.classifier_path)
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
        gray_img = Image.open(imagePath).convert('L')
        # 录入加入均衡化处理,识别也加该处理
        # gray_img = cv2.equalizeHist(gray_img)  # 直方图均衡化，增强对比度
        # 将图像转换为数组,以黑白深浅
        img_numpy = np.array(gray_img, 'uint8')
        # 获取图片人脸特征
        faces = face_detector.detectMultiScale(img_numpy)
        # 获取每张图片的id
        emp_id = int(imagePath[imagePath.rfind('\\')+1:].split('.')[0])

        # 预防无面容图片
        for x, y, w, h in faces:
            ids.append(emp_id)
            faces_samples.append(img_numpy[y:y + h, x:x + h])

        # 打印脸部特征和id
        print('id:', ids)
        print('fs:', faces_samples)
        return faces_samples, ids

    def train_model(self, old_model_path, jpg_path):
        # 加载识别器
        new_faces, new_ids = self.getImageAndLabels(jpg_path)
        # 训练 start
        # train()：从头开始训练，会覆盖原有模型（不保留历史数据）。
        # update()：仅LBPH识别器支持，可在已有模型基础上补充新数据，保留历史训练信息。
        # 2. 加载已有模型（若存在）
        if os.path.exists(old_model_path):
            self.recognizer.read(old_model_path)
            print(f"已加载原有模型：{old_model_path}")
        else:

            # 若模型不存在，则直接用新数据训练(相当于首次训练)
            print(f"未找到原有模型，将用新数据创建模型：{old_model_path}")
            self.recognizer.train(new_faces, np.array(new_ids))
            # 保存新模型 固定位置,位置写死
            self.recognizer.write(old_model_path)
            return

        # 3. 用新数据更新模型（增量训练）
        # 注意：new_faces必须是灰度图，且尺寸与旧模型训练数据一致
        self.recognizer.update(new_faces, np.array(new_ids))
        print("新数据已追加到原有模型")
        # 4. 保存更新后的模型（覆盖原文件）
        self.recognizer.write(old_model_path)

    def enter_database(self):
        name = self.lineEdit_8.text()
        department = self.lineEdit_7.text()

        # 拍照帧没有保存,则退出
        if self.captured_frame is None:
            self.show_warning_dialog("提示", "未拍照,请先拍照!")
            return
        # 判空,为空退出
        if not name or not department:
            self.show_warning_dialog("提示", "姓名和部门不能为空!")
            return
        # 获取当前时间戳,作为入职时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        max_id = 10000

        conn = self.connection.get_conn()
        print("get_conn")
        try:
            with conn.cursor as cursor:
                # 先将姓名,部门,入职时间 插入数据库
                sql = "INSERT INTO employeeinfo(ename, dname, hidrdate) VALUES (%s, %s, %s)"
                cursor.execute(sql, [name, department, current_time])  # 插入 id 字段
                conn.conn.commit()
                print("信息入库成功")
                # 查询当前员工信息表最大id,作为图片的名字
                sql = "SELECT MAX(id) FROM employeeinfo"
                cursor.execute(sql)
                result = cursor.fetchone()
                if result:
                    ids = list(result.values())
                    if ids[0] is None:
                        max_id = 10000
                    else:
                        max_id = ids[0]
                else:
                    self.show_warning_dialog("Error", "数据库查询失败")
        except Exception as e:
            # 发生错误时回滚事务
            conn.rollback()
            print(f"查询最大 id 失败: {e}")
            self.show_warning_dialog("Error", "数据库查询失败!")
            return
        finally:
            conn.close()
        print("sql is ok")
        # 保存拍照帧,指定保存路径为../detection
        detection_path = self.jpgs_save_path
        os.makedirs(detection_path, exist_ok=True)
        filename = f"{max_id}.jpg"
        save_path = os.path.join(detection_path, filename)
        cv2.imwrite(save_path, self.captured_frame)
        print(f"照片已保存为 {save_path}")
        # 训练模型
        self.train_model(self.train_path, save_path)

    # 数据修改
    def update_user(self):
        try:
            # 获取输入并校验
            emp_id = self.lineEdit_3.text().strip()  # 去掉首尾空格
            if not emp_id:
                self.show_warning_dialog("Error", "工号不能为空！")
                return

            new_ename = self.lineEdit_4.text().strip()
            if not new_ename:
                self.show_warning_dialog("Error", "姓名不能为空！")
                return

            new_dname = self.lineEdit_5.text().strip()
            if not new_dname:
                self.show_warning_dialog("Error", "部门不能为空！")
                return

            conn = self.connection.get_conn()
            with conn.cursor as cur:
                # 1. 先检查ID是否存在
                cur.execute("SELECT id FROM employeeinfo WHERE id = %s", emp_id)
                if not cur.fetchone():  # 如果查询不到结果
                    self.show_warning_dialog("Error", rf"工号 {emp_id} 不存在！")
                    return

                # 2. 执行更新（只有ID存在时才更新）
                sql = "UPDATE employeeinfo SET ename = %s, dname = %s WHERE id = %s"
                cur.execute(sql, [new_ename, new_dname, emp_id])
                conn.conn.commit()  # 提交事务
                self.show_warning_dialog("Tip", "信息修改成功！")
        except Exception as e:
            conn.rollback()  # 出错时回滚
            print("数据库错误:", e)
            self.show_warning_dialog("Error", "操作失败，请检查输入或联系管理员！")
        finally:
            conn.close()

    # 打卡查询
    def query_by_id(self):
        try:
            emp_id = self.lineEdit_6.text().strip()

            # 判断是否输入是空
            if not emp_id:
                # 内容为空弹窗报错
                self.show_warning_dialog("Error", "工号不能为空！")
                return
            # with连接数据库
            conn = self.connection.get_conn()
            with conn.cursor as cur:
                sql = """
                                      select 
                                      e.id,e.ename,e.dname,a.checkintime
                                      from employeeinfo as e
                                      left join attendancerecord as a
                                      on e.id = a.employeeid
                                      where e.id = %s
                                      """
                cur.execute(sql, emp_id)
                result = cur.fetchall()
                # 判断输入的工号是否存在
                if not result:
                    # 不存在报错
                    self.show_warning_dialog("Error", rf"未找到 {emp_id} 工号的员工信息")
                    return
                label_info = ""
                for item in result:
                    record = ""
                    for i in item.values():
                        record += (str(i) + "  ")
                    label_info += (record + '\n')
                self.textEdit.setText(label_info)

        except pymysql.MySQLError as e:
            print(e)
            conn.rollback()
            self.show_warning_dialog("Error", "查询数据库失败,请联系数据库管理员!")
        finally:
            conn.close()

    # 删除数据
    def delete_employee(self):
        """删除员工信息"""
        emp_id = self.lineEdit_9.text().strip()

        reply = QMessageBox.question(self, "删除", "确定要删除这条数据吗?")

        if reply == QMessageBox.Yes:
            conn = self.connection.get_conn()
            try:
                with conn.cursor as cur:
                    # 先查询
                    select_sql = "select id AS id, ename, dname from employeeinfo where id = %s"
                    cur.execute(select_sql, emp_id)
                    employee = cur.fetchone()
                    if not employee:
                        self.show_warning_dialog("Error", "未找到该员工信息")
                        return

                    departure_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # 开启事务
                    conn.begin()
                    # 插入到离职员工表
                    insert_sql = "INSERT INTO ex_employeeinfo (employeeid, ename, dname, departure_date) VALUES (%s, %s, %s,%s)"

                    cur.execute(insert_sql, (employee['id'], employee['ename'], employee['dname'], departure_time))

                    # 从在职员工表删除
                    delete_sql = "DELETE FROM employeeinfo WHERE id = %s"
                    cur.execute(delete_sql, emp_id)

                    # 提交事务
                    conn.conn.commit()
                    self.show_warning_dialog("成功", "员工信息已删除并移致离职员工表")
                    self.lineEdit_9.clear()

            except Exception as e:
                # 发生错误时回滚事务
                conn.rollback()
                self.show_warning_dialog("Error", rf"错误信息: {str(e)}")
            finally:
                conn.close()

    def closeEvent(self, event):
        """关闭子窗口时释放摄像头"""
        self.timer.stop()
        self.camera.release()
        self.connection.close()
        event.accept()


if __name__ == "__main__":
    import sys
    app = QApplication([])
    ui = Ui_Manager()
    ui.show()
    sys.exit(app.exec_())
