import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage, QRegion
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox
import cv2
from PyQt5.QtCore import QTimer, Qt
import numpy as np
import os
from datetime import datetime
from ui_resource.Camera import CameraManager
from ui_resource.passwd import Ui_Passwd
from ui_resource.Manager import Ui_Manager
from ui_resource.DB import SQLPool


class FaceRecognitionUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.line_edits = {}  # 存储输入框引用
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(self.project_root)  # 切换工作目录
        self.setup_ui()
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        # classifier_path 修改为本地的python环境中该文件的路径
        self.classifier_path = 'D:/python3.9.10/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml'
        self.train_path = rf'{self.project_root}/train_model/face_train.yml'
        self.jpgs_save_path = rf'{self.project_root}/detection'
        # 定时器和摄像头
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.camera = CameraManager()
        self.warningtime = 0
        self.connection = SQLPool()
        self.load_train()

    # 弹窗显示
    def show_warning_dialog(self, title, message):
        # 创建消息框
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.addButton(QMessageBox.Ok)
        # msg_box.addButton(QMessageBox.Cancel)
        # 设置消息框样式表（核心）
        msg_box.setStyleSheet("""
            /* 消息框整体背景 */
            QMessageBox {
                background-color: #faf9f5;  /* 雅白色背景 */
            }
            /* 消息框标题栏文本 */
            QMessageBox QLabel#qt_msgbox_title {
                color: #555;  /* 深灰色文字 */
                font: bold 12pt "微软雅黑";
            }
            /* 消息框内容文本 */
            QMessageBox QLabel#qt_msgbox_label {
                color: #333;  /* 黑色文字 */
                font: 10pt "微软雅黑";
                padding: 10px;
            }
            /* 消息框按钮 */
            QMessageBox QPushButton {
                background-color: transparent;  /* 按钮背景 */
                color: #ffffff;  /* 深灰色文字 */
                border: 2px solid #ddd;  /* 浅灰色边框 */
                border-radius: 6px;
                font: 10pt "微软雅黑";
                padding: 5px 15px;
                margin: 5px;
            }
            /* 按钮悬停效果 */
            QMessageBox QPushButton:hover {
                background-color: #e8e7e2;  /* 浅灰色悬停背景 */
            }
            /* 按钮点击效果 */
            QMessageBox QPushButton:pressed {
                background-color: #dcdbd6;  /* 深一点的灰色点击背景 */
            }
        """)

        # 显示消息框并获取点击结果
        result = msg_box.exec_()
        # if result == QMessageBox.Ok:
        #     print('点击了确定')
        # else:
        #     print('点击了取消')

    def load_train(self):
            if os.path.exists(self.train_path):
                self.recognizer.read(self.train_path)
                self.start_camera()
            else:
                self.show_warning_dialog("Tip", "暂无人脸模型,请先录入")
                # 打开子窗口并等待关闭
                manager_ui = Ui_Manager()
                manager_ui.exec_()
                self.recognizer.read(self.train_path)
                self.start_camera()

    def setup_ui(self):
        # 窗口配置
        self.setObjectName("FaceRecognition")
        self.setFixedSize(950, 650)

        # 背景图（最底层）
        self.background_label = QtWidgets.QLabel(self)
        self.background_label.setGeometry(QtCore.QRect(0, 0, self.width(), self.height()))
        self.background_label.setScaledContents(True)
        bg_pixmap = QtGui.QPixmap("t01.jpg")
        if bg_pixmap.isNull():
            self.background_label.setStyleSheet("background-color: rgb(250, 250, 245);")
        else:
            self.background_label.setPixmap(bg_pixmap)
        self.setStyleSheet("background-color: transparent;")

        # 主布局（左右分栏）
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # 左侧视频区
        self.video_widget = QtWidgets.QWidget()
        self.video_widget.setFixedWidth(580)
        self.video_widget.setStyleSheet("""
            background-color: rgba(250, 250, 245, 0.9);
            border-radius: 8px;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.18);
        """)
        video_layout = QtWidgets.QVBoxLayout(self.video_widget)
        video_layout.setContentsMargins(10, 10, 10, 10)
        video_layout.setAlignment(QtCore.Qt.AlignCenter)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(560, 520)
        self.video_label.setScaledContents(True)
        self.video_label.setStyleSheet("""
            background-color: rgb(245, 245, 240);
            border-radius: 6px;
            border: 1px solid #eee;
        """)
        video_layout.addWidget(self.video_label)

        self.tip_label = QtWidgets.QLabel(self.video_widget)
        self.tip_label.setGeometry(QtCore.QRect(175, 470, 230, 45))
        self.tip_label.setStyleSheet("""
            color: rgba(0, 255, 0, 0.95); ; 
            font: 20pt "Arial"; 
            border-radius: 8px; 
            padding: 5px;
            background-color: transparent;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
        """)
        self.tip_label.setAlignment(QtCore.Qt.AlignCenter)
        self.tip_label.hide()

        self.tip_label1 = QtWidgets.QLabel(self.video_widget)
        self.tip_label1.setGeometry(QtCore.QRect(175, 470, 230, 45))
        self.tip_label1.setStyleSheet("""
                    color: rgba(255, 0, 0, 0.95); 
                    font: 20pt "Arial"; 
                    border-radius: 8px; 
                    padding: 5px;
                    background-color: transparent;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
                """)
        self.tip_label1.setAlignment(QtCore.Qt.AlignCenter)
        self.tip_label1.hide()

        main_layout.addWidget(self.video_widget)

        # 右侧信息交互区（核心优化区域）
        self.info_widget = QtWidgets.QWidget()
        self.info_widget.setFixedWidth(300)  # 保持宽度不变，通过内部布局调整左移
        self.info_widget.setStyleSheet("""
            background-color: rgba(250, 250, 245, 0.8);
            border-radius: 8px;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08);
        """)
        info_layout = QtWidgets.QVBoxLayout(self.info_widget)
        # 调整左侧边距为10（原30），实现整体左移
        info_layout.setContentsMargins(10, 30, 30, 20)
        info_layout.setSpacing(15)

        # 标题
        self.title_label = QtWidgets.QLabel()
        self.title_label.setStyleSheet("""
            color: #333; 
            font: 18pt "微软雅黑"; 
            font-weight: bold;
            background-color: transparent;
        """)
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        info_layout.addWidget(self.title_label)

        # 人脸占位（圆圈）- 保持不被覆盖
        self.face_label = QtWidgets.QLabel()
        self.face_label.setFixedSize(160, 160)
        self.face_label.setStyleSheet("""
            border-radius: 80px; 
            background-color: #eee;
            border: 4px solid white;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
        """)
        self.face_label.setAlignment(QtCore.Qt.AlignCenter)
        self.face_label.raise_()
        info_layout.addWidget(self.face_label, alignment=QtCore.Qt.AlignCenter)

        # 输入项：工号、姓名、部门、时间（整体左移+输入框加长）
        self.create_input_row(info_layout, "工号", "lineEdit")
        self.create_input_row(info_layout, "姓名", "lineEdit_2")
        self.create_input_row(info_layout, "部门", "lineEdit_3")
        self.create_input_row(info_layout, "时间", "lineEdit_4")

        # 管理按钮
        self.manage_btn = QtWidgets.QPushButton()
        self.manage_btn.setMinimumHeight(45)
        self.manage_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #333; 
                font: 13pt "微软雅黑"; 
                border: 1px solid #ccc; 
                border-radius: 6px; 
                padding: 8px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.5);
            }
        """)
        self.manage_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.manage_btn.clicked.connect(self.active_manager)
        info_layout.addWidget(self.manage_btn, alignment=QtCore.Qt.AlignCenter)

        # 右侧底部：极限优化小队队标
        self.team_logo = QtWidgets.QLabel()
        self.team_logo.setFixedSize(105, 24)
        team_pixmap = QtGui.QPixmap("ui_resource/logo.png")
        if not team_pixmap.isNull():
            self.team_logo.setPixmap(team_pixmap.scaled(
                self.team_logo.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            ))
        else:
            self.team_logo.setText("极限优化小队")
            self.team_logo.setStyleSheet("color: #999; font: 9pt \"微软雅黑\";")
        self.team_logo.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)
        info_layout.addWidget(self.team_logo, alignment=QtCore.Qt.AlignRight)

        main_layout.addWidget(self.info_widget)

        # 确保背景图最底层
        self.background_label.lower()

        self.retranslate_ui()

    def create_input_row(self, layout, label_text, edit_name, read_only=False):
        row_widget = QtWidgets.QWidget()
        row_widget.setStyleSheet("background-color: transparent;")
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(10)  # 保持标签与输入框间距

        # 标签（略微缩短宽度，给输入框让出空间）
        label = QtWidgets.QLabel(label_text)
        label.setStyleSheet("""
            color: #666; 
            font: 13pt "微软雅黑"; 
            background-color: transparent;
        """)
        label.setFixedWidth(50)  # 原70，缩短10px
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        # 输入框（核心优化：增加长度+固定宽度）
        line_edit = QtWidgets.QLineEdit()
        line_edit.setObjectName(edit_name)
        line_edit.setMinimumHeight(40)
        line_edit.setFixedWidth(220)  # 固定宽度为180px（原自适应，约150px）
        self.line_edits[edit_name] = line_edit

        if read_only:
            line_edit.setReadOnly(True)
            line_edit.setStyleSheet("""
                background-color: #f5f5f5;
                color: #999; 
                border: 1px solid #eee;
                font: 13pt "微软雅黑"; 
                padding: 8px; 
                border-radius: 6px;
            """)
        else:
            line_edit.setStyleSheet(f"""
                background-color: rgba(255, 255, 254, 0.5);
                color: #333; 
                border: 1px solid #ddd;
                font: 11pt "微软雅黑"; 
                padding: 12px; 
                border-radius: 6px;
            """)

        row_layout.addWidget(label)
        row_layout.addWidget(line_edit)
        layout.addWidget(row_widget)

    def retranslate_ui(self):
        self.setWindowTitle("人脸识别考勤系统")
        self.title_label.setText("人脸识别考勤系统")
        self.manage_btn.setText("管理")
        self.tip_label.setText("打卡成功")
        self.tip_label1.setText("打卡失败")

    def resizeEvent(self, event):
        self.background_label.setGeometry(QtCore.QRect(0, 0, self.width(), self.height()))
        super().resizeEvent(event)

    def start_camera(self):
        """在主窗口中开启摄像头"""
        if self.camera.open():
            # 100ms
            self.timer.start(100)
        else:
            self.show_warning_dialog("open fail", "打开摄像头失败!")

    def active_manager_ui(self):
        self.timer.stop()  # 停止主窗口刷新
        self.camera.release()  # 释放摄像头
        # 打开子窗口并等待关闭
        manager_ui = Ui_Manager()
        manager_ui.exec_()

        self.lineEdit_clear()
        self.recognizer.read(self.train_path)
        self.start_camera()

    def active_manager(self):
        self.passwd = Ui_Passwd()
        if self.passwd.exec_() == QDialog.Accepted:
            user_input = self.passwd.lineEdit.text().strip()
            if user_input == "123456":
                self.active_manager_ui()
            else:
                self.show_warning_dialog("Waring", "密码错误")

    def lineEdit_clear(self):
        self.line_edits['lineEdit'].setText("")
        self.line_edits['lineEdit_2'].setText("")
        self.line_edits['lineEdit_3'].setText("")
        self.line_edits['lineEdit_4'].setText("")
        self.face_label.clear()

    def load_jpg(self, image_path):
        # 加载图片
        pixmap = QPixmap(image_path)
        # 检查图片是否加载成功
        if pixmap.isNull():
            # self.image_label.setText("图片加载失败，请检查路径是否正确")
            self.show_warning_dialog("Tip", "图片加载失败，请检查路径是否正确")
            return
        # 2. 缩放图片至标签大小（保持比例）
        scaled_pixmap = pixmap.scaled(
            self.face_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation  # 平滑缩放，画质更好
        )
        # 3. 创建圆形掩码（与图片尺寸一致）
        # 取图片宽高中的最小值作为圆的直径（确保圆形）
        diameter = min(scaled_pixmap.width(), scaled_pixmap.height())
        # 创建圆形区域（x, y, 宽, 高, 椭圆）
        circle_region = QRegion(
            (scaled_pixmap.width() - diameter) // 2,  # 圆心x坐标
            (scaled_pixmap.height() - diameter) // 2,  # 圆心y坐标
            diameter, diameter,  # 宽和高（相等则为圆）
            QRegion.Ellipse  # 形状为椭圆（宽高相等时为圆）
        )

        # 4. 设置掩码和图片
        self.face_label.setPixmap(scaled_pixmap)
        self.face_label.setMask(circle_region)  # 应用圆形掩码

    def update_info2ui(self, emp_id):
        # 当前界面右侧信息和当前识别人脸信息一致时, 不更新右侧信息界面
        if self.line_edits["lineEdit"].text() == "":
            pass
        elif str(emp_id) == self.line_edits["lineEdit"].text():
            return

        self.line_edits["lineEdit"].setText(rf'{emp_id}')
        conn = self.connection.get_conn()
        try:
            with conn.cursor as cur:
                sql = """
                                        SELECT
                                            ename, dname
                                        FROM employeeinfo where id = %s
                                        """
                cur.execute(sql, emp_id)
                res = cur.fetchone()
                print(res)
                self.line_edits["lineEdit_2"].setText(res["ename"])  # 姓名
                self.line_edits["lineEdit_3"].setText(res["dname"])  # 部门
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.line_edits["lineEdit_4"].setText(str(current_time))  # 时间
                jpgs = os.listdir(self.jpgs_save_path)
                for jpg in jpgs:
                    if str(emp_id) in jpg:
                        jpg_path = os.path.join(self.jpgs_save_path, jpg)
                        self.load_jpg(jpg_path)
                        break
                sql = "insert into attendancerecord(employeeid, checkintime) values(%s,%s)"
                cur.execute(sql, [emp_id, current_time])
                conn.conn.commit()  # 提交事务
        except Exception as e:
            print(f"[ERROR] {e}")
            self.show_warning_dialog("Error", f"数据库操作失败,请联系数据库管理员!")
            conn.rollback()  # 回滚事务
        finally:
            conn.close()  # 放回连接池

    # 识别的图片
    def face_detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度
        # 提高识别率
        # gray = cv2.equalizeHist(gray)  # 直方图均衡化，增强对比度
        scale = 1.5  # 放大1.5倍
        cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # 加载分类器
        face_detector = cv2.CascadeClassifier(self.classifier_path)
        face = face_detector.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (50, 50), (220, 220))

        # 画面中无人脸,清除右侧信息栏
        if len(face) == 0:
            self.tip_label.hide()
            self.tip_label1.hide()
            self.lineEdit_clear()
            return
        for x, y, w, h in face:
            # cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
            cv2.circle(img, center=(x + w // 2, y + h // 2), radius=w // 2+2, color=(171, 224, 183), thickness=2)
            cv2.circle(img, center=(x + w // 2, y + h // 2), radius=w // 2 + 8, color=(119, 192, 139), thickness=2)
            # face_recognition
            ids, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
            print('人脸坐标', x, y, w, h)
            print('标签id:', ids, '置信评分：', confidence)
            if confidence > 55:
                self.tip_label.hide()  # 识别成功 label
                self.tip_label1.show()  # 识别失败 label
                self.lineEdit_clear()
                self.warningtime += 1
                # 可疑度叠加,叠加到100,说明在本视频监控下多次出现同一个陌生人,此时可以采取发消息给用户或者发邮件给用户,暂不操作
                if self.warningtime > 100:
                    # code
                    self.warningtime = 0
            else:
                self.tip_label1.hide()
                self.tip_label.show()
                # 更新信息到右侧
                self.update_info2ui(ids)

    def update_frame(self):
        if self.camera.is_opened():
            flag, image = self.camera.read_frame()
            self.face_detect(image)
            if image.dtype == np.uint16:
                image = (image / 256).astype(np.uint8)  # 16位转8位
            # 将 BGR 格式转为 RGB 格式(OpenCV默认BGR，PyQt需要RGB)
            cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 转为QImage（参数：数据、宽度、高度、每行字节数、格式）
            # RGB888表示3通道，每个通道8位
            q_img = QImage(cv_image.data,
                          cv_image.shape[1],
                          cv_image.shape[0],
                          cv_image.strides[0],
                          QImage.Format_RGB888)
            # QImage 转 label支持的QPixmap格式
            pixmap = QPixmap(q_img)
            # 加载 pixmap
            self.video_label.setPixmap(pixmap)
        else:
            self.show_warning_dialog("Error", "摄像头未打开")

    def closeEvent(self, event):
        """关闭主窗口时释放摄像头"""
        self.timer.stop()
        self.camera.release()
        self.connection.close()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    font = QtGui.QFont("Microsoft YaHei", 10)
    app.setFont(font)
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    window = FaceRecognitionUI()
    window.show()
    sys.exit(app.exec_())