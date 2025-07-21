import sys, os
from ui_resource.FaceAtten import FaceRecognitionUI
from PyQt5.QtWidgets import QApplication

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)  # 切换工作目录
    print(f"工作目录已切换为：{os.getcwd()}")
    app = QApplication(sys.argv)
    ui = FaceRecognitionUI()
    ui.show()
    sys.exit(app.exec_())


