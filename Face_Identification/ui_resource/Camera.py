import cv2


class CameraManager:
    """摄像头管理单例类，确保全局只有一个摄像头实例"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.cap = None  # 摄像头实例
        return cls._instance

    def is_opened(self):
        """判断摄像头是否已打开"""
        return self.cap is not None and self.cap.isOpened()

    def open(self, camera_index=0):
        """打开摄像头（默认使用内置摄像头，index=0）"""
        if self.is_opened():
            self.release()  # 已打开则先释放
        self.cap = cv2.VideoCapture(camera_index)
        return self.is_opened()

    def read_frame(self):
        """读取一帧图像（返回BGR格式）"""
        if self.is_opened():
            ret, frame = self.cap.read()
            return ret, frame
        return False, None

    def release(self):
        """释放摄像头资源"""
        if self.is_opened():
            self.cap.release()
            self.cap = None