import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QGraphicsView, QGraphicsScene
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
from keras.models import load_model
from PIL import Image, ImageOps

def find_available_cameras():
    index = 0
    available_cameras = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            cap.release()
            break
        else:
            available_cameras.append(index)
        cap.release()
        index += 1
    return available_cameras

class WebcamApp(QMainWindow):
    def __init__(self, camera_index=0):
        super().__init__()
        loadUi('D:/AI_Model/TM002/capture.ui', self)  # 파일 경로 수정 필요
        self.model = load_model("./Model/keras_Model.h5", compile=False)
        self.class_names = open("./Model/labels.txt", "r").readlines()
        self.pushButton.clicked.connect(self.capture_and_predict)
        self.camera_index = camera_index
        self.init_webcam()

    def init_webcam(self):
        self.capture = cv2.VideoCapture(self.camera_index)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888).rgbSwapped()
            self.graphicsView.setScene(QGraphicsScene(self))
            self.graphicsView.scene().addPixmap(QPixmap.fromImage(image))

    def capture_and_predict(self):
        ret, frame = self.capture.read()
        if ret:
            # Convert captured image to PIL Image, resize and preprocess
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
            image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # Prediction
            prediction = self.model.predict(data)
            index = np.argmax(prediction)
            class_name = self.class_names[index].strip()
            confidence_score = prediction[0][index]

            # Update QLabel with the prediction
            self.label.setText(f"Class: {class_name}\nConfidence: {confidence_score:.2f}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    cameras = find_available_cameras()
    if cameras:
        print("Available cameras:", cameras)
        selected_camera_index = int(input("Enter the camera index to use: "))
        window = WebcamApp(selected_camera_index)
        window.show()
    else:
        print("No cameras found. Please connect a camera and try again.")
    sys.exit(app.exec_())