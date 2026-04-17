import sys
import cv2
import numpy as np
import os
from pathlib import Path
from collections import deque
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                              QFileDialog, QVBoxLayout, QMessageBox)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer
import insightface


class AgeGenderDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Age and Gender Detection")
        self.setGeometry(100, 100, 800, 600)

        self.message_label = QLabel("For better accuracy, please upload an image instead of using the webcam.", self)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label.setWordWrap(True)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(400)
        self.image_label.setStyleSheet("background-color: #1a1a1a;")

        self.upload_btn = QPushButton("Upload Image", self)
        self.upload_btn.clicked.connect(self.upload_image)

        self.webcam_btn = QPushButton("Start Webcam", self)
        self.webcam_btn.clicked.connect(self.toggle_webcam)

        self.result_label = QLabel("Age and Gender: ", self)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.message_label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.upload_btn)
        layout.addWidget(self.webcam_btn)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_frame)

        self.gender_buffer = deque(maxlen=30)  # Increased for better stability
        self.age_buffer = deque(maxlen=30)

        self.load_models()

    def load_models(self):
        # Use InsightFace for better face detection
        self.face_analyzer = insightface.app.FaceAnalysis(name='buffalo_s')  # Smaller, faster model
        self.face_analyzer.prepare(ctx_id=0, det_size=(320, 320))  # Smaller size for speed
        
        # Keep original models for age/gender prediction (fallback)
        self.ageNet = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
        self.genderNet = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")

        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.age_midpoints = [1, 5, 10, 17, 25, 40, 50, 80]
        self.genderList = ['Male', 'Female']

        self.custom_model = None
        self.custom_age_buckets = ['<=15', '16-17', '18-19', '20-22', '23-25', '26-29', '30+']
        self.custom_age_midpoints = [13.0, 16.5, 18.5, 21.0, 24.0, 27.5, 35.0]
        self.age_buckets = self.custom_age_buckets
        custom_model_path = Path("models/best_age_gender_model.onnx")
        if custom_model_path.exists():
            try:
                self.custom_model = cv2.dnn.readNetFromONNX(str(custom_model_path))
                print(f"Loaded custom ONNX model: {custom_model_path}")
            except Exception as e:
                print(f"Warning: failed to load custom ONNX model: {e}")

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def map_age_to_bucket(self, age: float) -> int:
        if age <= 15:
            return 0
        if age <= 17:
            return 1
        if age <= 19:
            return 2
        if age <= 22:
            return 3
        if age <= 25:
            return 4
        if age <= 29:
            return 5
        return 6

    def predict_custom_age_gender(self, face_crop):
        blob = cv2.dnn.blobFromImage(face_crop, 1.0 / 255.0, (224, 224), (0, 0, 0), swapRB=True, crop=False)
        self.custom_model.setInput(blob)
        out_names = self.custom_model.getUnconnectedOutLayersNames()
        outputs = self.custom_model.forward(out_names)
        if len(outputs) != 2:
            raise ValueError("Unexpected custom ONNX model outputs")

        age_logits = outputs[0]
        gender_logits = outputs[1]
        age_probs = self.softmax(age_logits[0])
        age_index = int(np.argmax(age_probs))
        gender_index = int(np.argmax(gender_logits[0]))
        return age_index, age_probs, gender_index

    def preprocess_face(self, face):
        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        return enhanced

    def estimate_image_quality(self, face_crop):
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast_value = float(np.std(gray))
        brightness_value = float(np.mean(gray))

        blur_score = np.clip(blur_value / 100.0, 0.0, 1.0)
        contrast_score = np.clip(contrast_value / 64.0, 0.0, 1.0)
        brightness_score = 1.0 - abs(brightness_value - 128.0) / 128.0
        brightness_score = np.clip(brightness_score, 0.0, 1.0)

    def adjust_for_camera_quality(self, face_crop, quality):
        if quality < 0.45:
            hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.equalizeHist(v)
            improved = cv2.merge((h, s, v))
            face_crop = cv2.cvtColor(improved, cv2.COLOR_HSV2BGR)

            # apply a mild sharpening if blur is the issue
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            face_crop = cv2.filter2D(face_crop, -1, kernel)
        return face_crop

    def highlightFace(self, frame, conf_threshold=0.75):
        frameOpencvDnn = frame.copy()
        
        # Use InsightFace for face detection
        faces = self.face_analyzer.get(frame)
        
        for face in faces:
            # InsightFace bbox is [x1, y1, x2, y2]
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return frameOpencvDnn, faces

    def upload_image(self):
        if self.timer.isActive():
            self.stop_webcam()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            frame = cv2.imread(file_name)
            if frame is not None:
                h, w = frame.shape[:2]
                if h > 1000 or w > 1000:
                    scale = min(1000 / h, 1000 / w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                self.predict_age_gender(frame)
            else:
                self.result_label.setText("Error: Unable to read the image.")

    def toggle_webcam(self):
        if self.timer.isActive():
            self.stop_webcam()
        else:
            self.confirm_webcam_usage()

    def confirm_webcam_usage(self):
        reply = QMessageBox.question(
            self, "Webcam Usage",
            "For better accuracy, we recommend uploading an image.\n"
            "Do you still want to continue with the webcam?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.start_webcam()

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.result_label.setText("Error: Could not open webcam.")
            return
        self.webcam_btn.setText("Stop Webcam")
        self.timer.start(30)

    def stop_webcam(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.webcam_btn.setText("Start Webcam")
        self.image_label.clear()
        self.result_label.setText("Webcam stopped.")
        self.gender_buffer.clear()
        self.age_buffer.clear()

    def capture_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self.predict_age_gender(frame, live=True)
        else:
            self.result_label.setText("Error: Failed to read from webcam.")
            self.stop_webcam()

    def predict_age_gender(self, frame, live=False):
        resultImg, faces = self.highlightFace(frame)

        if not faces:
            if live:
                self.gender_buffer.clear()
                self.age_buffer.clear()
            self.result_label.setText("No face detected.")
            self.display_image(resultImg)
            return

        results = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            fw = x2 - x1
            fh = y2 - y1
            
            # Skip small faces for better accuracy
            if fw < 80 or fh < 80:
                continue
                
            # Crop a more stable square around the detected face
            size = int(max(fw, fh) * 1.25)
            cx = x1 + fw // 2
            cy = y1 + fh // 2
            x1s = max(0, cx - size // 2)
            y1s = max(0, cy - size // 2)
            x2s = min(frame.shape[1], x1s + size)
            y2s = min(frame.shape[0], y1s + size)
            x1s = max(0, x2s - size)
            y1s = max(0, y2s - size)
            face_crop = frame[y1s:y2s, x1s:x2s]
            if face_crop.size == 0:
                continue

            face_crop = self.preprocess_face(face_crop)

            try:
                use_custom = self.custom_model is not None
                if use_custom:
                    age_index, age_probs, gender_index = self.predict_custom_age_gender(face_crop)
                    age_range = self.custom_age_buckets[age_index]
                    age = int(np.dot(age_probs, self.custom_age_midpoints))
                    gender = self.genderList[gender_index]
                    bucket_labels = self.custom_age_buckets
                else:
                    insight_age = getattr(face, 'age', None)
                    insight_sex = getattr(face, 'sex', None)

                    if insight_age is not None:
                        age_index = self.map_age_to_bucket(insight_age)
                        age_range = self.age_buckets[age_index]
                        age = int(insight_age)
                        gender = 'Male' if insight_sex == 'M' else 'Female' if insight_sex == 'F' else None
                        bucket_labels = self.age_buckets
                    else:
                        # Use the loaded Caffe age and gender nets
                        blob = cv2.dnn.blobFromImage(
                            face_crop, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)

                        self.ageNet.setInput(blob)
                        age_preds = self.ageNet.forward()
                        age_scores = age_preds[0].astype(np.float32)
                        exp_scores = np.exp(age_scores - np.max(age_scores))
                        age_probs = exp_scores / exp_scores.sum()
                        age_index = int(age_probs.argmax())
                        age_range = self.ageList[age_index]
                        age = int(np.dot(age_probs, self.age_midpoints))
                        bucket_labels = self.ageList

                        self.genderNet.setInput(blob)
                        gender_preds = self.genderNet.forward()
                        gender = self.genderList[int(gender_preds[0].argmax())]
                        if insight_sex is not None:
                            gender = 'Male' if insight_sex == 'M' else 'Female'

                if live:
                    self.gender_buffer.append(gender)
                    self.age_buffer.append(age_index)
                    # For webcam, always use stable value from buffer
                    if self.gender_buffer:
                        gender = max(set(self.gender_buffer), key=self.gender_buffer.count)
                    if self.age_buffer:
                        stable_age_index = max(set(self.age_buffer), key=self.age_buffer.count)
                        stable_age_range = bucket_labels[stable_age_index]
                        if use_custom:
                            age = int(self.custom_age_midpoints[stable_age_index])
                        elif bucket_labels is self.age_buckets:
                            age = int(age)
                        else:
                            stable_low, stable_high = [int(x) for x in stable_age_range.strip('()').split('-')]
                            age = int((stable_low + stable_high) / 2)
                display_age = max(0, age - 7)
                text = f"{gender}, Age: approx {display_age}"

                results.append(text)
                cv2.putText(resultImg, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            except Exception as e:
                print(f"Age/gender net analysis failed: {e}")
                continue

        self.result_label.setText(" | ".join(results) if results else "No face detected.")
        self.display_image(resultImg)

    def display_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        qt_image = QImage(image_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(
            pixmap.scaled(self.image_label.size(),
                          Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
        )

    def closeEvent(self, event):
        if self.timer.isActive():
            self.stop_webcam()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AgeGenderDetectionApp()
    window.show()
    sys.exit(app.exec())
