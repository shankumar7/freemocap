import logging
import platform
import time

import cv2
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp  # type: ignore

    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False


class LiveCaptureThread(QThread):
    frame_ready = Signal(QImage)
    status_message = Signal(str)

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720, parent=None):
        super().__init__(parent=parent)
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._running = False
        self._pose_worker = None

    def run(self):
        self._running = True
        backend = cv2.CAP_DSHOW if platform.system().lower().startswith("win") else 0
        cap = cv2.VideoCapture(self.camera_index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.status_message.emit(f"ERROR: Cannot open camera {self.camera_index}")
            self._running = False
            return

        if HAS_MEDIAPIPE:
            self.status_message.emit("MediaPipe enabled: live motion capture running")
            mp_holistic = mp.solutions.holistic
            mp_drawing = mp.solutions.drawing_utils
            with mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                enable_segmentation=False,
                refine_face_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as pose_worker:
                self._pose_worker = pose_worker
                self._camera_loop(cap, mp_drawing)
        else:
            self.status_message.emit("MediaPipe not installed: showing raw camera preview")
            self._camera_loop(cap, None)

        cap.release()
        self._pose_worker = None

    def _camera_loop(self, cap, mp_drawing):
        previous_frame_time = time.time()
        while self._running:
            frame_start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                self.status_message.emit("Failed to read frame from camera")
                break

            frame = cv2.flip(frame, 1)

            if self._pose_worker is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inference_frame = cv2.resize(rgb_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
                results = self._pose_worker.process(inference_frame)

                if results.pose_landmarks and mp_drawing is not None:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

                if results.left_hand_landmarks and mp_drawing is not None:
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

                if results.right_hand_landmarks and mp_drawing is not None:
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

                if results.face_landmarks and mp_drawing is not None:
                    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION)

            current_frame_time = time.time()
            fps = 1.0 / max(current_frame_time - previous_frame_time, 1e-6)
            previous_frame_time = current_frame_time

            latency_ms = (time.time() - frame_start_time) * 1000.0
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}  Latency: {latency_ms:.1f} ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_frame.shape
            bytes_per_line = channel * width
            q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            self.frame_ready.emit(q_image.copy())

    def stop(self):
        self._running = False
        if self.isRunning():
            self.wait(1000)


class LiveCaptureWidget(QWidget):
    def __init__(self, camera_index: int = 0, parent=None):
        super().__init__(parent=parent)
        self._camera_index = camera_index
        self._thread = None

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self._title_label = QLabel("Live Motion Capture")
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet("font-size: 28px; font-weight: bold;")
        self._layout.addWidget(self._title_label)

        self._status_label = QLabel("Ready")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self._status_label)

        self._image_label = QLabel("Camera preview will appear here")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("border: 1px solid #bbb; background: #111; color: white;")
        self._image_label.setMinimumSize(960, 540)
        self._layout.addWidget(self._image_label)

        self._stop_button = QPushButton("Stop Live Capture")
        self._stop_button.clicked.connect(self.stop_capture)
        self._layout.addWidget(self._stop_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self._layout.addStretch(1)

    def start_capture(self):
        if self._thread is not None and self._thread.isRunning():
            return

        self._thread = LiveCaptureThread(camera_index=self._camera_index, parent=self)
        self._thread.frame_ready.connect(self._handle_frame)
        self._thread.status_message.connect(self._status_label.setText)
        self._thread.start()

    def stop_capture(self):
        if self._thread is not None:
            self._thread.stop()
            self._thread = None
        self._status_label.setText("Stopped")

    def _handle_frame(self, q_image: QImage):
        pixmap = QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(
            self._image_label.width(),
            self._image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(pixmap)

    def showEvent(self, event):
        super().showEvent(event)
        self.start_capture()

    def hideEvent(self, event):
        self.stop_capture()
        super().hideEvent(event)

    def closeEvent(self, event):
        self.stop_capture()
        super().closeEvent(event)