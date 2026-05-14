import cv2
from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap
import numpy as np

from military_drill_ai.gui.control_panel import ControlPanel
from military_drill_ai.gui.video_thread import VideoThread
from military_drill_ai.tracking.vive_tracker import ViveTrackerClient

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Military Drill AI - Tracker Edition")
        self.setGeometry(100, 100, 1200, 800)
        
        # Apply dark theme similar to FreeMoCap
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #ffffff; }
            QWidget { background-color: #1e1e1e; color: #ffffff; font-family: 'Inter', sans-serif; }
            QGroupBox { border: 1px solid #3d3d3d; border-radius: 5px; margin-top: 10px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QPushButton { background-color: #0e639c; color: white; border: none; padding: 8px; border-radius: 4px; }
            QPushButton:hover { background-color: #1177bb; }
            QPushButton:disabled { background-color: #3d3d3d; color: #888888; }
            QLineEdit { background-color: #2d2d2d; border: 1px solid #3d3d3d; padding: 4px; color: white; }
        """)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        
        # --- Left Side: Video Viewport ---
        self.viewport_layout = QVBoxLayout()
        self.video_label = QLabel("Click 'Start Camera' to begin pipeline")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; border: 1px solid #3d3d3d;")
        self.video_label.setMinimumSize(800, 600)
        self.video_label.mousePressEvent = self.on_viewport_click
        self.viewport_layout.addWidget(self.video_label)
        self.main_layout.addLayout(self.viewport_layout, stretch=3)
        
        # --- Right Side: Control Panel ---
        self.control_panel = ControlPanel()
        self.main_layout.addWidget(self.control_panel, stretch=1)
        
        # Modules
        self.vive_client = ViveTrackerClient()
        self.video_thread = None
        
        # Connections
        self.control_panel.btn_start.clicked.connect(self.start_camera)
        self.control_panel.btn_stop.clicked.connect(self.stop_camera)
        self.control_panel.btn_connect_vr.clicked.connect(self.toggle_vr)

    @Slot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image label with a new OpenCV image."""
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)
        
    @Slot(str)
    def update_log(self, text):
        self.control_panel.log_message(text)
        
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # Scale keeping aspect ratio
        p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def on_viewport_click(self, event):
        """Handle clicks on the video label for calibration."""
        if self.control_panel.btn_calibrate.isChecked() and self.video_thread is not None:
            # We need to map the click position on the QLabel back to the original image coordinates
            # Since we scale the image using KeepAspectRatio, this requires some math.
            
            # Get actual displayed pixmap size
            pixmap = self.video_label.pixmap()
            if not pixmap:
                return
                
            pm_width = pixmap.width()
            pm_height = pixmap.height()
            
            # Get label size
            lbl_width = self.video_label.width()
            lbl_height = self.video_label.height()
            
            # Calculate offsets
            x_offset = (lbl_width - pm_width) // 2
            y_offset = (lbl_height - pm_height) // 2
            
            # Click pos
            click_x = event.position().x()
            click_y = event.position().y()
            
            # Check if click is inside the actual image
            if x_offset <= click_x <= x_offset + pm_width and y_offset <= click_y <= y_offset + pm_height:
                # Map to original image coordinates (assuming we don't scale up past original, 
                # but if we do, we need the original image shape... For now we just pass relative)
                
                # To be precise, we pass the pixel coordinate mapped to the original frame
                # Let's just pass the coordinate mapped to the pixmap size. The VideoThread 
                # receives the resized image or original? It receives original and we resized it.
                # So we must map back to the original frame size.
                
                # Let's assume the VideoThread is sending 640x480 or 1280x720. 
                # The thread has access to the frame size. 
                # To simplify, we just send proportional coordinates.
                rel_x = (click_x - x_offset) / pm_width
                rel_y = (click_y - y_offset) / pm_height
                
                # We need a small hack: VideoThread needs absolute. 
                # Since we don't store the exact frame size here easily, we can add a method.
                self.video_thread.add_click_relative(rel_x, rel_y)

    def toggle_vr(self):
        if self.vive_client.is_initialized:
            self.vive_client.shutdown()
            self.control_panel.set_vr_status(False)
            self.update_log("SteamVR disconnected.")
        else:
            self.update_log("Attempting to connect to SteamVR...")
            success = self.vive_client.initialize()
            self.control_panel.set_vr_status(success)
            if success:
                self.update_log("SteamVR connected successfully. Trackers will be polled during camera feed.")
            else:
                self.update_log("Failed to connect to SteamVR. Make sure it is running.")

    def start_camera(self):
        self.control_panel.btn_start.setEnabled(False)
        self.control_panel.btn_stop.setEnabled(True)
        self.control_panel.source_input.setEnabled(False)
        
        self.video_thread = VideoThread(vive_client=self.vive_client)
        self.video_thread.video_source = self.control_panel.source_input.text()
        
        # Setup signals
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.status_signal.connect(self.update_log)
        
        self.video_thread.start()

    def stop_camera(self):
        self.control_panel.btn_start.setEnabled(True)
        self.control_panel.btn_stop.setEnabled(False)
        self.control_panel.source_input.setEnabled(True)
        
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
            self.video_label.clear()
            self.video_label.setText("Camera Stopped")
            
    def closeEvent(self, event):
        self.stop_camera()
        if self.vive_client.is_initialized:
            self.vive_client.shutdown()
        event.accept()
