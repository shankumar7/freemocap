import cv2
import numpy as np
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap

from military_drill_ai.gui.control_panel import ControlPanel
from military_drill_ai.gui.video_thread import VideoThread

class MilitaryDrillAiWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        
        # --- Left Side: Video Viewport ---
        self.viewport_layout = QVBoxLayout()
        self.video_label = QLabel("Click 'Start Camera' to begin pipeline")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; border: 1px solid #3d3d3d;")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.mousePressEvent = self.on_viewport_click
        self.viewport_layout.addWidget(self.video_label)
        self.layout.addLayout(self.viewport_layout, stretch=3)
        
        # --- Right Side: Control Panel ---
        self.control_panel = ControlPanel()
        self.layout.addWidget(self.control_panel, stretch=1)
        
        # Modules
        self.video_thread = None
        
        # Connections
        self.control_panel.btn_start.clicked.connect(self.start_camera)
        self.control_panel.btn_stop.clicked.connect(self.stop_camera)

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
            pixmap = self.video_label.pixmap()
            if not pixmap: return
                
            pm_width = pixmap.width()
            pm_height = pixmap.height()
            lbl_width = self.video_label.width()
            lbl_height = self.video_label.height()
            
            x_offset = (lbl_width - pm_width) // 2
            y_offset = (lbl_height - pm_height) // 2
            
            click_x = event.position().x()
            click_y = event.position().y()
            
            if x_offset <= click_x <= x_offset + pm_width and y_offset <= click_y <= y_offset + pm_height:
                rel_x = (click_x - x_offset) / pm_width
                rel_y = (click_y - y_offset) / pm_height
                self.video_thread.add_click_relative(rel_x, rel_y)

    def start_camera(self):
        self.control_panel.btn_start.setEnabled(False)
        self.control_panel.btn_stop.setEnabled(True)
        self.control_panel.source_input.setEnabled(False)
        
        self.video_thread = VideoThread()
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
            
    def close(self):
        self.stop_camera()
        super().close()
