import cv2
import numpy as np
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QGroupBox
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap

from military_drill_ai.gui.control_panel import ControlPanel
from freemocap.gui.qt.widgets.live_3d_viewer import Live3DViewer
from military_drill_ai.gui.video_thread import VideoThread

class MilitaryDrillAiWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(30, 30, 30, 30)
        self.layout.setSpacing(30)
        self.setLayout(self.layout)
        
        # --- Left Side: Viewports (2D + 3D) ---
        self.viewports_container = QWidget()
        self.viewports_layout = QHBoxLayout(self.viewports_container)
        self.viewports_layout.setContentsMargins(0, 0, 0, 0)
        self.viewports_layout.setSpacing(20)
        
        # 2D View
        self.video_label = QLabel("Click 'Start Camera' to begin pipeline")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; border: 2px solid #3D5A80; border-radius: 15px;")
        self.video_label.setMinimumSize(480, 360)
        self.video_label.mousePressEvent = self.on_viewport_click
        self.viewports_layout.addWidget(self.video_label, stretch=1)
        
        # 3D View
        self.live_3d_viewer = Live3DViewer()
        self.live_3d_viewer.setMinimumSize(480, 360)
        self.live_3d_viewer.setStyleSheet("border: 2px solid #3D5A80; border-radius: 15px; background-color: #FFFFFF;")
        self.viewports_layout.addWidget(self.live_3d_viewer, stretch=1)
        
        self.layout.addWidget(self.viewports_container, stretch=4)
        
        # --- Right Side: Analytics & Control ---
        self.right_panel = QVBoxLayout()
        
        # Metrics Display
        self.metrics_group = QGroupBox("Live Cadet Analytics")
        self.metrics_layout = QVBoxLayout()
        self.metrics_group.setLayout(self.metrics_layout)
        self.metrics_display = QLabel("No cadets detected.")
        self.metrics_display.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.metrics_display.setStyleSheet("font-family: 'Consolas'; font-size: 14px; color: #3D5A80;")
        self.metrics_layout.addWidget(self.metrics_display)
        
        self.right_panel.addWidget(self.metrics_group, stretch=2)
        
        # Control Panel
        self.control_panel = ControlPanel()
        self.right_panel.addWidget(self.control_panel, stretch=1)
        
        self.layout.addLayout(self.right_panel, stretch=1)
        
        # Modules
        self.video_thread = None
        
        # Connections
        self.control_panel.btn_start.clicked.connect(self.start_camera)
        self.control_panel.btn_stop.clicked.connect(self.stop_camera)
        
        # Default state
        self.control_panel.btn_toggle_3d.setChecked(True)

    @Slot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image label with a new OpenCV image."""
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)
        
    @Slot(str)
    def update_log(self, text):
        self.control_panel.log_message(text)

    @Slot(dict)
    def update_metrics(self, metrics_data):
        """Updates the metrics panel with real-time cadet data."""
        if not metrics_data:
            self.metrics_display.setText("No cadets detected.")
            return
            
        text = ""
        for c_id, data in metrics_data.items():
            text += f"CADET ID: {c_id}\n"
            text += f"  - Height: {data.get('height', 'N/A')}\n"
            text += f"  - Angle:  {data.get('angle', 'N/A')}\n"
            text += f"  - Leg Dist: {data.get('leg_distance', 'N/A')}\n"
            text += "-" * 20 + "\n"
        self.metrics_display.setText(text)

    @Slot(dict)
    def update_3d_view(self, pose_3d_data):
        """Updates the 3D viewer if the toggle is enabled."""
        if self.control_panel.btn_toggle_3d.isChecked():
            self.live_3d_viewer.update_landmarks(pose_3d_data)
        
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
        self.video_thread.metrics_signal.connect(self.update_metrics)
        self.video_thread.pose_3d_signal.connect(self.update_3d_view)
        
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
            
    @property
    def blender_button(self):
        return self.control_panel.btn_blender

    def close(self):
        self.stop_camera()
        super().close()
