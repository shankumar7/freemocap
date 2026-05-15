from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QGroupBox, QLineEdit, QHBoxLayout
from PySide6.QtCore import Qt

class ControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        # Camera Settings
        self.camera_group = QGroupBox("Camera Controls")
        self.camera_layout = QVBoxLayout()
        self.camera_layout.setSpacing(10)
        self.camera_group.setLayout(self.camera_layout)
        
        source_layout = QHBoxLayout()
        source_label = QLabel("Source:")
        self.source_input = QLineEdit("0")
        source_layout.addWidget(source_label)
        source_layout.addWidget(self.source_input)
        
        self.btn_start = QPushButton("Start Camera")
        self.btn_stop = QPushButton("Stop Camera")
        self.btn_stop.setEnabled(False)
        
        self.camera_layout.addLayout(source_layout)
        self.camera_layout.addWidget(self.btn_start)
        self.camera_layout.addWidget(self.btn_stop)
        
        # Tracking & Calibration Settings
        self.tracking_group = QGroupBox("Military Drill AI")
        self.tracking_layout = QVBoxLayout()
        self.tracking_layout.setSpacing(10)
        self.tracking_group.setLayout(self.tracking_layout)
        
        self.btn_calibrate = QPushButton("Enable 15cm Scale Calibration")
        self.btn_calibrate.setCheckable(True)
        
        self.tracking_layout.addWidget(self.btn_calibrate)
        
        # Status Log
        self.log_group = QGroupBox("System Log")
        self.log_layout = QVBoxLayout()
        self.log_group.setLayout(self.log_layout)
        self.status_log = QLabel("Ready.")
        self.status_log.setWordWrap(True)
        self.status_log.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.log_layout.addWidget(self.status_log)
        
        # Add to main layout
        self.layout.addWidget(self.camera_group)
        self.layout.addWidget(self.tracking_group)
        self.layout.addWidget(self.log_group)
        self.layout.addStretch()

    def log_message(self, msg):
        self.status_log.setText(msg)

