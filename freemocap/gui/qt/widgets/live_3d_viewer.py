import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Slot
import numpy as np

class Live3DViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.fig.patch.set_facecolor('#F0F2F5')
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)
        
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#F0F2F5')
        
        # Connections for MediaPipe Pose
        self.connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Upper body
            (11, 23), (12, 24), (23, 24), # Torso
            (23, 25), (25, 27), (27, 29), (29, 31), # Left leg
            (24, 26), (26, 28), (28, 30), (30, 32), # Right leg
            (15, 17), (17, 19), (19, 15), (15, 21), # Left hand
            (16, 18), (18, 20), (20, 16), (16, 22)  # Right hand
        ]
        
        self.setup_plot()
        
    def setup_plot(self):
        self.ax.clear()
        self.ax.set_xlim3d([-1, 1])
        self.ax.set_ylim3d([-1, 1])
        self.ax.set_zlim3d([-1, 1])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Z')
        self.ax.set_zlabel('Y')
        self.ax.view_init(elev=20, azim=45)
        self.canvas.draw()

    @Slot(dict)
    def update_landmarks(self, pose_3d_data):
        """Updates the 3D plot with new landmarks."""
        if not pose_3d_data:
            return
            
        self.ax.clear()
        # Set stable limits
        self.ax.set_xlim3d([-0.5, 0.5])
        self.ax.set_ylim3d([-0.5, 0.5])
        self.ax.set_zlim3d([-1, 0.5]) # MediaPipe Y is down
        
        for c_id, landmarks in pose_3d_data.items():
            if not landmarks: continue
            
            # Convert to numpy array for easier indexing
            pts = np.array([(lm[0], lm[2], -lm[1]) if lm else (None, None, None) for lm in landmarks])
            
            # Draw points
            valid_pts = pts[~np.isnan(pts.astype(float)).any(axis=1)]
            if len(valid_pts) > 0:
                self.ax.scatter(valid_pts[:, 0], valid_pts[:, 1], valid_pts[:, 2], c='#3D5A80', s=10)
            
            # Draw connections
            for connection in self.connections:
                p1 = pts[connection[0]]
                p2 = pts[connection[1]]
                if None not in p1 and None not in p2:
                    self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='#EE6C4D', linewidth=2)
                    
        self.ax.set_title("Live 3D Skeleton")
        self.canvas.draw()
