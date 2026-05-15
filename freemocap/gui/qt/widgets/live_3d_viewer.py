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
            (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), # Face
            (11, 12), (11, 23), (12, 24), (23, 24), # Torso
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), # Left arm
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), # Right arm
            (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), # Left leg
            (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)  # Right leg
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
        
        # Set stable limits AFTER clearing
        self.ax.set_xlim3d([-1, 1])
        self.ax.set_ylim3d([-1, 1])
        self.ax.set_zlim3d([-1.5, 0.5]) # MediaPipe Y is down, origin at hips
        
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Z (m)')
        self.ax.set_zlabel('Y (m)')
        self.ax.set_title("Live 3D Skeleton")
        
        has_drawn = False
        for c_id, landmarks in pose_3d_data.items():
            if not landmarks: continue
            
            # Convert to numpy array for easier indexing
            # MediaPipe: X (left/right), Y (up/down), Z (forward/back)
            # Matplotlib: X (left/right), Y (forward/back), Z (up/down)
            # Map MP(X, Y, Z) -> MPL(X, Z, -Y)
            pts = []
            for lm in landmarks:
                if lm is not None:
                    pts.append([lm[0], lm[2], -lm[1]])
                else:
                    pts.append([np.nan, np.nan, np.nan])
            pts = np.array(pts)
            
            # Draw points
            valid_mask = ~np.isnan(pts[:, 0])
            if np.any(valid_mask):
                valid_pts = pts[valid_mask]
                self.ax.scatter(valid_pts[:, 0], valid_pts[:, 1], valid_pts[:, 2], c='#3D5A80', s=20)
                has_drawn = True
            
            # Draw connections
            for connection in self.connections:
                idx1, idx2 = connection
                if idx1 < len(pts) and idx2 < len(pts):
                    p1 = pts[idx1]
                    p2 = pts[idx2]
                    if not np.isnan(p1[0]) and not np.isnan(p2[0]):
                        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='#EE6C4D', linewidth=2)
        
        if has_drawn:
            self.canvas.draw_idle()
