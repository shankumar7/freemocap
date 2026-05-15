import cv2
import time
import math
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from military_drill_ai.utils.config import config
from military_drill_ai.detection.yolo_detector import YOLODetector
from military_drill_ai.tracking.byte_tracker import Tracker
from military_drill_ai.pose_estimation.mediapipe_estimator import MediaPipeEstimator
from military_drill_ai.analytics.posture_analyzer import PostureAnalyzer

import threading

class FrameGrabber:
    """Runs a background thread to continually grab the latest frame from the camera,
    preventing buffer buildup and eliminating latency."""
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        
        if self.cap.isOpened():
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()
        
    def update(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
            # Tiny sleep to not hog CPU but keep buffer empty
            time.sleep(0.005)
            
    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None
            
    def isOpened(self):
        return self.cap.isOpened()
        
    def release(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()

class VideoThread(QThread):
    # Signals to communicate with the main GUI thread
    change_pixmap_signal = Signal(np.ndarray)
    metrics_signal = Signal(dict)
    pose_3d_signal = Signal(dict)
    status_signal = Signal(str)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.video_source = "0"
        self.show_skeleton = True
        self.calibration_clicks = []
        
        # We will initialize models when the thread starts
        self.detector = None
        self.tracker = None
        self.pose_estimator = None
        self.analyzer = None

    def run(self):
        self.status_signal.emit("Initializing AI Models...")
        self.detector = YOLODetector(model_path=config.YOLO_MODEL_PATH, 
                                     conf=config.DETECTION_CONFIDENCE,
                                     classes=config.DETECTION_CLASSES)
        self.tracker = Tracker(self.detector)
        self.pose_estimator = MediaPipeEstimator()
        self.analyzer = PostureAnalyzer()
        
        source = int(self.video_source) if self.video_source.isdigit() else self.video_source
        
        # Use our new FrameGrabber to prevent latency/buffer buildup
        cap = FrameGrabber(source)
        
        if not cap.isOpened():
            self.status_signal.emit(f"Error: Could not open video source {self.video_source}")
            return

        self.status_signal.emit("Running Pipeline. Ready for calibration clicks.")
        
        while self._run_flag and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
                
            self.current_frame_shape = frame.shape

            # Optimization: Downscale frame for AI if it's too large (e.g. > 480p)
            # This significantly reduces latency in YOLO and MediaPipe
            h, w = frame.shape[:2]
            target_h = 480
            scale = target_h / h
            if scale < 1.0:
                frame_ai = cv2.resize(frame, (int(w * scale), target_h))
            else:
                frame_ai = frame
            
            # 1. Tracking (YOLO + ByteTrack)
            tracks = self.tracker.track_frame(frame_ai)
            
            # Rescale tracks back to original frame size
            if scale < 1.0:
                for i in range(len(tracks)):
                    tracks[i][0:4] = tracks[i][0:4] / scale
            
            frame_out = self.tracker.draw_tracks(frame, tracks)
            
            # 2. Pose Estimation (MediaPipe)
            frame_out, posture_data = self.pose_estimator.estimate_and_draw(frame_out, tracks, draw_skeleton=self.show_skeleton)
            
            # 3. Analytics Engine
            frame_out, metrics_data = self.analyzer.analyze_and_draw(frame_out, posture_data, tracks)
            
            # 4. Calibration Logic
            if len(self.calibration_clicks) == 2:
                self._handle_calibration(posture_data)
                
            # Draw clicks
            for click in self.calibration_clicks:
                cv2.circle(frame_out, click, 5, (0, 0, 255), -1)

            # Send frame and metrics to GUI
            self.change_pixmap_signal.emit(frame_out)
            self.metrics_signal.emit(metrics_data)
            
            # Extract 3D data for visualization
            pose_3d_data = {c_id: data.get('world_landmarks') for c_id, data in posture_data.items() if 'world_landmarks' in data}
            self.pose_3d_signal.emit(pose_3d_data)
            
            # Small sleep to yield to GUI thread
            time.sleep(0.01)
            
        cap.release()
        self.status_signal.emit("Camera Stopped")
        
    def _handle_calibration(self, posture_data):
        p1 = self.calibration_clicks[0]
        p2 = self.calibration_clicks[1]
        
        pixel_dist = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        if pixel_dist > 0 and len(posture_data) > 0:
            pixels_per_foot = pixel_dist * (30.48 / 15.0)
            cadet_id = list(posture_data.keys())[0]
            data = posture_data[cadet_id]
            
            nose = data.get('nose')
            l_ankle = data.get('left_ankle')
            r_ankle = data.get('right_ankle')
            w_nose = data.get('world_nose')
            w_l_ankle = data.get('world_left_ankle')
            w_r_ankle = data.get('world_right_ankle')
            
            if nose and (l_ankle or r_ankle) and w_nose and (w_l_ankle or w_r_ankle):
                ankles_y = [a[1] for a in [l_ankle, r_ankle] if a]
                height_px = (sum(ankles_y) / len(ankles_y)) - nose[1]
                calibrated_height_feet = height_px / pixels_per_foot
                
                dist_l = self.analyzer.calculate_3d_distance(w_nose, w_l_ankle) if w_l_ankle else 0
                dist_r = self.analyzer.calculate_3d_distance(w_nose, w_r_ankle) if w_r_ankle else 0
                dists = [d for d in [dist_l, dist_r] if d > 0]
                
                if dists:
                    total_world_height = (sum(dists) / len(dists)) + 0.15
                    config.WORLD_TO_REAL_RATIO = calibrated_height_feet / total_world_height
                    self.status_signal.emit("SUCCESS: Calibrated! 3D Anchor Locked.")
                
        self.calibration_clicks = [] # Reset clicks
        
    def add_click_relative(self, rel_x, rel_y):
        if hasattr(self, 'current_frame_shape') and self.current_frame_shape:
            h, w = self.current_frame_shape[:2]
            abs_x = int(rel_x * w)
            abs_y = int(rel_y * h)
            self.calibration_clicks.append((abs_x, abs_y))

    def stop(self):
        self._run_flag = False
        self.wait()

