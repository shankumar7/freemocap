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
    status_signal = Signal(str)
    
    def __init__(self, vive_client=None):
        super().__init__()
        self._run_flag = True
        self.video_source = "0"
        self.vive_client = vive_client
        self.calibration_clicks = []
        self.trigger_vr_calibration = False
        
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

            # 1. Tracking (YOLO + ByteTrack)
            tracks = self.tracker.track_frame(frame)
            frame_out = self.tracker.draw_tracks(frame, tracks)
            
            # 2. Pose Estimation (MediaPipe)
            frame_out, posture_data = self.pose_estimator.estimate_and_draw(frame_out, tracks)
            
            # 3. Analytics Engine
            frame_out = self.analyzer.analyze_and_draw(frame_out, posture_data, tracks)
            
            # 4. Integrate Vive Tracker Data if active
            if self.vive_client and self.vive_client.is_initialized:
                tracker_data = self.vive_client.poll_poses()
                if tracker_data:
                    y_offset = 30
                    for serial, data in tracker_data.items():
                        pos = data['position']
                        cls = data['device_class']
                        text = f"{cls} [{serial}]: X:{pos[0]:.2f} Y:{pos[1]:.2f} Z:{pos[2]:.2f}"
                        cv2.putText(frame_out, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        y_offset += 30
                else:
                    cv2.putText(frame_out, "VR Connected, but 0 devices tracking (Wake up headset!)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                if self.trigger_vr_calibration:
                    self._handle_vr_calibration(tracker_data, posture_data)
                    self.trigger_vr_calibration = False

            # 5. Calibration Logic
            if len(self.calibration_clicks) == 2:
                self._handle_calibration(posture_data)
                
            # Draw clicks
            for click in self.calibration_clicks:
                cv2.circle(frame_out, click, 5, (0, 0, 255), -1)

            # Send frame to GUI
            self.change_pixmap_signal.emit(frame_out)
            
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

    def auto_calibrate_vr(self):
        self.trigger_vr_calibration = True

    def _handle_vr_calibration(self, tracker_data, posture_data):
        if not tracker_data:
            self.status_signal.emit("Error: No Vive Tracker data available.")
            return
            
        # Find the highest tracker (Y axis is up in SteamVR)
        highest_y = -float('inf')
        for serial, data in tracker_data.items():
            if data['position'][1] > highest_y:
                highest_y = data['position'][1]
                
        if highest_y == -float('inf'):
            return
            
        # highest_y is in meters. Convert to feet (optional, but our original code used feet)
        real_height_feet = highest_y * 3.28084
        
        if len(posture_data) > 0:
            cadet_id = list(posture_data.keys())[0]
            data = posture_data[cadet_id]
            
            w_nose = data.get('world_nose')
            w_l_ankle = data.get('world_left_ankle')
            w_r_ankle = data.get('world_right_ankle')
            
            if w_nose and (w_l_ankle or w_r_ankle):
                dist_l = self.analyzer.calculate_3d_distance(w_nose, w_l_ankle) if w_l_ankle else 0
                dist_r = self.analyzer.calculate_3d_distance(w_nose, w_r_ankle) if w_r_ankle else 0
                dists = [d for d in [dist_l, dist_r] if d > 0]
                
                if dists:
                    total_world_height = (sum(dists) / len(dists)) + 0.15
                    config.WORLD_TO_REAL_RATIO = real_height_feet / total_world_height
                    self.status_signal.emit(f"SUCCESS: Auto-Calibrated using VR Head Tracker! Height locked to {highest_y:.2f} meters.")
                else:
                    self.status_signal.emit("Error: Could not calculate 3D height from MediaPipe.")
            else:
                self.status_signal.emit("Error: MediaPipe could not detect full body for calibration.")
        else:
            self.status_signal.emit("Error: No cadet detected in frame.")

    def stop(self):
        self._run_flag = False
        self.wait()
