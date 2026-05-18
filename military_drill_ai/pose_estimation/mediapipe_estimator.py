import cv2
import numpy as np
import mediapipe as mp

class MediaPipeEstimator:
    def __init__(self):
        """
        Initializes MediaPipe Pose for robust skeleton tracking.
        We use ROI-based processing for maximum accuracy and speed.
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.trackers = {} # cadet_id -> Pose instance

    def _get_tracker(self, cadet_id):
        if cadet_id not in self.trackers:
            self.trackers[cadet_id] = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
        return self.trackers[cadet_id]

    def estimate_and_draw(self, frame, tracks, draw_skeleton=True):
        """
        Processes each cadet ROI through MediaPipe Pose.
        """
        out_frame = frame.copy()
        frame_h, frame_w, _ = frame.shape
        
        active_ids = set()
        posture_data = {}
        
        for track in tracks:
            x1, y1, x2, y2, conf, cls, cadet_id = track
            if cadet_id == -1: continue
            active_ids.add(cadet_id)
            
            # 1. ROI Extraction with safety margins
            w_box = x2 - x1
            h_box = y2 - y1
            margin = 0.2
            
            roi_x1 = max(0, int(x1 - w_box * margin))
            roi_y1 = max(0, int(y1 - h_box * margin))
            roi_x2 = min(frame_w, int(x2 + w_box * margin))
            roi_y2 = min(frame_h, int(y2 + h_box * margin))
            
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            if roi.size == 0: continue
            
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            tracker = self._get_tracker(cadet_id)
            results = tracker.process(roi_rgb)
            
            if results.pose_landmarks:
                roi_h, roi_w = roi.shape[:2]
                
                # Helper to translate ROI landmarks to global pixels
                def get_global_px(lm):
                    if not lm or getattr(lm, 'visibility', 0) < 0.1: return None
                    gx = int(roi_x1 + lm.x * roi_w)
                    gy = int(roi_y1 + lm.y * roi_h)
                    return (gx, gy)
                
                lms = results.pose_landmarks.landmark
                cadet_posture = {
                    'nose': get_global_px(lms[self.mp_pose.PoseLandmark.NOSE]),
                    'right_shoulder': get_global_px(lms[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]),
                    'right_elbow': get_global_px(lms[self.mp_pose.PoseLandmark.RIGHT_ELBOW]),
                    'right_wrist': get_global_px(lms[self.mp_pose.PoseLandmark.RIGHT_WRIST]),
                    'left_ankle': get_global_px(lms[self.mp_pose.PoseLandmark.LEFT_ANKLE]),
                    'right_ankle': get_global_px(lms[self.mp_pose.PoseLandmark.RIGHT_ANKLE])
                }
                
                # Store world landmarks for 3D View
                if results.pose_world_landmarks:
                    def get_3d(lm):
                        if not lm or getattr(lm, 'visibility', 0) < 0.1: return None
                        return (lm.x, lm.y, lm.z)
                    cadet_posture['world_landmarks'] = [get_3d(lm) for lm in results.pose_world_landmarks.landmark]
                    
                    # For metrics
                    w_lms = results.pose_world_landmarks.landmark
                    cadet_posture['world_nose'] = get_3d(w_lms[self.mp_pose.PoseLandmark.NOSE])
                    cadet_posture['world_left_ankle'] = get_3d(w_lms[self.mp_pose.PoseLandmark.LEFT_ANKLE])
                    cadet_posture['world_right_ankle'] = get_3d(w_lms[self.mp_pose.PoseLandmark.RIGHT_ANKLE])
                
                posture_data[cadet_id] = cadet_posture
                
                # Draw skeleton on global frame
                if draw_skeleton:
                    # We create a temporary landmark list in global coords for drawing
                    # However, mp_drawing works on normalized coords. 
                    # So we convert back to global normalized.
                    for conn in self.mp_pose.POSE_CONNECTIONS:
                        p1_idx, p2_idx = conn
                        p1 = get_global_px(lms[p1_idx])
                        p2 = get_global_px(lms[p2_idx])
                        if p1 and p2:
                            cv2.line(out_frame, p1, p2, (255, 255, 255), 2)
                            cv2.circle(out_frame, p1, 3, (0, 0, 255), -1)
                            cv2.circle(out_frame, p2, 3, (0, 0, 255), -1)

        # Cleanup
        dead_ids = set(self.trackers.keys()) - active_ids
        for d_id in dead_ids:
            self.trackers[d_id].close()
            del self.trackers[d_id]
            
        return out_frame, posture_data
