import cv2
import numpy as np
import mediapipe as mp

class MediaPipeEstimator:
    def __init__(self):
        """
        Initializes MediaPipe Holistic for full-body and finger tracking.
        We use a dictionary to maintain a separate Holistic tracker for each Cadet ID.
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.trackers = {} # cadet_id -> Holistic instance

    def _get_tracker(self, cadet_id):
        if cadet_id not in self.trackers:
            self.trackers[cadet_id] = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                refine_face_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        return self.trackers[cadet_id]

    def estimate_and_draw(self, frame, tracks):
        """
        Extracts full body and finger keypoints using full-frame masking.
        This guarantees flawless aspect ratios and native MediaPipe accuracy.
        """
        out_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        active_ids = set()
        posture_data = {}
        frame_h, frame_w, _ = frame.shape
        
        for track in tracks:
            x1, y1, x2, y2, conf, cls, cadet_id = track
            if cadet_id == -1:
                continue
                
            active_ids.add(cadet_id)
            tracker = self._get_tracker(cadet_id)
            
            # To completely eliminate distortion/hand-cropping, we run MediaPipe 
            # on the FULL frame, but black out everything except this cadet's ROI.
            margin_x = int((x2 - x1) * 0.4)
            margin_y = int((y2 - y1) * 0.4)
            
            crop_x1 = max(0, int(x1) - margin_x)
            crop_y1 = max(0, int(y1) - margin_y)
            crop_x2 = min(frame_w, int(x2) + margin_x)
            crop_y2 = min(frame_h, int(y2) + margin_y)
            
            # Create a blank mask and copy the cadet's pixels into it
            masked_frame = np.zeros_like(frame_rgb)
            masked_frame[crop_y1:crop_y2, crop_x1:crop_x2] = frame_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Process the masked full-frame
            results = tracker.process(masked_frame)
            
            # Extract landmarks for Analytics Engine
            cadet_posture = {}
            if results.pose_landmarks:
                def get_px(landmark):
                    if not landmark or getattr(landmark, 'visibility', 0) < 0.3:
                        return None
                    # Coordinates are normalized 0.0 to 1.0 of the FULL frame!
                    return (int(landmark.x * frame_w), int(landmark.y * frame_h))
                
                landmarks = results.pose_landmarks.landmark
                mp_pose = mp.solutions.pose
                
                cadet_posture = {
                    'nose': get_px(landmarks[mp_pose.PoseLandmark.NOSE]),
                    'right_shoulder': get_px(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]),
                    'right_elbow': get_px(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]),
                    'right_wrist': get_px(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]),
                    'left_ankle': get_px(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]),
                    'right_ankle': get_px(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])
                }
                
            if results.pose_world_landmarks:
                def get_3d(landmark):
                    if not landmark or getattr(landmark, 'visibility', 0) < 0.3:
                        return None
                    return (landmark.x, landmark.y, landmark.z)
                    
                world_landmarks = results.pose_world_landmarks.landmark
                cadet_posture['world_nose'] = get_3d(world_landmarks[mp_pose.PoseLandmark.NOSE])
                cadet_posture['world_right_shoulder'] = get_3d(world_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
                cadet_posture['world_right_elbow'] = get_3d(world_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW])
                cadet_posture['world_right_wrist'] = get_3d(world_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])
                cadet_posture['world_left_ankle'] = get_3d(world_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])
                cadet_posture['world_right_ankle'] = get_3d(world_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])
                
            if cadet_posture:
                posture_data[cadet_id] = cadet_posture
            
            # Draw using Native MediaPipe Utilities for flawless rendering
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    out_frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            if results.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    out_frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
                )
            if results.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    out_frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
                )
            
        # Clean up trackers for cadets who have left the frame
        dead_ids = set(self.trackers.keys()) - active_ids
        for d_id in dead_ids:
            self.trackers[d_id].close()
            del self.trackers[d_id]
            
        return out_frame, posture_data
