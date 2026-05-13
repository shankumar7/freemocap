import cv2
import mediapipe as mp

class MediaPipeEstimator:
    def __init__(self):
        """
        Initializes MediaPipe Holistic for full-body and finger tracking.
        We use a dictionary to maintain a separate Holistic tracker for each Cadet ID,
        ensuring perfect temporal smoothing for multiple people.
        """
        self.mp_holistic = mp.solutions.holistic
        self.trackers = {} # cadet_id -> Holistic instance

    def _get_tracker(self, cadet_id):
        if cadet_id not in self.trackers:
            self.trackers[cadet_id] = self.mp_holistic.Holistic(
                static_image_mode=False, # False allows temporal smoothing
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
        Extracts full body and finger keypoints for each tracked bounding box.
        Draws them directly onto the frame.
        """
        out_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Keep track of active IDs to clean up old trackers and save memory
        active_ids = set()
        
        # Posture data to send to the Analytics Engine
        posture_data = {}
        
        for track in tracks:
            x1, y1, x2, y2, conf, cls, cadet_id = track
            if cadet_id == -1:
                continue
                
            active_ids.add(cadet_id)
            
            # Get the dedicated tracker for this specific cadet
            tracker = self._get_tracker(cadet_id)
            
            # MediaPipe expects a square-ish aspect ratio.
            width = x2 - x1
            height = y2 - y1
            center_x = x1 + width // 2
            center_y = y1 + height // 2
            
            box_size = max(width, height)
            padded_size = int(box_size * 1.4) # 40% margin to catch extended saluting hands
            half_size = padded_size // 2
            
            crop_x1 = max(0, center_x - half_size)
            crop_y1 = max(0, center_y - half_size)
            crop_x2 = min(frame.shape[1], center_x + half_size)
            crop_y2 = min(frame.shape[0], center_y + half_size)
            
            crop = frame_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                continue
                
            # Process the square crop with this cadet's personal MediaPipe tracker
            results = tracker.process(crop)
            
            crop_h, crop_w, _ = crop.shape
            
            # Extract landmarks for Analytics Engine
            if results.pose_landmarks:
                def get_px(landmark):
                    if not landmark or getattr(landmark, 'visibility', 0) < 0.3:
                        return None
                    return (int(landmark.x * crop_w) + crop_x1, int(landmark.y * crop_h) + crop_y1)
                
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
                posture_data[cadet_id] = cadet_posture
            
            # Draw offset landmarks
            self._draw_offset_landmarks(out_frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS, crop_x1, crop_y1, crop_w, crop_h, is_hand=False)
            self._draw_offset_landmarks(out_frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, crop_x1, crop_y1, crop_w, crop_h, is_hand=True)
            self._draw_offset_landmarks(out_frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, crop_x1, crop_y1, crop_w, crop_h, is_hand=True)
            
        # Clean up trackers for cadets who have left the frame
        dead_ids = set(self.trackers.keys()) - active_ids
        for d_id in dead_ids:
            self.trackers[d_id].close()
            del self.trackers[d_id]
            
        return out_frame, posture_data

    def _draw_offset_landmarks(self, frame, landmark_list, connections, offset_x, offset_y, crop_w, crop_h, is_hand=False):
        if not landmark_list:
            return
            
        # Draw connections (bones)
        if connections:
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                
                pt1 = landmark_list.landmark[start_idx]
                pt2 = landmark_list.landmark[end_idx]
                
                # Ignore points that are very low visibility (Hands do not have visibility calculated)
                if not is_hand:
                    if hasattr(pt1, 'visibility') and pt1.visibility < 0.3:
                        continue
                    if hasattr(pt2, 'visibility') and pt2.visibility < 0.3:
                        continue
                
                # Convert normalized coordinates to pixel coordinates in the crop, then add offset
                x1 = int(pt1.x * crop_w) + offset_x
                y1 = int(pt1.y * crop_h) + offset_y
                x2 = int(pt2.x * crop_w) + offset_x
                y2 = int(pt2.y * crop_h) + offset_y
                
                color = (0, 255, 0) if is_hand else (255, 0, 0)
                thickness = 1 if is_hand else 2
                cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
                
        # Draw landmarks (joints)
        for idx, landmark in enumerate(landmark_list.landmark):
            if not is_hand:
                if hasattr(landmark, 'visibility') and landmark.visibility < 0.3:
                    continue
                
            x = int(landmark.x * crop_w) + offset_x
            y = int(landmark.y * crop_h) + offset_y
            
            color = (0, 255, 255) if is_hand else (0, 0, 255)
            radius = 2 if is_hand else 3
            cv2.circle(frame, (x, y), radius, color, -1)
