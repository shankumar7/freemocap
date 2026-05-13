import cv2
import mediapipe as mp

class MediaPipeEstimator:
    def __init__(self):
        """
        Initializes MediaPipe Holistic for full-body and finger tracking.
        """
        self.mp_holistic = mp.solutions.holistic
        
        # Initialize Holistic model
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def estimate_and_draw(self, frame, tracks):
        """
        Extracts full body and finger keypoints for each tracked bounding box.
        Draws them directly onto the frame.
        """
        out_frame = frame.copy()
        
        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for track in tracks:
            x1, y1, x2, y2, conf, cls, cadet_id = track
            
            # Add a small margin to the bounding box to ensure hands/feet aren't clipped
            margin_x = int((x2 - x1) * 0.15)
            margin_y = int((y2 - y1) * 0.15)
            
            crop_x1 = max(0, x1 - margin_x)
            crop_y1 = max(0, y1 - margin_y)
            crop_x2 = min(frame.shape[1], x2 + margin_x)
            crop_y2 = min(frame.shape[0], y2 + margin_y)
            
            crop = frame_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                continue
                
            # Process the crop with MediaPipe
            results = self.holistic.process(crop)
            
            # We need to offset the drawing coordinates back to the full frame
            crop_h, crop_w, _ = crop.shape
            
            # Draw offset landmarks
            self._draw_offset_landmarks(out_frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS, crop_x1, crop_y1, crop_w, crop_h, is_hand=False)
            self._draw_offset_landmarks(out_frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, crop_x1, crop_y1, crop_w, crop_h, is_hand=True)
            self._draw_offset_landmarks(out_frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, crop_x1, crop_y1, crop_w, crop_h, is_hand=True)
            
        return out_frame

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
                
                # Ignore points that are very low visibility
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
            if hasattr(landmark, 'visibility') and landmark.visibility < 0.3:
                continue
                
            x = int(landmark.x * crop_w) + offset_x
            y = int(landmark.y * crop_h) + offset_y
            
            color = (0, 255, 255) if is_hand else (0, 0, 255)
            radius = 2 if is_hand else 3
            cv2.circle(frame, (x, y), radius, color, -1)
