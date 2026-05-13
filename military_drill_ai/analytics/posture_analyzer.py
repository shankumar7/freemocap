import math
import cv2
from military_drill_ai.utils.config import config

class PostureAnalyzer:
    def __init__(self):
        """
        Initializes the Posture Analytics Engine.
        Uses a Hybrid Approach:
        - 3D Real-World Depth for Height (Invariant to Camera Distance)
        - 2D Visual Perspective for Salute Angle (Matches physical ruler references)
        """
        pass

    def calculate_2d_angle(self, p1, p2, p3):
        """
        Calculates the 2D visual interior angle exactly as seen from the camera's perspective.
        """
        if not (p1 and p2 and p3):
            return None
            
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        angle = abs(angle)
        if angle > 180.0:
            angle = 360.0 - angle
            
        return angle

    def calculate_3d_distance(self, p1, p2):
        """
        Calculates absolute metric distance using MediaPipe's intrinsic depth network.
        """
        if not (p1 and p2):
            return None
        return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)

    def analyze_and_draw(self, frame, posture_data, tracks):
        """
        Calculates hybrid metrics and draws them on the frame.
        """
        out_frame = frame.copy()
        
        for track in tracks:
            x1, y1, x2, y2, conf, cls, cadet_id = track
            if cadet_id not in posture_data:
                continue
                
            data = posture_data[cadet_id]
            metrics = []
            
            # 1. DEPTH-INVARIANT HEIGHT & CALIBRATED 2D HEIGHT
            # Using 2D pixels so the physical 15cm scale calibration takes effect
            nose = data.get('nose')
            l_ankle = data.get('left_ankle')
            r_ankle = data.get('right_ankle')
            
            if nose and (l_ankle or r_ankle):
                ankles_y = []
                if l_ankle: ankles_y.append(l_ankle[1])
                if r_ankle: ankles_y.append(r_ankle[1])
                
                avg_ankle_y = sum(ankles_y) / len(ankles_y)
                height_px = avg_ankle_y - nose[1]
                
                # Convert using the live calibrated PIXELS_PER_FOOT
                if config.PIXELS_PER_FOOT > 0:
                    height_in_feet = height_px / config.PIXELS_PER_FOOT
                    feet = int(height_in_feet)
                    inches = int((height_in_feet - feet) * 12)
                    metrics.append(f"Calibrated Height: {feet}'{inches}\"")
                else:
                    metrics.append(f"Height: {int(height_px)}px (Uncalibrated)")
                
            # 2. VISUAL SALUTE ANGLE (Using 2D Pixels to match the 15cm physical ruler)
            r_shoulder = data.get('right_shoulder')
            r_elbow = data.get('right_elbow')
            r_wrist = data.get('right_wrist')
            
            if r_shoulder and r_elbow and r_wrist:
                elbow_angle = self.calculate_2d_angle(r_shoulder, r_elbow, r_wrist)
                
                if elbow_angle is not None:
                    # Append exact visual angle to metrics
                    metrics.append(f"Visual Angle: {int(elbow_angle)} deg")
                    
                    # Draw visual arc around the elbow
                    cv2.putText(out_frame, f"{int(elbow_angle)}", 
                                (r_elbow[0] + 15, r_elbow[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                
            # Draw all metrics slightly above the bounding box
            y_offset = max(y1 - 30, 0)
            for i, metric in enumerate(metrics):
                cv2.putText(out_frame, metric, (int(x1), int(y_offset) - (i * 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
        return out_frame
