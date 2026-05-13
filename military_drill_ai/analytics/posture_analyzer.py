import math
import cv2
from military_drill_ai.utils.config import config

class PostureAnalyzer:
    def __init__(self):
        """
        Initializes the Posture Analytics Engine.
        Responsible for calculating salute angles and approximate heights.
        """
        pass

    def calculate_angle(self, p1, p2, p3):
        """
        Calculates the interior angle between three points (p2 is the vertex).
        Points are tuples of (x, y).
        """
        if not (p1 and p2 and p3):
            return None
            
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        # Calculate angle using atan2
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        
        # Make it a positive interior angle <= 180
        angle = abs(angle)
        if angle > 180.0:
            angle = 360.0 - angle
            
        return angle

    def analyze_and_draw(self, frame, posture_data, tracks):
        """
        Calculates height and salute angle, and draws the metrics on the frame.
        """
        out_frame = frame.copy()
        
        for track in tracks:
            x1, y1, x2, y2, conf, cls, cadet_id = track
            if cadet_id not in posture_data:
                continue
                
            data = posture_data[cadet_id]
            metrics = []
            
            # 1. Calculate Approx Height (Nose to average of ankles)
            nose = data.get('nose')
            l_ankle = data.get('left_ankle')
            r_ankle = data.get('right_ankle')
            
            if nose and (l_ankle or r_ankle):
                # Use whichever ankle is visible, or average if both are visible
                ankles_y = []
                if l_ankle: ankles_y.append(l_ankle[1])
                if r_ankle: ankles_y.append(r_ankle[1])
                
                avg_ankle_y = sum(ankles_y) / len(ankles_y)
                
                # Height in pixels
                height_px = avg_ankle_y - nose[1]
                
                # Convert to feet and inches using camera calibration
                height_in_feet = height_px / config.PIXELS_PER_FOOT
                feet = int(height_in_feet)
                inches = int((height_in_feet - feet) * 12)
                
                metrics.append(f"Height: {feet}'{inches}\"")
                
            # 2. Calculate Salute Angle (Right Shoulder -> Right Elbow -> Right Wrist)
            r_shoulder = data.get('right_shoulder')
            r_elbow = data.get('right_elbow')
            r_wrist = data.get('right_wrist')
            
            if r_shoulder and r_elbow and r_wrist:
                elbow_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
                
                if elbow_angle is not None:
                    # Append exact angle to metrics
                    metrics.append(f"Salute Angle: {int(elbow_angle)} deg")
                    
                    # Draw visual arc around the elbow to highlight the analytics
                    cv2.putText(out_frame, f"{int(elbow_angle)}", 
                                (r_elbow[0] + 15, r_elbow[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                
            # Draw all metrics slightly above the bounding box
            y_offset = max(y1 - 30, 0)
            for i, metric in enumerate(metrics):
                cv2.putText(out_frame, metric, (x1, y_offset - (i * 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
        return out_frame
