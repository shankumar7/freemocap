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
        Returns the processed frame and a dictionary of metrics per cadet.
        """
        out_frame = frame.copy()
        all_cadet_metrics = {}
        
        for track in tracks:
            x1, y1, x2, y2, conf, cls, cadet_id = track
            if cadet_id not in posture_data:
                continue
                
            data = posture_data[cadet_id]
            metrics_labels = []
            metrics_values = {}
            
            # 1. DEPTH-INVARIANT HEIGHT (Anchored to 15cm Scale Calibration)
            w_nose = data.get('world_nose')
            w_l_ankle = data.get('world_left_ankle')
            w_r_ankle = data.get('world_right_ankle')
            
            if w_nose and (w_l_ankle or w_r_ankle):
                dist_l = self.calculate_3d_distance(w_nose, w_l_ankle) if w_l_ankle else 0
                dist_r = self.calculate_3d_distance(w_nose, w_r_ankle) if w_r_ankle else 0
                dists = [d for d in [dist_l, dist_r] if d > 0]
                
                avg_body_length_m = sum(dists) / len(dists)
                total_world_height_m = avg_body_length_m + 0.15
                
                # Apply the interactive 3D Anchor Ratio if calibrated
                if hasattr(config, 'WORLD_TO_REAL_RATIO') and config.WORLD_TO_REAL_RATIO > 0:
                    height_in_feet = total_world_height_m * config.WORLD_TO_REAL_RATIO
                    feet = int(height_in_feet)
                    inches = int((height_in_feet - feet) * 12)
                    metrics_labels.append(f"Height: {feet}'{inches}\"")
                    metrics_values['height'] = f"{feet}'{inches}\""
                else:
                    height_in_feet = total_world_height_m * 3.28084
                    feet = int(height_in_feet)
                    inches = int((height_in_feet - feet) * 12)
                    metrics_labels.append(f"Height: {feet}'{inches}\" (Uncalib)")
                    metrics_values['height'] = f"{feet}'{inches}\"*"
                
            # 2. VISUAL SALUTE ANGLE
            r_shoulder = data.get('right_shoulder')
            r_elbow = data.get('right_elbow')
            r_wrist = data.get('right_wrist')
            
            if r_shoulder and r_elbow and r_wrist:
                elbow_angle = self.calculate_2d_angle(r_shoulder, r_elbow, r_wrist)
                if elbow_angle is not None:
                    metrics_labels.append(f"Angle: {int(elbow_angle)} deg")
                    metrics_values['angle'] = f"{int(elbow_angle)}°"
                    cv2.putText(out_frame, f"{int(elbow_angle)}", 
                                (r_elbow[0] + 15, r_elbow[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 3. LEG DISTANCE (3D World Space)
            if w_l_ankle and w_r_ankle:
                leg_dist_m = self.calculate_3d_distance(w_l_ankle, w_r_ankle)
                if hasattr(config, 'WORLD_TO_REAL_RATIO') and config.WORLD_TO_REAL_RATIO > 0:
                    # Convert meters to inches (approx based on ratio)
                    # WORLD_TO_REAL_RATIO is height_feet / world_height_meters
                    # So ratio * 12 gives inches / meter
                    leg_dist_inches = leg_dist_m * (config.WORLD_TO_REAL_RATIO * 12 / 3.28084)
                    metrics_labels.append(f"Leg Dist: {int(leg_dist_inches)} in")
                    metrics_values['leg_distance'] = f"{int(leg_dist_inches)} in"
                else:
                    leg_dist_cm = leg_dist_m * 100
                    metrics_labels.append(f"Leg Dist: {int(leg_dist_cm)} cm (Rel)")
                    metrics_values['leg_distance'] = f"{int(leg_dist_cm)} cm"

            if metrics_values:
                all_cadet_metrics[cadet_id] = metrics_values
                    
            # Draw all metrics slightly above the bounding box
            y_offset = max(y1 - 30, 0)
            for i, label in enumerate(metrics_labels):
                cv2.putText(out_frame, label, (int(x1), int(y_offset) - (i * 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
        return out_frame, all_cadet_metrics
