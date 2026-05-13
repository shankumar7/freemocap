import math
import cv2

class PostureAnalyzer:
    def __init__(self):
        """
        Initializes the Posture Analytics Engine using True 3D World Landmarks.
        """
        pass

    def calculate_3d_angle(self, p1, p2, p3):
        """
        Calculates the 3D interior angle between three points (p2 is the vertex).
        Points are tuples of (x, y, z) in meters.
        """
        if not (p1 and p2 and p3):
            return None
            
        v1 = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
        v2 = (p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2])
        
        dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
        
        if mag1 == 0 or mag2 == 0:
            return None
            
        cos_theta = max(min(dot / (mag1 * mag2), 1.0), -1.0)
        return math.degrees(math.acos(cos_theta))

    def calculate_3d_distance(self, p1, p2):
        if not (p1 and p2):
            return None
        return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)

    def analyze_and_draw(self, frame, posture_data, tracks):
        """
        Calculates true real-world height and 3D salute angle.
        """
        out_frame = frame.copy()
        
        for track in tracks:
            x1, y1, x2, y2, conf, cls, cadet_id = track
            if cadet_id not in posture_data:
                continue
                
            data = posture_data[cadet_id]
            metrics = []
            
            # 1. Calculate Real-World Height using 3D World Landmarks (in meters)
            w_nose = data.get('world_nose')
            w_l_ankle = data.get('world_left_ankle')
            w_r_ankle = data.get('world_right_ankle')
            
            if w_nose and (w_l_ankle or w_r_ankle):
                dist_l = self.calculate_3d_distance(w_nose, w_l_ankle)
                dist_r = self.calculate_3d_distance(w_nose, w_r_ankle)
                
                dists = []
                if dist_l: dists.append(dist_l)
                if dist_r: dists.append(dist_r)
                
                avg_body_length_m = sum(dists) / len(dists)
                
                # Add 0.15 meters (~6 inches) to account for top of head (above nose) 
                # and bottom of foot (below ankle)
                total_height_m = avg_body_length_m + 0.15 
                
                # Convert meters to feet
                height_in_feet = total_height_m * 3.28084
                feet = int(height_in_feet)
                inches = int((height_in_feet - feet) * 12)
                
                metrics.append(f"Height: {feet}'{inches}\"")
                
            # 2. Calculate true 3D Salute Angle
            w_r_shoulder = data.get('world_right_shoulder')
            w_r_elbow = data.get('world_right_elbow')
            w_r_wrist = data.get('world_right_wrist')
            
            if w_r_shoulder and w_r_elbow and w_r_wrist:
                elbow_angle = self.calculate_3d_angle(w_r_shoulder, w_r_elbow, w_r_wrist)
                
                if elbow_angle is not None:
                    metrics.append(f"3D Salute Angle: {int(elbow_angle)} deg")
                    
                    # Draw on 2D screen using 2D elbow coordinate
                    r_elbow_2d = data.get('right_elbow')
                    if r_elbow_2d:
                        cv2.putText(out_frame, f"{int(elbow_angle)}", 
                                    (r_elbow_2d[0] + 15, r_elbow_2d[1]), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                
            # Draw all metrics slightly above the bounding box
            y_offset = max(y1 - 30, 0)
            for i, metric in enumerate(metrics):
                cv2.putText(out_frame, metric, (int(x1), int(y_offset) - (i * 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
        return out_frame
