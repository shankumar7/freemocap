import cv2
import argparse
import sys
import os

# Add the parent directory to sys.path so we can import military_drill_ai
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from military_drill_ai.utils.config import config
from military_drill_ai.detection.yolo_detector import YOLODetector
from military_drill_ai.tracking.byte_tracker import Tracker

from military_drill_ai.pose_estimation.mediapipe_estimator import MediaPipeEstimator
from military_drill_ai.analytics.posture_analyzer import PostureAnalyzer

import math

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param['clicks'].append((x, y))

def run_pipeline(video_source: str = "0"):
    """
    Test Pipeline
    """
    print("Initializing YOLO Detector (Stage 1)...")
    detector = YOLODetector(model_path=config.YOLO_MODEL_PATH, 
                            conf=config.DETECTION_CONFIDENCE,
                            classes=config.DETECTION_CLASSES)
                            
    print("Initializing ByteTracker (Stage 2)...")
    tracker = Tracker(detector)
    
    print("Initializing MediaPipe Holistic (Stage 3)...")
    pose_estimator = MediaPipeEstimator()
    
    print("Initializing Posture Analyzer (Stage 4)...")
    analyzer = PostureAnalyzer()
    
    # If source is an integer string, use it as camera index
    if video_source.isdigit():
        video_source = int(video_source)
        
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    cv2.namedWindow("Military Drill AI Pipeline - Live")
    click_data = {'clicks': []}
    cv2.setMouseCallback("Military Drill AI Pipeline - Live", mouse_callback, click_data)

    print("Starting video feed. Press 'q' to quit.")
    print("CALIBRATION: Hold your 15cm scale up to the camera and click both ends of it on the screen to calibrate!")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run tracking (includes detection)
        tracks = tracker.track_frame(frame)
        
        # Draw tracking bounding boxes
        frame_out = tracker.draw_tracks(frame, tracks)
        
        # Run Full Body & Finger Estimation on the tracked boxes
        frame_out, posture_data = pose_estimator.estimate_and_draw(frame_out, tracks)
        
        # Run Analytics Engine (Height & Salute Angle)
        frame_out = analyzer.analyze_and_draw(frame_out, posture_data, tracks)
        
        # Interactive Calibration Logic
        if len(click_data['clicks']) == 2:
            p1 = click_data['clicks'][0]
            p2 = click_data['clicks'][1]
            
            pixel_dist = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            if pixel_dist > 0 and len(posture_data) > 0:
                # 1. Calculate the 2D Pixels per foot using the 15cm scale
                pixels_per_foot = pixel_dist * (30.48 / 15.0)
                
                # 2. Grab the first visible cadet to anchor their 3D height
                cadet_id = list(posture_data.keys())[0]
                data = posture_data[cadet_id]
                
                nose = data.get('nose')
                l_ankle = data.get('left_ankle')
                r_ankle = data.get('right_ankle')
                w_nose = data.get('world_nose')
                w_l_ankle = data.get('world_left_ankle')
                w_r_ankle = data.get('world_right_ankle')
                
                if nose and (l_ankle or r_ankle) and w_nose and (w_l_ankle or w_r_ankle):
                    # Get their 2D pixel height
                    ankles_y = [a[1] for a in [l_ankle, r_ankle] if a]
                    height_px = (sum(ankles_y) / len(ankles_y)) - nose[1]
                    
                    # Calculate their actual true height in feet based on the scale
                    calibrated_height_feet = height_px / pixels_per_foot
                    
                    # Get their uncalibrated 3D depth-invariant height
                    dist_l = analyzer.calculate_3d_distance(w_nose, w_l_ankle) if w_l_ankle else 0
                    dist_r = analyzer.calculate_3d_distance(w_nose, w_r_ankle) if w_r_ankle else 0
                    dists = [d for d in [dist_l, dist_r] if d > 0]
                    total_world_height = (sum(dists) / len(dists)) + 0.15
                    
                    # Anchor the calibration to the 3D model!
                    config.WORLD_TO_REAL_RATIO = calibrated_height_feet / total_world_height
                    print(f"SUCCESS: Calibrated! 3D Anchor Locked. Height will now remain constant even if they walk away.")
            
            click_data['clicks'] = [] # Reset clicks
            
        # Draw clicks
        for click in click_data['clicks']:
            cv2.circle(frame_out, click, 5, (0, 0, 255), -1)
            
        cv2.imshow("Military Drill AI Pipeline - Live", frame_out)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Military Drill AI Pipeline")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or path to video file)")
    args = parser.parse_args()
    
    run_pipeline(args.source)
