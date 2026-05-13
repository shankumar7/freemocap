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

    print("Starting video feed. Press 'q' to quit.")
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
