import cv2
import os
from military_drill_ai.detection.yolo_detector import YOLODetector
from military_drill_ai.tracking.id_manager import IDManager
from typing import List, Tuple

class Tracker:
    def __init__(self, detector: YOLODetector):
        """
        Initialize the tracker using YOLOv8's built-in ByteTrack and our custom ReID IDManager.
        """
        self.detector = detector
        self.id_manager = IDManager()
        
    def track_frame(self, frame) -> List[Tuple[int, int, int, int, float, int, int]]:
        """
        Runs detection and tracking using ByteTrack.
        Returns a list of tracks: [x1, y1, x2, y2, conf, cls, cadet_id]
        """
        custom_tracker_path = os.path.join(os.path.dirname(__file__), "custom_tracker.yaml")
        
        results = self.detector.model.track(
            frame, 
            conf=self.detector.conf, 
            classes=self.detector.classes, 
            tracker=custom_tracker_path,  # Use our custom config with 600 frame buffer
            persist=True, 
            verbose=False
        )
        
        # Gather all YOLO IDs in current frame
        current_yolo_ids = []
        for r in results:
            if r.boxes is not None and r.boxes.id is not None:
                current_yolo_ids.extend(r.boxes.id.int().cpu().tolist())
                
        # Update the simple ID manager (resets if 0 people, assigns new sequential IDs otherwise)
        self.id_manager.update_frame_tracks(current_yolo_ids)
        
        tracks = []
        for r in results:
            boxes = r.boxes
            keypoints = r.keypoints # YOLOv8-pose keypoints
            
            if boxes is not None and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                for i, (box, yolo_track_id) in enumerate(zip(boxes, track_ids)):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    
                    # Map YOLO ID to our custom sequential Cadet ID
                    cadet_id = self.id_manager.get_cadet_id(yolo_track_id)
                    
                    # Extract keypoints if available
                    kpts = keypoints.data[i].cpu().numpy() if keypoints is not None else None
                    tracks.append((int(x1), int(y1), int(x2), int(y2), conf, cls, cadet_id, kpts))
            elif boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    kpts = keypoints.data[i].cpu().numpy() if keypoints is not None else None
                    tracks.append((int(x1), int(y1), int(x2), int(y2), conf, cls, -1, kpts))
        return tracks

    def draw_tracks(self, frame, tracks: List[Tuple]):
        """
        Draws bounding boxes, persistent IDs, and skeletons.
        """
        out_frame = frame.copy()
        
        # COCO 17 keypoint connections
        SKELETON = [
            (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
            (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
            (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
            (3, 5), (4, 6)
        ]
        
        for track in tracks:
            x1, y1, x2, y2, conf, cls, cadet_id, kpts = track
            color = (0, 255, 255) # Yellow for tracking
            label = f"Cadet ID: {cadet_id}" if cadet_id != -1 else "Cadet"
            
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out_frame, label, (x1, max(y1 - 10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
            if kpts is not None and len(kpts) == 17:
                # Draw joints
                for x, y, kpt_conf in kpts:
                    if kpt_conf > 0.25:
                        cv2.circle(out_frame, (int(x), int(y)), 4, (0, 0, 255), -1)
                        
                # Draw bones
                for j1, j2 in SKELETON:
                    if kpts[j1][2] > 0.25 and kpts[j2][2] > 0.25:
                        pt1 = (int(kpts[j1][0]), int(kpts[j1][1]))
                        pt2 = (int(kpts[j2][0]), int(kpts[j2][1]))
                        cv2.line(out_frame, pt1, pt2, (255, 0, 0), 2)
                        
        return out_frame
