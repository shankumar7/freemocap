import cv2
from military_drill_ai.detection.yolo_detector import YOLODetector
from typing import List, Tuple

class Tracker:
    def __init__(self, detector: YOLODetector):
        """
        Initialize the tracker using YOLOv8's built-in ByteTrack.
        """
        self.detector = detector
        
    def track_frame(self, frame) -> List[Tuple[int, int, int, int, float, int, int]]:
        """
        Runs detection and tracking using ByteTrack.
        Returns a list of tracks: [x1, y1, x2, y2, conf, cls, track_id]
        """
        results = self.detector.model.track(
            frame, 
            conf=self.detector.conf, 
            classes=self.detector.classes, 
            tracker="bytetrack.yaml", 
            persist=True, 
            verbose=False
        )
        
        tracks = []
        for r in results:
            boxes = r.boxes
            if boxes is not None and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    tracks.append((int(x1), int(y1), int(x2), int(y2), conf, cls, track_id))
            elif boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    tracks.append((int(x1), int(y1), int(x2), int(y2), conf, cls, -1))
        return tracks

    def draw_tracks(self, frame, tracks: List[Tuple]):
        """
        Draws bounding boxes and persistent track IDs.
        """
        out_frame = frame.copy()
        for track in tracks:
            x1, y1, x2, y2, conf, cls, track_id = track
            color = (0, 255, 255) # Yellow for tracking
            label = f"Cadet ID: {track_id}" if track_id != -1 else "Cadet (No ID)"
            
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out_frame, label, (x1, max(y1 - 10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return out_frame
