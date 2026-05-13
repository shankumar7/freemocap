import cv2
from ultralytics import YOLO
from typing import List, Tuple

class YOLODetector:
    def __init__(self, model_path: str, conf: float = 0.5, classes: List[int] = None):
        """
        Initialize the YOLOv8 detector.
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.classes = classes if classes is not None else [0]
    
    def detect(self, frame) -> List[Tuple[int, int, int, int, float, int]]:
        """
        Runs YOLOv8 detection on a frame.
        Returns a list of detections: [x1, y1, x2, y2, conf, cls]
        """
        results = self.model(frame, conf=self.conf, classes=self.classes, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                detections.append((int(x1), int(y1), int(x2), int(y2), conf, cls))
        return detections

    def draw_detections(self, frame, detections: List[Tuple[int, int, int, int, float, int]]):
        """
        Draws bounding boxes on the frame.
        """
        out_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out_frame, f"Person {conf:.2f}", (x1, max(y1 - 10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return out_frame
