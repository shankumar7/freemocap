from pydantic import BaseModel

class Config(BaseModel):
    # Detection
    YOLO_MODEL_PATH: str = "yolov8m.pt"
    DETECTION_CONFIDENCE: float = 0.5
    DETECTION_CLASSES: list[int] = [0]  # Only detect person (class 0 in COCO)
    
    # Analytics Calibration
    # Adjust this value based on your camera's distance to the cadets.
    # For example, if a 6-foot person takes up 600 pixels on screen, this should be 100.
    PIXELS_PER_FOOT: float = 110.0
    
    # Paths
    VIDEO_OUTPUT_PATH: str = "output.mp4"

config = Config()
