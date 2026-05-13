from pydantic import BaseModel

class Config(BaseModel):
    # Detection
    YOLO_MODEL_PATH: str = "yolov8m.pt"
    DETECTION_CONFIDENCE: float = 0.5
    DETECTION_CLASSES: list[int] = [0]  # Only detect person (class 0 in COCO)
    
    # Analytics Calibration
    # The 3D world ratio anchor that locks 2D scale to depth-invariant 3D tracking.
    WORLD_TO_REAL_RATIO: float = 0.0
    PIXELS_PER_FOOT: float = 203.2
    
    # Paths
    VIDEO_OUTPUT_PATH: str = "output.mp4"

config = Config()
