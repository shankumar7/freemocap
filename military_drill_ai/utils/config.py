from pydantic import BaseModel

class Config(BaseModel):
    # Detection
    YOLO_MODEL_PATH: str = "yolov8m.pt"
    DETECTION_CONFIDENCE: float = 0.5
    DETECTION_CLASSES: list[int] = [0]  # Only detect person (class 0 in COCO)
    
    # Analytics Calibration
    # To calibrate using a 15cm scale:
    # 1. Hold the 15cm scale at the exact distance the cadet will stand.
    # 2. Count how many pixels it takes up on screen (e.g., let's say 100 pixels).
    # 3. PIXELS_PER_FOOT = (100 pixels / 15 cm) * 30.48 cm
    # Example: If 15cm = 100px, then PIXELS_PER_FOOT = 203.2
    PIXELS_PER_FOOT: float = 203.2
    
    # Paths
    VIDEO_OUTPUT_PATH: str = "output.mp4"

config = Config()
