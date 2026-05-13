from pydantic import BaseModel

class Config(BaseModel):
    # Detection & Pose
    YOLO_MODEL_PATH: str = "yolov8m-pose.pt"
    DETECTION_CONFIDENCE: float = 0.5
    DETECTION_CLASSES: list[int] = [0]  # Only detect person (class 0 in COCO)
    
    # Paths
    VIDEO_OUTPUT_PATH: str = "output.mp4"

config = Config()
