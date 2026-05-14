import sys
import os
import argparse
from PySide6.QtWidgets import QApplication

# Add the parent directory to sys.path so we can import military_drill_ai
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from military_drill_ai.gui.main_window import MainWindow

def run_pipeline(video_source: str = "0"):
    """
    Test Pipeline
    """
    app = QApplication(sys.argv)
    
    window = MainWindow()
    # Set the initial video source from command line
    window.control_panel.source_input.setText(video_source)
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Military Drill AI Pipeline")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or path to video file)")
    args = parser.parse_args()
    
    run_pipeline(args.source)
