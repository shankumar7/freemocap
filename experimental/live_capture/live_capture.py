#!/usr/bin/env python3
"""PoC live capture script for FreeMoCap.

Usage:
    python experimental/live_capture/live_capture.py --camera 0

This script captures from a webcam, runs MediaPipe Pose when available,
draws landmarks, and overlays FPS and latency. It is intended as a minimal
proof-of-concept for live motion capture integration.
"""
from __future__ import annotations
import time
import argparse
import sys
import cv2

try:
    import mediapipe as mp  # type: ignore
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False


def main(camera_index: int = 0, width: int = 640, height: int = 480) -> int:
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        return 1

    pose = None
    if HAS_MEDIAPIPE:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        print("MediaPipe detected: using MediaPipe Pose for inference")
    else:
        print("MediaPipe not available: running camera preview only")

    prev_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame from camera")
                break

            t0 = time.time()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if HAS_MEDIAPIPE and pose is not None:
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )
            else:
                # ensure BGR for display
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            now = time.time()
            fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
            prev_time = now
            latency_ms = (time.time() - t0) * 1000.0

            overlay_text = f"FPS: {fps:.1f}  Latency: {latency_ms:.1f} ms"
            cv2.putText(image, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("FreeMoCap Live PoC", image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    finally:
        if pose is not None:
            pose.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="live_capture")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--width", type=int, default=640, help="Capture width")
    p.add_argument("--height", type=int, default=480, help="Capture height")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(camera_index=args.camera, width=args.width, height=args.height))
