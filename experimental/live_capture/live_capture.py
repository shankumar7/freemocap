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


def main(camera_index: int = 0, width: int = 640, height: int = 480, backend: str = "holistic") -> int:
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        return 1

    # Choose backend: 'pose' (Pose only) or 'holistic' (body + hands + face)
    worker = None
    mp_drawing = None
    mp_drawing_styles = None
    if HAS_MEDIAPIPE:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        if backend == "holistic":
            mp_holistic = mp.solutions.holistic
            worker = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                refine_face_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            print("MediaPipe detected: using Holistic (body+hands+face)")
        else:
            mp_pose = mp.solutions.pose
            worker = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            print("MediaPipe detected: using Pose (body only)")
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

            if HAS_MEDIAPIPE and worker is not None:
                image.flags.writeable = False
                # Holistic returns multiple landmark groups
                results = worker.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw landmarks depending on backend
                try:
                    if hasattr(results, "pose_landmarks") and results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            results.pose_landmarks,
                            mp.solutions.pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                        )

                    if hasattr(results, "left_hand_landmarks") and results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            results.left_hand_landmarks,
                            mp.solutions.holistic.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        )

                    if hasattr(results, "right_hand_landmarks") and results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            results.right_hand_landmarks,
                            mp.solutions.holistic.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        )

                    if hasattr(results, "face_landmarks") and results.face_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            results.face_landmarks,
                            mp.solutions.holistic.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                        )
                except Exception:
                    # Fallback: if drawing styles or connections not available for pose-only backend
                    if hasattr(results, "pose_landmarks") and results.pose_landmarks:
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
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
        if worker is not None:
            worker.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="live_capture")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--width", type=int, default=640, help="Capture width")
    p.add_argument("--height", type=int, default=480, help="Capture height")
    p.add_argument(
        "--backend",
        choices=["holistic", "pose"],
        default="holistic",
        help="Which MediaPipe backend to use: holistic (body+hands+face) or pose (body only)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(camera_index=args.camera, width=args.width, height=args.height, backend=args.backend))
