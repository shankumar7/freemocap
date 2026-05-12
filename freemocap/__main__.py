"""Package entry point for FreeMoCap.

Default behavior launches the Qt GUI.
Use --live-capture to start the MediaPipe live-capture proof of concept.
"""
import argparse
import sys
from multiprocessing import freeze_support
from pathlib import Path

try:
    from freemocap.gui.qt.freemocap_main import qt_gui_main
except Exception:
    base_package_path = Path(__file__).parent.parent
    print(f"adding base_package_path: {base_package_path} : to sys.path")
    sys.path.insert(0, str(base_package_path))  # add parent directory to sys.path
    from freemocap.gui.qt.freemocap_main import qt_gui_main


def parse_args():
    parser = argparse.ArgumentParser(prog="freemocap")
    parser.add_argument("--live-capture", action="store_true", help="Start the live capture proof-of-concept instead of the GUI")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for live capture")
    parser.add_argument("--width", type=int, default=640, help="Capture width for live capture")
    parser.add_argument("--height", type=int, default=480, help="Capture height for live capture")
    parser.add_argument(
        "--backend",
        choices=["holistic", "pose"],
        default="holistic",
        help="MediaPipe backend for live capture",
    )
    return parser.parse_args()


def main():
    # set up so you can change the taskbar icon - https://stackoverflow.com/a/74531530/14662833
    import ctypes
    import freemocap
    args = parse_args()

    if args.live_capture:
        from freemocap.live_capture import run

        return run(camera_index=args.camera, width=args.width, height=args.height, backend=args.backend)

    if sys.platform == "win32":
        myappid = f"{freemocap.__package_name__}_{freemocap.__version__}"  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    qt_gui_main()


if __name__ == "__main__":
    freeze_support()
    print(f"Running `freemocap.__main__` from - {__file__}")

    main()
