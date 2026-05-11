import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from runtime_config import RUNTIME_CONFIG
from tracking import create_pan_tilt_controller


def _test_config(
    backend: str,
    update_every_n_frames: int,
    invert_pan: bool,
    invert_tilt: bool,
):
    return replace(
        RUNTIME_CONFIG.pan_tilt,
        enabled=True,
        backend=backend,
        update_every_n_frames=update_every_n_frames,
        invert_pan=invert_pan,
        invert_tilt=invert_tilt,
    )


def run_simulation(args) -> None:
    frame_shape = (480, 640, 3)
    face_boxes = [
        (280, 200, 80, 80),  # center
        (80, 200, 80, 80),   # left
        (480, 200, 80, 80),  # right
        (280, 60, 80, 80),   # up
        (280, 340, 80, 80),  # down
        (100, 70, 80, 80),   # up-left
        (460, 330, 80, 80),  # down-right
    ]

    controller = create_pan_tilt_controller(
        _test_config(args.backend, args.update_every, args.invert_pan, args.invert_tilt)
    )
    try:
        for index in range(args.frames):
            bbox = face_boxes[index % len(face_boxes)]
            print(f"[TEST] frame={index + 1} bbox={bbox}")
            controller.update(bbox, frame_shape)
            time.sleep(args.delay)
    finally:
        controller.close()


def run_camera(args) -> None:
    import cv2

    from face_detection.yunet_face_detector import YuNetFaceDetector

    config = RUNTIME_CONFIG
    controller = create_pan_tilt_controller(
        _test_config(args.backend, args.update_every, args.invert_pan, args.invert_tilt)
    )
    detector = YuNetFaceDetector(
        model_path=config.yunet.model_path,
        score_threshold=config.yunet.score_threshold,
        nms_threshold=config.yunet.nms_threshold,
        top_k=config.yunet.top_k,
        max_detection_width=config.yunet.max_detection_width,
    )
    capture = cv2.VideoCapture(config.camera.camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)

    if not capture.isOpened():
        controller.close()
        raise RuntimeError("camera could not be opened")

    if args.show_window:
        cv2.namedWindow("Pan-Tilt Tracking Test")

    try:
        for index in range(args.frames):
            ok, frame = capture.read()
            if not ok or frame is None:
                print("[TEST] camera frame read failed")
                break

            detection = detector.detect_one(frame)
            if detection is None:
                print(f"[TEST] frame={index + 1} no_face")
                if args.show_window:
                    _draw_status(cv2, frame, "no face")
                    if _show_frame_and_should_quit(cv2, frame):
                        break
                time.sleep(args.delay)
                continue

            print(f"[TEST] frame={index + 1} bbox={detection.bbox}")
            controller.update(detection.bbox, frame.shape)
            if args.show_window:
                detector.draw(frame, detection)
                _draw_status(cv2, frame, f"bbox={detection.bbox}")
                if _show_frame_and_should_quit(cv2, frame):
                    break
            time.sleep(args.delay)
    finally:
        capture.release()
        controller.close()
        if args.show_window:
            cv2.destroyAllWindows()


def _draw_status(cv2, frame, text: str) -> None:
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )


def _show_frame_and_should_quit(cv2, frame) -> bool:
    cv2.imshow("Pan-Tilt Tracking Test", frame)
    key = cv2.waitKey(1) & 0xFF
    return key in (ord("q"), 27)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test pan-tilt face tracking without the main app."
    )
    parser.add_argument(
        "--mode",
        choices=("simulate", "camera"),
        default="simulate",
        help="simulate uses fixed bboxes; camera uses only camera + face detection.",
    )
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument(
        "--backend",
        choices=("mock", "arducam"),
        default="mock",
        help="mock is safe on laptop; arducam drives the real Raspberry Pi HAT.",
    )
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--invert-pan", action="store_true")
    parser.add_argument("--invert-tilt", action="store_true")
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="show the camera feed and face bbox in camera mode; quit with q or Esc.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "camera":
        run_camera(args)
    else:
        run_simulation(args)


if __name__ == "__main__":
    main()
