import platform
import time
from typing import Optional

import cv2

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


class CameraStream:
    """Thin camera wrapper with OpenCV and Raspberry Pi Picamera2 backends."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        backend: str = "auto",
        warmup_seconds: float = 0.2,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.backend = backend.lower()
        self.warmup_seconds = warmup_seconds
        self.cap = None
        self.picam2: Optional["Picamera2"] = None
        self.active_backend: Optional[str] = None

    def open(self) -> None:
        """Open the configured camera backend."""
        resolved_backend = self._resolve_backend()

        if resolved_backend == "picamera2":
            self._open_picamera2()
            return

        self._open_opencv()

    def read(self):
        if self.active_backend == "picamera2":
            if self.picam2 is None:
                raise RuntimeError("Picamera2 backend is not open.")

            frame = self.picam2.capture_array()
            if frame is None:
                return None

            if len(frame.shape) == 3 and frame.shape[2] == 3:
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if len(frame.shape) == 3 and frame.shape[2] == 4:
                return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            return frame

        if self.cap is None:
            raise RuntimeError("Camera is not open.")

        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        if self.picam2 is not None:
            self.picam2.stop()
            self.picam2.close()
            self.picam2 = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.active_backend = None

    def _resolve_backend(self) -> str:
        if self.backend in {"cv2", "opencv"}:
            return "opencv"

        if self.backend == "picamera2":
            if Picamera2 is None:
                raise RuntimeError(
                    "Picamera2 backend was requested, but the 'picamera2' package is not installed."
                )
            return "picamera2"

        if self.backend != "auto":
            raise ValueError(f"Unknown camera backend: {self.backend}")

        if self._should_use_picamera2():
            return "picamera2"

        return "opencv"

    def _should_use_picamera2(self) -> bool:
        return Picamera2 is not None and platform.system().lower() == "linux"

    def _open_opencv(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera. Check access and device availability.")

        self.active_backend = "opencv"

    def _open_picamera2(self) -> None:
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            buffer_count=4,
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(self.warmup_seconds)
        self.active_backend = "picamera2"
