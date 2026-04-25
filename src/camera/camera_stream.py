import platform
import time
from typing import Optional

import cv2

PICAMERA2_IMPORT_ERROR = None

try:
    from picamera2 import Picamera2
except ImportError as exc:
    Picamera2 = None
    PICAMERA2_IMPORT_ERROR = exc


class CameraStream:
    """Thin camera wrapper with OpenCV and Raspberry Pi Picamera2 backends."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        backend: str = "auto",
        warmup_seconds: float = 0.2,
        center_crop_enabled: bool = False,
        center_crop_scale: float = 1.0,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.backend = backend.lower()
        self.warmup_seconds = warmup_seconds
        self.center_crop_enabled = center_crop_enabled
        self.center_crop_scale = center_crop_scale
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
            self.cap.release()
            self.cap = None

            message = [
                "Cannot open camera through OpenCV.",
                f"Requested index: {self.camera_index}.",
            ]

            if platform.system().lower() == "linux":
                message.append(
                    "On Raspberry Pi CSI cameras, use Picamera2/libcamera instead of cv2.VideoCapture(0)."
                )
                message.append(
                    "Run with RUSTY_CAMERA_BACKEND=picamera2 after installing the Raspberry Pi camera packages."
                )

                if PICAMERA2_IMPORT_ERROR is not None:
                    message.append(
                        f"Picamera2 import failed: {PICAMERA2_IMPORT_ERROR}."
                    )

            raise RuntimeError(" ".join(message))

        self.active_backend = "opencv"

    def _open_picamera2(self) -> None:
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            buffer_count=4,
        )
        self.picam2.configure(config)
        if self.center_crop_enabled:
            self._apply_picamera2_center_crop()
        self.picam2.start()
        time.sleep(self.warmup_seconds)
        self.active_backend = "picamera2"

    def _apply_picamera2_center_crop(self) -> None:
        if self.picam2 is None:
            return

        scale = max(0.1, min(float(self.center_crop_scale), 1.0))
        crop_max = self.picam2.camera_properties.get("ScalerCropMaximum")
        if crop_max is None:
            return

        sensor_x, sensor_y, sensor_width, sensor_height = crop_max
        crop_width = int(round(sensor_width * scale))
        crop_height = int(round(sensor_height * scale))
        crop_x = sensor_x + (sensor_width - crop_width) // 2
        crop_y = sensor_y + (sensor_height - crop_height) // 2

        self.picam2.set_controls(
            {"ScalerCrop": (crop_x, crop_y, crop_width, crop_height)}
        )
