import time
from typing import Optional

from tracking.face_tracking_controller import FaceBbox, FaceTrackingController


class MockPanTiltController:
    def __init__(
        self,
        config,
        active: bool = False,
        disabled_reason: Optional[str] = None,
        log_startup: bool = True,
        backend_label: str = "mock",
    ):
        self.config = config
        self.active = active
        self.disabled_reason = disabled_reason
        self.pan = float(config.pan_start)
        self.tilt = float(config.tilt_start)
        self._frame_count = 0
        self._last_logged_position = (self.pan, self.tilt)
        self._last_log_time = 0.0
        self._tracker = FaceTrackingController(
            dead_zone_x=config.dead_zone_x,
            dead_zone_y=config.dead_zone_y,
            step_degrees=config.step_degrees,
            invert_pan=config.invert_pan,
            invert_tilt=config.invert_tilt,
            error_smoothing_alpha=config.error_smoothing_alpha,
            min_move_updates=config.min_move_updates,
        )

        if not log_startup:
            return

        if self.active:
            print(f"[PAN_TILT] enabled backend={backend_label}")
        else:
            reason = self.disabled_reason or "config_disabled"
            print(f"[PAN_TILT] disabled reason={reason} backend={backend_label}")

    def update(self, face_bbox: Optional[FaceBbox], frame_shape) -> None:
        if not self.active or face_bbox is None:
            return

        if not self._should_update_this_frame():
            return

        tracking_update = self._tracker.update(face_bbox, frame_shape)
        if not tracking_update.should_move:
            return

        previous_pan = self.pan
        previous_tilt = self.tilt
        self.pan = _clamp(
            self.pan + tracking_update.pan_delta,
            self.config.pan_min,
            self.config.pan_max,
        )
        self.tilt = _clamp(
            self.tilt + tracking_update.tilt_delta,
            self.config.tilt_min,
            self.config.tilt_max,
        )

        if self.pan != previous_pan or self.tilt != previous_tilt:
            self._log_update(tracking_update.error_x, tracking_update.error_y)

    def close(self) -> None:
        pass

    def _should_update_this_frame(self) -> bool:
        self._frame_count += 1
        update_every = max(1, int(self.config.update_every_n_frames))
        return self._frame_count % update_every == 0

    def _log_update(self, error_x: float, error_y: float) -> None:
        now = time.monotonic()
        position = (self.pan, self.tilt)
        position_changed = position != self._last_logged_position
        enough_time_elapsed = now - self._last_log_time >= 1.0
        if not position_changed and not enough_time_elapsed:
            return

        print(
            "[PAN_TILT] update",
            f"pan={self.pan:.1f}",
            f"tilt={self.tilt:.1f}",
            f"error_x={error_x:.1f}",
            f"error_y={error_y:.1f}",
        )
        self._last_logged_position = position
        self._last_log_time = now


class ArducamPanTiltController(MockPanTiltController):
    def __init__(self, config):
        super().__init__(
            config,
            active=True,
            log_startup=False,
            backend_label="arducam",
        )
        self._kit = None
        self._pan_servo = None
        self._tilt_servo = None

        try:
            self._initialize_hardware()
            self._apply_servo_angles()
            print("[PAN_TILT] enabled backend=arducam")
        except Exception as exc:
            self.active = False
            print(f"[PAN_TILT] disabled reason=arducam_init_failed backend=arducam error={exc}")

    def update(self, face_bbox: Optional[FaceBbox], frame_shape) -> None:
        if not self.active:
            return

        previous_pan = self.pan
        previous_tilt = self.tilt
        super().update(face_bbox, frame_shape)
        if self.pan != previous_pan or self.tilt != previous_tilt:
            try:
                self._apply_servo_angles()
            except Exception as exc:
                self.active = False
                print(f"[PAN_TILT] disabled reason=arducam_update_failed backend=arducam error={exc}")

    def close(self) -> None:
        self._kit = None
        self._pan_servo = None
        self._tilt_servo = None

    def _initialize_hardware(self) -> None:
        from adafruit_servokit import ServoKit

        self._kit = ServoKit(channels=16)
        self._pan_servo = self._kit.servo[int(self.config.pan_channel)]
        self._tilt_servo = self._kit.servo[int(self.config.tilt_channel)]

    def _apply_servo_angles(self) -> None:
        if self._pan_servo is None or self._tilt_servo is None:
            raise RuntimeError("servo channels are not initialized")

        self._pan_servo.angle = self.pan
        self._tilt_servo.angle = self.tilt


def create_pan_tilt_controller(config):
    if not config.enabled:
        return MockPanTiltController(config, active=False, disabled_reason="config_disabled")

    backend = (config.backend or "mock").lower()
    if backend == "mock":
        return MockPanTiltController(config, active=True)

    if backend == "arducam":
        return ArducamPanTiltController(config)

    return MockPanTiltController(
        config,
        active=False,
        disabled_reason="unknown_backend",
        backend_label=config.backend,
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(float(minimum), min(float(maximum), float(value)))
