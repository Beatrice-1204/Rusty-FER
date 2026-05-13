from dataclasses import dataclass
from typing import Optional, Tuple


FaceBbox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class FaceTrackingUpdate:
    pan_delta: float
    tilt_delta: float
    error_x: float
    error_y: float

    @property
    def should_move(self) -> bool:
        return self.pan_delta != 0.0 or self.tilt_delta != 0.0


class FaceTrackingController:
    def __init__(
        self,
        dead_zone_x: int,
        dead_zone_y: int,
        step_degrees: float,
        invert_pan: bool = False,
        invert_tilt: bool = False,
        error_smoothing_alpha: float = 0.25,
        min_move_updates: int = 1,
    ):
        self.dead_zone_x = max(0, int(dead_zone_x))
        self.dead_zone_y = max(0, int(dead_zone_y))
        self.step_degrees = abs(float(step_degrees))
        self.invert_pan = bool(invert_pan)
        self.invert_tilt = bool(invert_tilt)
        self.error_smoothing_alpha = _clamp(float(error_smoothing_alpha), 0.0, 1.0)
        self.min_move_updates = max(1, int(min_move_updates))
        self._smoothed_error_x: Optional[float] = None
        self._smoothed_error_y: Optional[float] = None
        self._pan_outside_dead_zone_updates = 0
        self._tilt_outside_dead_zone_updates = 0

    def reset(self) -> None:
        self._smoothed_error_x = None
        self._smoothed_error_y = None
        self._pan_outside_dead_zone_updates = 0
        self._tilt_outside_dead_zone_updates = 0

    def update(
        self,
        face_bbox: Optional[FaceBbox],
        frame_shape,
    ) -> FaceTrackingUpdate:
        if face_bbox is None or frame_shape is None:
            self.reset()
            return FaceTrackingUpdate(0.0, 0.0, 0.0, 0.0)

        frame_height, frame_width = frame_shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            return FaceTrackingUpdate(0.0, 0.0, 0.0, 0.0)

        x, y, width, height = face_bbox
        face_center_x = x + width / 2.0
        face_center_y = y + height / 2.0
        frame_center_x = frame_width / 2.0
        frame_center_y = frame_height / 2.0

        raw_error_x = face_center_x - frame_center_x
        raw_error_y = face_center_y - frame_center_y
        error_x, error_y = self._smooth_errors(raw_error_x, raw_error_y)

        pan_delta = 0.0
        if abs(error_x) > self.dead_zone_x:
            self._pan_outside_dead_zone_updates += 1
            if self._pan_outside_dead_zone_updates >= self.min_move_updates:
                pan_delta = self.step_degrees if error_x > 0 else -self.step_degrees
                if self.invert_pan:
                    pan_delta = -pan_delta
        else:
            self._pan_outside_dead_zone_updates = 0

        tilt_delta = 0.0
        if abs(error_y) > self.dead_zone_y:
            self._tilt_outside_dead_zone_updates += 1
            if self._tilt_outside_dead_zone_updates >= self.min_move_updates:
                tilt_delta = self.step_degrees if error_y > 0 else -self.step_degrees
                if self.invert_tilt:
                    tilt_delta = -tilt_delta
        else:
            self._tilt_outside_dead_zone_updates = 0

        return FaceTrackingUpdate(
            pan_delta=pan_delta,
            tilt_delta=tilt_delta,
            error_x=error_x,
            error_y=error_y,
        )

    def _smooth_errors(self, raw_error_x: float, raw_error_y: float) -> Tuple[float, float]:
        if self._smoothed_error_x is None or self._smoothed_error_y is None:
            self._smoothed_error_x = raw_error_x
            self._smoothed_error_y = raw_error_y
            return raw_error_x, raw_error_y

        alpha = self.error_smoothing_alpha
        self._smoothed_error_x = (
            alpha * raw_error_x + (1.0 - alpha) * self._smoothed_error_x
        )
        self._smoothed_error_y = (
            alpha * raw_error_y + (1.0 - alpha) * self._smoothed_error_y
        )
        return self._smoothed_error_x, self._smoothed_error_y


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
