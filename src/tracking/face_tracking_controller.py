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
    ):
        self.dead_zone_x = max(0, int(dead_zone_x))
        self.dead_zone_y = max(0, int(dead_zone_y))
        self.step_degrees = abs(float(step_degrees))
        self.invert_pan = bool(invert_pan)
        self.invert_tilt = bool(invert_tilt)

    def update(
        self,
        face_bbox: Optional[FaceBbox],
        frame_shape,
    ) -> FaceTrackingUpdate:
        if face_bbox is None or frame_shape is None:
            return FaceTrackingUpdate(0.0, 0.0, 0.0, 0.0)

        frame_height, frame_width = frame_shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            return FaceTrackingUpdate(0.0, 0.0, 0.0, 0.0)

        x, y, width, height = face_bbox
        face_center_x = x + width / 2.0
        face_center_y = y + height / 2.0
        frame_center_x = frame_width / 2.0
        frame_center_y = frame_height / 2.0

        error_x = face_center_x - frame_center_x
        error_y = face_center_y - frame_center_y

        pan_delta = 0.0
        if abs(error_x) > self.dead_zone_x:
            pan_delta = self.step_degrees if error_x > 0 else -self.step_degrees
            if self.invert_pan:
                pan_delta = -pan_delta

        tilt_delta = 0.0
        if abs(error_y) > self.dead_zone_y:
            tilt_delta = self.step_degrees if error_y > 0 else -self.step_degrees
            if self.invert_tilt:
                tilt_delta = -tilt_delta

        return FaceTrackingUpdate(
            pan_delta=pan_delta,
            tilt_delta=tilt_delta,
            error_x=error_x,
            error_y=error_y,
        )
