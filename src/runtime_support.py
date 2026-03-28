from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

from face_detection.yunet_face_detector import YuNetFaceDetection


@dataclass
class FaceQualityResult:
    is_valid: bool
    reason: Optional[str] = None


class DetectionSmoother:
    def __init__(self, alpha: float, max_center_shift_ratio: float, min_size_ratio: float):
        self.alpha = alpha
        self.max_center_shift_ratio = max_center_shift_ratio
        self.min_size_ratio = min_size_ratio
        self._previous: Optional[YuNetFaceDetection] = None

    def smooth(self, detection: Optional[YuNetFaceDetection]) -> Optional[YuNetFaceDetection]:
        if detection is None:
            self._previous = None
            return None

        if self._previous is None or not self._is_continuous(self._previous, detection):
            self._previous = detection
            return detection

        prev_x, prev_y, prev_w, prev_h = self._previous.bbox
        cur_x, cur_y, cur_w, cur_h = detection.bbox

        smoothed_bbox = (
            self._smooth_value(prev_x, cur_x),
            self._smooth_value(prev_y, cur_y),
            self._smooth_value(prev_w, cur_w),
            self._smooth_value(prev_h, cur_h),
        )

        smoothed = YuNetFaceDetection(
            bbox=smoothed_bbox,
            left_eye=detection.left_eye,
            right_eye=detection.right_eye,
            nose=detection.nose,
            mouth_left=detection.mouth_left,
            mouth_right=detection.mouth_right,
            score=detection.score,
        )
        self._previous = smoothed
        return smoothed

    def reset(self) -> None:
        self._previous = None

    def _is_continuous(self, previous: YuNetFaceDetection, current: YuNetFaceDetection) -> bool:
        prev_x, prev_y, prev_w, prev_h = previous.bbox
        cur_x, cur_y, cur_w, cur_h = current.bbox

        prev_center_x = prev_x + prev_w / 2.0
        prev_center_y = prev_y + prev_h / 2.0
        cur_center_x = cur_x + cur_w / 2.0
        cur_center_y = cur_y + cur_h / 2.0

        dx = abs(cur_center_x - prev_center_x)
        dy = abs(cur_center_y - prev_center_y)
        reference = max(prev_w, prev_h, cur_w, cur_h, 1)
        size_ratio = min(prev_w, cur_w) / max(prev_w, cur_w, 1)

        return (
            dx <= reference * self.max_center_shift_ratio
            and dy <= reference * self.max_center_shift_ratio
            and size_ratio >= self.min_size_ratio
        )

    def _smooth_value(self, previous: int, current: int) -> int:
        return int(round(previous + self.alpha * (current - previous)))


class FaceQualityValidator:
    def __init__(
        self,
        min_face_width_px: int,
        min_face_height_px: int,
        min_border_margin_px: int,
        require_keypoints: bool,
        min_eye_distance_ratio: float,
        max_eye_y_diff_ratio: float,
    ):
        self.min_face_width_px = min_face_width_px
        self.min_face_height_px = min_face_height_px
        self.min_border_margin_px = min_border_margin_px
        self.require_keypoints = require_keypoints
        self.min_eye_distance_ratio = min_eye_distance_ratio
        self.max_eye_y_diff_ratio = max_eye_y_diff_ratio

    def validate(self, frame_shape, detection: Optional[YuNetFaceDetection]) -> FaceQualityResult:
        if detection is None:
            return FaceQualityResult(False, "no_face")

        frame_h, frame_w = frame_shape[:2]
        x, y, w, h = detection.bbox

        if w < self.min_face_width_px or h < self.min_face_height_px:
            return FaceQualityResult(False, "face_too_small")

        if (
            x < self.min_border_margin_px
            or y < self.min_border_margin_px
            or x + w > frame_w - self.min_border_margin_px
            or y + h > frame_h - self.min_border_margin_px
        ):
            return FaceQualityResult(False, "face_near_border")

        if not self.require_keypoints:
            return FaceQualityResult(True)

        keypoints = [
            detection.left_eye,
            detection.right_eye,
            detection.nose,
            detection.mouth_left,
            detection.mouth_right,
        ]
        for point_x, point_y in keypoints:
            if point_x < 0 or point_y < 0 or point_x >= frame_w or point_y >= frame_h:
                return FaceQualityResult(False, "keypoint_out_of_frame")

        eye_dx = abs(detection.right_eye[0] - detection.left_eye[0])
        eye_dy = abs(detection.right_eye[1] - detection.left_eye[1])
        if eye_dx < w * self.min_eye_distance_ratio:
            return FaceQualityResult(False, "eye_distance_too_small")
        if eye_dy > h * self.max_eye_y_diff_ratio:
            return FaceQualityResult(False, "eye_line_too_steep")

        return FaceQualityResult(True)


class PerfTracker:
    STAGES = (
        "capture",
        "detect",
        "post_detect",
        "preprocess",
        "predict",
        "total",
    )

    def __init__(self, fps_window_size: int = 30):
        self.frame_count = 0
        self.totals: Dict[str, float] = {stage: 0.0 for stage in self.STAGES}
        self.recent_total_ms = deque(maxlen=fps_window_size)

    def record(self, stage_times: Dict[str, float]) -> None:
        self.frame_count += 1
        for stage in self.STAGES:
            self.totals[stage] += stage_times.get(stage, 0.0)
        self.recent_total_ms.append(stage_times.get("total", 0.0) * 1000.0)

    def summary(self) -> Dict[str, float]:
        frames = max(self.frame_count, 1)
        avg_total_ms = self.totals["total"] / frames * 1000.0
        rolling_total_ms = (
            sum(self.recent_total_ms) / len(self.recent_total_ms)
            if self.recent_total_ms
            else avg_total_ms
        )

        return {
            "frames": float(self.frame_count),
            "capture_ms": self.totals["capture"] / frames * 1000.0,
            "detect_ms": self.totals["detect"] / frames * 1000.0,
            "post_detect_ms": self.totals["post_detect"] / frames * 1000.0,
            "preprocess_ms": self.totals["preprocess"] / frames * 1000.0,
            "predict_ms": self.totals["predict"] / frames * 1000.0,
            "total_ms": avg_total_ms,
            "fps_avg": 1000.0 / avg_total_ms if avg_total_ms > 0 else 0.0,
            "fps_inst": 1000.0 / rolling_total_ms if rolling_total_ms > 0 else 0.0,
        }
