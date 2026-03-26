from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]


@dataclass
class YuNetFaceDetection:
    """
    Compact, detector-agnostic face detection result.

    Attributes:
        bbox: (x, y, w, h) face bounding box in image pixels.
        left_eye: Left eye center in image pixels.
        right_eye: Right eye center in image pixels.
        nose: Nose tip in image pixels.
        mouth_left: Left mouth corner in image pixels.
        mouth_right: Right mouth corner in image pixels.
        score: Detector confidence score.
    """

    bbox: Tuple[int, int, int, int]
    left_eye: Point
    right_eye: Point
    nose: Point
    mouth_left: Point
    mouth_right: Point
    score: float


class YuNetFaceDetector:
    """
    OpenCV YuNet-based face detector.

    This wrapper keeps the detector output explicit and stable so the rest of
    the project can later consume face boxes and keypoints without depending on
    MediaPipe-specific result objects.
    """

    def __init__(
        self,
        model_path: str = "models/yunet/face_detection_yunet.onnx",
        score_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 1,
        max_detection_width: int = 320,
    ):
        self.model_path = model_path
        self.max_detection_width = max_detection_width
        self.detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )

    def detect(self, frame_bgr: np.ndarray) -> List[YuNetFaceDetection]:
        """
        Detect faces in a BGR frame.

        Returns a list of YuNetFaceDetection objects sorted by detector score,
        highest first.
        """
        if frame_bgr is None:
            return []

        height, width = frame_bgr.shape[:2]
        if height == 0 or width == 0:
            return []

        detection_frame, scale_x, scale_y = self._prepare_detection_frame(frame_bgr)
        det_height, det_width = detection_frame.shape[:2]

        self.detector.setInputSize((det_width, det_height))
        _, detections = self.detector.detect(detection_frame)

        if detections is None or len(detections) == 0:
            return []

        results = [self._parse_detection(row, scale_x, scale_y) for row in detections]
        results.sort(key=lambda detection: detection.score, reverse=True)
        return results

    def detect_one(self, frame_bgr: np.ndarray) -> Optional[YuNetFaceDetection]:
        """
        Detect the highest-confidence face in a BGR frame.
        """
        detections = self.detect(frame_bgr)
        return detections[0] if detections else None

    def draw(self, frame_bgr: np.ndarray, detection: Optional[YuNetFaceDetection]) -> None:
        """
        Draw a simple debug overlay for one detection.
        """
        if frame_bgr is None or detection is None:
            return

        x, y, w, h = detection.bbox
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for point in [
            detection.left_eye,
            detection.right_eye,
            detection.nose,
            detection.mouth_left,
            detection.mouth_right,
        ]:
            cv2.circle(frame_bgr, point, 2, (0, 255, 255), -1)

    def _prepare_detection_frame(self, frame_bgr: np.ndarray):
        height, width = frame_bgr.shape[:2]

        if self.max_detection_width <= 0 or width <= self.max_detection_width:
            return frame_bgr, 1.0, 1.0

        scale = self.max_detection_width / float(width)
        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))

        detection_frame = cv2.resize(
            frame_bgr,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR
        )

        scale_x = width / float(resized_width)
        scale_y = height / float(resized_height)
        return detection_frame, scale_x, scale_y

    def _parse_detection(self, row: np.ndarray, scale_x: float, scale_y: float) -> YuNetFaceDetection:
        values = row.tolist()

        x = int(round(values[0] * scale_x))
        y = int(round(values[1] * scale_y))
        w = int(round(values[2] * scale_x))
        h = int(round(values[3] * scale_y))
        left_eye = self._to_point(values[4:6], scale_x, scale_y)
        right_eye = self._to_point(values[6:8], scale_x, scale_y)
        nose = self._to_point(values[8:10], scale_x, scale_y)
        mouth_left = self._to_point(values[10:12], scale_x, scale_y)
        mouth_right = self._to_point(values[12:14], scale_x, scale_y)
        score = float(values[14])

        return YuNetFaceDetection(
            bbox=(x, y, w, h),
            left_eye=left_eye,
            right_eye=right_eye,
            nose=nose,
            mouth_left=mouth_left,
            mouth_right=mouth_right,
            score=score,
        )

    @staticmethod
    def _to_point(values, scale_x: float, scale_y: float) -> Point:
        return (
            int(round(values[0] * scale_x)),
            int(round(values[1] * scale_y)),
        )
