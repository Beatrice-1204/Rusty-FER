import cv2
import numpy as np


class ImagePreprocessor:
    """
    - extragerea regiunii fetei (ROI)
    - aliniere simpla pe ochi
    - conv grayscale
    - redimensionare la 48x48
    - normalizare intensitate
    - histogram equalization
    """

    def __init__(self, target_size=48):
        self.target_size = target_size

    def extract_face_roi(self, frame_bgr, detection, margin=20):
        """
        Extrage bounding box-ul fetei din detectia curenta.
        """
        if frame_bgr is None or detection is None:
            return None

        h, w, _ = frame_bgr.shape
        x, y, box_w, box_h = detection.bbox

        x_min = max(0, x - margin)
        y_min = max(0, y - margin)
        x_max = min(w, x + box_w + margin)
        y_max = min(h, y + box_h + margin)

        if x_max <= x_min or y_max <= y_min:
            return None

        return frame_bgr[y_min:y_max, x_min:x_max]

    def align_face_roi(self, frame_bgr, detection, margin=20):
        """
        Extrage ROI-ul si aplica o corectie simpla de rotatie folosind ochii.
        """
        face_roi = self.extract_face_roi(frame_bgr, detection, margin=margin)
        if face_roi is None:
            return None

        left_eye = getattr(detection, "left_eye", None)
        right_eye = getattr(detection, "right_eye", None)
        if left_eye is None or right_eye is None:
            return face_roi

        x, y, _, _ = detection.bbox
        crop_x = max(0, x - margin)
        crop_y = max(0, y - margin)

        left_eye_px = (left_eye[0] - crop_x, left_eye[1] - crop_y)
        right_eye_px = (right_eye[0] - crop_x, right_eye[1] - crop_y)

        dx = right_eye_px[0] - left_eye_px[0]
        dy = right_eye_px[1] - left_eye_px[1]
        if dx == 0 and dy == 0:
            return face_roi

        angle_deg = np.degrees(np.arctan2(dy, dx))
        center = (face_roi.shape[1] // 2, face_roi.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

        return cv2.warpAffine(
            face_roi,
            rotation_matrix,
            (face_roi.shape[1], face_roi.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

    def prepare_face_roi(self, frame_bgr, detection, margin=20):
        """
        Pipeline scurt pentru runtime:
        extragere ROI -> aliniere pe ochi -> mic crop intern pentru model.
        """
        aligned_roi = self.align_face_roi(frame_bgr, detection, margin=margin)
        if aligned_roi is None:
            return None

        h, w = aligned_roi.shape[:2]
        if h == 0 or w == 0:
            return aligned_roi

        side = int(min(h, w) * 0.9)
        side = max(side, 1)

        center_x = w // 2
        center_y = int(h * 0.46)

        x1 = max(0, center_x - side // 2)
        y1 = max(0, center_y - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)

        if x2 - x1 < side:
            x1 = max(0, x2 - side)
        if y2 - y1 < side:
            y1 = max(0, y2 - side)

        return aligned_roi[y1:y2, x1:x2]

    def preprocess(self, face_roi, debug=False, use_equalization=True):
        """
        Pasii de procesare:
        - grayscale
        - histogram equalization
        - resize
        - normalizare
        - reshape pentru CNN
        """

        if face_roi is None:
            return None if not debug else (None, None)

        debug_data = {}

        debug_data["roi"] = face_roi.copy()

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        debug_data["gray"] = gray.copy()

        if use_equalization:
            equalized = cv2.equalizeHist(gray)
        else:
            equalized = gray.copy()

        debug_data["equalized"] = equalized.copy()

        resized = cv2.resize(equalized, (self.target_size, self.target_size))
        debug_data["resized_48"] = resized.copy()

        normalized = resized / 255.0
        normalized = np.expand_dims(normalized, axis=-1)
        normalized = np.expand_dims(normalized, axis=0)

        if debug:
            return normalized, debug_data

        return normalized
