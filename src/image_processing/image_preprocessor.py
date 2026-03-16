import cv2
import numpy as np


class ImagePreprocessor:
    """
    - extragerea regiunii feței (ROI)
    - conv grayscale
    - redimensionare la 48x48
    - normalizare intensitate
    - histogram equalization
    """

    def __init__(self, target_size=48):
        self.target_size = target_size

    def extract_face_roi(self, frame_bgr, result):
        """
        Extrage bounding box-ul feței folosind landmarkurile MediaPipe.
        """

        if result is None or not result.multi_face_landmarks:
            return None

        h, w, _ = frame_bgr.shape
        face_landmarks = result.multi_face_landmarks[0]

        xs = []
        ys = []

        for lm in face_landmarks.landmark:
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # margine de siguranță
        margin = 20
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)

        roi = frame_bgr[y_min:y_max, x_min:x_max]

        return roi

    def preprocess(self, face_roi, debug=False, use_equalization=True):
        """
        pașii de procesare:
        - grayscale
        - histogram equalization
        - resize
        - normalizare
        - reshape pentru CNN

        """

        if face_roi is None:
            return None if not debug else (None, None)

        debug_data = {}

        # 1. ROI original
        debug_data["roi"] = face_roi.copy()

        # 2. grayscale
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        debug_data["gray"] = gray.copy()

        # 3. histogram equalization
        if use_equalization:
            equalized = cv2.equalizeHist(gray)
        else:
            equalized = gray.copy()

        debug_data["equalized"] = equalized.copy()

        # 4. resize la target_size (48)
        resized = cv2.resize(equalized, (self.target_size, self.target_size))
        debug_data["resized_48"] = resized.copy()

        # 5. normalizare [0,1]
        normalized = resized / 255.0

        # 6. reshape pentru CNN
        normalized = np.expand_dims(normalized, axis=-1)
        normalized = np.expand_dims(normalized, axis=0)

        if debug:
            return normalized, debug_data
        else:
            return normalized

