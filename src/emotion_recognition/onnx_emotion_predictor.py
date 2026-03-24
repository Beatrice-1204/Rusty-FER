from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class OnnxEmotionPredictor:
    """
    ONNX Runtime predictor for the FERPlus emotion model.
    Preserves the raw 8-class FERPlus label and exposes a mapped 5-class label
    for comparison with the existing TensorFlow path.
    """

    RAW_EMOTION_LABELS = [
        "neutral",
        "happiness",
        "surprise",
        "sadness",
        "anger",
        "disgust",
        "fear",
        "contempt",
    ]

    MAPPED_EMOTION_LABELS = {
        "neutral": "neutral",
        "happiness": "happy",
        "surprise": "surprise",
        "sadness": "sad",
        "anger": "angry",
        "disgust": "neutral",
        "fear": "sad",
        "contempt": "neutral",
    }

    def __init__(self, model_path="models/onnx/emotion-ferplus-7.onnx", debug_dir="debug/onnx"):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.debug_save_limit = 5
        self.debug_save_count = 0

        self.last_raw_label = None
        self.last_mapped_label = None
        self.last_confidence = None
        self.last_top3 = []
        self.last_scores = None
        self.last_probabilities = None

    def _softmax(self, scores):
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        return exp_scores / np.sum(exp_scores)

    def _save_debug_image(self, filename, image):
        if self.debug_save_count >= self.debug_save_limit:
            return
        cv2.imwrite(str(self.debug_dir / filename), image)

    def _tighten_face_roi(self, face_roi):
        """
        Tighten the shared face ROI for the ONNX path only.
        The shared landmark box is fairly loose, so we crop to a slightly
        smaller square centered on the face with a small upward bias.
        """
        h, w = face_roi.shape[:2]
        if h == 0 or w == 0:
            return face_roi

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

        return face_roi[y1:y2, x1:x2]

    def _align_face_roi(self, face_roi, result):
        """
        Apply a small ONNX-only roll correction using MediaPipe eye landmarks.
        """
        if face_roi is None or result is None or not result.multi_face_landmarks:
            return face_roi

        h, w = face_roi.shape[:2]
        if h == 0 or w == 0:
            return face_roi

        face_landmarks = result.multi_face_landmarks[0]
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]

        left_eye_px = (int(left_eye.x * w), int(left_eye.y * h))
        right_eye_px = (int(right_eye.x * w), int(right_eye.y * h))

        dx = right_eye_px[0] - left_eye_px[0]
        dy = right_eye_px[1] - left_eye_px[1]
        angle_deg = np.degrees(np.arctan2(dy, dx))

        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

        return cv2.warpAffine(
            face_roi,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

    def _prepare_from_roi(self, face_roi, result=None, use_equalization=True):
        """
        Prepare ONNX input directly from the detected face ROI.
        The FERPlus ONNX model expects grayscale 64x64 in NCHW layout.
        This path keeps pixel values in the 0..255 range as float32.
        """
        if face_roi is None:
            return None, None, None, None, None

        tight_roi = self._tighten_face_roi(face_roi)
        aligned_roi = self._align_face_roi(tight_roi, result)

        gray = cv2.cvtColor(aligned_roi, cv2.COLOR_BGR2GRAY)
        if use_equalization:
            gray = cv2.equalizeHist(gray)

        processed_48 = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_LINEAR)
        onnx_input_64 = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_LINEAR)
        nchw = np.expand_dims(np.expand_dims(onnx_input_64.astype(np.float32), axis=0), axis=0)

        return nchw, processed_48, onnx_input_64, tight_roi, aligned_roi

    def _log_top3(self, probabilities):
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top3 = [
            (self.RAW_EMOTION_LABELS[idx], float(probabilities[idx]))
            for idx in top_indices
        ]
        self.last_top3 = top3
        top3_text = ", ".join(f"{label}={score:.4f}" for label, score in top3)
        print(f"[ONNX] top3: {top3_text}")

    def predict(self, face_roi, processed_face=None, result=None, use_equalization=True):
        """
        Returns:
            - mapped 5-class label
            - confidence

        Raw ONNX label is preserved in self.last_raw_label.
        """
        input_tensor, processed_48, onnx_input_64, tight_roi, aligned_roi = self._prepare_from_roi(
            face_roi,
            result=result,
            use_equalization=use_equalization
        )
        if input_tensor is None:
            return None, None

        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        scores = outputs[0]
        probabilities = self._softmax(scores)

        class_index = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))

        raw_label = self.RAW_EMOTION_LABELS[class_index]
        mapped_label = self.MAPPED_EMOTION_LABELS[raw_label]

        self.last_scores = scores.tolist()
        self.last_probabilities = probabilities.tolist()
        self.last_raw_label = raw_label
        self.last_mapped_label = mapped_label
        self.last_confidence = confidence
        self._log_top3(probabilities)

        if self.debug_save_count < self.debug_save_limit:
            self._save_debug_image(f"sample_{self.debug_save_count}_roi.png", face_roi)
            self._save_debug_image(f"sample_{self.debug_save_count}_roi_tight.png", tight_roi)
            self._save_debug_image(f"sample_{self.debug_save_count}_roi_aligned.png", aligned_roi)
            if processed_face is not None:
                processed_face_2d = (processed_face[0, :, :, 0] * 255.0).clip(0, 255).astype(np.uint8)
                self._save_debug_image(
                    f"sample_{self.debug_save_count}_processed_48_from_tf.png",
                    processed_face_2d
                )
            self._save_debug_image(f"sample_{self.debug_save_count}_processed_48_for_onnx.png", processed_48)
            self._save_debug_image(f"sample_{self.debug_save_count}_onnx_input_64.png", onnx_input_64)
            self.debug_save_count += 1

        return mapped_label, confidence
