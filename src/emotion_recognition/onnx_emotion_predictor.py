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

    def _prepare_from_roi(self, face_roi, use_equalization=True):
        """
        Prepare ONNX input directly from the already prepared face ROI.
        The FERPlus ONNX model expects grayscale 64x64 in NCHW layout.
        This path keeps pixel values in the 0..255 range as float32.
        """
        if face_roi is None:
            return None, None, None, None, None

        prepared_roi = face_roi
        gray = cv2.cvtColor(prepared_roi, cv2.COLOR_BGR2GRAY)
        if use_equalization:
            gray = cv2.equalizeHist(gray)

        processed_48 = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_LINEAR)
        onnx_input_64 = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_LINEAR)
        nchw = np.expand_dims(np.expand_dims(onnx_input_64.astype(np.float32), axis=0), axis=0)

        return nchw, processed_48, onnx_input_64, prepared_roi, prepared_roi

    def _log_top3(self, probabilities):
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top3 = [
            (self.RAW_EMOTION_LABELS[idx], float(probabilities[idx]))
            for idx in top_indices
        ]
        self.last_top3 = top3
        top3_text = ", ".join(f"{label}={score:.4f}" for label, score in top3)
        print(f"[ONNX] top3: {top3_text}")

    def predict(self, face_roi, processed_face=None, use_equalization=True):
        """
        Returns:
            - mapped 5-class label
            - confidence

        Raw ONNX label is preserved in self.last_raw_label.
        """
        input_tensor, processed_48, onnx_input_64, tight_roi, aligned_roi = self._prepare_from_roi(
            face_roi,
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
