import tensorflow as tf
import numpy as np


class EmotionPredictor:
    """
    - încărcarea modelului antrenat
    - realizarea predicției emoției
    """

    def __init__(self, model_path="models/emotion_model.h5"):
        self.model = tf.keras.models.load_model(model_path)  #incarcarea modelului

        self.emotion_labels = [
            "angry",
            "happy",
            "neutral",
            "sad",
            "surprise"
        ]

    def predict(self, processed_face):
        """
        Primește imagine procesată (48x48 grayscale)
        Returnează:
            - label emoție
            - probabilitate
        """

        if processed_face is None:
            return None, None

        predictions = self.model.predict(processed_face, verbose=0) #forward propagation
        class_index = np.argmax(predictions)  #pozitia celei mai mari valori
        confidence = float(np.max(predictions))  #probabilitatea ea mai marte

        return self.emotion_labels[class_index], confidence
