import cv2

from camera.camera_stream import CameraStream
from face_detection.MP_face_processor import MediaPipeFaceProcessor
from image_processing.image_preprocessor import ImagePreprocessor
from emotion_recognition.emotion_predictor import EmotionPredictor
import tensorflow as tf
from image_processing.face_aligner import FaceAligner
from image_processing.image_preprocessor import ImagePreprocessor
from collections import deque
import numpy as np

DEBUG_PIPELINE = False


def main():
    global DEBUG_PIPELINE

    # ==========================
    # INITIALIZARE MODULE
    # ==========================

    camera = CameraStream(camera_index=0)
    camera.open()

    face_processor = MediaPipeFaceProcessor(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    preprocessor = ImagePreprocessor(target_size=48)
    predictor = EmotionPredictor(model_path="models/emotion_model.h5")
    emotion_buffer = deque(maxlen=5)

    cv2.namedWindow("Camera")

    cv2.createTrackbar("Equalization", "Camera", 1, 1, lambda x: None)
    cv2.createTrackbar("Voting", "Camera", 1, 1, lambda x: None)
    # ==========================
    # LOOP PRINCIPAL
    # ==========================

    while True:

        frame = camera.read()
        if frame is None:
            print("Eroare citire frame.")
            break

        equalization_enabled = cv2.getTrackbarPos("Equalization", "Camera")
        voting_enabled = cv2.getTrackbarPos("Voting", "Camera")

        # MediaPipe detectează landmarkuri
        result = face_processor.process(frame)

        # Extrage ROI față
        face_roi = preprocessor.extract_face_roi(frame, result)

        # Preprocesare pentru CNN
       # processed_face = preprocessor.preprocess(face_roi)
        if DEBUG_PIPELINE:
            processed_face, debug_images = preprocessor.preprocess(
                face_roi,
                debug=True,
                use_equalization=bool(equalization_enabled)
            )
        else:
            processed_face = preprocessor.preprocess(
                face_roi,
                use_equalization=bool(equalization_enabled)
            )

        # Predicție emoție
        if processed_face is None:
            continue

        emotion_label, confidence = predictor.predict(processed_face)

        if emotion_label is not None:

            # Dacă voting este activ
            if voting_enabled:

                emotion_buffer.append((emotion_label, confidence))

                print("Buffer:", [e[0] for e in emotion_buffer])

                if len(emotion_buffer) == 5:
                    labels = [e[0] for e in emotion_buffer]
                    stable_label = max(set(labels), key=labels.count)

                    stable_conf = np.mean(
                        [e[1] for e in emotion_buffer if e[0] == stable_label]
                    )

                    print("Voting on:", labels)
                    print("Stable emotion:", stable_label, "Confidence:", stable_conf)
                    print("------")

                    emotion_label = stable_label
                    confidence = stable_conf

                    emotion_buffer.clear()

            else:
                # Dacă voting este dezactivat,foloseste emoția brută
                emotion_buffer.clear()

        # ==========================
        # AFIȘARE EMOȚIE
        # ==========================

        if emotion_label is not None:

            text = f"{emotion_label} ({confidence*100:.1f}%)"

            cv2.putText(
                frame,
                text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

        # Desenează mesh peste imagine
        face_processor.draw(frame, result)

        cv2.imshow("Rusty - Emotion Recognition", frame)
        if DEBUG_PIPELINE and 'debug_images' in locals():
            roi = cv2.resize(debug_images["roi"], (200, 200))

            gray = cv2.cvtColor(debug_images["gray"], cv2.COLOR_GRAY2BGR)
            gray = cv2.resize(gray, (200, 200))

            equalized = cv2.cvtColor(debug_images["equalized"], cv2.COLOR_GRAY2BGR)
            equalized = cv2.resize(equalized, (200, 200))

            resized = cv2.cvtColor(debug_images["resized_48"], cv2.COLOR_GRAY2BGR)
            resized = cv2.resize(resized, (200, 200))

            top = np.hstack((roi, gray))
            bottom = np.hstack((equalized, resized))
            combined = np.vstack((top, bottom))

            cv2.imshow("Emotion Pipeline Debug", combined)

        #if cv2.waitKey(1) & 0xFF == ord("q"):
        #break

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('d'):
            DEBUG_PIPELINE = not DEBUG_PIPELINE
            print("Debug mode:", DEBUG_PIPELINE)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
