import time
from collections import deque

import cv2
import numpy as np

from camera.camera_stream import CameraStream
from emotion_recognition.onnx_emotion_predictor import OnnxEmotionPredictor
from face_detection.yunet_face_detector import YuNetFaceDetector
from image_processing.image_preprocessor import ImagePreprocessor

DEBUG_PIPELINE = False
DEBUG_LOGGING = False
PERF_LOG_EVERY_N_FRAMES = 30
USE_ONNX_EQUALIZATION = False
DETECT_EVERY_N_FRAMES = 2
MIN_EMOTION_CONFIDENCE = 0.45
MIN_TOP1_TOP2_MARGIN = 0.00


def main():
    global DEBUG_PIPELINE

    camera = CameraStream(camera_index=0)
    camera.open()

    face_processor = YuNetFaceDetector(
        model_path="models/yunet/face_detection_yunet.onnx",
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=1,
    )

    preprocessor = ImagePreprocessor(target_size=48)
    predictor = OnnxEmotionPredictor(model_path="models/onnx/emotion-ferplus-7.onnx")
    emotion_buffer = deque(maxlen=5)
    frame_counter = 0
    last_detection = None
    last_stable_emotion = None
    last_stable_confidence = None
    perf_totals = {
        "camera_read": 0.0,
        "face_process": 0.0,
        "preprocess": 0.0,
        "predict": 0.0,
        "total": 0.0,
    }

    cv2.namedWindow("Camera")
    cv2.createTrackbar("Equalization", "Camera", 1, 1, lambda x: None)
    cv2.createTrackbar("Voting", "Camera", 1, 1, lambda x: None)

    while True:
        loop_start = time.perf_counter()

        read_start = time.perf_counter()
        frame = camera.read()
        perf_totals["camera_read"] += time.perf_counter() - read_start
        if frame is None:
            print("Eroare citire frame.")
            break

        equalization_enabled = cv2.getTrackbarPos("Equalization", "Camera")
        voting_enabled = cv2.getTrackbarPos("Voting", "Camera")

        face_start = time.perf_counter()
        should_detect = last_detection is None or frame_counter % DETECT_EVERY_N_FRAMES == 0
        if should_detect:
            current_detection = face_processor.detect_one(frame)
            if current_detection is not None:
                last_detection = current_detection
        detection = last_detection
        perf_totals["face_process"] += time.perf_counter() - face_start

        preprocess_start = time.perf_counter()
        face_roi = preprocessor.prepare_face_roi(frame, detection)

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
        perf_totals["preprocess"] += time.perf_counter() - preprocess_start

        if processed_face is None:
            frame_counter += 1
            perf_totals["total"] += time.perf_counter() - loop_start
            continue

        predict_start = time.perf_counter()
        emotion_label, confidence = predictor.predict(
            face_roi,
            processed_face=processed_face,
            use_equalization=USE_ONNX_EQUALIZATION
        )
        perf_totals["predict"] += time.perf_counter() - predict_start

        top3 = getattr(predictor, "last_top3", [])
        top1_score = float(top3[0][1]) if len(top3) >= 1 else (confidence or 0.0)
        top2_score = float(top3[1][1]) if len(top3) >= 2 else 0.0
        confidence_ok = confidence is not None and confidence >= MIN_EMOTION_CONFIDENCE
        margin_ok = (top1_score - top2_score) >= MIN_TOP1_TOP2_MARGIN
        prediction_is_confident = confidence_ok and margin_ok
        show_raw_label = False

        if emotion_label is not None:
            if prediction_is_confident:
                if voting_enabled:
                    emotion_buffer.append((emotion_label, confidence))

                    if DEBUG_LOGGING:
                        print("Buffer:", [e[0] for e in emotion_buffer])

                    if len(emotion_buffer) == 5:
                        labels = [e[0] for e in emotion_buffer]
                        stable_label = max(set(labels), key=labels.count)
                        stable_conf = np.mean(
                            [e[1] for e in emotion_buffer if e[0] == stable_label]
                        )

                        if DEBUG_LOGGING:
                            print("Voting on:", labels)
                            print("Stable emotion:", stable_label, "Confidence:", stable_conf)
                            print("------")

                        emotion_label = stable_label
                        confidence = stable_conf
                        last_stable_emotion = stable_label
                        last_stable_confidence = stable_conf
                        show_raw_label = True
                        emotion_buffer.clear()
                    elif last_stable_emotion is not None:
                        emotion_label = last_stable_emotion
                        confidence = last_stable_confidence
                    else:
                        emotion_label = None
                        confidence = None
                else:
                    last_stable_emotion = emotion_label
                    last_stable_confidence = confidence
                    show_raw_label = True
            else:
                emotion_buffer.clear()
                if last_stable_emotion is not None:
                    emotion_label = last_stable_emotion
                    confidence = last_stable_confidence
                else:
                    emotion_label = None
                    confidence = None

                if DEBUG_LOGGING:
                    print(
                        "[STABLE] reject",
                        f"label={predictor.last_raw_label}",
                        f"conf={top1_score:.3f}",
                        f"margin={(top1_score - top2_score):.3f}"
                    )

        if emotion_label is not None:
            text = f"{emotion_label} ({confidence*100:.1f}%)"
            if show_raw_label and getattr(predictor, "last_raw_label", None):
                text += f" | raw: {predictor.last_raw_label}"

            cv2.putText(
                frame,
                text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

        face_processor.draw(frame, detection)

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

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('d'):
            DEBUG_PIPELINE = not DEBUG_PIPELINE
            print("Debug mode:", DEBUG_PIPELINE)

        frame_counter += 1
        perf_totals["total"] += time.perf_counter() - loop_start

        if frame_counter % PERF_LOG_EVERY_N_FRAMES == 0:
            avg_camera_ms = perf_totals["camera_read"] / frame_counter * 1000
            avg_face_ms = perf_totals["face_process"] / frame_counter * 1000
            avg_preprocess_ms = perf_totals["preprocess"] / frame_counter * 1000
            avg_predict_ms = perf_totals["predict"] / frame_counter * 1000
            avg_total_ms = perf_totals["total"] / frame_counter * 1000
            fps = 1000.0 / avg_total_ms if avg_total_ms > 0 else 0.0

            print(
                "[PERF]",
                f"frames={frame_counter}",
                f"camera={avg_camera_ms:.1f}ms",
                f"face={avg_face_ms:.1f}ms",
                f"preprocess={avg_preprocess_ms:.1f}ms",
                f"predict={avg_predict_ms:.1f}ms",
                f"total={avg_total_ms:.1f}ms",
                f"fps={fps:.1f}"
            )

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
