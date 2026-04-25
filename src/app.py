import os
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from camera.camera_stream import CameraStream
from emotion_recognition.onnx_emotion_predictor import OnnxEmotionPredictor
from face_detection.yunet_face_detector import YuNetFaceDetection, YuNetFaceDetector
from image_processing.image_preprocessor import ImagePreprocessor
from runtime_config import RUNTIME_CONFIG
from runtime_support import DetectionSmoother, FaceQualityValidator, PerfTracker


def _should_accept_prediction(confidence: Optional[float], top3, config) -> bool:
    if confidence is None:
        return False

    if not config.stabilization.enabled:
        return True

    top1_score = float(top3[0][1]) if len(top3) >= 1 else confidence
    top2_score = float(top3[1][1]) if len(top3) >= 2 else 0.0

    confidence_ok = top1_score >= config.stabilization.min_confidence
    margin_ok = (top1_score - top2_score) >= config.stabilization.min_margin
    return confidence_ok and margin_ok


def _format_emotion_text(
    emotion_label: Optional[str],
    confidence: Optional[float],
    raw_label: Optional[str],
    show_raw_label: bool,
) -> Optional[str]:
    if emotion_label is None or confidence is None:
        return None

    text = f"{emotion_label} ({confidence * 100:.1f}%)"
    if show_raw_label and raw_label:
        text += f" | raw: {raw_label}"
    return text


def _draw_emotion_text(frame, text: Optional[str]) -> None:
    if text is None:
        return

    cv2.putText(
        frame,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )


def _show_debug_pipeline(debug_images) -> None:
    frame = cv2.resize(debug_images["frame"], (320, 240))
    roi = cv2.resize(debug_images["roi"], (240, 240))
    onnx_input_64 = cv2.cvtColor(debug_images["onnx_input_64"], cv2.COLOR_GRAY2BGR)
    onnx_input_64 = cv2.resize(onnx_input_64, (240, 240))

    combined = np.hstack((frame, roi, onnx_input_64))
    cv2.imshow("Emotion Pipeline Debug", combined)


def _log_perf(perf_tracker: PerfTracker) -> None:
    summary = perf_tracker.summary()
    print(
        "[PERF]",
        f"frames={int(summary['frames'])}",
        f"capture={summary['capture_ms']:.1f}ms",
        f"detect={summary['detect_ms']:.1f}ms",
        f"post_detect={summary['post_detect_ms']:.1f}ms",
        f"preprocess={summary['preprocess_ms']:.1f}ms",
        f"predict={summary['predict_ms']:.1f}ms",
        f"total={summary['total_ms']:.1f}ms",
        f"fps_avg={summary['fps_avg']:.1f}",
        f"fps_inst={summary['fps_inst']:.1f}",
    )


def main():
    config = RUNTIME_CONFIG
    debug_pipeline = config.logging.debug_pipeline

    camera_backend = os.getenv("RUSTY_CAMERA_BACKEND", config.camera.backend)
    camera_width = int(os.getenv("RUSTY_CAMERA_WIDTH", str(config.camera.width)))
    camera_height = int(os.getenv("RUSTY_CAMERA_HEIGHT", str(config.camera.height)))
    camera_index = int(os.getenv("RUSTY_CAMERA_INDEX", str(config.camera.camera_index)))

    camera = CameraStream(
        camera_index=camera_index,
        width=camera_width,
        height=camera_height,
        backend=camera_backend,
        center_crop_enabled=config.camera.center_crop_enabled,
        center_crop_scale=config.camera.center_crop_scale,
    )
    camera.open()

    print(
        "[CAMERA]",
        f"backend={camera.active_backend}",
        f"size={camera_width}x{camera_height}",
        f"index={camera_index}",
    )

    face_detector = YuNetFaceDetector(
        model_path=config.yunet.model_path,
        score_threshold=config.yunet.score_threshold,
        nms_threshold=config.yunet.nms_threshold,
        top_k=config.yunet.top_k,
        max_detection_width=config.yunet.max_detection_width,
    )
    preprocessor = ImagePreprocessor(
        face_padding_px=config.face_roi.padding_px,
        crop_scale=config.face_roi.crop_scale,
        crop_center_y_ratio=config.face_roi.crop_center_y_ratio,
    )
    predictor = OnnxEmotionPredictor(
        model_path=config.emotion_model.model_path,
        log_top3=config.logging.log_top3,
    )

    smoother = DetectionSmoother(
        alpha=config.smoothing.alpha,
        max_center_shift_ratio=config.smoothing.max_center_shift_ratio,
        min_size_ratio=config.smoothing.min_size_ratio,
    )
    validator = FaceQualityValidator(
        min_face_width_px=config.face_quality.min_face_width_px,
        min_face_height_px=config.face_quality.min_face_height_px,
        min_border_margin_px=config.face_quality.min_border_margin_px,
        require_keypoints=config.face_quality.require_keypoints,
        min_eye_distance_ratio=config.face_quality.min_eye_distance_ratio,
        max_eye_y_diff_ratio=config.face_quality.max_eye_y_diff_ratio,
    )

    perf_tracker = PerfTracker(fps_window_size=config.logging.perf_log_every_n_frames)
    emotion_buffer = deque(maxlen=config.stabilization.voting_window)

    frame_counter = 0
    last_detection: Optional[YuNetFaceDetection] = None
    last_stable_emotion: Optional[str] = None
    last_stable_confidence: Optional[float] = None
    last_stable_raw_label: Optional[str] = None

    cv2.namedWindow("Rusty - Emotion Recognition")

    while True:
        stage_times = {
            "capture": 0.0,
            "detect": 0.0,
            "post_detect": 0.0,
            "preprocess": 0.0,
            "predict": 0.0,
            "total": 0.0,
        }
        loop_start = time.perf_counter()

        capture_start = time.perf_counter()
        frame = camera.read()
        stage_times["capture"] = time.perf_counter() - capture_start
        if frame is None:
            print("Eroare citire frame.")
            break

        detect_start = time.perf_counter()
        should_detect = (
            last_detection is None
            or frame_counter % config.yunet.detect_every_n_frames == 0
        )
        raw_detection = face_detector.detect_one(frame) if should_detect else last_detection
        stage_times["detect"] = time.perf_counter() - detect_start

        post_detect_start = time.perf_counter()
        detection = raw_detection
        if should_detect:
            if config.smoothing.enabled:
                detection = smoother.smooth(raw_detection)
            else:
                detection = raw_detection

            quality = validator.validate(frame.shape, detection)
            if quality.is_valid:
                last_detection = detection
            else:
                if config.logging.debug_logging:
                    print(f"[FACE] reject reason={quality.reason}")
                detection = None
                last_detection = None
                smoother.reset()
                emotion_buffer.clear()
        else:
            quality = validator.validate(frame.shape, detection)
            if not quality.is_valid:
                if config.logging.debug_logging:
                    print(f"[FACE] reuse reject reason={quality.reason}")
                detection = None
                last_detection = None
                smoother.reset()
                emotion_buffer.clear()
        stage_times["post_detect"] = time.perf_counter() - post_detect_start

        preprocess_start = time.perf_counter()
        face_roi = preprocessor.prepare_face_roi(frame, detection)
        stage_times["preprocess"] = time.perf_counter() - preprocess_start

        display_emotion = last_stable_emotion if config.stabilization.hold_last_stable else None
        display_confidence = last_stable_confidence if config.stabilization.hold_last_stable else None
        display_raw_label = last_stable_raw_label if config.logging.show_raw_label else None
        show_raw_label = False
        debug_images = None

        if face_roi is not None:
            predict_start = time.perf_counter()
            if predictor.can_save_debug_sample():
                predictor.save_debug_frame(frame)
            emotion_label, confidence = predictor.predict(
                face_roi,
                use_equalization=config.face_roi.onnx_use_equalization,
            )
            stage_times["predict"] = time.perf_counter() - predict_start

            top3 = getattr(predictor, "last_top3", [])
            prediction_is_confident = _should_accept_prediction(confidence, top3, config)

            if emotion_label is not None and prediction_is_confident:
                if config.stabilization.enable_voting:
                    emotion_buffer.append((emotion_label, confidence))
                    if len(emotion_buffer) == config.stabilization.voting_window:
                        labels = [label for label, _ in emotion_buffer]
                        stable_label = max(set(labels), key=labels.count)
                        stable_conf = float(np.mean([
                            score for label, score in emotion_buffer if label == stable_label
                        ]))
                        last_stable_emotion = stable_label
                        last_stable_confidence = stable_conf
                        last_stable_raw_label = predictor.last_raw_label
                        emotion_buffer.clear()
                else:
                    last_stable_emotion = emotion_label
                    last_stable_confidence = confidence
                    last_stable_raw_label = predictor.last_raw_label

            elif emotion_label is not None:
                emotion_buffer.clear()
                if config.logging.debug_logging:
                    top1_score = float(top3[0][1]) if len(top3) >= 1 else 0.0
                    top2_score = float(top3[1][1]) if len(top3) >= 2 else 0.0
                    print(
                        "[STABLE] reject",
                        f"label={predictor.last_raw_label}",
                        f"conf={top1_score:.3f}",
                        f"margin={(top1_score - top2_score):.3f}",
                    )

            display_emotion = last_stable_emotion
            display_confidence = last_stable_confidence
            display_raw_label = last_stable_raw_label
            show_raw_label = config.logging.show_raw_label and display_emotion is not None
            if debug_pipeline:
                debug_images = preprocessor.get_debug_views(
                    frame,
                    predictor.last_roi,
                    predictor.last_onnx_input_64,
                )
        else:
            stage_times["predict"] = 0.0

        text = _format_emotion_text(
            display_emotion,
            display_confidence,
            display_raw_label,
            show_raw_label,
        )
        _draw_emotion_text(frame, text)

        if config.logging.draw_detection:
            face_detector.draw(frame, detection)

        cv2.imshow("Rusty - Emotion Recognition", frame)
        if debug_pipeline and debug_images is not None:
            _show_debug_pipeline(debug_images)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("d"):
            debug_pipeline = not debug_pipeline
            print("Debug mode:", debug_pipeline)

        stage_times["total"] = time.perf_counter() - loop_start
        perf_tracker.record(stage_times)
        frame_counter += 1

        if (
            config.logging.show_perf
            and perf_tracker.frame_count % config.logging.perf_log_every_n_frames == 0
        ):
            _log_perf(perf_tracker)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
