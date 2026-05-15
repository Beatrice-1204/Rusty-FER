import os
import time
from typing import Optional

import cv2
import numpy as np

from camera.camera_stream import CameraStream
from emotion_recognition.emotion_stabilizer import EmotionStabilizer
from emotion_recognition.onnx_emotion_predictor import OnnxEmotionPredictor
from face_detection.yunet_face_detector import YuNetFaceDetection, YuNetFaceDetector
from image_processing.image_preprocessor import ImagePreprocessor
from reactions.audio_controller import AudioController
from reactions.display_controller import DisplayController
from reactions.reaction_config import ReactionAudioConfig, ReactionDisplayConfig
from reactions.reaction_gate import ReactionGate, ReactionGateState
from reactions.reaction_manager import ReactionManager
from runtime_config import RUNTIME_CONFIG
from runtime_support import DetectionSmoother, FaceQualityValidator, PerfTracker
from tracking import create_pan_tilt_controller

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
    show_camera_window = config.logging.show_camera_window
    debug_pipeline = config.logging.debug_pipeline and show_camera_window
    app_started_at = time.perf_counter()

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
        undistort_enabled=config.undistort.enabled,
        camera_matrix=config.undistort.camera_matrix,
        dist_coeffs=config.undistort.dist_coeffs,
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
    emotion_stabilizer = EmotionStabilizer(
        config.stabilization,
        debug_logging=config.logging.debug_logging,
    )
    reaction_gate = ReactionGate(
        config.reaction_gate,
        debug_logging=config.reaction_gate.debug_logging,
    )
    reaction_display_config = ReactionDisplayConfig()
    reaction_audio_config = ReactionAudioConfig()
    reaction_manager = ReactionManager(
        DisplayController(reaction_display_config),
        AudioController(reaction_audio_config),
    )
    pan_tilt_controller = create_pan_tilt_controller(config.pan_tilt)
    reaction_manager.handle(reaction_display_config.idle_emotion)
    reaction_manager.play_startup()

    frame_counter = 0
    last_detection: Optional[YuNetFaceDetection] = None
    last_stable_emotion: Optional[str] = None
    last_stable_confidence: Optional[float] = None
    last_stable_raw_label: Optional[str] = None

    if show_camera_window:
        cv2.namedWindow("Rusty - Emotion Recognition")

    try:
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

            if (
                config.app.max_runtime_seconds is not None
                and loop_start - app_started_at >= config.app.max_runtime_seconds
            ):
                print("[APP] max_runtime reached, shutting down")
                break

            if not reaction_manager.update():
                break

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
                    if reaction_gate.is_detecting():
                        emotion_stabilizer.reset(clear_stable=True)
                        last_stable_emotion = None
                        last_stable_confidence = None
                        last_stable_raw_label = None
            else:
                quality = validator.validate(frame.shape, detection)
                if not quality.is_valid:
                    if config.logging.debug_logging:
                        print(f"[FACE] reuse reject reason={quality.reason}")
                    detection = None
                    last_detection = None
                    smoother.reset()
                    if reaction_gate.is_detecting():
                        emotion_stabilizer.reset(clear_stable=True)
                        last_stable_emotion = None
                        last_stable_confidence = None
                        last_stable_raw_label = None
            stage_times["post_detect"] = time.perf_counter() - post_detect_start

            pan_tilt_controller.update(
                detection.bbox if detection is not None else None,
                frame.shape,
            )

            preprocess_start = time.perf_counter()
            face_roi = preprocessor.prepare_face_roi(frame, detection)
            stage_times["preprocess"] = time.perf_counter() - preprocess_start

            display_emotion = last_stable_emotion if config.stabilization.hold_last_stable else None
            display_confidence = last_stable_confidence if config.stabilization.hold_last_stable else None
            display_raw_label = last_stable_raw_label if config.logging.show_raw_label else None
            show_raw_label = False
            debug_images = None
            reaction_emotion: Optional[str] = None
            previous_gate_state = reaction_gate.state

            if not reaction_gate.is_detecting():
                reaction_gate.update(None)
                if reaction_gate.is_detecting():
                    emotion_stabilizer.reset(clear_stable=True)
                    last_stable_emotion = None
                    last_stable_confidence = None
                    last_stable_raw_label = None

            if reaction_gate.is_detecting() and face_roi is not None:
                predict_start = time.perf_counter()
                emotion_label, confidence = predictor.predict(
                    face_roi,
                    use_equalization=config.face_roi.onnx_use_equalization,
                )
                if emotion_label is not None and predictor.can_save_debug_sample():
                    predictor.save_debug_sample(
                        frame,
                        predictor.last_roi,
                        predictor.last_onnx_input_64,
                    )
                stage_times["predict"] = time.perf_counter() - predict_start

                top3 = getattr(predictor, "last_top3", [])
                stable_result = emotion_stabilizer.update(
                    emotion_label,
                    confidence,
                    raw_label=predictor.last_raw_label,
                    top3=top3,
                )
                last_stable_emotion = stable_result.emotion
                last_stable_confidence = stable_result.confidence
                last_stable_raw_label = stable_result.raw_label
                if (
                    stable_result.reason is not None
                    and stable_result.reason.startswith("stable_window")
                    and stable_result.emotion != config.reaction_gate.neutral_label
                ):
                    reaction_emotion = stable_result.emotion

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

            if reaction_gate.is_detecting():
                should_log_summary = reaction_gate.detecting_has_elapsed()
                summary = None
                if should_log_summary:
                    summary = emotion_stabilizer.summarize(
                        neutral_label=config.reaction_gate.neutral_label
                    )
                    print("[REACTION] detecting_summary", *summary.to_log_parts())

                triggered_emotion = reaction_gate.update(reaction_emotion)
                if triggered_emotion is not None:
                    print(f"[REACTION] trigger emotion={triggered_emotion}")
                    reaction_manager.handle(triggered_emotion)
                elif should_log_summary and summary is not None:
                    print("[REACTION] no_trigger", *summary.to_log_parts())
                    if reaction_gate.is_detecting():
                        reaction_gate.restart_detecting_window(reason=summary.reason)

            if (
                previous_gate_state == ReactionGateState.COOLDOWN
                and reaction_gate.state == ReactionGateState.IDLE
            ):
                reaction_manager.handle(reaction_display_config.idle_emotion)

            text = _format_emotion_text(
                display_emotion,
                display_confidence,
                display_raw_label,
                show_raw_label,
            )
            _draw_emotion_text(frame, text)

            if config.logging.draw_detection:
                face_detector.draw(frame, detection)

            if show_camera_window:
                cv2.imshow("Rusty - Emotion Recognition", frame)
            if debug_pipeline and debug_images is not None:
                _show_debug_pipeline(debug_images)

            if show_camera_window:
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

    finally:
        camera.release()
        pan_tilt_controller.close()
        reaction_manager.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
