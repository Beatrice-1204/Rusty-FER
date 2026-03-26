import cv2

from src.camera.camera_stream import CameraStream
from src.face_detection.yunet_face_detector import YuNetFaceDetector


def main() -> None:
    camera = CameraStream(camera_index=0)
    camera.open()

    detector = YuNetFaceDetector(
        model_path="models/yunet/face_detection_yunet.onnx",
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=20,
    )

    while True:
        frame = camera.read()
        if frame is None:
            print("Nu s-a putut citi frame-ul.")
            break

        detections = detector.detect(frame)

        for detection in detections:
            detector.draw(frame, detection)

            x, y, _, _ = detection.bbox
            cv2.putText(
                frame,
                f"{detection.score:.2f}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("YuNet Live Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
