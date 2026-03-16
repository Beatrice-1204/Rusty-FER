import cv2


class HaarFaceDetector:
    """
    Detecteaz fețe folosind Haar Cascade
    """

    def __init__(self):
        # Calea către fișierul Haar livrat de OpenCV
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Dacă fișierul nu s-a încărcat corect
        if self.face_cascade.empty():
            raise RuntimeError("Nu s-a putut încărca Haar Cascade.")

    def detect(self, frame_bgr):
        """
        Primește un frame color (BGR) și returnează
        o listă de fețe: [(x, y, w, h), ...]
        """

        # Haar funcționează pe grayscale
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        return faces
