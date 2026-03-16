import cv2


class CameraStream:

    #CameraStream = modulul care se ocupă DOAR cu accesul la camera


    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):  #constructor
        """
        camera_index:
            0 = camera implicită

        # width / height: Pe Raspberry Pi, rezoluția mai mică
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None  # aici se păstreaza obiectul VideoCapture

    def open(self) -> None:  #metoda
        """Deschide camera."""
        self.cap = cv2.VideoCapture(self.camera_index)

        # Setez rezoluția
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Verificăm dacă s-a deschis camera cu succes
        if not self.cap.isOpened():
            raise RuntimeError("Eroare: nu pot deschide camera. Verifică permisiunile.")

    def read(self):
        # Citește un frame din fluxul video.
        if self.cap is None:
            raise RuntimeError("Camera nu este deschisă. ")

        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:

        if self.cap is not None:
            self.cap.release()
            self.cap = None
