import cv2
import math

# nu imi place optimizarea - nu folosim
class FaceAligner:
    """
    Modul responsabil pentru:
    - calcularea unghiului feței pe baza poziției ochilor
    - rotirea imaginii astfel încât ochii să fie pe linie orizontală
    """

    def __init__(self):
        pass

    def compute_roll_angle(self, left_eye_px, right_eye_px):
        """
        Calculează unghiul (în grade) dintre linia ochilor și orizontală.
        """

        (x1, y1) = left_eye_px
        (x2, y2) = right_eye_px

        dx = x2 - x1
        dy = y2 - y1

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def align(self, frame_bgr, left_eye_px, right_eye_px):
        """
        Rotește imaginea în jurul centrului dintre ochi
        astfel încât fața să fie aliniată.
        """

        if left_eye_px is None or right_eye_px is None:
            return frame_bgr

        angle = self.compute_roll_angle(left_eye_px, right_eye_px)

        # centru rotire = mijlocul dintre ochi
        eye_center_x = int((left_eye_px[0] + right_eye_px[0]) / 2)
        eye_center_y = int((left_eye_px[1] + right_eye_px[1]) / 2)
        center = (eye_center_x, eye_center_y)

        # matrice rotație
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # aplic rotația
        aligned = cv2.warpAffine(
            frame_bgr,
            rotation_matrix,
            (frame_bgr.shape[1], frame_bgr.shape[0]),
            flags=cv2.INTER_LINEAR
        )

        return aligned
