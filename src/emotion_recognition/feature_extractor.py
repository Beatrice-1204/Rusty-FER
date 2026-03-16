import math

class FeatureExtractor:
    def __init__(self):
        pass

    def _distance(self, p1, p2):
        #calculeaza distanta euclidiana intre 2 puncte
        return math.sqrt(
            (p1.x - p2.x) ** 2 +
            (p1.y - p2.y) ** 2
        )

    def extract(self, face_landmarks):
        #extrage un vector de features normalizat pentru fata
        lm = face_landmarks.landmark

        # Face width (pentru normalizare)
        face_left = lm[234]
        face_right = lm[454]
        face_width = self._distance(face_left, face_right)

        if face_width == 0:
            return None

        # === Gură ===
        top_lip = lm[13]
        bottom_lip = lm[14]
        left_mouth = lm[78]
        right_mouth = lm[308]

        mouth_height = self._distance(top_lip, bottom_lip) / face_width
        mouth_width = self._distance(left_mouth, right_mouth) / face_width
        mouth_ratio = mouth_height / (mouth_width + 1e-6)

        # Colturile gurii pentru unghi / asimetrie
        left_mouth_corner =lm[61]
        right_mouth_corner = lm[291]
        mouth_slope = (right_mouth_corner.y - left_mouth_corner.y)/(self._distance(left_mouth_corner, right_mouth_corner) + 1e-6)

        # === Ochi stâng ===
        left_eye_top = lm[159]
        left_eye_bottom = lm[145]
        left_eye_left = lm[33]
        left_eye_right = lm[133]

        left_eye_height = self._distance(left_eye_top, left_eye_bottom) / face_width
        left_eye_width = self._distance(left_eye_left, left_eye_right) / face_width
        left_eye_ratio = left_eye_height / (left_eye_width + 1e-6)

        # === Ochi drept ===
        right_eye_top = lm[386]
        right_eye_bottom = lm[374]
        right_eye_left = lm[362]
        right_eye_right = lm[263]

        right_eye_height = self._distance(right_eye_top, right_eye_bottom) / face_width
        right_eye_width = self._distance(right_eye_left, right_eye_right) / face_width
        right_eye_ratio = right_eye_height / (right_eye_width + 1e-6)

        # === Simetria ochilor ===
        eye_ratio_diff = abs(left_eye_ratio - right_eye_ratio)

        # === Sprâncene ===
        left_eyebrow_top = lm[105]
        left_eyebrow_bottom = lm[66]
        right_eyebrow_top = lm[334]
        right_eyebrow_bottom = lm[296]

        left_eyebrow_height = self._distance(left_eyebrow_top, left_eyebrow_bottom) / face_width
        right_eyebrow_height = self._distance(right_eyebrow_top, right_eyebrow_bottom) / face_width

        # Unghi sprâncene (diferență poziții verticale)
        eyebrow_slope = (right_eyebrow_top.y - left_eyebrow_top.y) / ( self._distance(left_eyebrow_top, right_eyebrow_top) + 1e-6)

        # === Nas ===
        nose_left = lm[93]
        nose_right = lm[323]
        nose_top = lm[6]
        nose_bottom = lm[195]

        nose_width = self._distance(nose_left, nose_right) / face_width
        nose_height = self._distance(nose_top, nose_bottom) / face_width

        # Raport nas / ochi
        nose_eye_ratio = nose_height / (max(left_eye_height, right_eye_height) + 1e-6)

        # === Distante între nas și gură ===
        nose_to_mouth = self._distance(nose_bottom, top_lip) / face_width

        # === Unghi gură vs nas ===
        nose_mouth_slope = (top_lip.y - nose_bottom.y) / (self._distance(top_lip, nose_bottom) + 1e-6)

        jaw_left = lm[152]
        jaw_right = lm[377]
        jaw_width = self._distance(jaw_left, jaw_right) / face_width

        face_top = lm[10]
        face_bottom = lm[152]
        face_height = self._distance(face_top, face_bottom) / face_width

        left_pupil_center = lm[468]
        right_pupil_center = lm[473]
        pupil_left_distance = self._distance(left_eye_top, left_pupil_center) / face_width
        pupil_right_distance = self._distance(right_eye_top, right_pupil_center) / face_width

        cheekbone_left = lm[127]
        cheekbone_right = lm[356]
        cheekbone_width = self._distance(cheekbone_left, cheekbone_right) / face_width

        lip_corner_distance = self._distance(left_mouth_corner, right_mouth_corner) / face_width

        eyebrow_eye_distance_left = self._distance(left_eyebrow_top, left_eye_top) / face_width
        eyebrow_eye_distance_right = self._distance(right_eyebrow_top, right_eye_top) / face_width

        nose_tip_slope = (nose_top.y - nose_bottom.y) / (self._distance(nose_top, nose_bottom) + 1e-6)

        chin_to_lip_distance = self._distance(face_bottom, top_lip) / face_width

        # === Vector (25 features) ===
        features = [
            mouth_height, mouth_width, mouth_ratio, mouth_slope,
            left_eye_ratio, right_eye_ratio, eye_ratio_diff,
            left_eyebrow_height, right_eyebrow_height, eyebrow_slope,
            nose_width, nose_height, nose_eye_ratio, nose_to_mouth, nose_mouth_slope,
            jaw_width, face_height, pupil_left_distance, pupil_right_distance,
            cheekbone_width, lip_corner_distance,
            eyebrow_eye_distance_left, eyebrow_eye_distance_right,
            nose_tip_slope, chin_to_lip_distance
        ]

        return features
