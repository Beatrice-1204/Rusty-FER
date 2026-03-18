import cv2
import mediapipe as mp
import math

class MediaPipeFaceProcessor:
    """
        MediaPipeFaceProcessor
          1) primește un frame din cameră
          2) convertește în RGB
          3) rulează MediaPipe FaceMesh pt landmarks faciali
          4) deseneaza mash
    """

    def __init__(
            self,
            max_num_faces:int=1,
            refine_landmarks: bool=True,
            min_detection_confidence: float=0.5,
            min_tracking_confidence: float=0.5,
    ):
        #mp.solutions contine solutii fata facute
        self.mp_face_mesh=mp.solutions.face_mesh
        #drawing_utils = pt a desena puncte /linii pe frame pt vizualizare
        self.mp_drawing=mp.solutions.drawing_utils
        #drawing_styles = stiluri predefinite pt cum arata liniile/contururile
        self.mp_styles=mp.solutions.drawing_styles

        #Initializez modelul FaceMash:
        #-static_image_mode=False ->modul video:foloseste tracking intre frameuri
        #-max_num_faces -> cate fete sa urmareasca
        #-refine_landmarks=True->landmarkuri mai detaliate pentru ochi
        #-min_detection_confidence-> prag pentru a gasi o fata (detectia initiala)
        #-min_tracking_confidence -> prag pentru tracking intre frame-uri, pentru stabilitate

        self.face_mesh=self.mp_face_mesh.FaceMesh(  #o clasa din MediaPipe care ruleaza un pipeline optimizat:detecteaza fata, decumeaza fona fetei, estimeaza landmarks, urmareste fata in timp
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr):
        #Ruleaza MediaPipe pe frame
        # frame_bgr -img OpenCV
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        result=self.face_mesh.process(frame_rgb)

        return result

    def draw(self, frame_bgr, result):
        #Deseneaza mashul pe frame
        #frame_bgr-farme OpenCV pe car eil afise

        if result is None:
            return

        if not result.multi_face_landmarks:
            return

        #pt fiecare fata detectata
        for face_landmarks in result.multi_face_landmarks:

            #1. Desenez "tesselation- plasa

            self.mp_drawing.draw_landmarks(
                image=frame_bgr,
                landmark_list=face_landmarks, #lista de puncte ale fetei
                connections=self.mp_face_mesh.FACEMESH_TESSELATION, #conexiuni pt plasa
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_styles.get_default_face_mesh_tesselation_style()

            )
            #2. Desenez contururile la ochi , sprancene, buze

            self.mp_drawing.draw_landmarks(
                image=frame_bgr,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS, #conexiuni pt conturuuri
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_styles.get_default_face_mesh_contours_style()

            )

    def get_landmarks_px(self, frame_bgr, result, idx:int):
        # Întoarce coordonatele (x, y) în pixeli pentru un landmark specific.
        #idx = index-ul landmark-ului din FaceMesh.

        if result is None or not result.multi_face_landmarks:
            return None

        h, w, _ = frame_bgr.shape
        face_landmarks = result.multi_face_landmarks[0] #se alege prima fata detectatat
        lm=face_landmarks.landmark[idx]

        x_px=int(lm.x * w)
        y_px=int(lm.y * h)
        return (x_px, y_px)

    def compute_roll_angle_deg(self, left_eye_px, right_eye_px):
        #calculeaza unghiul in grade dintre linia ochilor si orizontala  si il returneaza in grade

        (x1, y1)=left_eye_px
        (x2, y2)=right_eye_px

        #vectorul dintre ochi
        dx=x2-x1
        dy=y2-y1

        #atan2(dy, dx)- unghiul in radiani fata de Ox
        angle_rad=math.atan2(dy, dx)
        angle_deg=math.degrees(angle_rad)
        return angle_deg

















