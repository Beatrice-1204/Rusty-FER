from dataclasses import dataclass, field

from paths import project_path


@dataclass(frozen=True)
class CameraConfig:
    camera_index: int = 0
    width: int = 640
    height: int = 480


@dataclass(frozen=True)
class YuNetConfig:
    model_path: str = project_path("models", "YuNet", "face_detection_yunet.onnx")
    score_threshold: float = 0.6  # pragul de incredere pt a considera o detectie valida
    nms_threshold: float = 0.3 # pragul pentru Non-Maximum Suppression (NMS) pentru a elimina detectiile multiple ale aceleiasi fete
    top_k: int = 1
    max_detection_width: int = 320  # dim maxima a imag pt detectie, doar pt detectare
    detect_every_n_frames: int = 1


@dataclass(frozen=True)
class FaceRoiConfig:
    padding_px: int = 20
    crop_scale: float = 0.9
    crop_center_y_ratio: float = 0.46
    onnx_use_equalization: bool = False


@dataclass(frozen=True)
class EmotionModelConfig:
    model_path: str = project_path("models", "ONNX", "emotion-ferplus-7.onnx")


@dataclass(frozen=True)
class SmoothingConfig:
    enabled: bool = True
    alpha: float = 0.35  #cat de repede urmareste noul bbox detectat 
    max_center_shift_ratio: float = 0.75 #cat de mult poate sa se miste centrul fetei intre doua detectii consecutive 
    min_size_ratio: float = 0.6 #cat de mult poate sa se schimbe dimensiunea fetei intre doua detectii consecutive

# Config pt fetele instabile 
@dataclass(frozen=True)
class FaceQualityConfig:
    min_face_width_px: int = 80
    min_face_height_px: int = 80
    min_border_margin_px: int = 4
    require_keypoints: bool = True
    min_eye_distance_ratio: float = 0.12
    max_eye_y_diff_ratio: float = 0.35

# Config pt stabilizarea emotiilor 
@dataclass(frozen=True)
class StabilizationConfig:
    enabled: bool = True
    min_confidence: float = 0.45 # pragul min pt a fi luata in considerare o emotie noua 
    min_margin: float = 0.0  #top1-top2 
    hold_last_stable: bool = True #daca o predictie noua e slaba, se pastreaza emotia anterioara 
    enable_voting: bool = False
    voting_window: int = 5 # nr de predictii daca voting e activat


@dataclass(frozen=True)
class LoggingConfig:
    debug_pipeline: bool = False #arata imaginile intermediare
    debug_logging: bool = False 
    draw_detection: bool = True
    show_raw_label: bool = True
    log_top3: bool = False 
    show_perf: bool = True
    perf_log_every_n_frames: int = 30


@dataclass(frozen=True)
class RuntimeConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    yunet: YuNetConfig = field(default_factory=YuNetConfig)
    face_roi: FaceRoiConfig = field(default_factory=FaceRoiConfig)
    emotion_model: EmotionModelConfig = field(default_factory=EmotionModelConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    face_quality: FaceQualityConfig = field(default_factory=FaceQualityConfig)
    stabilization: StabilizationConfig = field(default_factory=StabilizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


RUNTIME_CONFIG = RuntimeConfig()
