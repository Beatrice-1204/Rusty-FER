from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from paths import project_path


@dataclass(frozen=True)
class AppConfig:
    max_runtime_seconds: Optional[float] = None


@dataclass(frozen=True)
class CameraConfig:
    backend: str = "picamera2"
    camera_index: int = 0
    width: int = 640 
    height: int =480
    center_crop_enabled: bool = True
    center_crop_scale: float = 0.75


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
    onnx_use_equalization: bool = True 


@dataclass(frozen=True)
class UndistortConfig:
    enabled: bool = False
    camera_matrix: Optional[Tuple[Tuple[float, float, float], ...]] = None
    dist_coeffs: Optional[Tuple[float, ...]] = None


@dataclass(frozen=True)
class PanTiltConfig:
    enabled: bool = True 
    backend: str = "arducam"
    pan_channel: int = 1 #canalul pt pan
    tilt_channel: int = 0 #canalul pt tilt
    invert_pan: bool = False    
    invert_tilt: bool = True    
    debug_tracking: bool = True
    update_every_n_frames: int = 2
    dead_zone_x: int = 70
    dead_zone_y: int = 50
    error_smoothing_alpha: float = 0.35#procentaj pt eroarea curenta (1 este fara smoothing)
    min_move_updates: int = 1 #la cate update-uri in afara dead zone se ia in considerare miscarea 
    proportional_control_enabled: bool = True
    proportional_gain_x: float = 2.5
    proportional_gain_y: float = 2.0
    min_step_degrees: float = 0.2
    max_step_degrees: float = 2.0
    search_enabled: bool = True 
    search_after_no_face_seconds: float = 2.5
    search_update_every_n_frames: int = 5
    search_step_degrees: float = 1.0
    pan_start: float = 90.0
    tilt_start: float = 90.0
    pan_min: float = 0.0
    pan_max: float = 180.0
    tilt_min: float = 50.0
    tilt_max: float = 140.0
    step_degrees: float = 3.0


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
    enable_voting: bool = True
    voting_window: int = 18 # nr de predictii daca voting e activat
    confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "neutral": 0.45,
        "happy": 0.60,
        "surprise": 0.55,
        "sad": 0.40,
        "angry": 0.40,
    })
    min_occurrences: Dict[str, int] = field(default_factory=lambda: {
        "neutral": 1,
        "happy": 4,
        "surprise": 4,
        "sad": 6,
        "angry": 6,
    })


@dataclass(frozen=True)
class ReactionGateConfig:
    enabled: bool = True
    initial_idle_seconds: float = 4.0
    idle_seconds: float = 1.5
    detecting_seconds: float = 2.0
    reacting_seconds: float = 3.0
    cooldown_seconds: float = 3.0
    neutral_label: str = "neutral"
    debug_logging: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    debug_pipeline: bool = False #arata imaginile intermediare
    debug_logging: bool = False   
    show_camera_window: bool = True 
    draw_detection: bool = True
    show_raw_label: bool = True
    log_top3: bool = True 
    show_perf: bool = True
    perf_log_every_n_frames: int = 30


@dataclass(frozen=True)
class RuntimeConfig:
    app: AppConfig = field(default_factory=AppConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    yunet: YuNetConfig = field(default_factory=YuNetConfig)
    face_roi: FaceRoiConfig = field(default_factory=FaceRoiConfig)
    undistort: UndistortConfig = field(default_factory=UndistortConfig)
    pan_tilt: PanTiltConfig = field(default_factory=PanTiltConfig)
    emotion_model: EmotionModelConfig = field(default_factory=EmotionModelConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    face_quality: FaceQualityConfig = field(default_factory=FaceQualityConfig)
    stabilization: StabilizationConfig = field(default_factory=StabilizationConfig)
    reaction_gate: ReactionGateConfig = field(default_factory=ReactionGateConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


RUNTIME_CONFIG = RuntimeConfig()
