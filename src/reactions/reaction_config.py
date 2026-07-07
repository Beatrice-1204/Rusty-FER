from dataclasses import dataclass, field
from typing import Dict

from paths import project_path


REACTION_IMAGE_NAMES = {
    "idle": "idle.png",
    "happy": "happy.png",
    "sad": "sad.png",
    "angry": "angry.png",
    "surprise": "surprise.png",
}

REACTION_AUDIO_NAMES = {
    "happy": "happy.wav",
    "sad": "sad.wav",
    "angry": "angry.wav",
    "surprise": "surprise.wav",
}


@dataclass(frozen=True)
class ReactionDisplayConfig:
    image_dir: str = project_path("assets", "reactions", "images")
    image_names: Dict[str, str] = field(default_factory=lambda: dict(REACTION_IMAGE_NAMES))
    idle_emotion: str = "idle"
    fullscreen: bool = True
    procedural_idle_enabled: bool = True
    procedural_happy_enabled: bool = True
    procedural_surprise_enabled: bool = True
    procedural_angry_enabled: bool = True
    procedural_sad_enabled: bool = True 


@dataclass(frozen=True)
class ReactionAudioConfig:
    audio_dir: str = project_path("assets", "reactions", "audio")
    audio_names: Dict[str, str] = field(default_factory=lambda: dict(REACTION_AUDIO_NAMES))
    idle_emotion: str = "idle"
    startup_audio_name: str = "startup.wav"
    play_startup_audio: bool = True
