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


@dataclass(frozen=True)
class ReactionDisplayConfig:
    image_dir: str = project_path("assets", "reactions", "images")
    image_names: Dict[str, str] = field(default_factory=lambda: dict(REACTION_IMAGE_NAMES))
    idle_emotion: str = "idle"
    fullscreen: bool = True

