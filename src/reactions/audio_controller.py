import os
from typing import Dict, Optional

from reactions.reaction_config import ReactionAudioConfig


class AudioController:
    def __init__(self, config: ReactionAudioConfig):
        self.config = config
        self._pygame = None
        self._sounds: Dict[str, object] = {}
        self._startup_sound: Optional[object] = None
        self._enabled = False

        self._initialize_audio()
        if self._enabled:
            self._preload_sounds()
            self._preload_startup_sound()

    def play(self, emotion: str) -> None:
        if not self._enabled or emotion == self.config.idle_emotion:
            return

        sound = self._sounds.get(emotion)
        if sound is None:
            print(f"[AUDIO] warning missing preloaded sound for emotion={emotion}")
            return

        sound.play()

    def play_startup(self) -> None:
        if (
            not self._enabled
            or not self.config.play_startup_audio
            or self._startup_sound is None
        ):
            return

        self._startup_sound.play()

    def close(self) -> None:
        if self._pygame is not None:
            try:
                self._pygame.mixer.quit()
            except self._pygame.error as error:
                print(f"[AUDIO] warning failed to close mixer error={error}")

    def _initialize_audio(self) -> None:
        try:
            import pygame

            self._pygame = pygame
            if not pygame.get_init():
                pygame.init()
            pygame.mixer.init()
            self._enabled = True
        except Exception as error:
            print(f"[AUDIO] warning audio disabled error={error}")
            self._enabled = False

    def _preload_sounds(self) -> None:
        for emotion, filename in self.config.audio_names.items():
            path = os.path.join(self.config.audio_dir, filename)
            sound = self._load_sound(path, f"emotion={emotion}")
            if sound is not None:
                self._sounds[emotion] = sound

    def _preload_startup_sound(self) -> None:
        if not self.config.play_startup_audio:
            return

        path = os.path.join(self.config.audio_dir, self.config.startup_audio_name)
        self._startup_sound = self._load_sound(path, "startup")

    def _load_sound(self, path: str, label: str):
        if not os.path.exists(path):
            print(f"[AUDIO] warning missing audio {label} path={path}")
            return None

        try:
            return self._pygame.mixer.Sound(path)
        except self._pygame.error as error:
            print(f"[AUDIO] warning failed to load audio {label} path={path} error={error}")
            return None
