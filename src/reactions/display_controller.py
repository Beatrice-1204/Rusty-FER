import os
import time
from typing import Dict, Optional, Tuple

from reactions.procedural_face_renderer import (
    SimpleOrangeAngryFace,
    SimpleOrangeFace,
    SimpleOrangeHappyFace,
    SimpleOrangeSadFace,
    SimpleOrangeSurpriseFace,
)
from reactions.reaction_config import ReactionDisplayConfig


class DisplayController:
    def __init__(self, config: ReactionDisplayConfig):
        self.config = config
        self._pygame = None
        self._screen = None
        self._images: Dict[str, object] = {}
        self._screen_size: Tuple[int, int] = (0, 0)
        self._current_emotion: Optional[str] = None
        self._procedural_face = (
            SimpleOrangeFace() if config.procedural_idle_enabled else None
        )
        self._procedural_happy_face = (
            SimpleOrangeHappyFace() if config.procedural_happy_enabled else None
        )
        self._procedural_surprise_face = (
            SimpleOrangeSurpriseFace() if config.procedural_surprise_enabled else None
        )
        self._procedural_angry_face = (
            SimpleOrangeAngryFace() if config.procedural_angry_enabled else None
        )
        self._procedural_sad_face = (
            SimpleOrangeSadFace() if config.procedural_sad_enabled else None
        )
        self._procedural_idle_active = False
        self._procedural_happy_active = False
        self._procedural_surprise_active = False
        self._procedural_angry_active = False
        self._procedural_sad_active = False
        self._last_update_time = time.perf_counter()

        self._initialize_display()
        self._preload_images()

    def show(self, emotion: str) -> None:
        if self._should_use_procedural_idle(emotion):
            self._procedural_idle_active = True
            self._procedural_happy_active = False
            self._procedural_surprise_active = False
            self._procedural_angry_active = False
            self._procedural_sad_active = False
            self._draw_procedural_idle()
            self._current_emotion = emotion
            return

        if self._should_use_procedural_happy(emotion):
            self._procedural_idle_active = False
            self._procedural_happy_active = True
            self._procedural_surprise_active = False
            self._procedural_angry_active = False
            self._procedural_sad_active = False
            self._draw_procedural_happy()
            self._current_emotion = emotion
            return

        if self._should_use_procedural_surprise(emotion):
            self._procedural_idle_active = False
            self._procedural_happy_active = False
            self._procedural_surprise_active = True
            self._procedural_angry_active = False
            self._procedural_sad_active = False
            self._draw_procedural_surprise()
            self._current_emotion = emotion
            return

        if self._should_use_procedural_angry(emotion):
            self._procedural_idle_active = False
            self._procedural_happy_active = False
            self._procedural_surprise_active = False
            self._procedural_angry_active = True
            self._procedural_sad_active = False
            self._draw_procedural_angry()
            self._current_emotion = emotion
            return

        if self._should_use_procedural_sad(emotion):
            self._procedural_idle_active = False
            self._procedural_happy_active = False
            self._procedural_surprise_active = False
            self._procedural_angry_active = False
            self._procedural_sad_active = True
            self._draw_procedural_sad()
            self._current_emotion = emotion
            return

        self._procedural_idle_active = False
        self._procedural_happy_active = False
        self._procedural_surprise_active = False
        self._procedural_angry_active = False
        self._procedural_sad_active = False
        image = self._images.get(emotion)
        selected_emotion = emotion

        if image is None:
            print(f"[DISPLAY] warning missing preloaded image for emotion={emotion}")
            selected_emotion = self.config.idle_emotion
            image = self._images.get(selected_emotion)

        if image is None:
            print("[DISPLAY] warning no image available to display")
            return

        self._draw_image(image)
        self._current_emotion = selected_emotion

    def update(self) -> bool:
        if self._pygame is None:
            return True

        now = time.perf_counter()
        dt = now - self._last_update_time
        self._last_update_time = now

        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                return False
            if event.type == self._pygame.KEYDOWN:
                if event.key in (self._pygame.K_q, self._pygame.K_ESCAPE):
                    print("[APP] exit requested by keyboard")
                    return False

        if self._procedural_idle_active:
            self._procedural_face.update(dt)
            self._draw_procedural_idle()
        elif self._procedural_happy_active:
            self._procedural_happy_face.update(dt)
            self._draw_procedural_happy()
        elif self._procedural_surprise_active:
            self._procedural_surprise_face.update(dt)
            self._draw_procedural_surprise()
        elif self._procedural_angry_active:
            self._procedural_angry_face.update(dt)
            self._draw_procedural_angry()
        elif self._procedural_sad_active:
            self._procedural_sad_face.update(dt)
            self._draw_procedural_sad()

        return True

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None

    def _initialize_display(self) -> None:
        import pygame

        self._pygame = pygame
        pygame.init()
        pygame.display.set_caption("Rusty - Reactions")

        flags = pygame.FULLSCREEN if self.config.fullscreen else 0
        self._screen = pygame.display.set_mode((0, 0), flags)
        self._screen_size = self._screen.get_size()

    def _preload_images(self) -> None:
        for emotion, filename in self.config.image_names.items():
            path = os.path.join(self.config.image_dir, filename)
            if not os.path.exists(path):
                print(f"[DISPLAY] warning missing image emotion={emotion} path={path}")
                continue

            try:
                image = self._pygame.image.load(path).convert()
            except self._pygame.error as error:
                print(f"[DISPLAY] warning failed to load image path={path} error={error}")
                continue

            self._images[emotion] = image

    def _draw_image(self, image) -> None:
        self._screen.fill((0, 0, 0))
        scaled_image, position = self._scale_to_screen(image)
        self._screen.blit(scaled_image, position)
        self._pygame.display.flip()

    def _should_use_procedural_idle(self, emotion: str) -> bool:
        return (
            self._procedural_face is not None
            and emotion in (self.config.idle_emotion, "neutral")
        )

    def _draw_procedural_idle(self) -> None:
        self._procedural_face.draw(self._screen)
        self._pygame.display.flip()

    def _should_use_procedural_happy(self, emotion: str) -> bool:
        return self._procedural_happy_face is not None and emotion == "happy"

    def _draw_procedural_happy(self) -> None:
        self._procedural_happy_face.draw(self._screen)
        self._pygame.display.flip()

    def _should_use_procedural_surprise(self, emotion: str) -> bool:
        return self._procedural_surprise_face is not None and emotion == "surprise"

    def _draw_procedural_surprise(self) -> None:
        self._procedural_surprise_face.draw(self._screen)
        self._pygame.display.flip()

    def _should_use_procedural_angry(self, emotion: str) -> bool:
        return self._procedural_angry_face is not None and emotion == "angry"

    def _draw_procedural_angry(self) -> None:
        self._procedural_angry_face.draw(self._screen)
        self._pygame.display.flip()

    def _should_use_procedural_sad(self, emotion: str) -> bool:
        return self._procedural_sad_face is not None and emotion == "sad"

    def _draw_procedural_sad(self) -> None:
        self._procedural_sad_face.draw(self._screen)
        self._pygame.display.flip()

    def _scale_to_screen(self, image):
        screen_width, screen_height = self._screen_size
        image_width, image_height = image.get_size()

        if image_width == 0 or image_height == 0:
            return image, (0, 0)

        scale = min(screen_width / image_width, screen_height / image_height)
        target_size = (
            max(1, int(image_width * scale)),
            max(1, int(image_height * scale)),
        )
        scaled_image = self._pygame.transform.smoothscale(image, target_size)
        position = (
            (screen_width - target_size[0]) // 2,
            (screen_height - target_size[1]) // 2,
        )
        return scaled_image, position
