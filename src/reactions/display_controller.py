import os
from typing import Dict, Optional, Tuple

from reactions.reaction_config import ReactionDisplayConfig


class DisplayController:
    def __init__(self, config: ReactionDisplayConfig):
        self.config = config
        self._pygame = None
        self._screen = None
        self._images: Dict[str, object] = {}
        self._screen_size: Tuple[int, int] = (0, 0)
        self._current_emotion: Optional[str] = None

        self._initialize_display()
        self._preload_images()

    def show(self, emotion: str) -> None:
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
        return self._pump_events()

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()

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
        self._pump_events()

        self._screen.fill((0, 0, 0))
        scaled_image, position = self._scale_to_screen(image)
        self._screen.blit(scaled_image, position)
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

    def _pump_events(self) -> bool:
        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                return False
            if event.type == self._pygame.KEYDOWN:
                if event.key in (self._pygame.K_q, self._pygame.K_ESCAPE):
                    return False
        return True
