from typing import Optional

from reactions.audio_controller import AudioController
from reactions.display_controller import DisplayController


class ReactionManager:
    def __init__(
        self,
        display_controller: DisplayController,
        audio_controller: Optional[AudioController] = None,
    ):
        self.display_controller = display_controller
        self.audio_controller = audio_controller

    def handle(self, emotion: str) -> None:
        self.display_controller.show(emotion)
        if self.audio_controller is not None:
            self.audio_controller.play(emotion)

    def update(self) -> bool:
        for controller in (self.display_controller, self.audio_controller):
            update = getattr(controller, "update", None)
            if update is not None and update() is False:
                return False
        return True

    def play_startup(self) -> None:
        if self.audio_controller is not None:
            self.audio_controller.play_startup()

    def close(self) -> None:
        self.display_controller.close()
        if self.audio_controller is not None:
            self.audio_controller.close()
