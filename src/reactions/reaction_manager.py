from reactions.display_controller import DisplayController


class ReactionManager:
    def __init__(self, display_controller: DisplayController):
        self.display_controller = display_controller

    def handle(self, emotion: str) -> None:
        self.display_controller.show(emotion)

    def update(self) -> bool:
        return self.display_controller.update()

    def close(self) -> None:
        self.display_controller.close()
