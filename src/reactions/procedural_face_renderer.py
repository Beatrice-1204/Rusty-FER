import math
import random
from dataclasses import dataclass

import pygame


BG_COLOR = (0, 0, 0)
ORANGE = (255, 140, 20)


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(value, maximum))


def smoothstep(value: float) -> float:
    value = clamp(value)
    return value * value * (3.0 - 2.0 * value)


@dataclass(frozen=True)
class Eye:
    center_x: float
    center_y: float
    width: float
    height: float

    def draw(self, surface, openness: float) -> None:
        w, h = surface.get_size()
        eye_w = int(w * self.width)
        eye_h = int(h * self.height * max(openness, 0.055))
        center_x = int(w * self.center_x)
        center_y = int(h * self.center_y)
        rect = pygame.Rect(
            center_x - eye_w // 2,
            center_y - eye_h // 2,
            eye_w,
            eye_h,
        )
        pygame.draw.ellipse(surface, ORANGE, rect)


class BlinkController:
    def __init__(
        self,
        min_interval: float,
        max_interval: float,
        blink_duration: float,
    ):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.blink_duration = blink_duration
        self.openness = 1.0
        self._elapsed = 0.0
        self._is_blinking = False
        self._time_until_blink = self._next_interval()

    def update(self, dt: float) -> None:
        if self._is_blinking:
            self._elapsed += max(0.0, dt)
            progress = min(1.0, self._elapsed / self.blink_duration)
            if progress < 0.5:
                close_amount = progress / 0.5
            else:
                close_amount = 1.0 - (progress - 0.5) / 0.5
            close_amount = smoothstep(close_amount)
            self.openness = 1.0 - close_amount

            if progress >= 1.0:
                self._is_blinking = False
                self._elapsed = 0.0
                self.openness = 1.0
                self._time_until_blink = self._next_interval()
            return

        self._time_until_blink -= max(0.0, dt)
        if self._time_until_blink <= 0.0:
            self.force_blink()

    def force_blink(self) -> None:
        self._is_blinking = True
        self._elapsed = 0.0
        self.openness = 1.0

    def _next_interval(self) -> float:
        return random.uniform(self.min_interval, self.max_interval)


class SimpleOrangeFace:
    def __init__(self):
        self.left_eye = Eye(0.35, 0.40, 0.19, 0.22)
        self.right_eye = Eye(0.65, 0.40, 0.19, 0.22)
        self.blink = BlinkController(
            min_interval=2.2,
            max_interval=5.0,
            blink_duration=0.16,
        )

    def update(self, dt: float) -> None:
        self.blink.update(dt)

    def draw(self, surface) -> None:
        surface.fill(BG_COLOR)
        self.left_eye.draw(surface, self.blink.openness)
        self.right_eye.draw(surface, self.blink.openness)
        self._draw_neutral_mouth(surface)

    def force_blink(self) -> None:
        self.blink.force_blink()

    def _draw_neutral_mouth(self, surface) -> None:
        w, h = surface.get_size()
        mouth_w = int(w * 0.14)
        thickness = max(5, int(h * 0.016))
        center_x = int(w * 0.50)
        y = int(h * 0.62)
        x_start = center_x - mouth_w // 2
        x_end = center_x + mouth_w // 2
        radius = thickness // 2

        pygame.draw.line(
            surface,
            ORANGE,
            (x_start, y),
            (x_end, y),
            thickness,
        )
        pygame.draw.circle(surface, ORANGE, (x_start, y), radius)
        pygame.draw.circle(surface, ORANGE, (x_end, y), radius)


class SimpleOrangeHappyFace:
    def __init__(self):
        self.time = 0.0

    def update(self, dt: float) -> None:
        self.time += dt

    def draw_happy_eye(
        self,
        surface,
        cx,
        cy,
        w,
        h,
        phase: float = 0.0,
    ) -> None:
        bob = math.sin(self.time * 3.5 + phase) * h * 0.03

        rect = pygame.Rect(0, 0, w, h)
        rect.center = (int(cx), int(cy + bob))

        thickness = max(6, int(h * 0.18))

        pygame.draw.arc(
            surface,
            ORANGE,
            rect,
            math.radians(20),
            math.radians(160),
            thickness,
        )

    def draw_eyes(self, surface, w, h) -> None:
        eye_w = w * 0.17
        eye_h = h * 0.13

        left_cx = w * 0.35
        right_cx = w * 0.65
        cy = h * 0.39

        self.draw_happy_eye(surface, left_cx, cy, eye_w, eye_h, phase=0.0)
        self.draw_happy_eye(surface, right_cx, cy, eye_w, eye_h, phase=0.45)

    def draw_mouth(self, surface, w, h) -> None:
        pulse = 0.5 + 0.5 * math.sin(self.time * 4.0)

        mouth_w = w * (0.28 + 0.015 * pulse)
        mouth_h = h * (0.19 + 0.018 * pulse)
        mouth_y = h * (0.61 + 0.006 * math.sin(self.time * 4.0))

        rect = pygame.Rect(0, 0, mouth_w, mouth_h)
        rect.center = (int(w * 0.50), int(mouth_y))

        thickness = max(7, int(h * 0.020))

        pygame.draw.arc(
            surface,
            ORANGE,
            rect,
            math.radians(185),
            math.radians(355),
            thickness,
        )

    def draw(self, surface) -> None:
        w, h = surface.get_size()
        surface.fill(BG_COLOR)

        self.draw_eyes(surface, w, h)
        self.draw_mouth(surface, w, h)


class SimpleOrangeSurpriseFace:
    def __init__(self):
        self.time = 0.0

    def update(self, dt: float) -> None:
        self.time += dt

    def draw_eye(
        self,
        surface,
        cx,
        cy,
        w,
        h,
        phase: float = 0.0,
    ) -> None:
        pulse = 0.5 + 0.5 * math.sin(self.time * 3.2 + phase)
        eye_w = w * (1.0 + 0.04 * pulse)
        eye_h = h * (1.0 + 0.08 * pulse)

        rect = pygame.Rect(0, 0, eye_w, eye_h)
        rect.center = (int(cx), int(cy))
        pygame.draw.ellipse(surface, ORANGE, rect, max(6, int(h * 0.12)))

    def draw_eyes(self, surface, w, h) -> None:
        eye_w = w * 0.14
        eye_h = h * 0.20

        left_cx = w * 0.35
        right_cx = w * 0.65
        cy = h * 0.39

        self.draw_eye(surface, left_cx, cy, eye_w, eye_h, phase=0.0)
        self.draw_eye(surface, right_cx, cy, eye_w, eye_h, phase=0.35)

    def draw_mouth(self, surface, w, h) -> None:
        pulse = 0.5 + 0.5 * math.sin(self.time * 3.0 + 0.4)

        mouth_w = w * (0.09 + 0.010 * pulse)
        mouth_h = h * (0.15 + 0.015 * pulse)
        mouth_y = h * (0.63 + 0.004 * math.sin(self.time * 3.0))

        rect = pygame.Rect(0, 0, mouth_w, mouth_h)
        rect.center = (int(w * 0.50), int(mouth_y))

        thickness = max(7, int(h * 0.020))
        pygame.draw.ellipse(surface, ORANGE, rect, thickness)

    def draw(self, surface) -> None:
        w, h = surface.get_size()
        surface.fill(BG_COLOR)

        self.draw_eyes(surface, w, h)
        self.draw_mouth(surface, w, h)


class SimpleOrangeAngryFace:
    def __init__(self):
        self.time = 0.0

    def update(self, dt: float) -> None:
        self.time += dt

    def draw_angry_eye(
        self,
        surface,
        cx,
        cy,
        w,
        h,
        left: bool = True,
        phase: float = 0.0,
    ) -> None:
        tension = math.sin(self.time * 5.0 + phase) * h * 0.015
        thickness = max(6, int(h * 0.18))

        rect = pygame.Rect(0, 0, w, h)
        rect.center = (int(cx), int(cy + tension))

        if left:
            start_deg, end_deg = 205, 335
        else:
            start_deg, end_deg = 205, 335

        pygame.draw.arc(
            surface,
            ORANGE,
            rect,
            math.radians(start_deg),
            math.radians(end_deg),
            thickness,
        )

        if left:
            x1 = int(cx - w * 0.48)
            y1 = int(cy - h * 0.55)
            x2 = int(cx + w * 0.38)
            y2 = int(cy - h * 0.18)
        else:
            x1 = int(cx - w * 0.38)
            y1 = int(cy - h * 0.18)
            x2 = int(cx + w * 0.48)
            y2 = int(cy - h * 0.55)

        pygame.draw.line(
            surface,
            ORANGE,
            (x1, y1),
            (x2, y2),
            max(6, int(h * 0.14)),
        )

    def draw_eyes(self, surface, w, h) -> None:
        eye_w = w * 0.17
        eye_h = h * 0.12

        left_cx = w * 0.35
        right_cx = w * 0.65
        cy = h * 0.42

        self.draw_angry_eye(surface, left_cx, cy, eye_w, eye_h, left=True, phase=0.0)
        self.draw_angry_eye(
            surface,
            right_cx,
            cy,
            eye_w,
            eye_h,
            left=False,
            phase=0.35,
        )

    def draw_mouth(self, surface, w, h) -> None:
        pulse = 0.5 + 0.5 * math.sin(self.time * 4.8)

        mouth_w = w * (0.18 + 0.008 * pulse)
        mouth_h = h * (0.08 + 0.004 * pulse)
        mouth_y = h * (0.66 + 0.002 * math.sin(self.time * 4.8))

        rect = pygame.Rect(0, 0, mouth_w, mouth_h)
        rect.center = (int(w * 0.50), int(mouth_y))

        thickness = max(7, int(h * 0.020))

        pygame.draw.arc(
            surface,
            ORANGE,
            rect,
            math.radians(20),
            math.radians(160),
            thickness,
        )

    def draw(self, surface) -> None:
        w, h = surface.get_size()
        surface.fill(BG_COLOR)

        self.draw_eyes(surface, w, h)
        self.draw_mouth(surface, w, h)


class SimpleOrangeSadFace:
    def __init__(self):
        self.time = 0.0

    def update(self, dt: float) -> None:
        self.time += dt

    def draw_sad_eye(
        self,
        surface,
        cx,
        cy,
        w,
        h,
        phase: float = 0.0,
    ) -> None:
        bob = math.sin(self.time * 1.8 + phase) * h * 0.02

        rect = pygame.Rect(0, 0, w, h)
        rect.center = (int(cx), int(cy + bob))

        thickness = max(6, int(h * 0.18))

        pygame.draw.arc(
            surface,
            ORANGE,
            rect,
            math.radians(200),
            math.radians(340),
            thickness,
        )

    def draw_eyes(self, surface, w, h) -> None:
        eye_w = w * 0.17
        eye_h = h * 0.13

        left_cx = w * 0.35
        right_cx = w * 0.65
        cy = h * 0.40

        self.draw_sad_eye(surface, left_cx, cy, eye_w, eye_h, phase=0.0)
        self.draw_sad_eye(surface, right_cx, cy, eye_w, eye_h, phase=0.45)

    def draw_mouth(self, surface, w, h) -> None:
        pulse = 0.5 + 0.5 * math.sin(self.time * 1.8)

        mouth_w = w * (0.20 + 0.008 * pulse)
        mouth_h = h * (0.12 + 0.008 * pulse)
        mouth_y = h * (0.67 + 0.004 * math.sin(self.time * 1.8))

        rect = pygame.Rect(0, 0, mouth_w, mouth_h)
        rect.center = (int(w * 0.50), int(mouth_y))

        thickness = max(7, int(h * 0.020))

        pygame.draw.arc(
            surface,
            ORANGE,
            rect,
            math.radians(20),
            math.radians(160),
            thickness,
        )

    def draw(self, surface) -> None:
        w, h = surface.get_size()
        surface.fill(BG_COLOR)

        self.draw_eyes(surface, w, h)
        self.draw_mouth(surface, w, h)
