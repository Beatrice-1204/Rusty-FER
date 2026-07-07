import time
from enum import Enum
from typing import Callable, Optional


class RobotMode(str, Enum):
    STANDBY = "STANDBY"
    ACTIVE = "ACTIVE"


class RobotModeManager:
    def __init__(
        self,
        config,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.config = config
        self.clock = clock
        self.mode = self._initial_mode(getattr(config, "initial_mode", RobotMode.STANDBY.value))
        self.last_activity_time = self.clock()
        self._standby_face_started_at: Optional[float] = None
        self._last_activation_log_second: Optional[int] = None

    def is_active(self) -> bool:
        return self.mode == RobotMode.ACTIVE

    def is_standby(self) -> bool:
        return self.mode == RobotMode.STANDBY

    def set_active(self) -> None:
        if self.mode == RobotMode.ACTIVE:
            return

        previous_mode = self.mode
        self.mode = RobotMode.ACTIVE
        self.mark_activity()
        self._reset_standby_activation()
        print(f"[MODE] {previous_mode.value} -> {self.mode.value}")

    def set_standby(self) -> None:
        if self.mode == RobotMode.STANDBY:
            return

        previous_mode = self.mode
        self.mode = RobotMode.STANDBY
        self._reset_standby_activation()
        print(f"[MODE] {previous_mode.value} -> {self.mode.value}")

    def toggle(self) -> None:
        if self.is_active():
            self.set_standby()
        else:
            self.set_active()

    def mark_activity(self) -> None:
        self.last_activity_time = self.clock()

    def update(self, face_detected: bool) -> RobotMode:
        if self.is_standby():
            if self.should_activate_from_standby(face_detected):
                self.set_active()
            return self.mode

        if face_detected:
            self.mark_activity()
            return self.mode

        elapsed = self.clock() - self.last_activity_time
        timeout = float(getattr(self.config, "inactivity_timeout_seconds", 30.0))
        if elapsed >= timeout:
            print(
                "[MODE] inactivity timeout reached",
                f"elapsed={elapsed:.1f}s",
                f"timeout={timeout:.1f}s",
            )
            self.set_standby()

        return self.mode

    def should_activate_from_standby(self, face_detected: bool) -> bool:
        if not face_detected:
            self._reset_standby_activation()
            return False

        now = self.clock()
        if self._standby_face_started_at is None:
            self._standby_face_started_at = now
            self._last_activation_log_second = None

        elapsed = now - self._standby_face_started_at
        required = float(getattr(self.config, "activation_required_seconds", 1.5))
        self._log_standby_activation_progress(elapsed, required)
        return elapsed >= required

    def _reset_standby_activation(self) -> None:
        self._standby_face_started_at = None
        self._last_activation_log_second = None

    def _log_standby_activation_progress(self, elapsed: float, required: float) -> None:
        current_second = int(elapsed)
        if self._last_activation_log_second == current_second and elapsed < required:
            return

        self._last_activation_log_second = current_second
        print(
            "[MODE] standby face detected",
            f"elapsed={elapsed:.1f}s",
            f"required={required:.1f}s",
        )

    def _initial_mode(self, value: str) -> RobotMode:
        normalized = str(value).strip().upper()
        if normalized == RobotMode.ACTIVE.value:
            return RobotMode.ACTIVE
        return RobotMode.STANDBY
