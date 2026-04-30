import time
from enum import Enum
from typing import Callable, Optional


class ReactionGateState(str, Enum):
    IDLE = "IDLE"
    DETECTING = "DETECTING"
    REACTING = "REACTING"
    COOLDOWN = "COOLDOWN"


class ReactionGate:
    def __init__(
        self,
        config,
        clock: Callable[[], float] = time.perf_counter,
        debug_logging: Optional[bool] = None,
    ):
        self.config = config
        self.clock = clock
        self.debug_logging = config.debug_logging if debug_logging is None else debug_logging
        self.state = ReactionGateState.IDLE
        self._state_started_at = self.clock()
        self._reaction_emotion: Optional[str] = None
        self._reaction_was_emitted = False
        self._candidate_emotion: Optional[str] = None

    def reset(self) -> None:
        self.state = ReactionGateState.IDLE
        self._state_started_at = self.clock()
        self._reaction_emotion = None
        self._reaction_was_emitted = False
        self._candidate_emotion = None
        self._log("reset", f"state={self.state.value}")

    def update(self, stable_emotion: Optional[str]) -> Optional[str]:
        if not self.config.enabled:
            return self._active_emotion_or_none(stable_emotion)

        now = self.clock()

        if self.state == ReactionGateState.IDLE:
            return self._update_idle(now)

        if self.state == ReactionGateState.DETECTING:
            return self._update_detecting(stable_emotion, now)

        if self.state == ReactionGateState.REACTING:
            return self._update_reacting()

        if self.state == ReactionGateState.COOLDOWN:
            return self._update_cooldown(now)

        return None

    def is_detecting(self) -> bool:
        return self.state == ReactionGateState.DETECTING

    def detecting_has_elapsed(self) -> bool:
        if not self.is_detecting():
            return False
        return (self.clock() - self._state_started_at) >= self.config.detecting_seconds

    def restart_detecting_window(self, reason: str = "window_summary") -> None:
        if self.is_detecting():
            self._restart_detecting(self.clock(), reason)

    def _update_idle(self, now: float) -> Optional[str]:
        elapsed = now - self._state_started_at
        if elapsed >= self.config.idle_seconds:
            self._transition(ReactionGateState.DETECTING, now)
        return None

    def _update_detecting(
        self,
        stable_emotion: Optional[str],
        now: float,
    ) -> Optional[str]:
        if self._is_active_emotion(stable_emotion):
            self._candidate_emotion = stable_emotion
            self._reaction_emotion = self._candidate_emotion
            self._reaction_was_emitted = False
            self._candidate_emotion = None
            self._transition(ReactionGateState.REACTING, now)
            return self._emit_reaction_once()

        if stable_emotion == self.config.neutral_label:
            self._candidate_emotion = None

        return None

    def _update_reacting(self) -> Optional[str]:
        now = self.clock()
        elapsed = now - self._state_started_at
        if elapsed >= self.config.reacting_seconds:
            self._transition(ReactionGateState.COOLDOWN, now)
            self._log("cooldown_start", f"duration={self.config.cooldown_seconds:.1f}s")
            return None

        return self._emit_reaction_once()

    def _update_cooldown(self, now: float) -> Optional[str]:
        elapsed = now - self._state_started_at
        if elapsed >= self.config.cooldown_seconds:
            self._transition(ReactionGateState.IDLE, now)
            self._log("cooldown_end")
        return None

    def _emit_reaction_once(self) -> Optional[str]:
        if self._reaction_was_emitted or self._reaction_emotion is None:
            return None

        self._reaction_was_emitted = True
        self._log("trigger", f"emotion={self._reaction_emotion}")
        return self._reaction_emotion

    def _transition(self, next_state: ReactionGateState, now: float) -> None:
        if self.state == next_state:
            return

        previous_state = self.state
        self.state = next_state
        self._state_started_at = now
        if next_state in (ReactionGateState.IDLE, ReactionGateState.COOLDOWN):
            self._reaction_emotion = None
            self._reaction_was_emitted = False
            self._candidate_emotion = None

        self._log("state", f"{previous_state.value}->{next_state.value}")

    def _restart_detecting(self, now: float, reason: str) -> None:
        self.state = ReactionGateState.DETECTING
        self._state_started_at = now
        self._reaction_emotion = None
        self._reaction_was_emitted = False
        self._candidate_emotion = None
        self._log("detecting_restart", f"reason={reason}")

    def _active_emotion_or_none(self, stable_emotion: Optional[str]) -> Optional[str]:
        if self._is_active_emotion(stable_emotion):
            return stable_emotion
        return None

    def _is_active_emotion(self, stable_emotion: Optional[str]) -> bool:
        return (
            stable_emotion is not None
            and stable_emotion != self.config.neutral_label
        )

    def _log(self, event: str, *parts: str) -> None:
        if self.debug_logging:
            print("[REACTION_GATE]", event, *parts)
