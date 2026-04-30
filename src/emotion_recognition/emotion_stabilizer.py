from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple


Prediction = Tuple[str, float, Optional[str]]


@dataclass(frozen=True)
class StableEmotionResult:
    emotion: Optional[str]
    confidence: Optional[float]
    raw_label: Optional[str]
    changed: bool = False
    reason: Optional[str] = None


@dataclass(frozen=True)
class StabilizationSummary:
    dominant_emotion: Optional[str]
    selected_emotion: Optional[str]
    count: int
    required: int
    average_confidence: Optional[float]
    threshold: Optional[float]
    accepted: bool
    reason: str
    total_predictions: int

    def to_log_parts(self) -> List[str]:
        avg_conf = (
            "None"
            if self.average_confidence is None
            else f"{self.average_confidence:.3f}"
        )
        threshold = "None" if self.threshold is None else f"{self.threshold:.3f}"
        return [
            f"dominant={self.dominant_emotion}",
            f"selected={self.selected_emotion}",
            f"count={self.count}/{self.total_predictions}",
            f"required={self.required}",
            f"avg_conf={avg_conf}",
            f"threshold={threshold}",
            f"accepted={self.accepted}",
            f"reason={self.reason}",
        ]


class EmotionStabilizer:
    def __init__(self, config, debug_logging: bool = False):
        self.config = config
        self.debug_logging = debug_logging
        self._predictions: Deque[Prediction] = deque(maxlen=config.voting_window)
        self._last_emotion: Optional[str] = None
        self._last_confidence: Optional[float] = None
        self._last_raw_label: Optional[str] = None

    def reset(self, clear_stable: bool = False) -> None:
        self._predictions.clear()
        if clear_stable:
            self._last_emotion = None
            self._last_confidence = None
            self._last_raw_label = None

    def summarize(self, neutral_label: str = "neutral") -> StabilizationSummary:
        if not self._predictions:
            return StabilizationSummary(
                dominant_emotion=None,
                selected_emotion=None,
                count=0,
                required=1,
                average_confidence=None,
                threshold=None,
                accepted=False,
                reason="insufficient_occurrences",
                total_predictions=0,
            )

        counts = Counter(label for label, _, _ in self._predictions)
        dominant_label, dominant_count = counts.most_common(1)[0]
        dominant_scores = [
            score for label, score, _ in self._predictions if label == dominant_label
        ]
        average_confidence = sum(dominant_scores) / len(dominant_scores)
        required = self._min_occurrences(dominant_label)
        threshold = self._confidence_threshold(dominant_label)
        grouped_scores: Dict[str, List[float]] = defaultdict(list)
        for label, score, _ in self._predictions:
            grouped_scores[label].append(score)

        accepted_candidates = []
        for label, scores in grouped_scores.items():
            occurrences = len(scores)
            candidate_average = sum(scores) / occurrences
            if (
                occurrences >= self._min_occurrences(label)
                and candidate_average >= self._confidence_threshold(label)
            ):
                accepted_candidates.append((label, occurrences, candidate_average))

        active_candidates = [
            candidate for candidate in accepted_candidates if candidate[0] != neutral_label
        ]
        if active_candidates:
            selected_label = max(
                active_candidates,
                key=lambda candidate: (candidate[1], candidate[2]),
            )[0]
            return StabilizationSummary(
                dominant_emotion=dominant_label,
                selected_emotion=selected_label,
                count=dominant_count,
                required=required,
                average_confidence=average_confidence,
                threshold=threshold,
                accepted=True,
                reason="accepted",
                total_predictions=len(self._predictions),
            )

        if dominant_label == neutral_label:
            accepted = False
            reason = "neutral_dominant"
        elif dominant_count < required:
            accepted = False
            reason = "insufficient_occurrences"
        elif average_confidence < threshold:
            accepted = False
            reason = "below_threshold"
        else:
            accepted = True
            reason = "accepted"

        return StabilizationSummary(
            dominant_emotion=dominant_label,
            selected_emotion=None,
            count=dominant_count,
            required=required,
            average_confidence=average_confidence,
            threshold=threshold,
            accepted=accepted,
            reason=reason,
            total_predictions=len(self._predictions),
        )

    def update(
        self,
        emotion_label: Optional[str],
        confidence: Optional[float],
        raw_label: Optional[str] = None,
        top3: Optional[Iterable[Tuple[str, float]]] = None,
    ) -> StableEmotionResult:
        if emotion_label is None or confidence is None:
            return self._current_result(reason="missing_prediction")

        if not self.config.enabled:
            return self._set_stable(emotion_label, confidence, raw_label, "disabled")

        margin = self._prediction_margin(top3, confidence)
        if margin < self.config.min_margin:
            self._log(
                "reject",
                f"label={emotion_label}",
                f"conf={confidence:.3f}",
                f"margin={margin:.3f}",
                f"min_margin={self.config.min_margin:.3f}",
            )
            return self._current_result(reason="low_margin")

        self._predictions.append((emotion_label, float(confidence), raw_label))
        self._log(
            "collect",
            f"label={emotion_label}",
            f"conf={confidence:.3f}",
            f"count={len(self._predictions)}/{self.config.voting_window}",
        )

        if not self.config.enable_voting:
            threshold = self._confidence_threshold(emotion_label)
            if confidence >= threshold:
                return self._set_stable(emotion_label, confidence, raw_label, "single_prediction")
            return self._current_result(reason="low_confidence")

        if len(self._predictions) < self.config.voting_window:
            return self._current_result(reason="window_not_full")

        return self._evaluate_window()

    def _evaluate_window(self) -> StableEmotionResult:
        grouped_scores: Dict[str, List[float]] = defaultdict(list)
        grouped_raw_labels: Dict[str, List[Optional[str]]] = defaultdict(list)

        for label, score, raw_label in self._predictions:
            grouped_scores[label].append(score)
            grouped_raw_labels[label].append(raw_label)

        counts = Counter(label for label, _, _ in self._predictions)
        dominant_label = counts.most_common(1)[0][0]
        self._log(
            "window",
            f"dominant={dominant_label}",
            f"predictions={dict(counts)}",
        )

        accepted_candidates = []
        for label, scores in grouped_scores.items():
            occurrences = len(scores)
            average_confidence = sum(scores) / occurrences
            min_occurrences = self._min_occurrences(label)
            threshold = self._confidence_threshold(label)

            if occurrences >= min_occurrences and average_confidence >= threshold:
                accepted_candidates.append((
                    label,
                    occurrences,
                    average_confidence,
                    self._most_common_raw_label(grouped_raw_labels[label]),
                ))
                self._log(
                    "accept_candidate",
                    f"label={label}",
                    f"occurrences={occurrences}/{min_occurrences}",
                    f"avg_conf={average_confidence:.3f}",
                    f"threshold={threshold:.3f}",
                )
            else:
                self._log(
                    "reject_candidate",
                    f"label={label}",
                    f"occurrences={occurrences}/{min_occurrences}",
                    f"avg_conf={average_confidence:.3f}",
                    f"threshold={threshold:.3f}",
                )

        non_neutral_candidates = [
            candidate for candidate in accepted_candidates if candidate[0] != "neutral"
        ]
        if non_neutral_candidates:
            label, occurrences, average_confidence, raw_label = max(
                non_neutral_candidates,
                key=lambda candidate: (candidate[1], candidate[2]),
            )
            return self._set_stable(
                label,
                average_confidence,
                raw_label,
                f"stable_window occurrences={occurrences}",
            )

        neutral_candidates = [
            candidate for candidate in accepted_candidates if candidate[0] == "neutral"
        ]
        if neutral_candidates:
            label, occurrences, average_confidence, raw_label = neutral_candidates[0]
            return self._set_stable(
                label,
                average_confidence,
                raw_label,
                f"neutral_rest occurrences={occurrences}",
            )

        return self._current_result(reason="no_stable_candidate")

    def _set_stable(
        self,
        emotion: str,
        confidence: float,
        raw_label: Optional[str],
        reason: str,
    ) -> StableEmotionResult:
        changed = emotion != self._last_emotion
        self._last_emotion = emotion
        self._last_confidence = float(confidence)
        self._last_raw_label = raw_label
        self._log(
            "stable",
            f"label={emotion}",
            f"conf={confidence:.3f}",
            f"reason={reason}",
        )
        return StableEmotionResult(
            emotion=self._last_emotion,
            confidence=self._last_confidence,
            raw_label=self._last_raw_label,
            changed=changed,
            reason=reason,
        )

    def _current_result(self, reason: Optional[str] = None) -> StableEmotionResult:
        return StableEmotionResult(
            emotion=self._last_emotion,
            confidence=self._last_confidence,
            raw_label=self._last_raw_label,
            changed=False,
            reason=reason,
        )

    def _prediction_margin(
        self,
        top3: Optional[Iterable[Tuple[str, float]]],
        confidence: float,
    ) -> float:
        if top3 is None:
            return confidence

        top_scores = [float(score) for _, score in top3]
        if len(top_scores) < 2:
            return top_scores[0] if top_scores else confidence
        return top_scores[0] - top_scores[1]

    def _confidence_threshold(self, label: str) -> float:
        return self.config.confidence_thresholds.get(label, self.config.min_confidence)

    def _min_occurrences(self, label: str) -> int:
        return self.config.min_occurrences.get(label, 1)

    def _most_common_raw_label(self, raw_labels: List[Optional[str]]) -> Optional[str]:
        valid_labels = [label for label in raw_labels if label is not None]
        if not valid_labels:
            return None
        return Counter(valid_labels).most_common(1)[0][0]

    def _log(self, event: str, *parts: str) -> None:
        if self.debug_logging:
            print("[STABLE]", event, *parts)
