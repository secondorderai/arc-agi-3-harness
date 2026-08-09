"""Generic trajectory signals and confidence-gated recovery decisions."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


PREDICTOR_PATH = Path(__file__).with_name("recovery_predictor.json")


def _grid_digest(grid: Sequence[Sequence[int]]) -> str:
    digest = hashlib.sha1()
    for row in grid:
        digest.update(bytes(max(0, min(255, int(cell))) for cell in row))
        digest.update(b"\xff")
    return digest.hexdigest()[:16]


def derive_alternate_seed(namespace: str, *, level: int, recovery_index: int) -> int:
    """Derive a stable, non-game-specific alternate sampling seed."""

    payload = f"{namespace}:level={int(level)}:recovery={int(recovery_index)}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big") & 0x7FFF_FFFF


@dataclass
class SessionSignals:
    """Compact, game-agnostic evidence used by the recovery gate."""

    level: int = 1
    prior_levels: int = 0
    level_started_elapsed_s: float = 0.0
    actions_since_progress: int = 0
    generated_tokens_at_progress: int = 0
    requests_at_progress: int = 0
    context_evictions_at_progress: int = 0
    latest_elapsed_s: float = 0.0
    latest_generated_tokens: int = 0
    latest_request_count: int = 0
    latest_context_evictions: int = 0
    prediction_matches: int = 0
    prediction_mismatches: int = 0
    contradiction_count: int = 0
    action_names: deque[str] = field(default_factory=lambda: deque(maxlen=256))
    gameplay_changes: deque[bool] = field(default_factory=lambda: deque(maxlen=64))
    state_action_keys: deque[str] = field(default_factory=lambda: deque(maxlen=256))

    def observe(
        self,
        *,
        action: str,
        before_grid: Sequence[Sequence[int]],
        after_grid: Sequence[Sequence[int]],
        payload: dict[str, Any],
        generated_tokens: int,
        request_count: int,
        context_evictions: int,
    ) -> None:
        elapsed = float(payload.get("run_elapsed_seconds", self.latest_elapsed_s) or 0.0)
        self.latest_elapsed_s = max(self.latest_elapsed_s, elapsed)
        self.latest_generated_tokens = max(0, int(generated_tokens))
        self.latest_request_count = max(0, int(request_count))
        self.latest_context_evictions = max(0, int(context_evictions))
        self.actions_since_progress += 1
        normalized_action = str(action).strip().upper()
        self.action_names.append(normalized_action)
        self.gameplay_changes.append(
            bool(payload.get("gameplay_changed", payload.get("board_changed", False)))
        )
        self.state_action_keys.append(
            f"{_grid_digest(before_grid)}:{normalized_action}:{_grid_digest(after_grid)}"
        )
        if "prediction_matched" in payload:
            if bool(payload.get("prediction_matched")):
                self.prediction_matches += 1
            else:
                self.prediction_mismatches += 1

        if bool(payload.get("level_completed")) or bool(payload.get("run_complete")):
            self.mark_progress(
                next_level=max(self.level + 1, int(payload.get("level", self.level + 1) or self.level + 1)),
                elapsed_s=elapsed,
                generated_tokens=generated_tokens,
                request_count=request_count,
                context_evictions=context_evictions,
            )

    def mark_progress(
        self,
        *,
        next_level: int,
        elapsed_s: float,
        generated_tokens: int,
        request_count: int,
        context_evictions: int,
    ) -> None:
        self.prior_levels = max(self.prior_levels + 1, int(next_level) - 1)
        self.level = max(1, int(next_level))
        self.level_started_elapsed_s = max(0.0, float(elapsed_s))
        self.actions_since_progress = 0
        self.generated_tokens_at_progress = max(0, int(generated_tokens))
        self.requests_at_progress = max(0, int(request_count))
        self.context_evictions_at_progress = max(0, int(context_evictions))
        self.prediction_matches = 0
        self.prediction_mismatches = 0
        self.contradiction_count = 0
        self.action_names.clear()
        self.gameplay_changes.clear()
        self.state_action_keys.clear()

    @property
    def elapsed_minutes(self) -> float:
        return max(0.0, self.latest_elapsed_s - self.level_started_elapsed_s) / 60.0

    @property
    def repeated_cycle(self) -> bool:
        counts = Counter(self.state_action_keys)
        return bool(counts and max(counts.values()) >= 2)

    @property
    def recent_gameplay_changes(self) -> int:
        return sum(self.gameplay_changes)

    @property
    def recent_gameplay_changes_16(self) -> int:
        return sum(list(self.gameplay_changes)[-16:])

    def predictor_features(self) -> dict[str, float]:
        actions = list(self.action_names)
        count = max(1, len(actions))
        requests = max(0, self.latest_request_count - self.requests_at_progress)
        generated = max(0, self.latest_generated_tokens - self.generated_tokens_at_progress)
        dominant = max(Counter(actions).values(), default=0) / count
        switches = sum(first != second for first, second in zip(actions, actions[1:]))
        return {
            "elapsed_minutes": self.elapsed_minutes,
            "generated_tokens_10k": generated / 10_000.0,
            "request_action_ratio": min(1.0, requests / max(1, self.actions_since_progress)),
            "zero_token_ratio": max(0.0, 1.0 - min(1.0, requests / max(1, self.actions_since_progress))),
            "dominant_action_ratio": dominant,
            "action_switch_rate": switches / max(1, len(actions) - 1),
            "unique_action_ratio": min(1.0, len(set(actions)) / 6.0),
            "mouse_ratio": sum(action in {"ACTION6", "MOUSE"} for action in actions) / count,
            "prior_levels": min(self.prior_levels, 4) / 4.0,
        }

    def diagnostic_payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["action_names"] = list(self.action_names)
        value["gameplay_changes"] = list(self.gameplay_changes)
        value["state_action_keys"] = list(self.state_action_keys)
        value["repeated_cycle"] = self.repeated_cycle
        value["predictor_features"] = self.predictor_features()
        return value


@dataclass(frozen=True)
class TrajectoryPredictor:
    features: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    intercept: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Path = PREDICTOR_PATH) -> "TrajectoryPredictor":
        payload = json.loads(path.read_text(encoding="utf-8"))
        features = tuple(str(value) for value in payload["features"])
        means = tuple(float(value) for value in payload["means"])
        scales = tuple(float(value) for value in payload["scales"])
        weights = tuple(float(value) for value in payload["weights"])
        if not (len(features) == len(means) == len(scales) == len(weights)):
            raise ValueError("recovery predictor vector lengths do not match")
        if any(value <= 0 for value in scales):
            raise ValueError("recovery predictor scales must be positive")
        return cls(
            features=features,
            means=means,
            scales=scales,
            weights=weights,
            intercept=float(payload["intercept"]),
            metadata=payload,
        )

    def probability(self, values: dict[str, float]) -> float:
        logit = self.intercept
        for name, mean, scale, weight in zip(
            self.features, self.means, self.scales, self.weights
        ):
            logit += weight * ((float(values.get(name, 0.0)) - mean) / scale)
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))


@dataclass(frozen=True)
class RecoveryDecision:
    triggered: bool
    probability: float
    reason: str = ""
    reset_recommended: bool = False


@dataclass
class RecoveryPolicy:
    predictor: TrajectoryPredictor = field(default_factory=TrajectoryPredictor.from_path)
    minimum_actions: int = 64
    warmup_seconds: float = 30 * 60
    low_probability_threshold: float = 0.25
    required_low_windows: int = 2
    maximum_recent_changes_for_reset: int = 2
    _low_windows: int = 0
    _last_evaluated_action: int = -1
    _observed_level: int | None = None
    _recovered_levels: set[int] = field(default_factory=set)

    def evaluate(
        self,
        signals: SessionSignals,
        *,
        contradiction: bool,
    ) -> RecoveryDecision:
        if self._observed_level != signals.level:
            self._observed_level = signals.level
            self._low_windows = 0
            self._last_evaluated_action = -1

        probability = self.predictor.probability(signals.predictor_features())
        if signals.level in self._recovered_levels:
            return RecoveryDecision(False, probability, "recovery already used on this level")
        if signals.actions_since_progress < max(1, self.minimum_actions):
            return RecoveryDecision(False, probability, "minimum action evidence not reached")
        if signals.elapsed_minutes * 60 < max(0.0, self.warmup_seconds):
            return RecoveryDecision(False, probability, "warmup time not reached")
        if signals.actions_since_progress == self._last_evaluated_action:
            return RecoveryDecision(False, probability, "no new action evidence")
        self._last_evaluated_action = signals.actions_since_progress
        if probability < self.low_probability_threshold:
            self._low_windows += 1
        else:
            self._low_windows = 0
        if self._low_windows < max(1, self.required_low_windows):
            return RecoveryDecision(False, probability, "low-confidence evidence is not consecutive")
        if not (signals.repeated_cycle or contradiction):
            return RecoveryDecision(False, probability, "no repeated cycle or contradicted hypothesis")

        self._recovered_levels.add(signals.level)
        self._low_windows = 0
        reason_parts = [
            f"success_probability={probability:.4f}",
            f"actions_without_progress={signals.actions_since_progress}",
        ]
        if signals.repeated_cycle:
            reason_parts.append("repeated_state_action_cycle")
        if contradiction:
            reason_parts.append("contradicted_hypothesis")
        return RecoveryDecision(
            True,
            probability,
            ", ".join(reason_parts),
            reset_recommended=bool(
                contradiction
                and signals.recent_gameplay_changes_16
                <= self.maximum_recent_changes_for_reset
            ),
        )

    @property
    def recovered_levels(self) -> tuple[int, ...]:
        return tuple(sorted(self._recovered_levels))


def engine_actions_from_history(
    history: Iterable[dict[str, Any]],
) -> list[str]:
    """Public helper shared with the offline trainer and its tests."""

    actions: list[str] = []
    for record in history:
        action = record.get("action") if isinstance(record, dict) else None
        if isinstance(action, dict):
            name = str(action.get("id", "")).strip()
        else:
            name = str(action or "").strip()
        if name:
            actions.append(name)
    return actions
