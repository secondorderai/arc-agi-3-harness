"""Stock Duck trajectory with one confidence-gated recovery fork per level."""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Any

from inference.agent.tool_agent import AnalyzerTurnResult, ToolAgent
from ouro3.ledger import HypothesisLedger
from ouro3.trajectory import (
    RecoveryDecision,
    RecoveryPolicy,
    SessionSignals,
    derive_alternate_seed,
)


class RecoveryPhase(StrEnum):
    NORMAL = "normal"
    HYPOTHESIS = "hypothesis"
    EXECUTION = "execution"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


class DuckRobustToolAgent(ToolAgent):
    """Duck agent that forks only after measured stagnation."""

    def __init__(
        self,
        *,
        session_namespace: str,
        recovery_policy: RecoveryPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.session_namespace = str(session_namespace)
        self.signals = SessionSignals()
        self.recovery_policy = recovery_policy or RecoveryPolicy(
            minimum_actions=max(
                1, _env_int("OURO3_ROBUST_MIN_ACTIONS", 64)
            ),
            warmup_seconds=max(
                0.0, _env_float("OURO3_ROBUST_WARMUP_SECONDS", 30 * 60)
            ),
            low_probability_threshold=_env_float(
                "OURO3_ROBUST_SUCCESS_THRESHOLD", 0.25
            ),
            required_low_windows=max(
                1, _env_int("OURO3_ROBUST_REQUIRED_LOW_WINDOWS", 2)
            ),
        )
        self.phase = RecoveryPhase.NORMAL
        self._primary_seed = self._seed
        self._hypothesis_temperature = _env_float(
            "OURO3_ROBUST_HYPOTHESIS_TEMPERATURE", 0.8
        )
        self._execution_temperature = _env_float(
            "OURO3_ROBUST_EXECUTION_TEMPERATURE", 0.2
        )
        self._max_execution_batch = max(
            1, min(8, _env_int("OURO3_ROBUST_MAX_EXECUTION_BATCH", 8))
        )
        self._recovery_count = 0
        self._recovery_successes = 0
        self._recovery_resets = 0
        self._prediction_matches = 0
        self._prediction_mismatches = 0
        self._last_decision: RecoveryDecision | None = None
        self._recovery_events: list[dict[str, Any]] = []

    @property
    def augmented_features_enabled(self) -> bool:
        return self.phase != RecoveryPhase.NORMAL

    @property
    def prediction_verification_enabled(self) -> bool:
        return self.augmented_features_enabled

    @property
    def ledger(self) -> HypothesisLedger:
        return HypothesisLedger.from_duck_memory(self._summarized_knowledge)

    @property
    def has_active_contradiction(self) -> bool:
        return self.ledger.has_active_contradiction

    @property
    def maximum_action_batch_size(self) -> int | None:
        if self.phase == RecoveryPhase.HYPOTHESIS:
            return 1
        if self.phase == RecoveryPhase.EXECUTION:
            return self._max_execution_batch
        return None

    @property
    def telemetry(self) -> dict[str, int]:
        value = dict(super().telemetry)
        value.update(
            {
                "recovery_count": self._recovery_count,
                "recovery_successes": self._recovery_successes,
                "recovery_resets": self._recovery_resets,
                "prediction_matches": self._prediction_matches,
                "prediction_mismatches": self._prediction_mismatches,
            }
        )
        return value

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "recovered_levels": list(self.recovery_policy.recovered_levels),
            "signals": self.signals.diagnostic_payload(),
            "events": list(self._recovery_events),
            "predictor_validation": dict(
                self.recovery_policy.predictor.metadata.get("validation", {})
            ),
        }

    def maybe_begin_recovery(
        self,
        *,
        action_count: int,
        level: int,
        elapsed_seconds: float,
    ) -> RecoveryDecision:
        if self.phase != RecoveryPhase.NORMAL:
            probability = self.recovery_policy.predictor.probability(
                self.signals.predictor_features()
            )
            return RecoveryDecision(False, probability, "recovery already active")
        self.signals.level = max(1, int(level))
        if (
            self.signals.actions_since_progress == 0
            and not self.signals.action_names
            and self.signals.prior_levels == 0
        ):
            self.signals.actions_since_progress = max(0, int(action_count))
        self.signals.latest_elapsed_s = max(
            self.signals.latest_elapsed_s, float(elapsed_seconds)
        )
        decision = self.recovery_policy.evaluate(
            self.signals,
            contradiction=self.has_active_contradiction,
        )
        self._last_decision = decision
        if decision.triggered:
            self._begin_recovery(decision)
        return decision

    def _begin_recovery(self, decision: RecoveryDecision) -> None:
        self._recovery_count += 1
        alternate_seed = derive_alternate_seed(
            self.session_namespace,
            level=self.signals.level,
            recovery_index=self._recovery_count,
        )
        ledger = self.ledger.compact()
        self._summarized_knowledge.update(ledger.to_duck_memory())
        self._history_messages = []
        self._seed = alternate_seed
        self._sampling_temperature_override = self._hypothesis_temperature
        self.phase = RecoveryPhase.HYPOTHESIS
        self._recovery_events.append(
            {
                "event": "recovery_started",
                "level": self.signals.level,
                "action": self.signals.actions_since_progress,
                "elapsed_seconds": round(self.signals.latest_elapsed_s, 3),
                "probability": decision.probability,
                "reason": decision.reason,
                "alternate_seed": alternate_seed,
                "hypothesis_temperature": self._hypothesis_temperature,
                "execution_temperature": self._execution_temperature,
                "reset_recommended": decision.reset_recommended,
                "preserved_ledger": ledger.to_payload(),
            }
        )

    def register_recovery_reset(self) -> None:
        self._recovery_resets += 1
        self._recovery_events.append(
            {
                "event": "recovery_reset",
                "level": self.signals.level,
                "action": self.signals.actions_since_progress,
            }
        )

    def register_prediction_mismatch(self, reason: str) -> None:
        ledger = self.ledger
        ledger.contradict(reason)
        self._summarized_knowledge.update(ledger.to_duck_memory())
        self.signals.contradiction_count += 1
        self.signals.prediction_mismatches += 1
        self._prediction_mismatches += 1
        self._recovery_events.append(
            {
                "event": "prediction_mismatch",
                "level": self.signals.level,
                "reason": str(reason),
            }
        )
        self._end_recovery("prediction_mismatch")

    def register_prediction_match(self) -> None:
        self.signals.prediction_matches += 1
        self._prediction_matches += 1

    def observe_transition(
        self,
        *,
        action: str,
        before_grid: Any,
        after_grid: Any,
        payload: dict[str, Any],
    ) -> None:
        telemetry = super().telemetry
        self.signals.observe(
            action=action,
            before_grid=before_grid,
            after_grid=after_grid,
            payload=payload,
            generated_tokens=self.generated_tokens,
            request_count=telemetry["request_count"],
            context_evictions=telemetry["context_evictions"],
        )
        if self.phase == RecoveryPhase.NORMAL:
            return
        if payload.get("level_completed") or payload.get("run_complete"):
            self._recovery_successes += 1
            self._recovery_events.append(
                {
                    "event": "recovery_succeeded",
                    "level": self.signals.prior_levels,
                    "action": payload.get("action_num"),
                }
            )
            self._end_recovery("level_progress")
        elif payload.get("game_over"):
            self._recovery_events.append(
                {
                    "event": "recovery_failed",
                    "level": self.signals.level,
                    "reason": "game_over",
                }
            )
            self._end_recovery("game_over")

    def _end_recovery(self, reason: str) -> None:
        if self.phase == RecoveryPhase.NORMAL:
            return
        self._recovery_events.append(
            {
                "event": "recovery_ended",
                "level": self.signals.level,
                "reason": reason,
            }
        )
        self.phase = RecoveryPhase.NORMAL
        self._seed = self._primary_seed
        self._sampling_temperature_override = None

    def _build_user_prompt(self, *args: Any, **kwargs: Any) -> str:
        prompt = super()._build_user_prompt(*args, **kwargs)
        if self.phase == RecoveryPhase.NORMAL:
            return prompt
        if self.phase == RecoveryPhase.HYPOTHESIS:
            return (
                f"{prompt}\n\nRECOVERY FORK (success probability "
                f"{self._last_decision.probability if self._last_decision else 0.0:.4f}):\n"
                "- The previous trajectory is low-confidence. Treat its plan as falsified, but retain verified evidence.\n"
                "- State two genuinely competing causal explanations for the scene.\n"
                "- For each explanation, state supporting evidence and evidence against it.\n"
                "- Choose exactly one low-risk action that best distinguishes the explanations.\n"
                "- Execute that one action with an `expect` object predicting its frame-change signature.\n"
                "- Do not execute a longer plan on this turn."
            )
        return (
            f"{prompt}\n\nVERIFIED RECOVERY EXECUTION:\n"
            f"- Use at most {self._max_execution_batch} actions per batch.\n"
            "- Every action must include an `expect` object.\n"
            "- Prefer temperature-stable execution of the best surviving hypothesis.\n"
            "- Stop and reconsider immediately after any prediction mismatch."
        )

    def _normalize_python_actions(self, value: Any) -> list[dict[str, Any]]:
        normalized = super()._normalize_python_actions(value)
        limit = self.maximum_action_batch_size
        if limit is not None and len(normalized) > limit:
            raise ValueError(
                f"Recovery permits at most {limit} action(s) in this batch."
            )
        if self.augmented_features_enabled:
            for index, action in enumerate(normalized, start=1):
                if not isinstance(action.get("expect"), dict):
                    raise ValueError(
                        f"Recovery action {index} requires an `expect` object."
                    )
        return normalized

    def analyze(
        self,
        state_path: Path,
        action_num: int,
        **kwargs: Any,
    ) -> AnalyzerTurnResult | None:
        result = super().analyze(state_path, action_num, **kwargs)
        if (
            self.phase == RecoveryPhase.HYPOTHESIS
            and result is not None
            and result.step_executed
        ):
            self.phase = RecoveryPhase.EXECUTION
            self._sampling_temperature_override = self._execution_temperature
            self._recovery_events.append(
                {
                    "event": "recovery_execution_started",
                    "level": self.signals.level,
                    "temperature": self._execution_temperature,
                }
            )
        return result
