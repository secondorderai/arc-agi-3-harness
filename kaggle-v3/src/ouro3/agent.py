"""Model adapters with compact memory and a deterministic failure floor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from inference.agent.runtime_state import load_runtime_state
from inference.agent.tool_agent import AnalyzerTurnResult, ToolAgent
from ouro3.fallback import DeterministicExplorer
from ouro3.ledger import HypothesisLedger


class HybridToolAgent(ToolAgent):
    """Duck ToolAgent plus explicit ledger and deterministic recovery."""

    def __init__(
        self,
        *,
        game_key: str,
        failure_floor: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.game_key = game_key
        self.failure_floor = max(1, int(failure_floor))
        self.consecutive_failures = 0
        self.fallback_count = 0
        self._explorer = DeterministicExplorer(game_key)

    @property
    def ledger(self) -> HypothesisLedger:
        return HypothesisLedger.from_duck_memory(self._summarized_knowledge)

    @property
    def has_active_contradiction(self) -> bool:
        return self.ledger.has_active_contradiction

    def register_prediction_mismatch(self, reason: str) -> None:
        ledger = self.ledger
        ledger.contradict(reason)
        self._summarized_knowledge.update(ledger.to_duck_memory())

    def analyze(self, *args: Any, **kwargs: Any) -> AnalyzerTurnResult | None:
        result = super().analyze(*args, **kwargs)
        if result is not None and result.step_executed and not result.retryable_failure:
            self.consecutive_failures = 0
            return result
        self.consecutive_failures += 1
        if self.consecutive_failures < self.failure_floor:
            return result

        state_path = _state_path_from_call(args, kwargs)
        step_env = kwargs.get("step_env")
        valid_actions = list(kwargs.get("valid_actions") or [])
        if state_path is None or not callable(step_env) or not valid_actions:
            return result
        current_frame, _history = load_runtime_state(state_path)
        if current_frame is None:
            return result
        action = self._explorer.choose(grid=current_frame.grid, valid_actions=valid_actions)
        payload = step_env(action)
        executed = bool(payload.get("executed")) if isinstance(payload, dict) else False
        self.fallback_count += 1
        self.consecutive_failures = 0
        return AnalyzerTurnResult(
            step_executed=executed,
            retryable_failure=False,
            reasoning=f"deterministic failure floor selected {action}",
        )


class ScriptedAnalyzer:
    """No-network analyzer used only for transport and scheduler rehearsals."""

    def __init__(self, game_key: str) -> None:
        self.game_key = game_key
        self._explorer = DeterministicExplorer(game_key)
        self.total_tokens = 0
        self.generated_tokens = 0
        self.has_active_contradiction = False

    def analyze(
        self,
        state_path: Path,
        action_num: int,
        valid_actions: list[str] | None = None,
        step_env: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        **_kwargs: Any,
    ) -> AnalyzerTurnResult:
        del action_num
        current_frame, _history = load_runtime_state(state_path)
        if current_frame is None or step_env is None or not valid_actions:
            return AnalyzerTurnResult(step_executed=False, retryable_failure=False)
        action = self._explorer.choose(grid=current_frame.grid, valid_actions=valid_actions)
        payload = step_env(action)
        return AnalyzerTurnResult(
            step_executed=bool(payload.get("executed")),
            retryable_failure=False,
            reasoning="scripted rehearsal action",
        )


def _state_path_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Path | None:
    value = args[0] if args else kwargs.get("state_path")
    return value if isinstance(value, Path) else None
