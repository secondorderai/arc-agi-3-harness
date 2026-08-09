from __future__ import annotations

from pathlib import Path

from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.agent.tool_agent import AnalyzerTurnResult, ToolAgent, _is_context_length_error
from ouro3.agent import HybridToolAgent


def test_context_overflow_detection() -> None:
    assert _is_context_length_error(
        RuntimeError("maximum context length exceeded; reduce the length of the input prompt")
    )


def test_runtime_environment_controls_local_tool_budget(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_ANALYZER_TOOL_STEPS", "2")
    monkeypatch.setenv("LOCAL_ANALYZER_MAX_OUTPUT", "1024")
    monkeypatch.setenv("LOCAL_ANALYZER_CONTEXT_WINDOW", "16384")
    agent = ToolAgent(
        model="test-model",
        base_url="http://127.0.0.1:1/v1",
        provider="openai",
    )
    assert agent._tool_steps == 2
    assert agent._max_output_tokens == 1024
    assert agent._context_budget_tokens == 16384 - 1024 - 512


def test_model_timeout_uses_deterministic_failure_floor(monkeypatch, tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    frame = Frame(grid=((0, 0), (0, 1)), step=0, level=1)
    write_runtime_state(
        state_path,
        current_frame=frame,
        history=[HistoryEntry(action="", frame=frame)],
    )
    monkeypatch.setattr(
        ToolAgent,
        "analyze",
        lambda self, *args, **kwargs: AnalyzerTurnResult(
            step_executed=False, retryable_failure=True
        ),
    )
    agent = HybridToolAgent(
        game_key="synthetic",
        failure_floor=1,
        model="test-model",
        base_url="http://127.0.0.1:1/v1",
        provider="openai",
    )
    actions = []

    def step_env(action):
        actions.append(action)
        return {"executed": True}

    result = agent.analyze(
        state_path,
        0,
        valid_actions=["RIGHT"],
        step_env=step_env,
    )
    assert result is not None and result.step_executed
    assert actions == [{"action": "RIGHT"}]
    assert agent.fallback_count == 1
