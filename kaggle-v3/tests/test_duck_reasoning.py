from __future__ import annotations

from pathlib import Path

from duck_reasoning.agent import DuckReasoningToolAgent
from duck_reasoning.solver import DuckReasoningHarnessSolver
from ouro3.config import HarnessConfig, RuntimeProfile
from ouro3.mode import HarnessMode
from ouro3.runner import make_solver


def _agent(tmp_path: Path) -> DuckReasoningToolAgent:
    agent = DuckReasoningToolAgent(
        model="test",
        base_url="http://127.0.0.1:1/v1",
        provider="vllm",
    )
    agent._ensure_session(tmp_path / "game_state.json")
    return agent


def test_reasoning_mode_normalizes_history_without_compaction(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OURO3_REASONING_TEMPLATE_VERIFIED", "true")
    agent = _agent(tmp_path)
    output = agent._trim_messages_for_context(
        [
            {"role": "system", "content": "Stock Duck"},
            {"role": "user", "content": "observe"},
            {
                "role": "assistant",
                "reasoning": "private route",
                "content": None,
            },
        ],
        tools=[],
    )

    assert output[-1] == {
        "role": "assistant",
        "reasoning_content": "private route",
        "content": "",
    }
    assert agent.telemetry["reasoning_turns"] == 1
    assert agent.telemetry["reasoning_template_verified"] == 1
    assert agent.telemetry["compaction_count"] == 0
    assert agent.telemetry["emergency_trims"] == 0
    assert agent.diagnostics["history_policy"] == "stock-duck"
    assert agent.diagnostics["semantic_compaction"] is False


def test_reasoning_mode_preserves_stock_context_eviction_policy(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    messages = [{"role": "system", "content": "Stock Duck"}]
    for index in range(40):
        messages.extend(
            [
                {"role": "user", "content": f"turn {index}"},
                {
                    "role": "assistant",
                    "reasoning": f"private {index}",
                    "content": "",
                },
            ]
        )

    output = agent._trim_messages_for_context(messages, tools=[])
    assert all("reasoning" not in message for message in output)
    assert all(
        "reasoning_content" in message
        for message in output
        if message.get("role") == "assistant"
    )
    assert agent.telemetry["compaction_count"] == 0


def test_reasoning_config_and_solver_are_isolated_from_reference() -> None:
    config = HarnessConfig.reasoning(seed=0)
    assert config.mode == HarnessMode.DUCK_REASONING
    assert config.experiment == "duck-reasoning-v1"
    assert config.config_hash != HarnessConfig.reference(seed=0).config_hash

    solver = make_solver(config)
    assert type(solver) is DuckReasoningHarnessSolver
    assert solver.max_runtime_s_per_game == 7_920
    assert solver.analyzer_timeout == 900

    local = HarnessConfig.reasoning(seed=0, profile=RuntimeProfile.LOCAL_MLX)
    assert local.model_id == "qwen3.5:4b-mlx"
    assert local.local_workers == 2
