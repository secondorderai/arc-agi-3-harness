from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from duck_memory.agent import DuckMemoryToolAgent
from duck_memory.memory import (
    CompactionMemory,
    LIST_SUMMARY_FIELDS,
    MEMORY_MARKER,
    compaction_prompt,
    covered_action_range,
    covered_message_count,
    is_memory_message,
    memory_message,
    partition_for_compaction,
    validate_summary,
)
from duck_memory.reasoning import (
    REASONING_SENTINEL,
    assert_reasoning_sentinel_rendered,
    normalize_reasoning_history,
    render_and_verify_reasoning,
)
from duck_memory.solver import DuckMemoryHarnessSolver
from inference.agent.tool_agent import _ChatCompletionResult
from inference.utils.openai_compat import build_chat_payload
from ouro3.config import HarnessConfig, RuntimeProfile
from ouro3.metrics import summarize_runs
from ouro3.mode import HarnessMode
from ouro3.runner import _write_memory_trace_bundle, make_solver


REFERENCE_CONFIG_HASH = (
    "f3f5e6447c2ce96ec60ae2303d85e18fdf3140329aeb0fc5c29701f3e2a0b4b5"
)


def _summary(**overrides):
    value = {
        "level_state": "level 1, still active",
        **{field: [] for field in LIST_SUMMARY_FIELDS},
    }
    value.update(overrides)
    return value


def _history_block(index: int, *, chars: int = 900):
    return [
        {"role": "user", "content": f"Action {index}: inspect frame " + "u" * chars},
        {
            "role": "assistant",
            "reasoning": f"private plan {index} " + "r" * chars,
            "content": "",
            "tool_calls": [
                {
                    "id": f"call-{index}",
                    "type": "function",
                    "function": {"name": "python", "arguments": '{"code":"pass"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"call-{index}",
            "content": json.dumps({"action": index, "changed": bool(index % 2)}),
        },
    ]


def _memory_agent(monkeypatch, tmp_path: Path) -> DuckMemoryToolAgent:
    monkeypatch.setenv("OURO3_COMPACTION_TRIGGER_TOKENS", "2000")
    monkeypatch.setenv("OURO3_COMPACTION_TARGET_TOKENS", "1200")
    monkeypatch.setenv("OURO3_COMPACTION_RECENT_ASSISTANT_TURNS", "2")
    monkeypatch.setenv("OURO3_COMPACTION_MAX_OUTPUT_TOKENS", "256")
    monkeypatch.setenv("OURO3_COMPACTION_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("TAAF_MINIMAL_DIAGNOSTICS", "false")
    agent = DuckMemoryToolAgent(
        model="test",
        base_url="http://127.0.0.1:1/v1",
        provider="vllm",
    )
    agent._ensure_session(tmp_path / "game_state.json")
    return agent


def test_reasoning_adapter_uses_qwen_history_field_and_preserve_flag() -> None:
    history = normalize_reasoning_history(
        [
            {"role": "user", "content": "observe"},
            {
                "role": "assistant",
                "reasoning": "private causal plan",
                "content": "acting",
            },
        ]
    )
    assert "reasoning" not in history[1]
    assert history[1]["reasoning_content"] == "private causal plan"

    payload = build_chat_payload(
        provider="vllm",
        model="qwen",
        messages=history,
        max_tokens=None,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        thinking=True,
        preserve_thinking=True,
    )
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": True,
        "preserve_thinking": True,
    }


def test_reasoning_adapter_normalizes_reasoning_only_content() -> None:
    history = normalize_reasoning_history(
        [
            {
                "role": "assistant",
                "reasoning": "private causal plan",
                "content": None,
            }
        ]
    )

    assert history == [
        {
            "role": "assistant",
            "reasoning_content": "private causal plan",
            "content": "",
        }
    ]


def test_exact_template_sentinel_detects_empty_historical_thinking() -> None:
    valid = (
        "<|im_start|>assistant\n<think>\n"
        f"{REASONING_SENTINEL}\n</think>\n\nMarker stored."
    )
    assert_reasoning_sentinel_rendered(valid)
    with pytest.raises(RuntimeError, match="not rendered exactly once"):
        assert_reasoning_sentinel_rendered(
            "<|im_start|>assistant\n<think>\n\n</think>"
        )

    calls = []

    def fake_renderer(messages, **kwargs):
        calls.append((messages, kwargs))
        return valid

    assert REASONING_SENTINEL in render_and_verify_reasoning(fake_renderer)
    assert calls[0][0][1]["reasoning_content"] == REASONING_SENTINEL
    assert calls[0][1]["preserve_thinking"] is True


def test_partition_preserves_recent_reasoning_and_tool_pairs() -> None:
    history = [
        message
        for index in range(5)
        for message in _history_block(index, chars=20)
    ]
    partition = partition_for_compaction(
        history,
        recent_assistant_turns=2,
    )
    assert partition.prefix
    assert partition.suffix
    assert partition.suffix[0]["role"] == "user"
    assert partition.retained_assistant_turns == 2
    for messages in (partition.prefix, partition.suffix):
        call_ids = {
            call["id"]
            for message in messages
            for call in message.get("tool_calls", [])
        }
        tool_ids = {
            message["tool_call_id"]
            for message in messages
            if message.get("role") == "tool"
        }
        assert call_ids == tool_ids
    assert all(
        "reasoning" not in message
        for message in (*partition.prefix, *partition.suffix)
    )


def test_structured_summary_validation_and_previous_summary_merging() -> None:
    summary = _summary(
        mechanics=["RIGHT moves the blue object one cell"],
        goal_hypotheses=["0.7: reach the green region; evidence: prior reward"],
    )
    validated = validate_summary(json.dumps(summary))
    first = memory_message(
        validated,
        generation=1,
        covered_messages=12,
        action_range=[1, 4],
    )
    assert is_memory_message(first)
    prompt = compaction_prompt(
        (
            first,
            {"role": "user", "content": "Action 5"},
            {
                "role": "assistant",
                "reasoning_content": "revise the route",
                "content": "",
            },
        )
    )
    assert MEMORY_MARKER in prompt[1]["content"]
    assert "revise the route" in prompt[1]["content"]
    compacted_prefix = (
        first,
        {"role": "user", "content": "Action 5"},
        {
            "role": "assistant",
            "reasoning_content": "revise the route",
            "content": "",
        },
    )
    assert covered_message_count(compacted_prefix) == 14
    assert covered_action_range(compacted_prefix) == [1, 5]
    with pytest.raises(ValueError, match="missing fields"):
        validate_summary('{"level_state":"active"}')


def test_successful_compaction_replaces_old_history_without_eviction(
    monkeypatch,
    tmp_path: Path,
) -> None:
    agent = _memory_agent(monkeypatch, tmp_path)
    summary = _summary(
        current_plan=["test the right-hand corridor"],
        failed_experiments=["UP at action 2 did not change gameplay"],
    )
    monkeypatch.setattr(
        agent,
        "_compaction_completion",
        lambda _messages: _ChatCompletionResult(
            message={"content": json.dumps(summary)},
            usage={"prompt_tokens": 1000, "completion_tokens": 120},
        ),
    )
    messages = [
        {"role": "system", "content": "stock Duck prompt"},
        *[
            message
            for index in range(6)
            for message in _history_block(index)
        ],
    ]
    output = agent._trim_messages_for_context(messages, tools=[])
    agent._history_messages = output[1:]
    assert output[0] == messages[0]
    assert any(is_memory_message(message) for message in output)
    assert agent.telemetry["compaction_count"] == 1
    assert agent.telemetry["context_evictions"] == 0
    assert agent.telemetry["emergency_trims"] == 0
    assert agent.telemetry["reasoning_turns"] == 6
    assert agent.telemetry["reasoning_compacted_turns"] > 0
    assert agent.telemetry["reasoning_accounted_turns"] == 6
    assert agent.telemetry["reasoning_unaccounted_turns"] == 0
    assert isinstance(agent.memory, CompactionMemory)
    assert agent.memory.summary == summary
    assert agent.memory.covered_messages > 0
    assert len(agent.memory.uncompacted_raw_messages) > 0
    assert len(agent.memory.latest_assistant_turns) == 2
    assert all(
        "reasoning" not in message
        for message in agent.memory.latest_assistant_turns
    )
    trace_path = tmp_path / "game_state_memory_trace.jsonl.gz"
    with gzip.open(trace_path, "rt", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle]
    assert any(event["event"] == "compaction_succeeded" for event in events)
    assert sum(event["event"] == "message" for event in events) >= len(messages) - 1


def test_compaction_retries_once_then_emergency_trims(
    monkeypatch,
    tmp_path: Path,
) -> None:
    agent = _memory_agent(monkeypatch, tmp_path)
    attempts = []

    def fail(prefix):
        attempts.append(len(prefix))
        raise TimeoutError("synthetic compactor timeout")

    monkeypatch.setattr(agent, "_summarize_prefix", fail)
    messages = [
        {"role": "system", "content": "stock Duck prompt"},
        *[
            message
            for index in range(6)
            for message in _history_block(index)
        ],
    ]
    output = agent._trim_messages_for_context(messages, tools=[])
    agent._history_messages = output[1:]
    assert len(attempts) == 2
    assert agent.telemetry["compaction_retries"] == 1
    assert agent.telemetry["compaction_failures"] == 1
    assert agent.telemetry["emergency_trims"] > 0
    assert agent.telemetry["context_evictions"] == agent.telemetry["emergency_trims"]
    assert agent._estimate_request_input_tokens(output, tools=[]) <= 1200


def test_invalid_compaction_retries_with_smaller_prefix_and_recovers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    agent = _memory_agent(monkeypatch, tmp_path)
    attempts = []
    summary = _summary(
        cross_level_knowledge=["blue objects stayed controllable after level 1"],
        current_plan=["verify whether the same control applies on level 2"],
    )

    def recover(prefix):
        attempts.append(len(prefix))
        if len(attempts) == 1:
            raise ValueError("synthetic structured-output rejection")
        return summary

    monkeypatch.setattr(agent, "_summarize_prefix", recover)
    history = [
        message
        for index in range(4)
        for message in _history_block(index, chars=350)
    ]
    partition = partition_for_compaction(
        history,
        recent_assistant_turns=2,
    )
    agent._trace_new_messages(history)
    output = agent._compact_partition(partition, pre_tokens=2_500)
    assert output is not None
    agent._history_messages = output
    assert len(attempts) == 2
    assert attempts[1] < attempts[0]
    assert any(is_memory_message(message) for message in output)
    assert agent.telemetry["compaction_count"] == 1
    assert agent.telemetry["compaction_retries"] == 1
    assert agent.telemetry["compaction_failures"] == 0
    assert agent.telemetry["emergency_trims"] == 0
    assert agent.telemetry["reasoning_unaccounted_turns"] == 0


def test_repeated_compaction_merges_previous_memory_and_coverage(
    monkeypatch,
    tmp_path: Path,
) -> None:
    agent = _memory_agent(monkeypatch, tmp_path)
    summaries = [
        _summary(
            current_plan=["test the first corridor"],
            cross_level_knowledge=["level 1 uses directional movement"],
        ),
        _summary(
            current_plan=["reuse the directional model on level 2"],
            cross_level_knowledge=[
                "levels 1 and 2 both use directional movement"
            ],
        ),
    ]
    prompts = []

    def complete(messages):
        prompts.append(messages)
        summary = summaries[min(len(prompts) - 1, len(summaries) - 1)]
        return _ChatCompletionResult(
            message={"content": json.dumps(summary)},
            usage={"prompt_tokens": 1_000, "completion_tokens": 100},
        )

    monkeypatch.setattr(agent, "_compaction_completion", complete)
    system = {"role": "system", "content": "stock Duck prompt"}
    first_messages = [
        system,
        *[
            message
            for index in range(6)
            for message in _history_block(index)
        ],
    ]
    first_output = agent._trim_messages_for_context(first_messages, tools=[])
    agent._history_messages = first_output[1:]
    first_coverage = agent.memory.covered_messages
    assert agent.memory.generation == 1

    second_messages = [
        system,
        *first_output[1:],
        *[
            message
            for index in range(6, 12)
            for message in _history_block(index)
        ],
    ]
    second_output = agent._trim_messages_for_context(second_messages, tools=[])
    agent._history_messages = second_output[1:]
    assert agent.memory.generation == 2
    assert agent.telemetry["compaction_count"] == 2
    assert agent.memory.covered_messages > first_coverage
    assert agent.memory.covered_action_range == [0, 9]
    assert MEMORY_MARKER in prompts[1][1]["content"]
    assert agent.telemetry["reasoning_unaccounted_turns"] == 0


def test_identical_reasoning_after_compaction_is_counted_as_a_new_turn(
    monkeypatch,
    tmp_path: Path,
) -> None:
    agent = _memory_agent(monkeypatch, tmp_path)
    repeated = {
        "role": "assistant",
        "reasoning_content": "same private reasoning",
        "content": "",
    }
    agent._trace_new_messages([repeated])
    agent._set_trace_active_history([])
    agent._trace_new_messages([repeated])
    assert agent.telemetry["reasoning_turns"] == 2


def test_validation_memory_trace_bundle_is_visible_and_complete(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / ".hidden-work"
    artifact_dir = job_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    for game_id in ("game-a", "game-b"):
        with gzip.open(
            artifact_dir / f"{game_id}_p0_memory_trace.jsonl.gz",
            "wt",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "message",
                        "message": {
                            "role": "assistant",
                            "reasoning_content": f"private-{game_id}",
                        },
                    }
                )
                + "\n"
            )
    output = tmp_path / "validation_metrics.json"
    bundle = _write_memory_trace_bundle(job_dir=job_dir, output_path=output)
    assert bundle == tmp_path / "validation_metrics-memory-trace.jsonl.gz"
    assert bundle is not None and bundle.is_file()
    with gzip.open(bundle, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert [row["game_id"] for row in rows] == ["game-a", "game-b"]
    assert [row["message"]["reasoning_content"] for row in rows] == [
        "private-game-a",
        "private-game-b",
    ]


def test_hidden_metrics_keep_only_aggregate_memory_telemetry() -> None:
    run = SimpleNamespace(
        game_id="hidden-game",
        state="gave_up",
        levels_completed=0,
        number_of_levels=1,
        actions_per_level=[0],
        history=[],
        final_generated_tokens=0,
        final_uncached_input_tokens=0,
        final_wallclock_seconds=1.0,
        final_score=0.0,
        solver_note="tokens=0",
        solver_telemetry={
            "reasoning_turns": 3,
            "reasoning_chars": 120,
            "reasoning_accounted_turns": 3,
            "reasoning_unaccounted_turns": 0,
        },
        solver_diagnostics={
            "compaction_memory": {
                "summary": {"current_plan": ["private plan"]}
            }
        },
    )
    metrics = summarize_runs(
        [run],
        experiment="duck-memory-v1",
        seed=0,
        config_hash="a" * 64,
        elapsed_seconds=1.0,
        mode="duck-memory",
        aggregate_memory_telemetry_only=True,
    )
    assert metrics["memory_telemetry_scope"] == "aggregate-only"
    assert metrics["telemetry"]["reasoning_turns"] == 3
    assert "telemetry" not in metrics["games"][0]
    assert "memory_diagnostics" not in metrics["games"][0]
    assert "memory_diagnostics" not in metrics
    assert "private plan" not in json.dumps(metrics)


def test_reference_hash_and_solver_remain_isolated(monkeypatch) -> None:
    assert HarnessConfig.reference().config_hash == REFERENCE_CONFIG_HASH
    memory = HarnessConfig.memory(seed=0)
    memory.apply_environment()
    solver = make_solver(memory)
    assert memory.mode == HarnessMode.DUCK_MEMORY
    assert type(solver) is DuckMemoryHarnessSolver
    assert solver.max_runtime_s_per_game == 7_920
    assert solver.analyzer_timeout == 900

    local = HarnessConfig.memory(
        seed=0,
        profile=RuntimeProfile.LOCAL_MLX,
    )
    assert local.model_id == "qwen3.5:4b-mlx"
    assert local.local_workers == 2
    assert local.local_game_cap_s == 360
