from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

from duck_retrodict.agent import RETRODICT_SYSTEM_ADDENDUM, DuckRetrodictToolAgent
from duck_retrodict.solver import DuckRetrodictHarnessSolver
from inference.agent.runtime_state import Frame
from ouro3.config import HarnessConfig, RuntimeProfile
from ouro3.fingerprint import prompt_sha256, runtime_fingerprint
from ouro3.mode import HarnessMode
from ouro3.runner import _write_retrodict_trace_bundle, make_solver


def _agent() -> DuckRetrodictToolAgent:
    return DuckRetrodictToolAgent(
        game_key="test-game",
        model="test",
        base_url="http://127.0.0.1:1/v1",
        provider="vllm",
    )


def test_retrodict_agent_owns_rules_and_limits_every_batch() -> None:
    agent = _agent()
    assert RETRODICT_SYSTEM_ADDENDUM in agent._system_prompt
    assert agent.augmented_features_enabled
    assert agent.verified_actions_enabled
    assert agent.maximum_action_batch_size is None
    agent.observe_transition(
        action="RIGHT",
        before_grid=((1, 0, 0),),
        after_grid=((0, 1, 0),),
        payload={"level": 1},
    )
    prompt = agent._build_user_prompt(
        1,
        valid_actions=["RIGHT"],
        current_frame=Frame(grid=((0, 1, 0),), step=1, level=1),
        history_entries=[],
    )
    assert "Host retrodictive verifier" not in prompt
    assert agent.telemetry["retrodict_transitions"] == 1


def test_retrodict_config_solver_notebook_runtime_and_fingerprint_are_isolated(
    monkeypatch,
) -> None:
    reference = HarnessConfig.reference(seed=0)
    config = HarnessConfig.retrodict(seed=0)
    assert config.mode == HarnessMode.DUCK_RETRODICT
    assert config.experiment == "duck-retrodict-v1"
    assert config.config_hash != reference.config_hash
    solver = make_solver(config)
    assert type(solver) is DuckRetrodictHarnessSolver
    assert solver.retrodict_max_rules == 256
    local = HarnessConfig.retrodict(
        seed=0,
        profile=RuntimeProfile.LOCAL_MLX,
    )
    assert local.model_id == "qwen3.5:4b-mlx"
    config.apply_environment()
    retrodict_prompt = prompt_sha256()
    assert os.environ["OURO3_RETRODICT_TRACE"] == "true"
    fingerprint = runtime_fingerprint(config)
    assert fingerprint["retrodict"]["full_log_replay"] is True
    assert fingerprint["retrodict"]["automatic_batch_size"] == 1
    assert fingerprint["retrodict"]["fallback_batch_limit"] is None
    reference.apply_environment()
    assert retrodict_prompt != prompt_sha256()
    assert os.environ["OURO3_RETRODICT_TRACE"] == "false"


def test_retrodict_submission_disables_transition_trace() -> None:
    config = HarnessConfig.retrodict(
        seed=0,
        profile=RuntimeProfile.KAGGLE_SUBMISSION,
    )
    config.apply_environment()
    assert os.environ["OURO3_RETRODICT_TRACE"] == "false"


def test_retrodict_executes_one_exact_action_without_model_call(
    monkeypatch,
    tmp_path: Path,
) -> None:
    agent = _agent()
    first = ((1, 0, 0),)
    second = ((0, 1, 0),)
    goal = ((0, 0, 1),)
    for _ in range(2):
        agent.world_model.observe(
            level=1,
            action="RIGHT",
            before=first,
            after=second,
        )
        agent.world_model.observe(
            level=1,
            action="SPACE",
            before=second,
            after=goal,
            payload={"level_completed": True},
        )
    state_path = tmp_path / "state.json"
    state_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "duck_retrodict.agent.load_runtime_state",
        lambda _path: (Frame(grid=first, step=0, level=1), []),
    )
    executed: list[dict[str, object]] = []

    def step_env(action: dict[str, object]) -> dict[str, object]:
        executed.append(action)
        return {"executed": True}

    result = agent.analyze(
        state_path=state_path,
        action_num=0,
        valid_actions=["RIGHT"],
        step_env=step_env,
    )
    assert result is not None and result.step_executed
    assert executed == [{"action": "RIGHT"}]
    assert agent.telemetry["retrodict_host_plan_actions"] == 1


def test_retrodict_writes_compressed_transition_evidence(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OURO3_RETRODICT_TRACE", "true")
    agent = _agent()
    agent._session_runtime_dir = tmp_path
    agent.observe_transition(
        action="RIGHT",
        before_grid=((1, 0),),
        after_grid=((0, 1),),
        payload={"level": 1},
    )
    path = tmp_path / "test-game_retrodict_trace.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.loads(handle.readline())
    assert payload["game_id"] == "test-game"
    assert payload["before"] == [[1, 0]]
    assert payload["after"] == [[0, 1]]


def test_retrodict_trace_bundle_skips_corrupt_members(tmp_path: Path) -> None:
    good = tmp_path / "good_retrodict_trace.jsonl.gz"
    with gzip.open(good, "wt", encoding="utf-8") as handle:
        handle.write('{"game_id":"good"}\n')
    (tmp_path / "bad_retrodict_trace.jsonl.gz").write_bytes(b"not-gzip")

    bundle = _write_retrodict_trace_bundle(
        job_dir=tmp_path,
        output_path=tmp_path / "validation_metrics.json",
    )

    assert bundle is not None
    with gzip.open(bundle, "rt", encoding="utf-8") as handle:
        assert [json.loads(line) for line in handle] == [{"game_id": "good"}]
