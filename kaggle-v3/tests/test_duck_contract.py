from __future__ import annotations

import pytest

from duck_contract.agent import (
    CONTRACT_SYSTEM_ADDENDUM,
    CONTRACT_USER_ADDENDUM,
    DuckContractToolAgent,
)
from duck_contract.solver import DuckContractHarnessSolver
from duck_contract.repair_agent import DuckContractRepairToolAgent
from duck_contract.repair_solver import DuckContractRepairHarnessSolver
from inference.agent.python_tool_sandbox import _sandbox_env, run_sandboxed_python
from ouro3.config import HarnessConfig, RuntimeProfile
from ouro3.mode import HarnessMode
from ouro3.runner import make_solver


def _agent() -> DuckContractToolAgent:
    return DuckContractToolAgent(
        model="test",
        base_url="http://127.0.0.1:1/v1",
        provider="vllm",
    )


def test_contract_mode_isolated_and_prompt_is_executable(monkeypatch) -> None:
    config = HarnessConfig.contract(seed=0)
    assert config.mode == HarnessMode.DUCK_CONTRACT
    assert config.experiment == "duck-contract-v1"
    agent = _agent()
    assert agent.augmented_features_enabled is False
    assert agent.prediction_verification_enabled is True
    prompt = agent._build_user_prompt(0, valid_actions=["RIGHT"])
    assert CONTRACT_SYSTEM_ADDENDUM in agent._system_prompt
    assert CONTRACT_USER_ADDENDUM in prompt
    assert "result = action([{'action': '<one valid action>'" in prompt
    assert "changed_regions" not in prompt
    monkeypatch.setenv("OURO3_HARNESS_MODE", "duck-contract")
    assert _sandbox_env()["OURO3_HARNESS_MODE"] == "duck-contract"


def test_contract_rejects_bare_and_batched_actions() -> None:
    agent = _agent()
    with pytest.raises(ValueError, match="non-empty generic"):
        agent._normalize_python_actions([{"action": "RIGHT"}])
    with pytest.raises(ValueError, match="generic observable"):
        agent._normalize_python_actions(
            [{"action": "RIGHT", "expect": {"private_rule": True}}]
        )
    with pytest.raises(ValueError, match="exactly one"):
        agent._normalize_python_actions(
            [
                {"action": "RIGHT", "expect": {"board_changed": True}},
                {"action": "LEFT", "expect": {"board_changed": True}},
            ]
        )
    assert agent.telemetry["deliberate_proposals"] == 0


def test_contract_accepts_one_expectation_and_preserves_telemetry() -> None:
    agent = _agent()
    value = agent._normalize_python_actions(
        [{"action": "RIGHT", "expect": {"board_changed": True}}]
    )
    assert value == [{"action": "RIGHT", "expect": {"board_changed": True}}]
    assert agent.telemetry["deliberate_proposals"] == 1
    agent.register_prediction_match()
    assert agent.telemetry["prediction_matches"] == 1


def test_contract_solver_keeps_stock_runtime_limits() -> None:
    solver = make_solver(HarnessConfig.contract(seed=0))
    assert type(solver) is DuckContractHarnessSolver
    assert solver.max_runtime_s_per_game == 7920
    assert solver.analyzer_timeout == 900
    local = HarnessConfig.contract(seed=0, profile=RuntimeProfile.LOCAL_MLX)
    assert local.model_id == "qwen3.5:4b-mlx"
    assert local.local_game_cap_s == 1200


def test_contract_repair_logs_missing_expect_and_truncates_batches() -> None:
    agent = DuckContractRepairToolAgent(
        model="test",
        base_url="http://127.0.0.1:1/v1",
        provider="vllm",
    )
    value = agent._normalize_python_actions(
        [{"action": "MOUSE", "row": 4, "col": 5}, {"action": "MOUSE", "row": 6, "col": 7}]
    )
    assert value == [{"action": "MOUSE", "row": 4, "col": 5, "expect": {"board_changed": True}}]
    assert agent.telemetry["deliberate_proposals"] == 1
    assert agent.telemetry["contract_repairs"] == 1
    assert agent.telemetry["contract_batch_truncations"] == 1
    explicit = agent._normalize_python_actions(
        [{"action": "MOUSE", "row": 4, "col": 5, "expect": {"board_changed": False}}]
    )
    assert explicit[0]["expect"] == {"board_changed": False}
    assert agent.telemetry["deliberate_proposals"] == 2
    assert agent.telemetry["contract_repairs"] == 1


def test_contract_repair_solver_and_config_are_isolated() -> None:
    config = HarnessConfig.contract_repair(seed=0)
    assert config.mode == HarnessMode.DUCK_CONTRACT_REPAIR
    assert config.experiment == "duck-contract-repair-v1"
    solver = make_solver(config)
    assert type(solver) is DuckContractRepairHarnessSolver
    assert solver.max_runtime_s_per_game == 7920


def test_contract_sandbox_preserves_expectation_payload(monkeypatch) -> None:
    monkeypatch.setenv("OURO3_HARNESS_MODE", "duck-contract-repair")
    seen: list[dict[str, object]] = []

    def handle(actions):
        seen.extend(actions)
        return {"action_result": {"executed": True}, "state": {}}

    result = run_sandboxed_python(
        code="result = action([{'action':'RIGHT','expect':{'board_changed':True}}])",
        timeout_seconds=5,
        initial_state={
            "current_frame": {
                "ascii": "",
                "step": 0,
                "level": 1,
                "shape": [1, 1],
                "segmentation": {"nodes": [], "adjacency_list": []},
            },
            "history": [],
            "valid_actions": ["RIGHT"],
            "last_action_result": {},
        },
        action_handler=handle,
    )
    assert not result.get("error")
    assert seen[0]["expect"] == {"board_changed": True}
