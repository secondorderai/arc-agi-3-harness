from __future__ import annotations

from duck_deliberate.agent import (
    DELIBERATE_SYSTEM_ADDENDUM,
    DuckDeliberateToolAgent,
)
from duck_deliberate.solver import DuckDeliberateHarnessSolver
from ouro3.config import HarnessConfig, RuntimeProfile
from ouro3.mode import HarnessMode
from ouro3.runner import make_solver


def _agent() -> DuckDeliberateToolAgent:
    return DuckDeliberateToolAgent(
        model="test",
        base_url="http://127.0.0.1:1/v1",
        provider="vllm",
    )


def test_deliberate_mode_isolated_from_reference_and_preserves_stock_surface() -> None:
    reference = HarnessConfig.reference(seed=0)
    deliberate = HarnessConfig.deliberate(seed=0)
    assert deliberate.mode == HarnessMode.DUCK_DELIBERATE
    assert deliberate.experiment == "duck-deliberate-v1"
    assert deliberate.config_hash != reference.config_hash
    agent = _agent()
    assert agent.augmented_features_enabled is False
    assert agent.prediction_verification_enabled is True
    assert DELIBERATE_SYSTEM_ADDENDUM in agent._system_prompt
    prompt = agent._build_user_prompt(0, valid_actions=["RIGHT"])
    assert "Hypothesis:" in prompt
    assert "prediction_matched" in prompt
    assert "changed_regions" not in prompt
    assert "hypothesis_ledger" not in prompt


def test_deliberate_actions_keep_expectations_and_count_predictions() -> None:
    agent = _agent()
    actions = agent._normalize_python_actions(
        [
            {"action": "RIGHT", "expect": {"board_changed": True}},
            {"action": "LEFT"},
        ]
    )
    assert actions[0]["expect"] == {"board_changed": True}
    assert "expect" not in actions[1]
    assert agent.telemetry["deliberate_proposals"] == 1
    compact = agent._compact_action_result(
        {
            "executed": True,
            "action_num": 1,
            "level": 1,
            "score": 0,
            "reward": 0,
            "state": "PLAYING",
            "valid_actions": ["RIGHT"],
            "board_changed": True,
            "prediction_matched": True,
        }
    )
    assert compact["prediction_matched"] is True
    agent.register_prediction_match()
    agent.register_prediction_mismatch("wrong board delta")
    assert agent.telemetry["prediction_matches"] == 1
    assert agent.telemetry["prediction_mismatches"] == 1
    assert agent.telemetry["hypothesis_revisions"] == 1


def test_deliberate_solver_uses_stock_runtime_limits() -> None:
    config = HarnessConfig.deliberate(seed=0)
    solver = make_solver(config)
    assert type(solver) is DuckDeliberateHarnessSolver
    assert solver.max_runtime_s_per_game == 7_920
    assert solver.analyzer_timeout == 900
    local = HarnessConfig.deliberate(seed=0, profile=RuntimeProfile.LOCAL_MLX)
    assert local.model_id == "qwen3.5:4b-mlx"
    assert local.local_game_cap_s == 1_200
