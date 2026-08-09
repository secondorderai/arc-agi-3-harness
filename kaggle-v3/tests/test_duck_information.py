from __future__ import annotations

from duck_information.agent import INFORMATION_USER_ADDENDUM, DuckInformationToolAgent
from duck_information.solver import DuckInformationHarnessSolver
from inference.agent.runtime_state import Frame, HistoryEntry
from ouro3.config import HarnessConfig, RuntimeProfile
from ouro3.mode import HarnessMode
from ouro3.runner import make_solver


def _agent() -> DuckInformationToolAgent:
    return DuckInformationToolAgent(
        model="test",
        base_url="http://127.0.0.1:1/v1",
        provider="vllm",
    )


def _unchanged_history() -> list[HistoryEntry]:
    frame = Frame(grid=((1, 0), (0, 1)), step=0, level=1)
    return [
        HistoryEntry(action="", frame=frame),
        HistoryEntry(action="RIGHT", frame=Frame(grid=frame.grid, step=1, level=1)),
        HistoryEntry(action="LEFT", frame=Frame(grid=frame.grid, step=2, level=1)),
    ]


def test_information_request_is_sparse_and_keeps_stock_surface() -> None:
    agent = _agent()
    initial = agent._build_user_prompt(0, valid_actions=["RIGHT"])
    assert INFORMATION_USER_ADDENDUM not in initial
    assert agent.augmented_features_enabled is False
    prompt = agent._build_user_prompt(
        2,
        valid_actions=["RIGHT", "LEFT"],
        history_entries=_unchanged_history(),
    )
    assert "Targeted information request" in prompt
    assert "changed_regions" not in prompt
    assert agent.telemetry["information_trigger_count"] == 1
    assert agent.telemetry["information_no_change_triggers"] == 1


def test_information_does_not_trigger_across_level_transition_or_after_cap() -> None:
    agent = DuckInformationToolAgent(
        model="test",
        base_url="http://127.0.0.1:1/v1",
        provider="vllm",
        information_max_triggers=1,
    )
    transitioned = [
        HistoryEntry(action="", frame=Frame(grid=((1,),), step=0, level=1)),
        HistoryEntry(action="RIGHT", frame=Frame(grid=((1,),), step=1, level=2)),
        HistoryEntry(action="LEFT", frame=Frame(grid=((1,),), step=2, level=2)),
    ]
    assert "Targeted information request" not in agent._build_user_prompt(
        2, valid_actions=["RIGHT"], history_entries=transitioned
    )
    assert "Targeted information request" in agent._build_user_prompt(
        3, valid_actions=["RIGHT"], history_entries=_unchanged_history()
    )
    assert "Targeted information request" not in agent._build_user_prompt(
        4, valid_actions=["RIGHT"], history_entries=_unchanged_history()
    )


def test_information_config_and_solver_are_isolated() -> None:
    config = HarnessConfig.information(seed=0)
    assert config.mode == HarnessMode.DUCK_INFORMATION
    assert config.experiment == "duck-information-v1"
    assert config.config_hash != HarnessConfig.reference(seed=0).config_hash
    solver = make_solver(config)
    assert type(solver) is DuckInformationHarnessSolver
    assert solver.max_runtime_s_per_game == 7920
    assert solver.analyzer_timeout == 900
    local = HarnessConfig.information(seed=0, profile=RuntimeProfile.LOCAL_MLX)
    assert local.model_id == "qwen3.5:4b-mlx"
    assert local.local_game_cap_s == 360
    assert local.analyzer_timeout_s == 180
