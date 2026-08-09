from __future__ import annotations

from duck_hierarchy.agent import HIERARCHY_USER_ADDENDUM, DuckHierarchyToolAgent
from duck_hierarchy.solver import DuckHierarchyHarnessSolver
from inference.agent.runtime_state import Frame, HistoryEntry
from ouro3.config import HarnessConfig, RuntimeProfile
from ouro3.mode import HarnessMode
from ouro3.runner import make_solver


def _agent() -> DuckHierarchyToolAgent:
    return DuckHierarchyToolAgent(
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


def test_hierarchy_is_sparse_and_bounded() -> None:
    agent = _agent()
    initial = agent._build_user_prompt(0, valid_actions=["RIGHT"])
    assert HIERARCHY_USER_ADDENDUM not in initial
    assert agent.augmented_features_enabled is False
    prompt = agent._build_user_prompt(
        2,
        valid_actions=["RIGHT", "LEFT"],
        history_entries=_unchanged_history(),
    )
    assert "Bounded candidate search trigger" in prompt
    assert "at most three" in prompt
    assert "changed_regions" not in prompt
    assert agent.telemetry["hierarchy_trigger_count"] == 1
    assert agent.telemetry["hierarchy_no_change_triggers"] == 1


def test_hierarchy_triggers_at_level_start_and_stops_at_cap() -> None:
    agent = DuckHierarchyToolAgent(
        model="test",
        base_url="http://127.0.0.1:1/v1",
        provider="vllm",
        hierarchy_max_triggers=1,
    )
    transition = [
        HistoryEntry(action="", frame=Frame(grid=((1,),), step=0, level=1)),
        HistoryEntry(action="RIGHT", frame=Frame(grid=((1,),), step=1, level=2)),
    ]
    prompt = agent._build_user_prompt(
        1, valid_actions=["RIGHT"], history_entries=transition
    )
    assert "Bounded candidate search trigger" in prompt
    assert agent.telemetry["hierarchy_level_start_triggers"] == 1
    assert "Bounded candidate search trigger" not in agent._build_user_prompt(
        2, valid_actions=["RIGHT"], history_entries=_unchanged_history()
    )


def test_hierarchy_config_and_solver_are_isolated() -> None:
    reference = HarnessConfig.reference(seed=0)
    hierarchy = HarnessConfig.hierarchy(seed=0)
    assert hierarchy.mode == HarnessMode.DUCK_HIERARCHY
    assert hierarchy.experiment == "duck-hierarchy-v1"
    assert hierarchy.config_hash != reference.config_hash
    solver = make_solver(hierarchy)
    assert type(solver) is DuckHierarchyHarnessSolver
    assert solver.max_runtime_s_per_game == 7920
    assert solver.analyzer_timeout == 900
    local = HarnessConfig.hierarchy(seed=0, profile=RuntimeProfile.LOCAL_MLX)
    assert local.model_id == "qwen3.5:4b-mlx"
    assert local.local_game_cap_s == 360
    assert local.analyzer_timeout_s == 180
