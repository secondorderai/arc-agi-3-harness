from __future__ import annotations

from duck_diversity.agent import DIVERSITY_USER_ADDENDUM, DuckDiversityToolAgent
from duck_diversity.solver import DuckDiversityHarnessSolver
from inference.agent.runtime_state import Frame, HistoryEntry
from ouro3.config import HarnessConfig, RuntimeProfile
from ouro3.mode import HarnessMode
from ouro3.runner import make_solver


def _history() -> list[HistoryEntry]:
    frame = Frame(grid=((1, 0), (0, 1)), step=0, level=1)
    return [
        HistoryEntry(action="", frame=frame),
        HistoryEntry(action="RIGHT", frame=Frame(grid=frame.grid, step=1, level=1)),
        HistoryEntry(action="LEFT", frame=Frame(grid=frame.grid, step=2, level=1)),
    ]


def test_diversity_uses_one_bounded_alternate_seed() -> None:
    agent = DuckDiversityToolAgent(
        model="test",
        base_url="http://127.0.0.1:1/v1",
        provider="vllm",
        seed=0,
        diversity_seed_offset=17,
    )
    prompt = agent._build_user_prompt(
        2, valid_actions=["RIGHT", "LEFT"], history_entries=_history()
    )
    assert "Controlled diversity trigger" in prompt
    assert agent.telemetry["diversity_trigger_count"] == 1
    assert agent.telemetry["diversity_seed_uses"] == 1
    assert agent._diversity_seed_override == 17


def test_diversity_is_not_always_on_or_beyond_cap() -> None:
    agent = DuckDiversityToolAgent(
        model="test",
        base_url="http://127.0.0.1:1/v1",
        provider="vllm",
        diversity_max_triggers=1,
    )
    assert DIVERSITY_USER_ADDENDUM not in agent._build_user_prompt(
        0, valid_actions=["RIGHT"]
    )
    assert "Controlled diversity trigger" in agent._build_user_prompt(
        1, valid_actions=["RIGHT"], history_entries=_history()
    )
    assert "Controlled diversity trigger" not in agent._build_user_prompt(
        2, valid_actions=["RIGHT"], history_entries=_history()
    )


def test_diversity_config_and_solver_are_isolated() -> None:
    reference = HarnessConfig.reference(seed=0)
    diversity = HarnessConfig.diversity(seed=0)
    assert diversity.mode == HarnessMode.DUCK_DIVERSITY
    assert diversity.experiment == "duck-diversity-v1"
    assert diversity.config_hash != reference.config_hash
    solver = make_solver(diversity)
    assert type(solver) is DuckDiversityHarnessSolver
    assert solver.max_runtime_s_per_game == 7920
    assert solver.analyzer_timeout == 900
    local = HarnessConfig.diversity(seed=0, profile=RuntimeProfile.LOCAL_MLX)
    assert local.model_id == "qwen3.5:4b-mlx"
    assert local.local_game_cap_s == 360
    assert local.analyzer_timeout_s == 180
