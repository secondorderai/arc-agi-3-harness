from __future__ import annotations

import os

from duck_poetiq.agent import (
    POETIQ_INTERVENTION_ADDENDUM,
    POETIQ_SYSTEM_ADDENDUM,
    DuckPoetiqToolAgent,
)
from duck_poetiq.solver import DuckPoetiqHarnessSolver
from inference.agent.runtime_state import Frame, HistoryEntry
from ouro3.config import HarnessConfig, RuntimeProfile
from ouro3.fingerprint import prompt_sha256, runtime_fingerprint
from ouro3.mode import HarnessMode
from ouro3.runner import make_solver


def _agent(**kwargs: object) -> DuckPoetiqToolAgent:
    return DuckPoetiqToolAgent(
        model="test",
        base_url="http://127.0.0.1:1/v1",
        provider="vllm",
        **kwargs,
    )


def _unchanged_history(count: int = 4) -> list[HistoryEntry]:
    frame = Frame(grid=((1, 0), (0, 1)), step=0, level=1)
    return [
        HistoryEntry(action="", frame=frame),
        *[
            HistoryEntry(
                action="RIGHT",
                frame=Frame(grid=frame.grid, step=index, level=1),
            )
            for index in range(1, count + 1)
        ],
    ]


def test_poetiq_is_one_persistent_protocol_with_stock_surface() -> None:
    agent = _agent(primary_seed=0)
    assert POETIQ_SYSTEM_ADDENDUM in agent._system_prompt
    assert agent.augmented_features_enabled is False
    assert agent.verified_actions_enabled is True
    prompt = agent._build_user_prompt(
        4,
        valid_actions=["RIGHT"],
        history_entries=_unchanged_history(),
    )
    assert POETIQ_INTERVENTION_ADDENDUM.split(":", 1)[0] in prompt
    assert "at most three" in prompt
    assert agent.maximum_action_batch_size == 1
    assert agent.telemetry["poetiq_intervention_triggers"] == 1
    assert agent.telemetry["poetiq_information_requests"] == 1


def test_poetiq_cooldown_and_second_attempt_use_alternate_seed() -> None:
    agent = _agent(primary_seed=3, intervention_cooldown_actions=12)
    history = _unchanged_history()
    agent._build_user_prompt(4, valid_actions=["RIGHT"], history_entries=history)
    assert agent._active_seed == 3
    agent.finish_intervention(
        {"executed": True, "gameplay_changed": False, "board_changed": False}
    )
    assert agent.telemetry["poetiq_failed_interventions"] == 1
    assert agent.diagnostics["intervention_events"][0]["seed"] == 3
    assert agent.diagnostics["intervention_events"][0]["outcome"] == "failure"
    assert "Poetiq intervention trigger" not in agent._build_user_prompt(
        5, valid_actions=["RIGHT"], history_entries=history
    )
    agent._last_intervention_action_count = -100
    agent._build_user_prompt(20, valid_actions=["RIGHT"], history_entries=history)
    assert agent._active_seed == 20
    assert agent.telemetry["poetiq_diverse_retries"] == 1
    assert agent.diagnostics["intervention_events"][1]["seed"] == 20


def test_poetiq_prediction_mismatch_aborts_and_does_not_count_success() -> None:
    agent = _agent(primary_seed=0)
    agent._build_user_prompt(4, valid_actions=["RIGHT"], history_entries=_unchanged_history())
    agent.register_prediction({"gameplay_changed": True})
    agent.register_prediction_mismatch("wrong")
    agent.finish_intervention(
        {"executed": True, "gameplay_changed": True, "board_changed": True}
    )
    assert agent.telemetry["prediction_mismatches"] == 1
    assert agent.telemetry["poetiq_intervention_successes"] == 0
    assert agent.telemetry["poetiq_failed_interventions"] == 1
    assert agent.maximum_action_batch_size is None


def test_poetiq_stall_trigger_uses_gameplay_changes_not_hud_animation() -> None:
    agent = _agent()
    history = _unchanged_history(3)
    for _ in range(3):
        agent.observe_transition(
            action="RIGHT",
            before_grid=((1,),),
            after_grid=((1,),),
            payload={"gameplay_changed": False, "hud_changed": True},
        )
    prompt = agent._build_user_prompt(4, valid_actions=["RIGHT"], history_entries=history)
    assert "gameplay unchanged" in prompt
    assert agent.telemetry["poetiq_intervention_triggers"] == 1


def test_poetiq_stalled_yield_has_exact_progress_guards() -> None:
    agent = _agent()
    assert not agent.should_yield_stalled_game(
        action_count=64,
        levels_completed=1,
        elapsed_seconds=1800,
        gameplay_change_history=[False] * 16,
    )
    agent._failed_interventions = 2
    assert not agent.should_yield_stalled_game(
        action_count=63,
        levels_completed=0,
        elapsed_seconds=1800,
        gameplay_change_history=[False] * 16,
    )
    assert not agent.should_yield_stalled_game(
        action_count=64,
        levels_completed=0,
        elapsed_seconds=1799,
        gameplay_change_history=[False] * 16,
    )
    assert not agent.should_yield_stalled_game(
        action_count=64,
        levels_completed=0,
        elapsed_seconds=1800,
        gameplay_change_history=[False] * 15 + [True],
    )
    assert agent.should_yield_stalled_game(
        action_count=64,
        levels_completed=0,
        elapsed_seconds=1800,
        gameplay_change_history=[False] * 16,
    )
    assert agent.telemetry["poetiq_stalled_yields"] == 1


def test_poetiq_config_solver_and_fingerprint_are_isolated() -> None:
    reference = HarnessConfig.reference(seed=0)
    poetiq = HarnessConfig.poetiq(seed=0)
    assert poetiq.mode == HarnessMode.DUCK_POETIQ
    assert poetiq.experiment == "duck-poetiq-v1"
    assert poetiq.config_hash != reference.config_hash
    solver = make_solver(poetiq)
    assert type(solver) is DuckPoetiqHarnessSolver
    assert solver.max_runtime_s_per_game == 7920
    assert solver.analyzer_timeout == 900
    local = HarnessConfig.poetiq(seed=0, profile=RuntimeProfile.LOCAL_MLX)
    assert local.model_id == "qwen3.5:4b-mlx"
    assert local.local_game_cap_s == 360
    assert local.analyzer_timeout_s == 180
    poetiq.apply_environment()
    poetiq_prompt = prompt_sha256()
    reference.apply_environment()
    assert poetiq_prompt != prompt_sha256()
    poetiq.apply_environment()
    fingerprint = runtime_fingerprint(poetiq)
    assert fingerprint["poetiq"]["maximum_candidates"] == 3
    assert fingerprint["poetiq"]["second_model_request"] is False
    os.environ["OURO3_HARNESS_MODE"] = HarnessMode.DUCK_REFERENCE.value
