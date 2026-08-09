from __future__ import annotations

from duck_audit.agent import AUDIT_USER_ADDENDUM, DuckAuditToolAgent
from duck_audit.solver import DuckAuditHarnessSolver
from inference.agent.runtime_state import Frame, HistoryEntry
from ouro3.config import HarnessConfig, RuntimeProfile
from ouro3.mode import HarnessMode
from ouro3.runner import make_solver


def _agent() -> DuckAuditToolAgent:
    return DuckAuditToolAgent(
        model="test",
        base_url="http://127.0.0.1:1/v1",
        provider="vllm",
    )


def _history(*, changed: bool = False) -> list[HistoryEntry]:
    initial = Frame(grid=((1, 0), (0, 1)), step=0, level=1)
    current = Frame(
        grid=((1, 1), (0, 1)) if changed else initial.grid,
        step=1,
        level=1,
    )
    return [
        HistoryEntry(action="", frame=initial),
        HistoryEntry(action="RIGHT", frame=current),
        HistoryEntry(action="RIGHT", frame=current),
        HistoryEntry(action="RIGHT", frame=current),
    ]


def test_audit_is_sparse_and_keeps_stock_surface() -> None:
    agent = _agent()
    initial = agent._build_user_prompt(0, valid_actions=["RIGHT"])
    assert AUDIT_USER_ADDENDUM not in initial
    assert agent.augmented_features_enabled is False
    assert agent._system_prompt.startswith(
        "You are a coding agent solving a grid-based puzzle game."
    )
    triggered = agent._build_user_prompt(
        3,
        valid_actions=["RIGHT"],
        history_entries=_history(),
    )
    assert "Sparse self-audit trigger" in triggered
    assert "changed_regions" not in triggered
    assert "hypothesis_ledger" not in triggered
    assert agent.telemetry["audit_trigger_count"] == 1
    assert agent.telemetry["audit_repeat_triggers"] == 1
    assert agent.telemetry["audit_no_change_triggers"] == 0


def test_audit_triggers_on_no_change_without_repeating_action() -> None:
    agent = DuckAuditToolAgent(
        model="test",
        base_url="http://127.0.0.1:1/v1",
        provider="vllm",
        audit_repeat_threshold=5,
        audit_no_change_threshold=2,
    )
    initial = Frame(grid=((1, 0),), step=0, level=1)
    history = [
        HistoryEntry(action="", frame=initial),
        HistoryEntry(action="RIGHT", frame=Frame(grid=initial.grid, step=1, level=1)),
        HistoryEntry(action="LEFT", frame=Frame(grid=initial.grid, step=2, level=1)),
    ]
    prompt = agent._build_user_prompt(
        2,
        valid_actions=["RIGHT", "LEFT"],
        history_entries=history,
    )
    assert "gameplay frame unchanged" in prompt
    assert agent.telemetry["audit_no_change_triggers"] == 1


def test_audit_does_not_trigger_across_level_transition_or_after_cap() -> None:
    agent = DuckAuditToolAgent(
        model="test",
        base_url="http://127.0.0.1:1/v1",
        provider="vllm",
        audit_max_triggers=1,
    )
    level_transition = [
        HistoryEntry(action="", frame=Frame(grid=((1,),), step=0, level=1)),
        HistoryEntry(action="RIGHT", frame=Frame(grid=((1,),), step=1, level=2)),
        HistoryEntry(action="RIGHT", frame=Frame(grid=((1,),), step=2, level=2)),
    ]
    prompt = agent._build_user_prompt(
        2,
        valid_actions=["RIGHT"],
        history_entries=level_transition,
    )
    assert "Sparse self-audit trigger" not in prompt
    first = agent._build_user_prompt(3, valid_actions=["RIGHT"], history_entries=_history())
    second = agent._build_user_prompt(4, valid_actions=["RIGHT"], history_entries=_history())
    assert "Sparse self-audit trigger" in first
    assert "Sparse self-audit trigger" not in second
    assert agent.telemetry["audit_trigger_count"] == 1


def test_audit_config_and_solver_are_isolated() -> None:
    reference = HarnessConfig.reference(seed=0)
    audit = HarnessConfig.audit(seed=0)
    assert audit.mode == HarnessMode.DUCK_AUDIT
    assert audit.experiment == "duck-audit-v1"
    assert audit.config_hash != reference.config_hash
    solver = make_solver(audit)
    assert type(solver) is DuckAuditHarnessSolver
    assert solver.max_runtime_s_per_game == 7920
    assert solver.analyzer_timeout == 900
    local = HarnessConfig.audit(seed=0, profile=RuntimeProfile.LOCAL_MLX)
    assert local.model_id == "qwen3.5:4b-mlx"
    assert local.local_game_cap_s == 360
    assert local.analyzer_timeout_s == 180
