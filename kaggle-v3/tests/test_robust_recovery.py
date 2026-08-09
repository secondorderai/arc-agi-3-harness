from __future__ import annotations

import importlib.util
import json
from collections import deque
from pathlib import Path

import nbformat
import pytest

from duck_robust.agent import DuckRobustToolAgent, RecoveryPhase
from duck_robust.solver import DuckRobustHarnessSolver
from inference.agent.tool_agent import AnalyzerTurnResult, ToolAgent
from ouro3.config import HarnessConfig
from ouro3.mode import HarnessMode
from ouro3.runner import make_solver
from ouro3.trajectory import (
    RecoveryPolicy,
    SessionSignals,
    TrajectoryPredictor,
    derive_alternate_seed,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CONFIG_HASH = (
    "f3f5e6447c2ce96ec60ae2303d85e18fdf3140329aeb0fc5c29701f3e2a0b4b5"
)


class _FixedPredictor:
    metadata = {"validation": {"promotion_target_met": False}}

    def __init__(self, probability: float) -> None:
        self.value = probability

    def probability(self, _values: dict[str, float]) -> float:
        return self.value


def _stagnant_signals(*, level: int = 1) -> SessionSignals:
    signals = SessionSignals(
        level=level,
        actions_since_progress=64,
        latest_elapsed_s=1_801,
    )
    signals.action_names = deque(["ACTION1"] * 64, maxlen=256)
    signals.gameplay_changes = deque([False] * 16, maxlen=64)
    signals.state_action_keys = deque(
        ["same-transition", "same-transition"], maxlen=256
    )
    return signals


def test_reference_hash_and_solver_remain_immutable() -> None:
    reference = HarnessConfig.reference()
    assert reference.config_hash == REFERENCE_CONFIG_HASH
    assert reference.mode == HarnessMode.DUCK_REFERENCE

    robust = HarnessConfig.robust(seed=0)
    robust.apply_environment()
    solver = make_solver(robust)
    assert type(solver) is DuckRobustHarnessSolver
    assert solver.primary_seed == 0
    assert robust.mode == HarnessMode.DUCK_ROBUST
    assert robust.config_hash != reference.config_hash


def test_alternate_seed_is_stable_and_level_scoped() -> None:
    first = derive_alternate_seed("session-7", level=1, recovery_index=1)
    assert first == derive_alternate_seed(
        "session-7", level=1, recovery_index=1
    )
    assert first != derive_alternate_seed(
        "session-7", level=2, recovery_index=1
    )
    assert 0 <= first <= 0x7FFF_FFFF


def test_recovery_requires_two_lows_and_cycle_or_contradiction() -> None:
    policy = RecoveryPolicy(
        predictor=_FixedPredictor(0.1),  # type: ignore[arg-type]
        minimum_actions=64,
        warmup_seconds=1_800,
        low_probability_threshold=0.25,
        required_low_windows=2,
    )
    signals = _stagnant_signals()

    first = policy.evaluate(signals, contradiction=False)
    assert not first.triggered
    signals.actions_since_progress += 1
    second = policy.evaluate(signals, contradiction=False)
    assert second.triggered
    assert "repeated_state_action_cycle" in second.reason
    assert not second.reset_recommended

    signals.actions_since_progress += 1
    assert not policy.evaluate(signals, contradiction=True).triggered


def test_recovery_confidence_windows_do_not_leak_across_levels() -> None:
    policy = RecoveryPolicy(
        predictor=_FixedPredictor(0.1),  # type: ignore[arg-type]
        minimum_actions=64,
        warmup_seconds=0,
        low_probability_threshold=0.25,
        required_low_windows=2,
    )
    signals = _stagnant_signals(level=1)
    assert not policy.evaluate(signals, contradiction=True).triggered

    signals.level = 2
    signals.actions_since_progress += 1
    assert not policy.evaluate(signals, contradiction=True).triggered
    signals.actions_since_progress += 1
    assert policy.evaluate(signals, contradiction=True).triggered


def test_recovery_reset_requires_contradiction_and_low_change_window() -> None:
    policy = RecoveryPolicy(
        predictor=_FixedPredictor(0.1),  # type: ignore[arg-type]
        minimum_actions=1,
        warmup_seconds=0,
        required_low_windows=1,
    )
    signals = _stagnant_signals(level=3)
    decision = policy.evaluate(signals, contradiction=True)
    assert decision.triggered
    assert decision.reset_recommended


def test_robust_agent_is_stock_until_gate_then_requires_predictions() -> None:
    policy = RecoveryPolicy(
        predictor=_FixedPredictor(0.1),  # type: ignore[arg-type]
        minimum_actions=1,
        warmup_seconds=0,
        required_low_windows=1,
    )
    agent = DuckRobustToolAgent(
        session_namespace="session-2",
        recovery_policy=policy,
        seed=0,
        model="test",
        base_url="http://127.0.0.1:1/v1",
        provider="openai",
    )
    normal_prompt = agent._build_user_prompt(0, valid_actions=["RIGHT"])
    assert agent.phase == RecoveryPhase.NORMAL
    assert "RECOVERY FORK" not in normal_prompt
    assert "hypothesis_ledger" not in normal_prompt

    agent.signals = _stagnant_signals()
    decision = agent.maybe_begin_recovery(
        action_count=64,
        level=1,
        elapsed_seconds=1_801,
    )
    assert decision.triggered
    assert agent.phase == RecoveryPhase.HYPOTHESIS
    assert agent.augmented_features_enabled
    assert agent._sampling_temperature_override == 0.8
    recovery_prompt = agent._build_user_prompt(
        64, valid_actions=["RIGHT"]
    )
    assert "two genuinely competing causal explanations" in recovery_prompt

    with pytest.raises(ValueError, match="requires an `expect` object"):
        agent._normalize_python_actions([{"action": "RIGHT"}])
    assert agent._normalize_python_actions(
        [{"action": "RIGHT", "expect": {"gameplay_changed": True}}]
    )[0]["expect"] == {"gameplay_changed": True}
    with pytest.raises(ValueError, match="at most 1 action"):
        agent._normalize_python_actions(
            [
                {"action": "RIGHT", "expect": {}},
                {"action": "LEFT", "expect": {}},
            ]
        )


def test_recovery_switches_to_execution_and_mismatch_returns_to_duck(
    monkeypatch,
) -> None:
    policy = RecoveryPolicy(
        predictor=_FixedPredictor(0.1),  # type: ignore[arg-type]
        minimum_actions=1,
        warmup_seconds=0,
        required_low_windows=1,
    )
    agent = DuckRobustToolAgent(
        session_namespace="session-3",
        recovery_policy=policy,
        seed=0,
        model="test",
        base_url="http://127.0.0.1:1/v1",
        provider="openai",
    )
    agent.signals = _stagnant_signals()
    assert agent.maybe_begin_recovery(
        action_count=64, level=1, elapsed_seconds=1_801
    ).triggered
    monkeypatch.setattr(
        ToolAgent,
        "analyze",
        lambda self, *args, **kwargs: AnalyzerTurnResult(
            step_executed=True
        ),
    )
    result = agent.analyze(Path("unused.json"), 64)
    assert result is not None and result.step_executed
    assert agent.phase == RecoveryPhase.EXECUTION
    assert agent._sampling_temperature_override == 0.2
    assert agent.maximum_action_batch_size == 8

    agent.register_prediction_mismatch("RIGHT moved the wrong object")
    assert agent.phase == RecoveryPhase.NORMAL
    assert agent._sampling_temperature_override is None
    assert agent.telemetry["prediction_mismatches"] == 1
    assert agent.ledger.current_plan == []


def test_packaged_predictor_is_pinned_and_fail_closed() -> None:
    predictor = TrajectoryPredictor.from_path()
    source = predictor.metadata["source"]
    validation = predictor.metadata["validation"]
    assert source["trajectory_count"] == 500
    assert source["game_count"] == 25
    assert source["sha256"] == (
        "95ad9c4f98fe65b5639f86221a13164420b6cba75b85777eb323403bc8c371d4"
    )
    assert validation["method"] == "leave-one-game-out"
    assert validation["promotion_target_met"] is False
    probability = predictor.probability(
        _stagnant_signals().predictor_features()
    )
    assert 0 < probability < 1


def test_robust_notebooks_select_candidate_without_changing_default(
    tmp_path: Path,
) -> None:
    path = ROOT / "scripts" / "build_notebooks.py"
    spec = importlib.util.spec_from_file_location("robust_notebook_builder", path)
    assert spec and spec.loader
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    reference_dir = tmp_path / "reference"
    builder.build(reference_dir)
    reference = nbformat.read(
        reference_dir / "submission" / "submission.ipynb", as_version=4
    )
    assert reference.metadata["ouro3"]["mode"] == "duck-reference"

    robust_dir = tmp_path / "robust"
    builder.build(robust_dir, validation_seed=0, mode="duck-robust")
    for kind in ("validation", "submission"):
        notebook = nbformat.read(
            robust_dir / kind / f"{kind}.ipynb", as_version=4
        )
        code = "\n".join(
            cell.source
            for cell in notebook.cells
            if cell.cell_type == "code"
        )
        assert notebook.metadata["ouro3"]["mode"] == "duck-robust"
        assert '"OURO3_HARNESS_MODE": "duck-robust"' in code
        assert "HarnessConfig.robust" in code


def test_predictor_artifact_is_valid_json() -> None:
    payload = json.loads(
        (ROOT / "src" / "ouro3" / "recovery_predictor.json").read_text()
    )
    assert len(payload["features"]) == len(payload["weights"])
