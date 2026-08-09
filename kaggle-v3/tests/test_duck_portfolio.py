from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

from duck_contract.agent import CONTRACT_USER_ADDENDUM
from duck_contract.repair_agent import REPAIR_SYSTEM_ADDENDUM
from duck_portfolio.agent import DuckPortfolioToolAgent
from duck_portfolio.router import (
    FEATURE_NAMES,
    POLICY_PRIORITY,
    PortfolioDecision,
    PortfolioPolicy,
    PortfolioRouter,
    PortfolioTransition,
    extract_portfolio_features,
)
from duck_portfolio.solver import DuckPortfolioHarnessSolver
from inference.agent.runtime_state import Frame, HistoryEntry
from ouro3.config import HarnessConfig, RuntimeProfile
from ouro3.fingerprint import prompt_sha256, runtime_fingerprint
from ouro3.mode import HarnessMode
from ouro3.runner import make_solver


class _FixedRouter:
    artifact_hash = "a" * 64

    def __init__(self, policy: PortfolioPolicy) -> None:
        self.policy = policy
        self.payload = {
            "candidate_order": [item.value for item in POLICY_PRIORITY],
        }

    def decide(self, features):
        del features
        adjusted = {
            PortfolioPolicy.STOCK.value: 0.5,
            PortfolioPolicy.AUDIT.value: 3.0,
            PortfolioPolicy.DELIBERATE.value: 2.0,
            PortfolioPolicy.CONTRACT_REPAIR.value: 1.0,
        }
        if self.policy != PortfolioPolicy.AUDIT:
            adjusted[self.policy.value] = 4.0
        return PortfolioDecision(
            policy=self.policy,
            raw_scores=dict(adjusted),
            adjusted_scores=adjusted,
            stock_fallback=False,
            confidence_margin=1.0,
        )

    def next_policy(self, adjusted_scores, current):
        del adjusted_scores
        return next(policy for policy in POLICY_PRIORITY if policy != current)


def _agent(policy: PortfolioPolicy = PortfolioPolicy.AUDIT) -> DuckPortfolioToolAgent:
    return DuckPortfolioToolAgent(
        router=_FixedRouter(policy),
        model="test",
        base_url="http://127.0.0.1:1/v1",
        provider="vllm",
    )


def _observe(
    agent: DuckPortfolioToolAgent,
    action_num: int,
    *,
    changed: bool = False,
    progress: bool = False,
    remaining: float = 4000,
) -> None:
    before = ((0, 1), (0, 0))
    after = ((0, 1), (0, 2)) if changed else before
    agent.observe_transition(
        action="RIGHT",
        before_grid=before,
        after_grid=after,
        payload={
            "action_num": action_num,
            "score": int(progress),
            "level_completed": progress,
            "gameplay_changed": changed,
            "hud_changed": False,
            "changed_regions": ([{"area": 1}] if changed else []),
            "time_remaining_seconds": remaining,
        },
    )


def _artifact_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "candidate_order": [policy.value for policy in POLICY_PRIORITY],
        "feature_names": list(FEATURE_NAMES),
        "score_clip": 10.0,
        "uncertainty_penalty": 0.5,
        "stock_margin": 0.25,
        "feature_means": {name: 0.0 for name in FEATURE_NAMES},
        "feature_scales": {name: 1.0 for name in FEATURE_NAMES},
        "models": {
            policy.value: {
                "intercept": 1.0,
                "coefficients": {name: 0.0 for name in FEATURE_NAMES},
                "loo_rmse": 0.0,
            }
            for policy in POLICY_PRIORITY
        },
    }


def test_feature_vector_is_generic_normalized_and_fixed_width() -> None:
    initial = ((0, 1, 0), (2, 0, 2), (0, 1, 0))
    transitions = [
        PortfolioTransition(
            action="MOUSE 1 2" if index < 4 else "RIGHT",
            before_grid=initial,
            after_grid=initial,
            gameplay_changed=index == 0,
            hud_changed=index == 1,
            changed_area=index,
        )
        for index in range(8)
    ]
    features = extract_portfolio_features(initial, transitions)
    assert tuple(features) == FEATURE_NAMES
    assert all(0.0 <= value <= 1.0 for value in features.values())
    assert features["mouse_action_fraction"] == 0.5
    assert features["gameplay_change_fraction"] == 0.125
    assert not any(
        token in key
        for key in features
        for token in ("game_id", "coordinate", "hash", "public")
    )


def test_router_ties_and_stock_margin_resolve_to_stock() -> None:
    payload = _artifact_payload()
    router = PortfolioRouter(payload)
    features = {name: 0.0 for name in FEATURE_NAMES}
    assert router.decide(features).policy == PortfolioPolicy.STOCK

    payload["models"]["audit"]["intercept"] = 1.2  # type: ignore[index]
    router = PortfolioRouter(payload)
    decision = router.decide(features)
    assert decision.policy == PortfolioPolicy.STOCK
    assert decision.stock_fallback is True


def test_router_artifact_rejects_game_lookup_inputs() -> None:
    payload = _artifact_payload()
    payload["game_ids"] = ["forbidden"]
    with pytest.raises(ValueError, match="forbidden key"):
        PortfolioRouter(payload)
    payload = _artifact_payload()
    payload["lookup"] = {"unknown-game": "audit"}
    with pytest.raises(ValueError, match="unsupported fields"):
        PortfolioRouter(payload)
    payload = _artifact_payload()
    payload["models"]["stock"]["per_game_scores"] = [1.0]  # type: ignore[index]
    with pytest.raises(ValueError, match="unsupported fields"):
        PortfolioRouter(payload)
    committed = json.loads(PortfolioRouter.default_path().read_text())
    keys: set[str] = set()

    def collect_keys(value) -> None:
        if isinstance(value, dict):
            keys.update(str(key).lower() for key in value)
            for item in value.values():
                collect_keys(item)
        elif isinstance(value, list):
            for item in value:
                collect_keys(item)

    collect_keys(committed)
    assert not keys.intersection(
        {"game_id", "game_ids", "coordinates", "frame_hash", "public_rules"}
    )


def test_router_can_load_an_explicit_candidate_artifact_without_changing_v1() -> None:
    parity_path = Path(__file__).resolve().parents[1] / "src" / "duck_portfolio" / "router_model_parity.json"
    parity = PortfolioRouter.load(parity_path)
    reference = PortfolioRouter.load()
    assert parity.artifact_hash != reference.artifact_hash
    assert parity.payload["experiment"] == "duck-portfolio-parity-v1"
    assert set(parity.payload["relative_models"]) == {
        "audit",
        "deliberate",
        "contract-repair",
    }
    assert reference.payload["experiment"] == "duck-portfolio-v1"


def test_router_relative_guardrail_rejects_uncertain_candidate() -> None:
    payload = _artifact_payload()
    payload["relative_models"] = {
        policy.value: {
            "intercept": 0.1,
            "coefficients": {name: 0.0 for name in FEATURE_NAMES},
            "loo_rmse": 0.1,
        }
        for policy in POLICY_PRIORITY
        if policy != PortfolioPolicy.STOCK
    }
    payload["relative_uncertainty_penalty"] = 0.5
    payload["relative_stock_margin"] = 0.25
    payload["models"]["audit"]["intercept"] = 3.0  # type: ignore[index]
    router = PortfolioRouter(payload)
    decision = router.decide({name: 0.0 for name in FEATURE_NAMES})
    assert decision.policy == PortfolioPolicy.STOCK
    assert decision.stock_fallback is True


def test_eight_action_stock_warmup_routes_without_replacing_conversation() -> None:
    agent = _agent(PortfolioPolicy.DELIBERATE)
    messages_identity = id(agent._history_messages)
    for action_num in range(1, 8):
        _observe(agent, action_num)
        assert agent.maximum_action_batch_size == 8 - action_num
        assert agent.diagnostics["initial_policy"] is None
    _observe(agent, 8)
    assert agent.diagnostics["initial_policy"] == "deliberate"
    assert agent.diagnostics["selection_action"] == 8
    assert agent.diagnostics["active_policy"] == "stock"
    assert agent.maximum_action_batch_size == 0
    prompt = agent._build_user_prompt(8, valid_actions=["RIGHT"])
    assert "Deterministic portfolio route: deliberate" in prompt
    assert "Deliberate loop for this turn" in prompt
    assert agent.diagnostics["active_policy"] == "deliberate"
    assert id(agent._history_messages) == messages_identity
    assert agent.verified_actions_enabled is True


def test_warmup_progress_permanently_locks_stock() -> None:
    agent = _agent(PortfolioPolicy.CONTRACT_REPAIR)
    _observe(agent, 1)
    _observe(agent, 2, progress=True)
    for action_num in range(3, 80):
        _observe(agent, action_num)
    diagnostics = agent.diagnostics
    assert diagnostics["initial_policy"] == "stock"
    assert diagnostics["active_policy"] == "stock"
    assert diagnostics["switch_count"] == 0
    assert diagnostics["route_events"][0]["event"] == "warmup-progress-lock"


def test_positive_reward_without_level_completion_does_not_lock_stock() -> None:
    agent = _agent(PortfolioPolicy.CONTRACT_REPAIR)
    for action_num in range(1, 9):
        before = ((0, 1), (0, 0))
        agent.observe_transition(
            action="RIGHT",
            before_grid=before,
            after_grid=before,
            payload={
                "action_num": action_num,
                "score": 1,
                "level_completed": False,
                "gameplay_changed": False,
                "hud_changed": False,
                "changed_regions": [],
                "time_remaining_seconds": 4000,
            },
        )
    assert agent.diagnostics["initial_policy"] == "contract-repair"
    assert agent.diagnostics["level_progress_seen"] is False


def test_policy_activation_covers_audit_deliberate_and_contract_repair() -> None:
    audit = _agent(PortfolioPolicy.AUDIT)
    for action_num in range(1, 9):
        _observe(audit, action_num)
    repeated = [
        HistoryEntry(
            action="RIGHT",
            frame=Frame(grid=((0,),), step=index, level=1),
        )
        for index in range(4)
    ]
    audit_prompt = audit._build_user_prompt(
        8,
        valid_actions=["RIGHT"],
        history_entries=repeated,
    )
    assert "Sparse self-audit trigger" in audit_prompt

    contract = _agent(PortfolioPolicy.CONTRACT_REPAIR)
    for action_num in range(1, 9):
        _observe(contract, action_num)
    contract_prompt = contract._build_user_prompt(8, valid_actions=["RIGHT"])
    assert contract_prompt.startswith(REPAIR_SYSTEM_ADDENDUM)
    assert contract_prompt.endswith(REPAIR_SYSTEM_ADDENDUM)
    assert contract_prompt.index(CONTRACT_USER_ADDENDUM) < contract_prompt.index(
        "Current state:"
    )
    normalized = contract._normalize_python_actions(
        [{"action": "RIGHT"}, {"action": "LEFT"}]
    )
    assert normalized == [
        {"action": "RIGHT", "expect": {"board_changed": True}}
    ]
    assert contract.maximum_action_batch_size == 1
    assert contract.telemetry["portfolio_contract_repairs"] == 1
    assert contract.telemetry["portfolio_contract_batch_truncations"] == 1


def test_stock_route_preserves_the_stock_prompt_and_still_closes_warmup() -> None:
    agent = _agent(PortfolioPolicy.STOCK)
    base_system_prompt = agent._base_system_prompt
    for action_num in range(1, 9):
        _observe(agent, action_num)
    assert agent.maximum_action_batch_size == 0
    prompt = agent._build_user_prompt(8, valid_actions=["RIGHT"])
    assert "Deterministic portfolio route" not in prompt
    assert agent._system_prompt == base_system_prompt
    assert agent.diagnostics["active_policy"] == "stock"


def test_single_switch_uses_original_ranking_and_exact_guards() -> None:
    agent = _agent(PortfolioPolicy.AUDIT)
    for action_num in range(1, 9):
        _observe(agent, action_num)
    agent._build_user_prompt(8, valid_actions=["RIGHT"])
    for action_num in range(9, 72):
        _observe(agent, action_num)
    assert agent.diagnostics["switch_count"] == 0
    _observe(agent, 72)
    assert agent.diagnostics["switch_count"] == 1
    prompt = agent._build_user_prompt(72, valid_actions=["RIGHT"])
    assert "Portfolio stall switch: audit -> stock" in prompt
    for action_num in range(73, 150):
        _observe(agent, action_num)
    assert agent.diagnostics["switch_count"] == 1

    guarded = _agent(PortfolioPolicy.AUDIT)
    for action_num in range(1, 9):
        _observe(guarded, action_num)
    guarded._build_user_prompt(8, valid_actions=["RIGHT"])
    for action_num in range(9, 73):
        _observe(guarded, action_num, remaining=1799)
    assert guarded.diagnostics["switch_count"] == 0


def test_portfolio_config_solver_fingerprint_and_stock_hash_isolation() -> None:
    reference = HarnessConfig.reference(seed=0)
    assert reference.config_hash == (
        "c93d036c59698bd9edb9a2aa96eafead1caf702edab0598c1a51fce1fa1f48a2"
    )
    portfolio = HarnessConfig.portfolio(seed=0)
    assert portfolio.mode == HarnessMode.DUCK_PORTFOLIO
    assert portfolio.experiment == "duck-portfolio-v1"
    solver = make_solver(portfolio)
    assert type(solver) is DuckPortfolioHarnessSolver
    assert solver.max_runtime_s_per_game == 7920
    assert solver.analyzer_timeout == 900
    local = HarnessConfig.portfolio(seed=0, profile=RuntimeProfile.LOCAL_MLX)
    assert local.model_id == "qwen3.5:4b-mlx"
    assert local.local_game_cap_s == 360

    portfolio.apply_environment()
    portfolio_prompt = prompt_sha256()
    fingerprint = runtime_fingerprint(portfolio)
    assert fingerprint["portfolio"]["router_artifact_sha256"] == (
        PortfolioRouter.load().artifact_hash
    )
    assert fingerprint["portfolio"]["one_persistent_conversation"] is True
    reference.apply_environment()
    assert portfolio_prompt != prompt_sha256()


def test_committed_router_passes_offline_gate_and_has_provenance() -> None:
    router = PortfolioRouter.load()
    cross_validation = router.cross_validation
    assert cross_validation["passed"] is True
    assert cross_validation["mean_lift"] >= 0.10
    assert cross_validation["routed_nonzero_games"] >= cross_validation[
        "stock_nonzero_games"
    ]
    assert cross_validation["distinct_non_stock_policies"] >= 2
    hashes = router.payload["training_artifact_sha256"]
    assert len(hashes) == 8
    assert all(len(value) == 64 for value in hashes.values())


def test_router_training_is_deterministic_and_leave_one_out_isolated() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_portfolio_router.py"
    spec = importlib.util.spec_from_file_location("portfolio_training_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first = module.build_payload()
    second = module.build_payload()
    committed = json.loads(PortfolioRouter.default_path().read_text())
    assert first == second == committed
    assert first["cross_validation"]["method"] == "leave-one-game-out"
    assert first["cross_validation"]["uncertainty_method"] == (
        "outer-holdout-safe-inner-loo-rmse"
    )
    assert first["cross_validation"]["game_count"] == 25


def test_runtime_parity_training_uses_the_live_guardrail_rule() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_portfolio_router.py"
    spec = importlib.util.spec_from_file_location("portfolio_training_parity_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = module.build_parity_payload()
    cv = payload["cross_validation"]
    assert cv["method"] == "leave-one-game-out-runtime-parity"
    assert cv["uncertainty_method"] == "training-fold-loo-rmse-global-per-policy"
    assert cv["passed"] is True
    assert cv["routed_clipped_mean"] == pytest.approx(1.9201007869749132)
    assert cv["selected_policy_counts"] == {
        "audit": 5,
        "contract-repair": 1,
        "stock": 19,
    }


def test_parity_artifact_decisions_match_builder_runtime_decisions() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_portfolio_router.py"
    spec = importlib.util.spec_from_file_location("portfolio_training_runtime_match", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    game_ids, target_rows, _provenance = module._load_targets()
    matrix, _trace_hash = module._load_features(game_ids)
    arrays = {
        policy: module.np.array(values, dtype=float)
        for policy, values in target_rows.items()
    }
    models, rmse = module._fit_runtime_models(matrix, arrays)
    relative_models, relative_rmse = module._fit_relative_models(matrix, arrays)
    router = PortfolioRouter.load(
        Path(__file__).resolve().parents[1]
        / "src"
        / "duck_portfolio"
        / "router_model_parity.json"
    )
    for index, row in enumerate(matrix):
        features = {name: float(row[offset]) for offset, name in enumerate(FEATURE_NAMES)}
        expected, _raw, _adjusted = module._runtime_decision(
            row, models, rmse, relative_models, relative_rmse
        )
        assert router.decide(features).policy == expected
