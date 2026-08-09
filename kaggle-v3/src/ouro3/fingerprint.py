"""Reproducibility fingerprints for Duck reference artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from inference.agent.tool_agent import (
    _build_system_prompt,
    python_tool_description,
)
from ouro3.config import HarnessConfig
from ouro3.scheduler import compute_submission_budget
from duck_deliberate.agent import DELIBERATE_SYSTEM_ADDENDUM, DELIBERATE_USER_ADDENDUM
from duck_contract.agent import CONTRACT_SYSTEM_ADDENDUM, CONTRACT_USER_ADDENDUM
from duck_contract.repair_agent import REPAIR_SYSTEM_ADDENDUM
from duck_audit.agent import AUDIT_USER_ADDENDUM
from duck_information.agent import INFORMATION_USER_ADDENDUM
from duck_hierarchy.agent import HIERARCHY_USER_ADDENDUM
from duck_diversity.agent import DIVERSITY_USER_ADDENDUM
from duck_poetiq.agent import POETIQ_SYSTEM_ADDENDUM, POETIQ_INTERVENTION_ADDENDUM
from duck_portfolio.agent import PORTFOLIO_SELECTION_NOTICE, PORTFOLIO_SWITCH_NOTICE
from duck_portfolio.router import PortfolioRouter
from duck_retrodict.agent import RETRODICT_SYSTEM_ADDENDUM


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_sha256(*, tool_output_tokens: int = 1_024) -> str:
    payload = {
        "system": _build_system_prompt(tool_output_tokens=tool_output_tokens),
        "python_tool": python_tool_description(),
    }
    if os.environ.get("OURO3_HARNESS_MODE", "").strip() == "duck-deliberate":
        payload["deliberate_system_addendum"] = DELIBERATE_SYSTEM_ADDENDUM
    if os.environ.get("OURO3_HARNESS_MODE", "").strip() == "duck-contract":
        payload["contract_system_addendum"] = CONTRACT_SYSTEM_ADDENDUM
    if os.environ.get("OURO3_HARNESS_MODE", "").strip() == "duck-contract-repair":
        payload["contract_system_addendum"] = CONTRACT_SYSTEM_ADDENDUM
        payload["repair_system_addendum"] = REPAIR_SYSTEM_ADDENDUM
    if os.environ.get("OURO3_HARNESS_MODE", "").strip() == "duck-audit":
        payload["audit_user_addendum"] = AUDIT_USER_ADDENDUM
    if os.environ.get("OURO3_HARNESS_MODE", "").strip() == "duck-information":
        payload["information_user_addendum"] = INFORMATION_USER_ADDENDUM
    if os.environ.get("OURO3_HARNESS_MODE", "").strip() == "duck-hierarchy":
        payload["hierarchy_user_addendum"] = HIERARCHY_USER_ADDENDUM
    if os.environ.get("OURO3_HARNESS_MODE", "").strip() == "duck-diversity":
        payload["diversity_user_addendum"] = DIVERSITY_USER_ADDENDUM
    if os.environ.get("OURO3_HARNESS_MODE", "").strip() == "duck-poetiq":
        payload["poetiq_system_addendum"] = POETIQ_SYSTEM_ADDENDUM
        payload["poetiq_intervention_addendum"] = POETIQ_INTERVENTION_ADDENDUM
    if os.environ.get("OURO3_HARNESS_MODE", "").strip() == "duck-portfolio":
        payload.update(
            {
                "portfolio_selection_notice": PORTFOLIO_SELECTION_NOTICE,
                "portfolio_switch_notice": PORTFOLIO_SWITCH_NOTICE,
                "audit_user_addendum": AUDIT_USER_ADDENDUM,
                "deliberate_system_addendum": DELIBERATE_SYSTEM_ADDENDUM,
                "deliberate_user_addendum": DELIBERATE_USER_ADDENDUM,
                "contract_system_addendum": CONTRACT_SYSTEM_ADDENDUM,
                "contract_user_addendum": CONTRACT_USER_ADDENDUM,
                "repair_system_addendum": REPAIR_SYSTEM_ADDENDUM,
            }
        )
    if os.environ.get("OURO3_HARNESS_MODE", "").strip() == "duck-retrodict":
        payload["retrodict_system_addendum"] = RETRODICT_SYSTEM_ADDENDUM
    return _sha256_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )


def runtime_fingerprint(config: HarnessConfig) -> dict[str, Any]:
    inference_root = Path(__file__).resolve().parents[1] / "inference"
    value = {
        "mode": config.mode.value,
        "model_id": config.model_id,
        "vllm": config.vllm_version,
        "torch": config.torch_version,
        "flashinfer": config.flashinfer_version,
        "gpu": config.kaggle_gpu,
        "max_model_len": config.max_model_len,
        "active_context": config.context_window,
        "image_scale": config.image_scale,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "thinking": config.enable_thinking,
        "tool_turns": "unlimited",
        "model_output_tokens": "server-default",
        "python_timeout_s": config.python_timeout_s,
        "python_output_tokens": config.python_output_tokens,
        "concurrency": config.concurrency,
        "per_game_cap_s": config.reference_game_cap_s,
        "analyzer_timeout_s": config.analyzer_timeout_s,
        "only_reset_levels": config.only_reset_levels,
        "prefix_caching": config.enable_prefix_caching,
        "tool_call_parser": config.tool_call_parser,
        "reasoning_parser": config.reasoning_parser,
        "seed": config.seed,
        "prompt_sha256": prompt_sha256(
            tool_output_tokens=config.python_output_tokens
        ),
        "tool_agent_source_sha256": _sha256_file(
            inference_root / "agent" / "tool_agent.py"
        ),
        "solver_source_sha256": _sha256_file(
            inference_root / "framework" / "solver.py"
        ),
        "source_manifest_sha256": os.environ.get(
            "OURO3_SOURCE_MANIFEST_SHA256", ""
        ).strip(),
    }
    if config.mode.value == "duck-memory":
        value["memory"] = {
            "reasoning_history_field": "reasoning_content",
            "preserve_thinking": True,
            "reasoning_template_verified": (
                os.environ.get(
                    "OURO3_REASONING_TEMPLATE_VERIFIED", ""
                ).strip().lower()
                in {"1", "true", "yes", "on"}
            ),
            "reasoning_template_sha256": os.environ.get(
                "OURO3_REASONING_TEMPLATE_SHA256", ""
            ).strip(),
            "compaction_trigger_tokens": config.compaction_trigger_tokens,
            "compaction_target_tokens": config.compaction_target_tokens,
            "compaction_recent_assistant_turns": (
                config.compaction_recent_assistant_turns
            ),
            "compaction_max_output_tokens": (
                config.compaction_max_output_tokens
            ),
            "compaction_timeout_s": config.compaction_timeout_s,
            "compaction_temperature": config.compaction_temperature,
            "compaction_top_p": config.compaction_top_p,
            "compaction_top_k": config.compaction_top_k,
            "compaction_max_concurrency": config.compaction_max_concurrency,
        }
    if config.mode.value == "duck-reasoning":
        value["reasoning"] = {
            "reasoning_history_field": "reasoning_content",
            "preserve_thinking": True,
            "reasoning_template_verified": (
                os.environ.get(
                    "OURO3_REASONING_TEMPLATE_VERIFIED", ""
                ).strip().lower()
                in {"1", "true", "yes", "on"}
            ),
            "reasoning_template_sha256": os.environ.get(
                "OURO3_REASONING_TEMPLATE_SHA256", ""
            ).strip(),
            "history_policy": "stock-duck",
            "semantic_compaction": False,
            "auxiliary_model_calls": 0,
        }
    if config.mode.value == "duck-retrodict":
        value["retrodict"] = {
            "persistent_evidence": True,
            "rule_language": "typed-host-owned-v1",
            "full_log_replay": True,
            "multi_ontology": ["color-4", "color-8", "color-4-all"],
            "automatic_batch_size": 1,
            "fallback_batch_limit": None,
            "max_rules": config.retrodict_max_rules,
            "prediction_threshold": config.retrodict_prediction_threshold,
        }
    if config.mode.value == "duck-audit":
        value["audit"] = {
            "trigger": "repeated-action-or-unchanged-gameplay-frame",
            "repeat_threshold": config.audit_repeat_threshold,
            "no_change_threshold": config.audit_no_change_threshold,
            "max_triggers": config.audit_max_triggers,
            "second_model_request": False,
            "stock_history_policy": True,
        }
    if config.mode.value == "duck-information":
        value["information"] = {
            "trigger": "unchanged-gameplay-frame",
            "no_change_threshold": config.information_no_change_threshold,
            "max_triggers": config.information_max_triggers,
            "second_model_request": False,
            "stock_history_policy": True,
        }
    if config.mode.value == "duck-hierarchy":
        value["hierarchy"] = {
            "trigger": "new-level-or-unchanged-gameplay-frame",
            "no_change_threshold": config.hierarchy_no_change_threshold,
            "max_triggers": config.hierarchy_max_triggers,
            "maximum_candidates": 3,
            "second_model_request": False,
            "stock_history_policy": True,
        }
    if config.mode.value == "duck-diversity":
        value["diversity"] = {
            "trigger": "unchanged-gameplay-frame",
            "no_change_threshold": config.diversity_no_change_threshold,
            "max_triggers": config.diversity_max_triggers,
            "seed_offset": config.diversity_seed_offset,
            "second_model_request": False,
            "stock_history_policy": True,
        }
    if config.mode.value == "duck-poetiq":
        value["poetiq"] = {
            "trigger": "repeated-action-or-unchanged-gameplay-frame",
            "repeat_threshold": config.poetiq_repeat_threshold,
            "no_change_threshold": config.poetiq_no_change_threshold,
            "cooldown_actions": config.poetiq_intervention_cooldown_actions,
            "max_interventions_per_level": config.poetiq_max_interventions_per_level,
            "diversity_seed_offset": config.poetiq_diversity_seed_offset,
            "yield_min_actions": config.poetiq_yield_min_actions,
            "yield_min_elapsed_s": config.poetiq_yield_min_elapsed_s,
            "yield_window": config.poetiq_yield_window,
            "yield_max_changes": config.poetiq_yield_max_changes,
            "maximum_candidates": 3,
            "second_model_request": False,
            "stock_tool_surface": True,
            "stock_history_policy": True,
        }
    if config.mode.value == "duck-portfolio":
        router = PortfolioRouter.load()
        value["portfolio"] = {
            "candidate_order": list(router.payload["candidate_order"]),
            "warmup_actions": config.portfolio_warmup_actions,
            "score_clip": config.portfolio_score_clip,
            "ridge_alpha": config.portfolio_ridge_alpha,
            "uncertainty_penalty": config.portfolio_uncertainty_penalty,
            "stock_margin": config.portfolio_stock_margin,
            "relative_guardrail_enabled": bool(router.relative_models),
            "relative_uncertainty_penalty": router.relative_uncertainty_penalty,
            "relative_stock_margin": router.relative_stock_margin,
            "switch_min_actions": config.portfolio_switch_min_actions,
            "switch_window": config.portfolio_switch_window,
            "switch_max_changes": config.portfolio_switch_max_changes,
            "switch_min_remaining_s": config.portfolio_switch_min_remaining_s,
            "max_switches": config.portfolio_max_switches,
            "router_artifact_sha256": router.artifact_hash,
            "router_schema_version": int(router.payload["schema_version"]),
            "one_persistent_conversation": True,
            "parallel_model_trajectories": 0,
            "game_identifiers_allowed": False,
        }
    if config.profile.value == "kaggle-submission":
        budget = compute_submission_budget(
            total_games=110,
            concurrency=config.concurrency,
            configured_game_cap_s=config.reference_game_cap_s,
            soft_deadline_s=config.soft_deadline_s,
            setup_teardown_reserve_s=config.setup_teardown_reserve_s,
        )
        value["submission_budget"] = {
            "total_games": budget.total_games,
            "concurrency": budget.concurrency,
            "waves": budget.waves,
            "per_game_cap_s": budget.per_game_cap_s,
            "worst_case_gameplay_s": budget.worst_case_gameplay_s,
            "soft_deadline_s": budget.soft_deadline_s,
            "setup_teardown_reserve_s": budget.setup_teardown_reserve_s,
            "safety_fraction": budget.safety_fraction,
        }
    return value
