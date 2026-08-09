"""Validated configuration for local, validation, and competition runs."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from ouro3.mode import HarnessMode

_ROBUST_CONFIG_FIELDS = {
    "robust_warmup_s",
    "robust_min_actions",
    "robust_success_threshold",
    "robust_required_low_windows",
    "robust_hypothesis_temperature",
    "robust_execution_temperature",
    "robust_max_execution_batch",
}

_MEMORY_CONFIG_FIELDS = {
    "compaction_trigger_tokens",
    "compaction_target_tokens",
    "compaction_recent_assistant_turns",
    "compaction_max_output_tokens",
    "compaction_timeout_s",
    "compaction_temperature",
    "compaction_top_p",
    "compaction_top_k",
    "compaction_max_concurrency",
}

_AUDIT_CONFIG_FIELDS = {
    "audit_repeat_threshold",
    "audit_no_change_threshold",
    "audit_max_triggers",
}

_INFORMATION_CONFIG_FIELDS = {
    "information_no_change_threshold",
    "information_max_triggers",
}

_HIERARCHY_CONFIG_FIELDS = {
    "hierarchy_no_change_threshold",
    "hierarchy_max_triggers",
}

_DIVERSITY_CONFIG_FIELDS = {
    "diversity_no_change_threshold",
    "diversity_max_triggers",
    "diversity_seed_offset",
}

_POETIQ_CONFIG_FIELDS = {
    "poetiq_repeat_threshold",
    "poetiq_no_change_threshold",
    "poetiq_intervention_cooldown_actions",
    "poetiq_max_interventions_per_level",
    "poetiq_diversity_seed_offset",
    "poetiq_yield_min_actions",
    "poetiq_yield_min_elapsed_s",
    "poetiq_yield_window",
    "poetiq_yield_max_changes",
}

_PORTFOLIO_CONFIG_FIELDS = {
    "portfolio_warmup_actions",
    "portfolio_score_clip",
    "portfolio_ridge_alpha",
    "portfolio_uncertainty_penalty",
    "portfolio_stock_margin",
    "portfolio_switch_min_actions",
    "portfolio_switch_window",
    "portfolio_switch_max_changes",
    "portfolio_switch_min_remaining_s",
    "portfolio_max_switches",
}

_RETRODICT_CONFIG_FIELDS = {
    "retrodict_max_rules",
    "retrodict_prediction_threshold",
}


class RuntimeProfile(StrEnum):
    LOCAL_MLX = "local-mlx"
    RTX_VALIDATION = "rtx-validation"
    KAGGLE_SUBMISSION = "kaggle-submission"
    SCRIPTED_REHEARSAL = "scripted-rehearsal"


@dataclass(frozen=True)
class HarnessConfig:
    """One immutable run configuration.

    The reference fields intentionally reproduce the public Duck run before
    any v3 experiment is promoted.
    """

    experiment: str = "duck-reference"
    mode: HarnessMode = HarnessMode.DUCK_REFERENCE
    profile: RuntimeProfile = RuntimeProfile.RTX_VALIDATION
    model_id: str = "vrfai/Qwen3.6-27B-FP8"
    base_url: str = "http://127.0.0.1:1234/v1"
    concurrency: int = 28
    local_workers: int = 2
    context_window: int = 32_768
    max_model_len: int = 65_536
    image_scale: int = 4
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    seed: int | None = None
    analyzer_timeout_s: float = 900.0
    python_timeout_s: int = 30
    python_output_tokens: int = 1_024
    reference_game_cap_s: int = 7_920
    local_game_cap_s: int = 360
    setup_teardown_reserve_s: int = 20 * 60
    soft_deadline_s: int = 8 * 60 * 60 + 40 * 60
    model_failure_floor: int = 3
    strategic_reset_min_actions: int = 64
    strategic_reset_window: int = 16
    strategic_reset_max_changes: int = 2
    robust_warmup_s: int = 30 * 60
    robust_min_actions: int = 64
    robust_success_threshold: float = 0.25
    robust_required_low_windows: int = 2
    robust_hypothesis_temperature: float = 0.8
    robust_execution_temperature: float = 0.2
    robust_max_execution_batch: int = 8
    compaction_trigger_tokens: int = 24_576
    compaction_target_tokens: int = 16_384
    compaction_recent_assistant_turns: int = 8
    compaction_max_output_tokens: int = 2_048
    compaction_timeout_s: float = 300.0
    compaction_temperature: float = 0.2
    compaction_top_p: float = 0.9
    compaction_top_k: int = 20
    compaction_max_concurrency: int = 4
    audit_repeat_threshold: int = 3
    audit_no_change_threshold: int = 2
    audit_max_triggers: int = 8
    information_no_change_threshold: int = 2
    information_max_triggers: int = 8
    hierarchy_no_change_threshold: int = 2
    hierarchy_max_triggers: int = 8
    diversity_no_change_threshold: int = 2
    diversity_max_triggers: int = 8
    diversity_seed_offset: int = 17
    poetiq_repeat_threshold: int = 4
    poetiq_no_change_threshold: int = 3
    poetiq_intervention_cooldown_actions: int = 12
    poetiq_max_interventions_per_level: int = 2
    poetiq_diversity_seed_offset: int = 17
    poetiq_yield_min_actions: int = 64
    poetiq_yield_min_elapsed_s: float = 30 * 60
    poetiq_yield_window: int = 16
    poetiq_yield_max_changes: int = 0
    portfolio_warmup_actions: int = 8
    portfolio_score_clip: float = 10.0
    portfolio_ridge_alpha: float = 10.0
    portfolio_uncertainty_penalty: float = 0.5
    portfolio_stock_margin: float = 0.25
    portfolio_switch_min_actions: int = 64
    portfolio_switch_window: int = 16
    portfolio_switch_max_changes: int = 0
    portfolio_switch_min_remaining_s: float = 1800.0
    portfolio_max_switches: int = 1
    retrodict_max_rules: int = 256
    retrodict_prediction_threshold: float = 0.90
    vllm_version: str = "0.19.0"
    torch_version: str = "2.10.0"
    flashinfer_version: str = "0.6.6"
    enable_thinking: bool = True
    enable_prefix_caching: bool = True
    only_reset_levels: bool = True
    tool_call_parser: str = "qwen3_coder"
    reasoning_parser: str = "qwen3"
    kaggle_gpu: str = "NvidiaRtxPro6000"
    wheelhouse_dataset: str = "driessmit1/arc3-vllm-h100-wheelhouse-v3"
    model_dataset: str = "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"
    source_dataset: str = "kinwochan/ouroboros-arc-agi-3-v3-source"
    validation_kernel: str = "kinwochan/ouroboros-arc-agi-3-v3-validation"
    submission_kernel: str = "kinwochan/ouroboros-arc-agi-3-v3"

    def validate(self) -> "HarnessConfig":
        if not self.experiment.strip():
            raise ValueError("experiment must be non-empty")
        if self.concurrency < 1 or self.local_workers < 1:
            raise ValueError("worker counts must be positive")
        if self.context_window < 1_024 or self.context_window > self.max_model_len:
            raise ValueError("context_window must be between 1024 and max_model_len")
        if self.image_scale < 1:
            raise ValueError("image_scale must be positive")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be in [0, 2]")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.analyzer_timeout_s <= 0 or self.reference_game_cap_s <= 0:
            raise ValueError("reference timeouts must be positive")
        if self.python_timeout_s != 30 or self.python_output_tokens != 1_024:
            raise ValueError("Duck Python tool limits must remain 30s and 1024 tokens")
        if self.setup_teardown_reserve_s >= self.soft_deadline_s:
            raise ValueError("setup/teardown reserve must be smaller than the soft deadline")
        if self.model_failure_floor < 1:
            raise ValueError("model_failure_floor must be positive")
        if self.retrodict_max_rules < 8:
            raise ValueError("retrodict_max_rules must be at least eight")
        if not 0.5 <= self.retrodict_prediction_threshold <= 1.0:
            raise ValueError(
                "retrodict_prediction_threshold must be in [0.5, 1.0]"
            )
        if self.strategic_reset_window < 1:
            raise ValueError("strategic_reset_window must be positive")
        if self.strategic_reset_max_changes >= self.strategic_reset_window:
            raise ValueError("strategic reset change threshold must be smaller than its window")
        if self.robust_warmup_s < 0 or self.robust_min_actions < 1:
            raise ValueError("robust recovery warmup and action threshold are invalid")
        if not 0 < self.robust_success_threshold < 1:
            raise ValueError("robust_success_threshold must be in (0, 1)")
        if self.robust_required_low_windows < 1:
            raise ValueError("robust_required_low_windows must be positive")
        if not 0 <= self.robust_hypothesis_temperature <= 2:
            raise ValueError("robust_hypothesis_temperature must be in [0, 2]")
        if not 0 <= self.robust_execution_temperature <= 2:
            raise ValueError("robust_execution_temperature must be in [0, 2]")
        if not 1 <= self.robust_max_execution_batch <= 8:
            raise ValueError("robust_max_execution_batch must be between 1 and 8")
        if self.compaction_trigger_tokens <= self.compaction_target_tokens:
            raise ValueError("compaction trigger must exceed its target")
        if self.compaction_trigger_tokens >= self.context_window:
            raise ValueError("compaction trigger must be below the active context")
        if self.compaction_recent_assistant_turns < 2:
            raise ValueError("compaction must retain at least two assistant turns")
        if self.compaction_max_output_tokens < 256:
            raise ValueError("compaction output budget is too small")
        if self.compaction_timeout_s <= 0:
            raise ValueError("compaction timeout must be positive")
        if not 0 <= self.compaction_temperature <= 2:
            raise ValueError("compaction temperature must be in [0, 2]")
        if not 0 < self.compaction_top_p <= 1:
            raise ValueError("compaction top-p must be in (0, 1]")
        if self.compaction_top_k < 1:
            raise ValueError("compaction top-k must be positive")
        if self.compaction_max_concurrency != 4:
            raise ValueError("duck-memory v1 requires four compaction slots")
        if self.audit_repeat_threshold < 2:
            raise ValueError("duck-audit repeat threshold must be at least two")
        if self.audit_no_change_threshold < 2:
            raise ValueError("duck-audit no-change threshold must be at least two")
        if self.audit_max_triggers < 1:
            raise ValueError("duck-audit max triggers must be positive")
        if self.information_no_change_threshold < 2:
            raise ValueError(
                "duck-information no-change threshold must be at least two"
            )
        if self.information_max_triggers < 1:
            raise ValueError("duck-information max triggers must be positive")
        if self.hierarchy_no_change_threshold < 2:
            raise ValueError(
                "duck-hierarchy no-change threshold must be at least two"
            )
        if self.hierarchy_max_triggers < 1:
            raise ValueError("duck-hierarchy max triggers must be positive")
        if self.diversity_no_change_threshold < 2:
            raise ValueError(
                "duck-diversity no-change threshold must be at least two"
            )
        if self.diversity_max_triggers < 1:
            raise ValueError("duck-diversity max triggers must be positive")
        if self.diversity_seed_offset < 1:
            raise ValueError("duck-diversity seed offset must be positive")
        if self.poetiq_repeat_threshold < 2:
            raise ValueError("duck-poetiq repeat threshold must be at least two")
        if self.poetiq_no_change_threshold < 2:
            raise ValueError("duck-poetiq no-change threshold must be at least two")
        if self.poetiq_intervention_cooldown_actions < 0:
            raise ValueError("duck-poetiq intervention cooldown must not be negative")
        if self.poetiq_max_interventions_per_level < 1:
            raise ValueError("duck-poetiq must allow at least one intervention per level")
        if self.poetiq_diversity_seed_offset < 1:
            raise ValueError("duck-poetiq diversity seed offset must be positive")
        if self.poetiq_yield_min_actions < 1 or self.poetiq_yield_min_elapsed_s < 0:
            raise ValueError("duck-poetiq stalled-yield thresholds are invalid")
        if self.poetiq_yield_window < 1:
            raise ValueError("duck-poetiq stalled-yield window must be positive")
        if self.poetiq_yield_max_changes < 0:
            raise ValueError("duck-poetiq stalled-yield change threshold must not be negative")
        if self.portfolio_warmup_actions != 8:
            raise ValueError("duck-portfolio-v1 requires exactly eight warm-up actions")
        if self.portfolio_score_clip != 10.0 or self.portfolio_ridge_alpha != 10.0:
            raise ValueError("duck-portfolio-v1 router training constants changed")
        if not 0 <= self.portfolio_uncertainty_penalty <= 2:
            raise ValueError("duck-portfolio uncertainty penalty must be in [0, 2]")
        if self.portfolio_stock_margin < 0:
            raise ValueError("duck-portfolio Stock margin must not be negative")
        if self.portfolio_switch_min_actions < 1 or self.portfolio_switch_window < 1:
            raise ValueError("duck-portfolio switch thresholds must be positive")
        if self.portfolio_switch_max_changes < 0:
            raise ValueError("duck-portfolio switch change threshold must not be negative")
        if self.portfolio_switch_min_remaining_s < 0:
            raise ValueError("duck-portfolio remaining-time guard must not be negative")
        if self.portfolio_max_switches != 1:
            raise ValueError("duck-portfolio-v1 permits exactly one policy switch")
        expected_pins = ("0.19.0", "2.10.0", "0.6.6")
        if (self.vllm_version, self.torch_version, self.flashinfer_version) != expected_pins:
            raise ValueError(
                "runtime pins changed; expected vLLM 0.19.0, Torch 2.10.0, FlashInfer 0.6.6"
            )
        if self.kaggle_gpu != "NvidiaRtxPro6000":
            raise ValueError("Kaggle kernels must use NvidiaRtxPro6000")
        if self.mode == HarnessMode.DUCK_REFERENCE:
            exact_reference = {
                "model_id": "vrfai/Qwen3.6-27B-FP8",
                "concurrency": 28,
                "context_window": 32_768,
                "max_model_len": 65_536,
                "image_scale": 4,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "analyzer_timeout_s": 900.0,
                "python_timeout_s": 30,
                "python_output_tokens": 1_024,
                "reference_game_cap_s": 7_920,
                "soft_deadline_s": 31_200,
                "enable_thinking": True,
                "enable_prefix_caching": True,
                "only_reset_levels": True,
                "tool_call_parser": "qwen3_coder",
                "reasoning_parser": "qwen3",
            }
            mismatches = {
                name: {"expected": expected, "actual": getattr(self, name)}
                for name, expected in exact_reference.items()
                if getattr(self, name) != expected
            }
            if mismatches:
                raise ValueError(
                    "duck-reference runtime changed: "
                    + json.dumps(mismatches, sort_keys=True)
                )
        for name in (
            self.wheelhouse_dataset,
            self.model_dataset,
            self.source_dataset,
            self.validation_kernel,
            self.submission_kernel,
        ):
            if name.count("/") != 1:
                raise ValueError(f"invalid Kaggle reference: {name!r}")
        return self

    @classmethod
    def reference(cls, *, seed: int | None = None) -> "HarnessConfig":
        return cls(seed=seed).validate()

    @classmethod
    def local(cls, *, seed: int = 0) -> "HarnessConfig":
        return cls(
            experiment="ouro-hybrid-local",
            mode=HarnessMode.OURO_HYBRID,
            profile=RuntimeProfile.LOCAL_MLX,
            model_id="qwen3.5:4b-mlx",
            base_url="http://127.0.0.1:11434/v1",
            concurrency=2,
            local_workers=2,
            seed=seed,
            analyzer_timeout_s=90,
        ).validate()

    @classmethod
    def robust(
        cls,
        *,
        seed: int | None = None,
        profile: RuntimeProfile = RuntimeProfile.RTX_VALIDATION,
    ) -> "HarnessConfig":
        return cls(
            experiment="duck-robust",
            mode=HarnessMode.DUCK_ROBUST,
            profile=profile,
            seed=seed,
        ).validate()

    @classmethod
    def memory(
        cls,
        *,
        seed: int | None = 0,
        profile: RuntimeProfile = RuntimeProfile.RTX_VALIDATION,
    ) -> "HarnessConfig":
        values: dict[str, Any] = {
            "experiment": "duck-memory-v1",
            "mode": HarnessMode.DUCK_MEMORY,
            "profile": profile,
            "seed": seed,
        }
        if profile == RuntimeProfile.LOCAL_MLX:
            values.update(
                {
                    "model_id": "qwen3.5:4b-mlx",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "concurrency": 2,
                    "local_workers": 2,
                    "analyzer_timeout_s": 90.0,
                }
            )
        return cls(**values).validate()

    @classmethod
    def reasoning(
        cls,
        *,
        seed: int | None = 0,
        profile: RuntimeProfile = RuntimeProfile.RTX_VALIDATION,
    ) -> "HarnessConfig":
        """Stock Duck with only the Qwen reasoning wire-field correction."""

        values: dict[str, Any] = {
            "experiment": "duck-reasoning-v1",
            "mode": HarnessMode.DUCK_REASONING,
            "profile": profile,
            "seed": seed,
        }
        if profile == RuntimeProfile.LOCAL_MLX:
            values.update(
                {
                    "model_id": "qwen3.5:4b-mlx",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "concurrency": 2,
                    "local_workers": 2,
                    "analyzer_timeout_s": 90.0,
                }
            )
        return cls(**values).validate()

    @classmethod
    def deliberate(
        cls,
        *,
        seed: int | None = 0,
        profile: RuntimeProfile = RuntimeProfile.RTX_VALIDATION,
    ) -> "HarnessConfig":
        """Stock Duck with falsification-first proposals and verification."""

        values: dict[str, Any] = {
            "experiment": "duck-deliberate-v1",
            "mode": HarnessMode.DUCK_DELIBERATE,
            "profile": profile,
            "seed": seed,
        }
        if profile == RuntimeProfile.LOCAL_MLX:
            values.update(
                {
                    "model_id": "qwen3.5:4b-mlx",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "concurrency": 2,
                    "local_workers": 2,
                    "analyzer_timeout_s": 180.0,
                    "local_game_cap_s": 1200,
                }
            )
        return cls(**values).validate()

    @classmethod
    def contract(
        cls,
        *,
        seed: int | None = 0,
        profile: RuntimeProfile = RuntimeProfile.RTX_VALIDATION,
    ) -> "HarnessConfig":
        """Stock Duck with a mandatory one-step executable action contract."""

        values: dict[str, Any] = {
            "experiment": "duck-contract-v1",
            "mode": HarnessMode.DUCK_CONTRACT,
            "profile": profile,
            "seed": seed,
        }
        if profile == RuntimeProfile.LOCAL_MLX:
            values.update(
                {
                    "model_id": "qwen3.5:4b-mlx",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "concurrency": 2,
                    "local_workers": 2,
                    "analyzer_timeout_s": 180.0,
                    "local_game_cap_s": 1200,
                }
            )
        return cls(**values).validate()

    @classmethod
    def contract_repair(
        cls,
        *,
        seed: int | None = 0,
        profile: RuntimeProfile = RuntimeProfile.RTX_VALIDATION,
    ) -> "HarnessConfig":
        """Stock Duck with logged one-step expectation repairs."""

        values: dict[str, Any] = {
            "experiment": "duck-contract-repair-v1",
            "mode": HarnessMode.DUCK_CONTRACT_REPAIR,
            "profile": profile,
            "seed": seed,
        }
        if profile == RuntimeProfile.LOCAL_MLX:
            values.update(
                {
                    "model_id": "qwen3.5:4b-mlx",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "concurrency": 2,
                    "local_workers": 2,
                    "analyzer_timeout_s": 180.0,
                    "local_game_cap_s": 360,
                }
            )
        return cls(**values).validate()

    @classmethod
    def audit(
        cls,
        *,
        seed: int | None = 0,
        profile: RuntimeProfile = RuntimeProfile.RTX_VALIDATION,
    ) -> "HarnessConfig":
        """Stock Duck with a sparse, event-triggered self-audit sidecar."""

        values: dict[str, Any] = {
            "experiment": "duck-audit-v1",
            "mode": HarnessMode.DUCK_AUDIT,
            "profile": profile,
            "seed": seed,
        }
        if profile == RuntimeProfile.LOCAL_MLX:
            values.update(
                {
                    "model_id": "qwen3.5:4b-mlx",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "concurrency": 2,
                    "local_workers": 2,
                    "analyzer_timeout_s": 180.0,
                    "local_game_cap_s": 360,
                }
            )
        return cls(**values).validate()

    @classmethod
    def information(
        cls,
        *,
        seed: int | None = 0,
        profile: RuntimeProfile = RuntimeProfile.RTX_VALIDATION,
    ) -> "HarnessConfig":
        """Stock Duck with a sparse, event-triggered information request."""

        values: dict[str, Any] = {
            "experiment": "duck-information-v1",
            "mode": HarnessMode.DUCK_INFORMATION,
            "profile": profile,
            "seed": seed,
        }
        if profile == RuntimeProfile.LOCAL_MLX:
            values.update(
                {
                    "model_id": "qwen3.5:4b-mlx",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "concurrency": 2,
                    "local_workers": 2,
                    "analyzer_timeout_s": 180.0,
                    "local_game_cap_s": 360,
                }
            )
        return cls(**values).validate()

    @classmethod
    def hierarchy(
        cls,
        *,
        seed: int | None = 0,
        profile: RuntimeProfile = RuntimeProfile.RTX_VALIDATION,
    ) -> "HarnessConfig":
        """Stock Duck with bounded candidate search on generic stalls."""

        values: dict[str, Any] = {
            "experiment": "duck-hierarchy-v1",
            "mode": HarnessMode.DUCK_HIERARCHY,
            "profile": profile,
            "seed": seed,
        }
        if profile == RuntimeProfile.LOCAL_MLX:
            values.update(
                {
                    "model_id": "qwen3.5:4b-mlx",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "concurrency": 2,
                    "local_workers": 2,
                    "analyzer_timeout_s": 180.0,
                    "local_game_cap_s": 360,
                }
            )
        return cls(**values).validate()

    @classmethod
    def diversity(
        cls,
        *,
        seed: int | None = 0,
        profile: RuntimeProfile = RuntimeProfile.RTX_VALIDATION,
    ) -> "HarnessConfig":
        """Stock Duck with a sparse alternate sampling path on stalls."""

        values: dict[str, Any] = {
            "experiment": "duck-diversity-v1",
            "mode": HarnessMode.DUCK_DIVERSITY,
            "profile": profile,
            "seed": seed,
        }
        if profile == RuntimeProfile.LOCAL_MLX:
            values.update(
                {
                    "model_id": "qwen3.5:4b-mlx",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "concurrency": 2,
                    "local_workers": 2,
                    "analyzer_timeout_s": 180.0,
                    "local_game_cap_s": 360,
                }
            )
        return cls(**values).validate()

    @classmethod
    def poetiq(
        cls,
        *,
        seed: int | None = 0,
        profile: RuntimeProfile = RuntimeProfile.RTX_VALIDATION,
    ) -> "HarnessConfig":
        """Stock Duck with one compact Poetiq-inspired intervention protocol."""

        values: dict[str, Any] = {
            "experiment": "duck-poetiq-v1",
            "mode": HarnessMode.DUCK_POETIQ,
            "profile": profile,
            "seed": seed,
        }
        if profile == RuntimeProfile.LOCAL_MLX:
            values.update(
                {
                    "model_id": "qwen3.5:4b-mlx",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "concurrency": 2,
                    "local_workers": 2,
                    "analyzer_timeout_s": 180.0,
                    "local_game_cap_s": 360,
                }
            )
        return cls(**values).validate()

    @classmethod
    def portfolio(
        cls,
        *,
        seed: int | None = 0,
        profile: RuntimeProfile = RuntimeProfile.RTX_VALIDATION,
    ) -> "HarnessConfig":
        """Stock Duck with a deterministic generic policy router."""

        values: dict[str, Any] = {
            "experiment": "duck-portfolio-v1",
            "mode": HarnessMode.DUCK_PORTFOLIO,
            "profile": profile,
            "seed": seed,
        }
        if profile == RuntimeProfile.LOCAL_MLX:
            values.update(
                {
                    "model_id": "qwen3.5:4b-mlx",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "concurrency": 2,
                    "local_workers": 2,
                    "analyzer_timeout_s": 180.0,
                    "local_game_cap_s": 360,
                }
            )
        return cls(**values).validate()

    @classmethod
    def retrodict(
        cls,
        *,
        seed: int | None = 0,
        profile: RuntimeProfile = RuntimeProfile.RTX_VALIDATION,
    ) -> "HarnessConfig":
        """Stock Duck with persistent typed replay and verifier-owned search."""

        values: dict[str, Any] = {
            "experiment": "duck-retrodict-v1",
            "mode": HarnessMode.DUCK_RETRODICT,
            "profile": profile,
            "seed": seed,
        }
        if profile == RuntimeProfile.LOCAL_MLX:
            values.update(
                {
                    "model_id": "qwen3.5:4b-mlx",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "concurrency": 2,
                    "local_workers": 2,
                    "analyzer_timeout_s": 180.0,
                    "local_game_cap_s": 360,
                }
            )
        return cls(**values).validate()

    @classmethod
    def retrodict_challenger(
        cls,
        *,
        model_dataset: str,
        model_id: str = "vrfai/Qwen3.6-35B-A3B-FP8",
        seed: int | None = 0,
        profile: RuntimeProfile = RuntimeProfile.RTX_VALIDATION,
    ) -> "HarnessConfig":
        """Retrodict A/B arm; callers must name the attached Kaggle snapshot."""

        return cls.retrodict(seed=seed, profile=profile).with_overrides(
            experiment="duck-retrodict-qwen36-35b-a3b-v1",
            model_id=model_id,
            model_dataset=model_dataset,
        )

    @classmethod
    def scripted(cls, *, seed: int = 0) -> "HarnessConfig":
        return cls(
            experiment="ouro-hybrid-scripted",
            mode=HarnessMode.OURO_HYBRID,
            profile=RuntimeProfile.SCRIPTED_REHEARSAL,
            model_id="scripted",
            base_url="",
            concurrency=28,
            seed=seed,
            analyzer_timeout_s=5,
        ).validate()

    @classmethod
    def from_json(cls, path: Path) -> "HarnessConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("configuration must be a JSON object")
        if "profile" in raw:
            raw["profile"] = RuntimeProfile(str(raw["profile"]))
        if "mode" in raw:
            raw["mode"] = HarnessMode(str(raw["mode"]))
        return cls(**raw).validate()

    def with_overrides(self, **overrides: Any) -> "HarnessConfig":
        if "profile" in overrides and not isinstance(overrides["profile"], RuntimeProfile):
            overrides["profile"] = RuntimeProfile(str(overrides["profile"]))
        if "mode" in overrides and not isinstance(overrides["mode"], HarnessMode):
            overrides["mode"] = HarnessMode(str(overrides["mode"]))
        return replace(self, **overrides).validate()

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["profile"] = self.profile.value
        value["mode"] = self.mode.value
        # Preserve audited hashes by excluding mode-specific fields everywhere
        # except the mode that consumes them.
        if self.mode == HarnessMode.DUCK_REFERENCE:
            for name in _ROBUST_CONFIG_FIELDS:
                value.pop(name, None)
        if self.mode != HarnessMode.DUCK_ROBUST:
            for name in _ROBUST_CONFIG_FIELDS:
                value.pop(name, None)
        if self.mode != HarnessMode.DUCK_MEMORY:
            for name in _MEMORY_CONFIG_FIELDS:
                value.pop(name, None)
        if self.mode != HarnessMode.DUCK_AUDIT:
            for name in _AUDIT_CONFIG_FIELDS:
                value.pop(name, None)
        if self.mode != HarnessMode.DUCK_INFORMATION:
            for name in _INFORMATION_CONFIG_FIELDS:
                value.pop(name, None)
        if self.mode != HarnessMode.DUCK_HIERARCHY:
            for name in _HIERARCHY_CONFIG_FIELDS:
                value.pop(name, None)
        if self.mode != HarnessMode.DUCK_DIVERSITY:
            for name in _DIVERSITY_CONFIG_FIELDS:
                value.pop(name, None)
        if self.mode != HarnessMode.DUCK_POETIQ:
            for name in _POETIQ_CONFIG_FIELDS:
                value.pop(name, None)
        if self.mode != HarnessMode.DUCK_PORTFOLIO:
            for name in _PORTFOLIO_CONFIG_FIELDS:
                value.pop(name, None)
        if self.mode != HarnessMode.DUCK_RETRODICT:
            for name in _RETRODICT_CONFIG_FIELDS:
                value.pop(name, None)
        return value

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def apply_environment(self) -> None:
        """Publish only the variables consumed by the attributed Duck core."""

        values = {
            "INFERENCE_ANALYZER_MODEL": self.model_id,
            "LOCAL_ANALYZER_MODEL_ID": self.model_id,
            "LOCAL_ANALYZER_BASE_URL": self.base_url,
            "LOCAL_ANALYZER_PROVIDER": (
                "openai" if self.profile == RuntimeProfile.LOCAL_MLX else "vllm"
            ),
            "LOCAL_ANALYZER_CONTEXT_WINDOW": str(self.context_window),
            "LOCAL_ANALYZER_TEMPERATURE": str(self.temperature),
            "LOCAL_ANALYZER_TOP_P": str(self.top_p),
            "LOCAL_ANALYZER_TOP_K": str(self.top_k),
            "LOCAL_ANALYZER_TIMEOUT": str(self.analyzer_timeout_s),
            "LOCAL_ANALYZER_TOOL_TIMEOUT": str(self.python_timeout_s),
            # The small local model otherwise spends most of its six-minute
            # integration budget thinking. The 27B reference keeps an
            # unlimited output/tool loop to reproduce Duck before promotion.
            "LOCAL_ANALYZER_MAX_OUTPUT": (
                str(self.python_output_tokens)
                if self.profile == RuntimeProfile.LOCAL_MLX
                else "0"
            ),
            "LOCAL_ANALYZER_TOOL_STEPS": (
                "2" if self.profile == RuntimeProfile.LOCAL_MLX else "0"
            ),
            "LOCAL_ANALYZER_YIELD_SECONDS": (
                "0" if self.profile == RuntimeProfile.LOCAL_MLX else "60"
            ),
            "MULTIMODAL_CONTEXT": "current_grid",
            "MULTIMODAL_UPSCALE": str(self.image_scale),
            "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": str(self.python_output_tokens),
            "LOCAL_ANALYZER_ENABLE_THINKING": str(self.enable_thinking).lower(),
            "OURO3_HARNESS_MODE": self.mode.value,
            "OURO3_RETRODICT_TRACE": str(
                self.mode == HarnessMode.DUCK_RETRODICT
                and self.profile != RuntimeProfile.KAGGLE_SUBMISSION
            ).lower(),
            "ONLY_RESET_LEVELS": str(self.only_reset_levels).lower(),
            "VLLM_ENABLE_PREFIX_CACHING": str(self.enable_prefix_caching).lower(),
            "VLLM_TOOL_CALL_PARSER": self.tool_call_parser,
            "VLLM_REASONING_PARSER": self.reasoning_parser,
            "OURO3_MODEL_FAILURE_FLOOR": str(self.model_failure_floor),
            "OURO3_STRATEGIC_RESET_MIN_ACTIONS": str(self.strategic_reset_min_actions),
            "OURO3_STRATEGIC_RESET_WINDOW": str(self.strategic_reset_window),
            "OURO3_STRATEGIC_RESET_MAX_CHANGES": str(self.strategic_reset_max_changes),
            "OURO3_ROBUST_WARMUP_SECONDS": str(self.robust_warmup_s),
            "OURO3_ROBUST_MIN_ACTIONS": str(self.robust_min_actions),
            "OURO3_ROBUST_SUCCESS_THRESHOLD": str(
                self.robust_success_threshold
            ),
            "OURO3_ROBUST_REQUIRED_LOW_WINDOWS": str(
                self.robust_required_low_windows
            ),
            "OURO3_ROBUST_HYPOTHESIS_TEMPERATURE": str(
                self.robust_hypothesis_temperature
            ),
            "OURO3_ROBUST_EXECUTION_TEMPERATURE": str(
                self.robust_execution_temperature
            ),
            "OURO3_ROBUST_MAX_EXECUTION_BATCH": str(
                self.robust_max_execution_batch
            ),
            "OURO3_COMPACTION_TRIGGER_TOKENS": str(
                self.compaction_trigger_tokens
            ),
            "OURO3_COMPACTION_TARGET_TOKENS": str(
                self.compaction_target_tokens
            ),
            "OURO3_COMPACTION_RECENT_ASSISTANT_TURNS": str(
                self.compaction_recent_assistant_turns
            ),
            "OURO3_COMPACTION_MAX_OUTPUT_TOKENS": str(
                self.compaction_max_output_tokens
            ),
            "OURO3_COMPACTION_TIMEOUT_SECONDS": str(
                self.compaction_timeout_s
            ),
            "OURO3_COMPACTION_TEMPERATURE": str(
                self.compaction_temperature
            ),
            "OURO3_COMPACTION_TOP_P": str(self.compaction_top_p),
            "OURO3_COMPACTION_TOP_K": str(self.compaction_top_k),
            "OURO3_COMPACTION_MAX_CONCURRENCY": str(
                self.compaction_max_concurrency
            ),
            "OURO3_POETIQ_REPEAT_THRESHOLD": str(self.poetiq_repeat_threshold),
            "OURO3_POETIQ_NO_CHANGE_THRESHOLD": str(self.poetiq_no_change_threshold),
            "OURO3_POETIQ_INTERVENTION_COOLDOWN_ACTIONS": str(
                self.poetiq_intervention_cooldown_actions
            ),
            "OURO3_POETIQ_MAX_INTERVENTIONS_PER_LEVEL": str(
                self.poetiq_max_interventions_per_level
            ),
            "OURO3_POETIQ_DIVERSITY_SEED_OFFSET": str(
                self.poetiq_diversity_seed_offset
            ),
            "OURO3_POETIQ_YIELD_MIN_ACTIONS": str(self.poetiq_yield_min_actions),
            "OURO3_POETIQ_YIELD_MIN_ELAPSED_SECONDS": str(
                self.poetiq_yield_min_elapsed_s
            ),
            "OURO3_POETIQ_YIELD_WINDOW": str(self.poetiq_yield_window),
            "OURO3_POETIQ_YIELD_MAX_CHANGES": str(self.poetiq_yield_max_changes),
        }
        for key, value in values.items():
            os.environ[key] = value
        if self.seed is None:
            os.environ.pop("LOCAL_ANALYZER_SEED", None)
        else:
            os.environ["LOCAL_ANALYZER_SEED"] = str(self.seed)
