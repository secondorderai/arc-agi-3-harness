"""Publish, gate, submit, and record exact Kaggle v3 artifact versions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_ROOT))

from ouro3.config import HarnessConfig
from ouro3.fingerprint import prompt_sha256
from ouro3.metrics import aggregate_metric_files, aggregate_two_seed_runs, write_metrics
from duck_portfolio.router import FEATURE_NAMES, PortfolioRouter
from ouro3.promotion import (
    evaluate_poetiq_promotion,
    evaluate_portfolio_promotion,
    evaluate_retrodict_offline_promotion,
    evaluate_retrodict_promotion,
)

COMPETITION = "arc-prize-2026-arc-agi-3"
SOURCE_REF = "kinwochan/ouroboros-arc-agi-3-v3-source"
VALIDATION_REF = "kinwochan/ouroboros-arc-agi-3-v3-validation"
SUBMISSION_REF = "kinwochan/ouroboros-arc-agi-3-v3"
KAGGLE = REPO_ROOT / "kaggle-v2" / ".venv" / "bin" / "kaggle"
TOKEN_FILE = REPO_ROOT / "kaggle" / ".kaggle" / "access_token"
LEDGER = ROOT / "submission-ledger.json"
REFERENCE_BASELINE = ROOT / "baselines" / "reference.json"
REFERENCE_PROGRESS = ROOT / "results" / "duck-reference-progress.json"
EARLY_BASELINE = ROOT / "results" / "duck-reference-early-private-baseline.json"
ROBUST_PROGRESS = ROOT / "results" / "duck-robust-progress.json"
MEMORY_PROGRESS = ROOT / "results" / "duck-memory-progress.json"
MEMORY_LOCAL_PUBLIC = ROOT / "results" / "duck-memory-local-public-25.json"
MEMORY_REHEARSAL = ROOT / "results" / "duck-memory-rehearsal-110.json"
REASONING_PROGRESS = ROOT / "results" / "duck-reasoning-progress.json"
CONTRACT_PROGRESS = ROOT / "results" / "duck-contract-progress.json"
CONTRACT_REPAIR_PROGRESS = ROOT / "results" / "duck-contract-repair-progress.json"
AUDIT_PROGRESS = ROOT / "results" / "duck-audit-progress.json"
INFORMATION_PROGRESS = ROOT / "results" / "duck-information-progress.json"
HIERARCHY_PROGRESS = ROOT / "results" / "duck-hierarchy-progress.json"
DIVERSITY_PROGRESS = ROOT / "results" / "duck-diversity-progress.json"
POETIQ_PROGRESS = ROOT / "results" / "duck-poetiq-progress.json"
PORTFOLIO_PROGRESS = ROOT / "results" / "duck-portfolio-progress.json"
PORTFOLIO_LOCAL_PUBLIC = ROOT / "results" / "duck-portfolio-local-public-25.json"
PORTFOLIO_REHEARSAL = ROOT / "results" / "duck-portfolio-rehearsal-110.json"
RETRODICT_PROGRESS = ROOT / "results" / "duck-retrodict-progress.json"
RETRODICT_LOCAL_PUBLIC = ROOT / "results" / "duck-retrodict-local-public-25.json"
RETRODICT_REHEARSAL = ROOT / "results" / "duck-retrodict-rehearsal-110.json"
RETRODICT_OFFLINE_REPORT = ROOT / "results" / "duck-retrodict-offline-gate.json"
REASONING_LOCAL_PUBLIC = ROOT / "results" / "duck-reasoning-local-public-25.json"
REASONING_REHEARSAL = ROOT / "results" / "duck-reasoning-rehearsal-110.json"
SOFT_DEADLINE_S = 8 * 60 * 60 + 40 * 60
POETIQ_PUBLIC_GPU_RESERVE_HOURS = 4.5
PORTFOLIO_PUBLIC_GPU_RESERVE_HOURS = 4.5
RETRODICT_PUBLIC_GPU_RESERVE_HOURS = 4.5
REFERENCE_MODE = "duck-reference"
ROBUST_MODE = "duck-robust"
MEMORY_MODE = "duck-memory"
REASONING_MODE = "duck-reasoning"
DELIBERATE_MODE = "duck-deliberate"
CONTRACT_MODE = "duck-contract"
CONTRACT_REPAIR_MODE = "duck-contract-repair"
AUDIT_MODE = "duck-audit"
INFORMATION_MODE = "duck-information"
HIERARCHY_MODE = "duck-hierarchy"
DIVERSITY_MODE = "duck-diversity"
POETIQ_MODE = "duck-poetiq"
PORTFOLIO_MODE = "duck-portfolio"
RETRODICT_MODE = "duck-retrodict"
SUPPORTED_MODES = (
    REFERENCE_MODE,
    ROBUST_MODE,
    MEMORY_MODE,
    REASONING_MODE,
    DELIBERATE_MODE,
    CONTRACT_MODE,
    CONTRACT_REPAIR_MODE,
    AUDIT_MODE,
    INFORMATION_MODE,
    HIERARCHY_MODE,
    DIVERSITY_MODE,
    POETIQ_MODE,
    PORTFOLIO_MODE,
    RETRODICT_MODE,
)
MEMORY_PUBLIC_SCORE_GATE = 1.20
MEMORY_VISIBLE_SCORE_TARGET = 0.80
REASONING_PUBLIC_SCORE_GATE = 1.20
REASONING_VISIBLE_SCORE_TARGET = 0.80


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    stdout: str


def kaggle_env() -> dict[str, str]:
    env = os.environ.copy()
    if not env.get("KAGGLE_API_TOKEN") and TOKEN_FILE.exists():
        env["KAGGLE_API_TOKEN"] = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not env.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("Kaggle API token is unavailable")
    return env


def run_kaggle(args: Sequence[str], *, check: bool = True) -> CommandResult:
    completed = subprocess.run(
        [str(KAGGLE), *args],
        env=kaggle_env(),
        text=True,
        capture_output=True,
    )
    output = "\n".join(
        value.strip() for value in (completed.stdout, completed.stderr) if value.strip()
    )
    if check and completed.returncode:
        raise RuntimeError(f"Kaggle command failed ({completed.returncode}): {' '.join(args)}\n{output}")
    return CommandResult(tuple(args), output)


def publish_dataset(dataset_dir: Path) -> str:
    owned = run_kaggle(["datasets", "list", "--mine", "--csv"])
    refs = {
        line.split(",", 1)[0].strip()
        for line in owned.stdout.splitlines()[1:]
        if line.strip()
    }
    if SOURCE_REF in refs:
        result = run_kaggle(
            [
                "datasets",
                "version",
                "-p",
                str(dataset_dir),
                "-m",
                f"kaggle-v3 {source_hash()[:12]}",
                "--dir-mode",
                "zip",
            ]
        )
    else:
        result = run_kaggle(
            ["datasets", "create", "-p", str(dataset_dir), "--dir-mode", "zip"]
        )
    return result.stdout


def push_kernel(kernel_dir: Path) -> int:
    result = run_kaggle(["kernels", "push", "-p", str(kernel_dir)])
    versions = [
        int(value)
        for value in re.findall(r"(?:version|Version)\s*[:#]?\s*(\d+)", result.stdout)
    ]
    if not versions:
        raise RuntimeError(
            "Kaggle accepted the kernel push but did not report its exact version; "
            f"response was:\n{result.stdout}"
        )
    return versions[-1]


def wait_kernel(kernel_ref: str, *, timeout_s: float = 9 * 60 * 60) -> str:
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        result = run_kaggle(["kernels", "status", kernel_ref])
        last = result.stdout
        lowered = last.lower()
        if "complete" in lowered:
            return last
        if any(value in lowered for value in ("error", "failed", "cancelled")):
            raise RuntimeError(f"kernel {kernel_ref} failed:\n{last}")
        time.sleep(60)
    raise TimeoutError(f"kernel {kernel_ref} did not complete before timeout; last status:\n{last}")


def pull_kernel_output(kernel_ref: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_kaggle(["kernels", "output", kernel_ref, "-p", str(output_dir), "--force"])
    return output_dir


def _validate_validation_artifact(
    metrics: dict[str, Any],
    *,
    expected_seed: int | None,
    expected_mode: str,
) -> None:
    if expected_mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported validation mode: {expected_mode}")
    if metrics.get("mode") != expected_mode:
        raise RuntimeError(
            f"validation artifact did not use {expected_mode} mode"
        )
    if metrics.get("seed") != expected_seed:
        raise RuntimeError(
            f"validation seed mismatch: expected {expected_seed!r}, "
            f"found {metrics.get('seed')!r}"
        )
    if int(metrics.get("game_count", 0)) != 25:
        raise RuntimeError("each validation kernel must run exactly 25 public games")
    games = list(metrics.get("games") or [])
    if len(games) != 25:
        raise RuntimeError("each validation artifact must record all 25 public games")
    failures = list(metrics.get("infrastructure_failures") or [])
    if failures:
        raise RuntimeError(
            f"{expected_mode} infrastructure failures: "
            f"{sorted(set(map(str, failures)))}"
        )
    if float(metrics.get("elapsed_seconds", SOFT_DEADLINE_S + 1)) >= SOFT_DEADLINE_S:
        raise RuntimeError("validation kernel exceeded the 8h40 soft deadline")
    fingerprint = dict(metrics.get("runtime_fingerprint") or {})
    expected = {
        "mode": expected_mode,
        "model_id": "vrfai/Qwen3.6-27B-FP8",
        "vllm": "0.19.0",
        "torch": "2.10.0",
        "flashinfer": "0.6.6",
        "gpu": "NvidiaRtxPro6000",
        "max_model_len": 65_536,
        "active_context": 32_768,
        "image_scale": 4,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "thinking": True,
        "tool_turns": "unlimited",
        "model_output_tokens": "server-default",
        "python_timeout_s": 30,
        "python_output_tokens": 1_024,
        "concurrency": 28,
        "per_game_cap_s": 7_920,
        "analyzer_timeout_s": 900.0,
        "only_reset_levels": True,
        "prefix_caching": True,
        "tool_call_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
        "seed": expected_seed,
    }
    mismatches = {
        key: {"expected": value, "actual": fingerprint.get(key)}
        for key, value in expected.items()
        if fingerprint.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"{expected_mode} runtime fingerprint mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    for key in (
        "prompt_sha256",
        "tool_agent_source_sha256",
        "solver_source_sha256",
        "source_manifest_sha256",
    ):
        value = str(fingerprint.get(key, ""))
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise RuntimeError(
                f"{expected_mode} fingerprint is missing valid {key}"
            )
    if expected_mode == ROBUST_MODE and not isinstance(
        metrics.get("recovery_diagnostics"), dict
    ):
        raise RuntimeError(
            "duck-robust validation artifact is missing recovery diagnostics"
        )
    if expected_mode == MEMORY_MODE:
        memory_fingerprint = dict(fingerprint.get("memory") or {})
        expected_memory = {
            "reasoning_history_field": "reasoning_content",
            "preserve_thinking": True,
            "reasoning_template_verified": True,
            "compaction_trigger_tokens": 24_576,
            "compaction_target_tokens": 16_384,
            "compaction_recent_assistant_turns": 8,
            "compaction_max_output_tokens": 2_048,
            "compaction_timeout_s": 300.0,
            "compaction_temperature": 0.2,
            "compaction_top_p": 0.9,
            "compaction_top_k": 20,
            "compaction_max_concurrency": 4,
        }
        memory_mismatches = {
            key: {"expected": value, "actual": memory_fingerprint.get(key)}
            for key, value in expected_memory.items()
            if memory_fingerprint.get(key) != value
        }
        if memory_mismatches:
            raise RuntimeError(
                "duck-memory runtime fingerprint mismatch: "
                + json.dumps(memory_mismatches, sort_keys=True)
            )
        if not re.fullmatch(
            r"[0-9a-f]{64}",
            str(memory_fingerprint.get("reasoning_template_sha256", "")),
        ):
            raise RuntimeError(
                "duck-memory runtime fingerprint is missing the verified "
                "reasoning template hash"
            )
        if not isinstance(metrics.get("memory_diagnostics"), dict):
            raise RuntimeError(
                "duck-memory validation artifact is missing memory diagnostics"
            )
    if expected_mode == REASONING_MODE:
        reasoning_fingerprint = dict(fingerprint.get("reasoning") or {})
        expected_reasoning = {
            "reasoning_history_field": "reasoning_content",
            "preserve_thinking": True,
            "reasoning_template_verified": True,
            "history_policy": "stock-duck",
            "semantic_compaction": False,
            "auxiliary_model_calls": 0,
        }
        reasoning_mismatches = {
            key: {"expected": value, "actual": reasoning_fingerprint.get(key)}
            for key, value in expected_reasoning.items()
            if reasoning_fingerprint.get(key) != value
        }
        if reasoning_mismatches:
            raise RuntimeError(
                "duck-reasoning runtime fingerprint mismatch: "
                + json.dumps(reasoning_mismatches, sort_keys=True)
            )
        if not re.fullmatch(
            r"[0-9a-f]{64}",
            str(reasoning_fingerprint.get("reasoning_template_sha256", "")),
        ):
            raise RuntimeError(
                "duck-reasoning runtime fingerprint is missing the verified "
                "reasoning template hash"
            )
    if expected_mode == POETIQ_MODE:
        poetiq_fingerprint = dict(fingerprint.get("poetiq") or {})
        expected_poetiq = {
            "trigger": "repeated-action-or-unchanged-gameplay-frame",
            "repeat_threshold": 4,
            "no_change_threshold": 3,
            "cooldown_actions": 12,
            "max_interventions_per_level": 2,
            "diversity_seed_offset": 17,
            "yield_min_actions": 64,
            "yield_min_elapsed_s": 1800.0,
            "yield_window": 16,
            "yield_max_changes": 0,
            "maximum_candidates": 3,
            "second_model_request": False,
            "stock_tool_surface": True,
            "stock_history_policy": True,
        }
        poetiq_mismatches = {
            key: {"expected": value, "actual": poetiq_fingerprint.get(key)}
            for key, value in expected_poetiq.items()
            if poetiq_fingerprint.get(key) != value
        }
        if poetiq_mismatches:
            raise RuntimeError(
                "duck-poetiq runtime fingerprint mismatch: "
                + json.dumps(poetiq_mismatches, sort_keys=True)
            )
        diagnostics = metrics.get("poetiq_diagnostics")
        if not isinstance(diagnostics, dict):
            raise RuntimeError("duck-poetiq validation artifact is missing diagnostics")
        if len(diagnostics) != int(metrics.get("game_count", 0)):
            raise RuntimeError(
                "duck-poetiq validation diagnostics are not present for every game"
            )
        missing_events = [
            str(game_id)
            for game_id, value in diagnostics.items()
            if not isinstance(value, dict)
            or not isinstance(value.get("intervention_events"), list)
        ]
        if missing_events:
            raise RuntimeError(
                "duck-poetiq diagnostics are missing auditable intervention events: "
                + ", ".join(sorted(missing_events))
            )
    if expected_mode == PORTFOLIO_MODE:
        router = PortfolioRouter.load()
        portfolio_fingerprint = dict(fingerprint.get("portfolio") or {})
        expected_portfolio = {
            "candidate_order": [
                "stock",
                "audit",
                "deliberate",
                "contract-repair",
            ],
            "warmup_actions": 8,
            "score_clip": 10.0,
            "ridge_alpha": 10.0,
            "uncertainty_penalty": 0.5,
            "stock_margin": 0.25,
            "switch_min_actions": 64,
            "switch_window": 16,
            "switch_max_changes": 0,
            "switch_min_remaining_s": 1800.0,
            "max_switches": 1,
            "router_artifact_sha256": router.artifact_hash,
            "router_schema_version": 1,
            "one_persistent_conversation": True,
            "parallel_model_trajectories": 0,
            "game_identifiers_allowed": False,
        }
        portfolio_mismatches = {
            key: {"expected": value, "actual": portfolio_fingerprint.get(key)}
            for key, value in expected_portfolio.items()
            if portfolio_fingerprint.get(key) != value
        }
        if portfolio_mismatches:
            raise RuntimeError(
                "duck-portfolio runtime fingerprint mismatch: "
                + json.dumps(portfolio_mismatches, sort_keys=True)
            )
        diagnostics = metrics.get("portfolio_diagnostics")
        if not isinstance(diagnostics, dict) or len(diagnostics) != 25:
            raise RuntimeError(
                "duck-portfolio validation diagnostics are not present for every game"
            )
        incomplete = []
        for game_id, value in diagnostics.items():
            if not isinstance(value, dict):
                incomplete.append(str(game_id))
                continue
            features = value.get("features")
            events = value.get("route_events")
            action_counts = value.get("policy_action_counts")
            if (
                not isinstance(features, dict)
                or not set(features).issubset(set(FEATURE_NAMES))
                or not isinstance(events, list)
                or not isinstance(action_counts, dict)
                or value.get("one_persistent_conversation") is not True
                or int(value.get("parallel_model_trajectories", -1)) != 0
                or int(value.get("switch_count", 0)) > 1
                or value.get("router_artifact_sha256") != router.artifact_hash
            ):
                incomplete.append(str(game_id))
        if incomplete:
            raise RuntimeError(
                "duck-portfolio diagnostics are incomplete or unauditable: "
                + ", ".join(sorted(incomplete))
            )
    if expected_mode == RETRODICT_MODE:
        retrodict_fingerprint = dict(fingerprint.get("retrodict") or {})
        expected_retrodict = {
            "persistent_evidence": True,
            "rule_language": "typed-host-owned-v1",
            "full_log_replay": True,
            "multi_ontology": ["color-4", "color-8", "color-4-all"],
            "automatic_batch_size": 1,
            "fallback_batch_limit": None,
            "max_rules": 256,
            "prediction_threshold": 0.9,
        }
        retrodict_mismatches = {
            key: {"expected": value, "actual": retrodict_fingerprint.get(key)}
            for key, value in expected_retrodict.items()
            if retrodict_fingerprint.get(key) != value
        }
        if retrodict_mismatches:
            raise RuntimeError(
                "duck-retrodict runtime fingerprint mismatch: "
                + json.dumps(retrodict_mismatches, sort_keys=True)
            )
        diagnostics = metrics.get("retrodict_diagnostics")
        if not isinstance(diagnostics, dict) or len(diagnostics) != 25:
            raise RuntimeError(
                "duck-retrodict validation diagnostics are not present for every game"
            )
        incomplete = [
            str(game_id)
            for game_id, value in diagnostics.items()
            if not isinstance(value, dict)
            or value.get("mode") != RETRODICT_MODE
            or not isinstance(value.get("world_model"), dict)
        ]
        if incomplete:
            raise RuntimeError(
                "duck-retrodict diagnostics are incomplete for games: "
                + ", ".join(sorted(incomplete))
            )


def enforce_memory_gate(metrics_path: Path) -> dict[str, Any]:
    """Promote one seed-0 memory run only after score and memory health pass."""

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    _validate_validation_artifact(
        metrics,
        expected_seed=0,
        expected_mode=MEMORY_MODE,
    )
    score = float(metrics.get("mean_engine_score", 0.0))
    if score < MEMORY_PUBLIC_SCORE_GATE:
        raise RuntimeError(
            f"duck-memory public engine score {score:.4f} is below "
            f"{MEMORY_PUBLIC_SCORE_GATE:.2f}"
        )
    telemetry = {
        str(key): int(value)
        for key, value in dict(metrics.get("telemetry") or {}).items()
    }
    expected_verified = int(metrics.get("game_count", 0))
    if telemetry.get("reasoning_template_verified", 0) != expected_verified:
        raise RuntimeError(
            "duck-memory reasoning tokenizer sentinel did not pass for every game"
        )
    if telemetry.get("reasoning_turns", 0) <= 0:
        raise RuntimeError("duck-memory did not retain any reasoning-bearing turns")
    if telemetry.get("reasoning_accounted_turns", 0) != telemetry.get(
        "reasoning_turns", 0
    ) or telemetry.get("reasoning_unaccounted_turns", 0):
        raise RuntimeError(
            "duck-memory did not account for every reasoning-bearing turn "
            "in retained or successfully compacted memory"
        )
    if telemetry.get("compaction_count", 0) <= 0:
        raise RuntimeError("duck-memory did not exercise semantic compaction")
    for key in (
        "compaction_failures",
        "emergency_trims",
        "context_evictions",
        "context_overflow_recoveries",
    ):
        if telemetry.get(key, 0):
            raise RuntimeError(
                f"duck-memory promotion requires {key}=0, found "
                f"{telemetry.get(key)}"
            )
    if telemetry.get("compaction_post_tokens", 0) >= telemetry.get(
        "compaction_pre_tokens", 0
    ):
        raise RuntimeError("duck-memory compaction did not reduce active context")
    return metrics


def enforce_reasoning_gate(metrics_path: Path) -> dict[str, Any]:
    """Gate the reasoning-only ablation without requiring compaction."""

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    _validate_validation_artifact(
        metrics,
        expected_seed=0,
        expected_mode=REASONING_MODE,
    )
    score = float(metrics.get("mean_engine_score", 0.0))
    if score < REASONING_PUBLIC_SCORE_GATE:
        raise RuntimeError(
            f"duck-reasoning public engine score {score:.4f} is below "
            f"{REASONING_PUBLIC_SCORE_GATE:.2f}"
        )
    telemetry = {
        str(key): int(value)
        for key, value in dict(metrics.get("telemetry") or {}).items()
    }
    expected_verified = int(metrics.get("game_count", 0))
    if telemetry.get("reasoning_template_verified", 0) != expected_verified:
        raise RuntimeError(
            "duck-reasoning tokenizer sentinel did not pass for every game"
        )
    if telemetry.get("reasoning_turns", 0) <= 0:
        raise RuntimeError("duck-reasoning did not receive reasoning-bearing turns")
    if any(
        telemetry.get(key, 0)
        for key in (
            "compaction_count",
            "compaction_retries",
            "compaction_failures",
            "emergency_trims",
        )
    ):
        raise RuntimeError(
            "duck-reasoning unexpectedly exercised semantic compaction"
        )
    return metrics


def _validate_reference_artifact(
    metrics: dict[str, Any],
    *,
    expected_seed: int | None,
) -> None:
    _validate_validation_artifact(
        metrics,
        expected_seed=expected_seed,
        expected_mode=REFERENCE_MODE,
    )


def enforce_fidelity_gate(metrics_path: Path) -> dict[str, Any]:
    """Gate the unseeded fidelity run on shape and health, never on score."""

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    _validate_reference_artifact(metrics, expected_seed=None)
    return metrics


def enforce_reference_gate(metrics_path: Path) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    seeds = list(metrics.get("seed_runs") or [])
    if len(seeds) != 5:
        raise RuntimeError("reference gate requires five complete seed runs")
    for seed, run in enumerate(seeds):
        _validate_reference_artifact(run, expected_seed=seed)
    mean = float(metrics.get("mean_engine_score", 0.0))
    if mean < 1.20:
        raise RuntimeError(
            f"reference public engine-score mean {mean:.4f} is below 1.20"
        )
    if metrics.get("infrastructure_failures"):
        raise RuntimeError("reference aggregate contains infrastructure failures")
    if not metrics.get("runtime_fingerprint_consistent"):
        raise RuntimeError("reference seed runtime fingerprints are inconsistent")
    if not metrics.get("prompt_fingerprint_consistent"):
        raise RuntimeError("reference seed prompt fingerprints are inconsistent")
    if float(metrics.get("max_kernel_elapsed_seconds", SOFT_DEADLINE_S + 1)) >= SOFT_DEADLINE_S:
        raise RuntimeError("a reference seed kernel exceeded the 8h40 soft deadline")
    return metrics


def freeze_reference(
    metrics_path: Path,
    metrics: dict[str, Any],
    *,
    kernel_versions: dict[int, int],
) -> dict[str, Any]:
    """Persist the immutable comparison data without committing huge traces."""

    seed_runs = []
    for run in metrics["seed_runs"]:
        seed_runs.append(
            {
                "seed": run["seed"],
                "mean_engine_score": run["mean_engine_score"],
                "mean_completed_levels": run["mean_completed_levels"],
                "median_completed_levels": run["median_completed_levels"],
                "total_completed_levels": run["total_completed_levels"],
                "infrastructure_failures": run["infrastructure_failures"],
                "games": [
                    {
                        "game_id": game["game_id"],
                        "state": game["state"],
                        "levels_completed": game["levels_completed"],
                        "actions": game["actions"],
                        "final_score": game["final_score"],
                    }
                    for game in run["games"]
                ],
            }
        )
    frozen = {
        "schema_version": 2,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "experiment": metrics["experiment"],
        "kernel": VALIDATION_REF,
        "kernel_versions": {
            str(seed): int(version) for seed, version in kernel_versions.items()
        },
        "config_hashes": metrics["config_hashes"],
        "prompt_sha256": metrics["prompt_sha256"],
        "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        "max_kernel_elapsed_seconds": metrics["max_kernel_elapsed_seconds"],
        "mean_engine_score": metrics["mean_engine_score"],
        "mean_completed_levels": metrics["mean_completed_levels"],
        "infrastructure_failures": metrics["infrastructure_failures"],
        "seed_runs": seed_runs,
    }
    REFERENCE_BASELINE.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_BASELINE.write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return frozen


def refresh_leaderboard_best() -> float:
    result = run_kaggle(
        ["competitions", "leaderboard", "-c", COMPETITION, "--show", "-v"]
    )
    scores: list[float] = []
    for line in result.stdout.splitlines():
        columns = [value.strip() for value in line.split(",")]
        if len(columns) < 3 or columns[0].lower() in {"teamid", "rank"}:
            continue
        for value in columns[2:4]:
            try:
                score = float(value)
            except ValueError:
                continue
            if 0 <= score <= 100:
                scores.append(score)
                break
    if not scores:
        raise RuntimeError("could not parse current leaderboard scores")
    return max(scores)


def parse_submission_rows(output: str) -> list[dict[str, str]]:
    """Parse Kaggle's verbose CSV without being confused by quoted messages."""

    reader = csv.DictReader(io.StringIO(output))
    return [
        {str(key): str(value or "") for key, value in row.items() if key is not None}
        for row in reader
        if row.get("ref")
    ]


def ensure_daily_quota() -> set[str]:
    result = run_kaggle(["competitions", "submissions", "-c", COMPETITION, "-v"])
    rows = parse_submission_rows(result.stdout)
    today = datetime.now(timezone.utc).date().isoformat()
    if any(row.get("date", "").startswith(today) for row in rows):
        raise RuntimeError(f"a competition submission already exists for {today}")
    return {row["ref"] for row in rows}


def submit_exact_kernel(
    *,
    kernel_version: int,
    message: str,
) -> str:
    result = run_kaggle(
        [
            "competitions",
            "submit",
            "-c",
            COMPETITION,
            "-k",
            SUBMISSION_REF,
            "-v",
            str(kernel_version),
            "-f",
            "submission.parquet",
            "-m",
            message,
        ]
    )
    return result.stdout


def wait_submission(
    *,
    previous_refs: set[str] | None = None,
    timeout_s: float = 10 * 60 * 60,
) -> dict[str, str]:
    """Wait for the newly created submission, never an older completed row."""

    deadline = time.monotonic() + timeout_s
    last = ""
    excluded = set(previous_refs or ())
    while time.monotonic() < deadline:
        result = run_kaggle(["competitions", "submissions", "-c", COMPETITION, "-v"])
        last = result.stdout
        rows = [
            row
            for row in parse_submission_rows(last)
            if row.get("ref") not in excluded
        ]
        if rows:
            newest = max(rows, key=lambda row: row.get("date", ""))
            lowered = newest.get("status", "").lower()
            if "complete" in lowered:
                return newest
            if any(value in lowered for value in ("error", "failed", "cancelled")):
                raise RuntimeError(
                    f"submission failed: {json.dumps(newest, sort_keys=True)}"
                )
        time.sleep(60)
    raise TimeoutError(f"submission did not complete; last response:\n{last}")


def source_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def config_hash(mode: str = REFERENCE_MODE) -> str:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported configuration mode: {mode}")
    name = {
        REFERENCE_MODE: "reference.json",
        ROBUST_MODE: "robust.json",
        MEMORY_MODE: "duck-memory-v1.json",
        REASONING_MODE: "duck-reasoning-v1.json",
        DELIBERATE_MODE: "duck-deliberate-v1.json",
        CONTRACT_MODE: "duck-contract-v1.json",
        CONTRACT_REPAIR_MODE: "duck-contract-repair-v1.json",
        AUDIT_MODE: "duck-audit-v1.json",
        INFORMATION_MODE: "duck-information-v1.json",
        HIERARCHY_MODE: "duck-hierarchy-v1.json",
        DIVERSITY_MODE: "duck-diversity-v1.json",
        POETIQ_MODE: "duck-poetiq-v1.json",
        PORTFOLIO_MODE: "duck-portfolio-v1.json",
        RETRODICT_MODE: "duck-retrodict-v1.json",
    }[mode]
    return HarnessConfig.from_json(ROOT / "configs" / name).config_hash


def source_manifest_hash() -> str:
    path = ROOT / "dist" / "source-dataset" / "manifest.sha256.json"
    if not path.is_file():
        raise RuntimeError("source dataset manifest has not been built")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portfolio_packaged_identity() -> dict[str, str]:
    """Return the three mutable identities checked before submission."""

    config = HarnessConfig.from_json(
        ROOT / "configs" / "duck-portfolio-v1.json"
    )
    router = PortfolioRouter.load()
    previous_mode = os.environ.get("OURO3_HARNESS_MODE")
    os.environ["OURO3_HARNESS_MODE"] = PORTFOLIO_MODE
    try:
        packaged_prompt_hash = prompt_sha256(
            tool_output_tokens=config.python_output_tokens
        )
    finally:
        if previous_mode is None:
            os.environ.pop("OURO3_HARNESS_MODE", None)
        else:
            os.environ["OURO3_HARNESS_MODE"] = previous_mode
    return {
        "config_hash": config.config_hash,
        "prompt_sha256": packaged_prompt_hash,
        "router_artifact_sha256": router.artifact_hash,
    }


def retrodict_packaged_identity() -> dict[str, str]:
    """Return the mutable retrodict config and prompt identities."""

    config = HarnessConfig.retrodict(seed=0)
    previous_mode = os.environ.get("OURO3_HARNESS_MODE")
    os.environ["OURO3_HARNESS_MODE"] = RETRODICT_MODE
    try:
        packaged_prompt_hash = prompt_sha256(
            tool_output_tokens=config.python_output_tokens
        )
    finally:
        if previous_mode is None:
            os.environ.pop("OURO3_HARNESS_MODE", None)
        else:
            os.environ["OURO3_HARNESS_MODE"] = previous_mode
    return {
        "config_hash": config.config_hash,
        "prompt_sha256": packaged_prompt_hash,
    }


def append_ledger(record: dict[str, Any]) -> None:
    payload = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {"submissions": []}
    payload.setdefault("submissions", []).append(record)
    LEDGER.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_reference_progress(record: dict[str, Any]) -> None:
    """Atomically checkpoint long-running reference validation progress."""

    REFERENCE_PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    temporary = REFERENCE_PROGRESS.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(REFERENCE_PROGRESS)


def write_robust_progress(record: dict[str, Any]) -> None:
    """Atomically checkpoint the single-run robust candidate evaluation."""

    ROBUST_PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    temporary = ROBUST_PROGRESS.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(ROBUST_PROGRESS)


def write_memory_progress(record: dict[str, Any]) -> None:
    """Atomically checkpoint the retained-reasoning candidate."""

    MEMORY_PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    temporary = MEMORY_PROGRESS.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(MEMORY_PROGRESS)


def write_reasoning_progress(record: dict[str, Any]) -> None:
    """Atomically checkpoint the reasoning-only candidate."""

    REASONING_PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    temporary = REASONING_PROGRESS.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(REASONING_PROGRESS)


def write_contract_progress(record: dict[str, Any], *, repair: bool) -> None:
    """Atomically checkpoint a public-only verified-action ablation."""

    destination = CONTRACT_REPAIR_PROGRESS if repair else CONTRACT_PROGRESS
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def write_audit_progress(record: dict[str, Any]) -> None:
    """Atomically checkpoint the sparse self-audit candidate."""

    AUDIT_PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    temporary = AUDIT_PROGRESS.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(AUDIT_PROGRESS)


def write_poetiq_progress(record: dict[str, Any]) -> None:
    """Atomically checkpoint the two-seed composite Poetiq experiment."""

    POETIQ_PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    temporary = POETIQ_PROGRESS.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(POETIQ_PROGRESS)


def write_portfolio_progress(record: dict[str, Any]) -> None:
    """Atomically checkpoint the deterministic portfolio experiment."""

    PORTFOLIO_PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    temporary = PORTFOLIO_PROGRESS.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(PORTFOLIO_PROGRESS)


def write_retrodict_progress(record: dict[str, Any]) -> None:
    """Atomically checkpoint the two-seed retrodictive experiment."""

    RETRODICT_PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    temporary = RETRODICT_PROGRESS.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(RETRODICT_PROGRESS)


def query_kaggle_gpu_hours_remaining() -> float:
    """Read the authenticated Kaggle GPU quota in hours.

    The CLI returns values such as ``"13.25h"``. Keep this parser small and
    strict so a malformed account response cannot accidentally authorize a
    public or hidden run.
    """

    result = run_kaggle(["quota", "--format", "json"])
    payload = json.loads(result.stdout)
    if not isinstance(payload, list):
        raise ValueError("Kaggle quota response is not a list")
    gpu = next(
        (
            row
            for row in payload
            if isinstance(row, dict) and str(row.get("resource", "")).upper() == "GPU"
        ),
        None,
    )
    if gpu is None:
        raise ValueError("Kaggle quota response did not contain a GPU row")
    raw_remaining = str(gpu.get("remaining", "")).strip().lower()
    if raw_remaining.endswith("h"):
        raw_remaining = raw_remaining[:-1].strip()
    remaining = float(raw_remaining)
    if remaining < 0:
        raise ValueError("Kaggle GPU quota cannot be negative")
    return remaining


def ensure_gpu_hours_remaining(
    remaining_hours: float | None,
    *,
    required_hours: float,
) -> float:
    """Fail closed when the caller cannot prove enough weekly GPU remains."""

    value = remaining_hours
    if value is None:
        raw = os.environ.get("OURO3_KAGGLE_GPU_HOURS_REMAINING", "").strip()
        value = float(raw) if raw else None
    if value is None:
        try:
            value = query_kaggle_gpu_hours_remaining()
        except Exception as exc:
            raise RuntimeError(
                "remaining Kaggle GPU hours are unknown; pass "
                "--gpu-hours-remaining after checking the account quota"
            ) from exc
    if float(value) < float(required_hours):
        raise RuntimeError(
            f"the candidate requires at least {required_hours:.1f} remaining "
            f"Kaggle GPU hours, found {float(value):.1f}"
        )
    return float(value)


def validate_memory_local_prerequisites(
    *,
    public_path: Path | None = None,
    rehearsal_path: Path | None = None,
) -> dict[str, Any]:
    """Require clean local 25-game and competition-shaped rehearsals."""

    public_path = public_path or MEMORY_LOCAL_PUBLIC
    rehearsal_path = rehearsal_path or MEMORY_REHEARSAL
    if not public_path.is_file() or not rehearsal_path.is_file():
        raise RuntimeError(
            "duck-memory local prerequisites are missing; run the 25-game "
            "integration and 110-game gateway rehearsal first"
        )
    public = json.loads(public_path.read_text(encoding="utf-8"))
    rehearsal = json.loads(rehearsal_path.read_text(encoding="utf-8"))
    if public.get("mode") != MEMORY_MODE or int(public.get("game_count", 0)) != 25:
        raise RuntimeError("duck-memory local public artifact has the wrong shape")
    if list(public.get("infrastructure_failures") or []):
        raise RuntimeError("duck-memory local public integration contains failures")
    if int(rehearsal.get("game_count", 0)) != 110:
        raise RuntimeError("duck-memory rehearsal must contain 110 games")
    if int(rehearsal.get("unique_game_ids", 0)) != 110:
        raise RuntimeError("duck-memory rehearsal did not preserve 110 unique IDs")
    if rehearsal.get("gateway_transport") != "competition-http":
        raise RuntimeError("duck-memory rehearsal did not use competition HTTP")
    if list(rehearsal.get("infrastructure_failures") or []):
        raise RuntimeError("duck-memory rehearsal contains infrastructure failures")
    return {
        "local_public": str(public_path),
        "local_public_score": float(public.get("mean_engine_score", 0.0)),
        "rehearsal": str(rehearsal_path),
        "rehearsal_unique_game_ids": 110,
    }


def validate_reasoning_local_prerequisites(
    *,
    public_path: Path | None = None,
    rehearsal_path: Path | None = None,
) -> dict[str, Any]:
    """Require the reasoning-only local public and gateway rehearsals."""

    public_path = public_path or REASONING_LOCAL_PUBLIC
    rehearsal_path = rehearsal_path or REASONING_REHEARSAL
    if not public_path.is_file() or not rehearsal_path.is_file():
        raise RuntimeError(
            "duck-reasoning local prerequisites are missing; run the 25-game "
            "integration and 110-game gateway rehearsal first"
        )
    public = json.loads(public_path.read_text(encoding="utf-8"))
    rehearsal = json.loads(rehearsal_path.read_text(encoding="utf-8"))
    if public.get("mode") != REASONING_MODE or int(public.get("game_count", 0)) != 25:
        raise RuntimeError("duck-reasoning local public artifact has the wrong shape")
    if list(public.get("infrastructure_failures") or []):
        raise RuntimeError("duck-reasoning local public integration contains failures")
    if int(rehearsal.get("game_count", 0)) != 110:
        raise RuntimeError("duck-reasoning rehearsal must contain 110 games")
    if int(rehearsal.get("unique_game_ids", 0)) != 110:
        raise RuntimeError("duck-reasoning rehearsal did not preserve 110 unique IDs")
    if rehearsal.get("gateway_transport") != "competition-http":
        raise RuntimeError("duck-reasoning rehearsal did not use competition HTTP")
    if list(rehearsal.get("infrastructure_failures") or []):
        raise RuntimeError("duck-reasoning rehearsal contains infrastructure failures")
    return {
        "local_public": str(public_path),
        "local_public_score": float(public.get("mean_engine_score", 0.0)),
        "rehearsal": str(rehearsal_path),
        "rehearsal_unique_game_ids": 110,
    }


def validate_poetiq_local_prerequisites(
    *,
    public_path: Path | None = None,
    rehearsal_path: Path | None = None,
) -> dict[str, Any]:
    """Require clean local integration and gateway rehearsal artifacts."""

    public_path = public_path or ROOT / "results" / "duck-poetiq-local-public-25.json"
    rehearsal_path = rehearsal_path or ROOT / "results" / "duck-poetiq-rehearsal-110.json"
    if not public_path.is_file() or not rehearsal_path.is_file():
        raise RuntimeError(
            "duck-poetiq local prerequisites are missing; run the 25-game "
            "integration and 110-game gateway rehearsal first"
        )
    public = json.loads(public_path.read_text(encoding="utf-8"))
    rehearsal = json.loads(rehearsal_path.read_text(encoding="utf-8"))
    if public.get("mode") != POETIQ_MODE or int(public.get("game_count", 0)) != 25:
        raise RuntimeError("duck-poetiq local public artifact has the wrong shape")
    if list(public.get("infrastructure_failures") or []):
        raise RuntimeError("duck-poetiq local integration contains failures")
    if int(rehearsal.get("game_count", 0)) != 110:
        raise RuntimeError("duck-poetiq rehearsal must contain 110 games")
    if int(rehearsal.get("unique_game_ids", 0)) != 110:
        raise RuntimeError("duck-poetiq rehearsal did not preserve 110 unique IDs")
    if rehearsal.get("gateway_transport") != "competition-http":
        raise RuntimeError("duck-poetiq rehearsal did not use competition HTTP")
    if list(rehearsal.get("infrastructure_failures") or []):
        raise RuntimeError("duck-poetiq rehearsal contains infrastructure failures")
    return {
        "local_public": str(public_path),
        "local_public_score": float(public.get("mean_engine_score", 0.0)),
        "rehearsal": str(rehearsal_path),
        "rehearsal_unique_game_ids": 110,
        "rehearsal_elapsed_seconds": float(rehearsal.get("elapsed_seconds", 0.0)),
    }


def validate_portfolio_local_prerequisites(
    *,
    public_path: Path | None = None,
    rehearsal_path: Path | None = None,
) -> dict[str, Any]:
    """Require a clean local portfolio integration and 110-game rehearsal."""

    public_path = public_path or PORTFOLIO_LOCAL_PUBLIC
    rehearsal_path = rehearsal_path or PORTFOLIO_REHEARSAL
    if not public_path.is_file() or not rehearsal_path.is_file():
        raise RuntimeError(
            "duck-portfolio local prerequisites are missing; run the 25-game "
            "MLX integration and 110-game gateway rehearsal first"
        )
    public = json.loads(public_path.read_text(encoding="utf-8"))
    rehearsal = json.loads(rehearsal_path.read_text(encoding="utf-8"))
    if public.get("mode") != PORTFOLIO_MODE or int(public.get("game_count", 0)) != 25:
        raise RuntimeError("duck-portfolio local public artifact has the wrong shape")
    if list(public.get("infrastructure_failures") or []):
        raise RuntimeError("duck-portfolio local integration contains failures")
    if int(rehearsal.get("game_count", 0)) != 110:
        raise RuntimeError("duck-portfolio rehearsal must contain 110 games")
    if int(rehearsal.get("unique_game_ids", 0)) != 110:
        raise RuntimeError("duck-portfolio rehearsal did not preserve 110 unique IDs")
    if rehearsal.get("gateway_transport") != "competition-http":
        raise RuntimeError("duck-portfolio rehearsal did not use competition HTTP")
    if list(rehearsal.get("infrastructure_failures") or []):
        raise RuntimeError("duck-portfolio rehearsal contains infrastructure failures")
    router = PortfolioRouter.load()
    cross_validation = router.cross_validation
    if cross_validation.get("passed") is not True:
        raise RuntimeError("duck-portfolio offline leave-one-game-out gate failed")
    if float(cross_validation.get("mean_lift", 0.0)) < 0.10:
        raise RuntimeError("duck-portfolio offline mean lift is below 0.10")
    if int(cross_validation.get("routed_nonzero_games", 0)) < int(
        cross_validation.get("stock_nonzero_games", 0)
    ):
        raise RuntimeError("duck-portfolio offline routing reduced breadth")
    if int(cross_validation.get("distinct_non_stock_policies", 0)) < 2:
        raise RuntimeError("duck-portfolio did not select two non-Stock policies")
    return {
        "local_public": str(public_path),
        "local_public_score": float(public.get("mean_engine_score", 0.0)),
        "rehearsal": str(rehearsal_path),
        "rehearsal_unique_game_ids": 110,
        "rehearsal_elapsed_seconds": float(rehearsal.get("elapsed_seconds", 0.0)),
        "router_artifact_sha256": router.artifact_hash,
        "cross_validation": cross_validation,
    }


def validate_retrodict_local_prerequisites(
    *,
    public_path: Path | None = None,
    rehearsal_path: Path | None = None,
    offline_report_path: Path | None = None,
    require_offline_pass: bool = True,
) -> dict[str, Any]:
    """Require local gameplay, chronological replay, and 110-game rehearsal."""

    public_path = public_path or RETRODICT_LOCAL_PUBLIC
    rehearsal_path = rehearsal_path or RETRODICT_REHEARSAL
    offline_report_path = offline_report_path or RETRODICT_OFFLINE_REPORT
    missing = [
        str(path)
        for path in (public_path, rehearsal_path, offline_report_path)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "duck-retrodict local prerequisites are missing: "
            + ", ".join(missing)
        )
    public = json.loads(public_path.read_text(encoding="utf-8"))
    rehearsal = json.loads(rehearsal_path.read_text(encoding="utf-8"))
    offline_report = json.loads(offline_report_path.read_text(encoding="utf-8"))
    if public.get("mode") != RETRODICT_MODE or int(public.get("game_count", 0)) != 25:
        raise RuntimeError("duck-retrodict local public artifact has the wrong shape")
    if list(public.get("infrastructure_failures") or []):
        raise RuntimeError("duck-retrodict local public integration contains failures")
    request_failures = int(
        dict(public.get("telemetry") or {}).get("request_failures", 0)
    )
    if request_failures:
        raise RuntimeError(
            "duck-retrodict local public integration contains "
            f"{request_failures} actor request failures"
        )
    diagnostics = public.get("retrodict_diagnostics")
    if not isinstance(diagnostics, dict) or len(diagnostics) != 25:
        raise RuntimeError(
            "duck-retrodict local public diagnostics are not present for every game"
        )
    offline_decision = evaluate_retrodict_offline_promotion(offline_report)
    if require_offline_pass and not offline_decision.passed:
        raise RuntimeError(
            "duck-retrodict offline gate failed: "
            + "; ".join(offline_decision.reasons)
        )
    if int(rehearsal.get("game_count", 0)) != 110:
        raise RuntimeError("duck-retrodict rehearsal must contain 110 games")
    if int(rehearsal.get("unique_game_ids", 0)) != 110:
        raise RuntimeError("duck-retrodict rehearsal did not preserve 110 unique IDs")
    if rehearsal.get("gateway_transport") != "competition-http":
        raise RuntimeError("duck-retrodict rehearsal did not use competition HTTP")
    if list(rehearsal.get("infrastructure_failures") or []):
        raise RuntimeError("duck-retrodict rehearsal contains infrastructure failures")
    rehearsal_elapsed = float(rehearsal.get("elapsed_seconds", SOFT_DEADLINE_S + 1))
    if rehearsal_elapsed >= SOFT_DEADLINE_S:
        raise RuntimeError("duck-retrodict rehearsal exceeded the 8h40 soft deadline")
    return {
        "local_public": str(public_path),
        "local_public_score": float(public.get("mean_engine_score", 0.0)),
        "offline_report": str(offline_report_path),
        "offline": offline_report,
        "rehearsal": str(rehearsal_path),
        "rehearsal_unique_game_ids": 110,
        "rehearsal_elapsed_seconds": rehearsal_elapsed,
    }


def _notebooks_root(mode: str) -> Path:
    if mode == REFERENCE_MODE:
        return ROOT / "notebooks"
    if mode == ROBUST_MODE:
        return ROOT / "notebooks" / "robust"
    if mode == MEMORY_MODE:
        return ROOT / "notebooks" / "memory"
    if mode == REASONING_MODE:
        return ROOT / "notebooks" / "reasoning"
    if mode == DELIBERATE_MODE:
        return ROOT / "notebooks" / "deliberate"
    if mode == CONTRACT_MODE:
        return ROOT / "notebooks" / "contract"
    if mode == CONTRACT_REPAIR_MODE:
        return ROOT / "notebooks" / "contract-repair"
    if mode == AUDIT_MODE:
        return ROOT / "notebooks" / "audit"
    if mode == INFORMATION_MODE:
        return ROOT / "notebooks" / "information"
    if mode == HIERARCHY_MODE:
        return ROOT / "notebooks" / "hierarchy"
    if mode == DIVERSITY_MODE:
        return ROOT / "notebooks" / "diversity"
    if mode == POETIQ_MODE:
        return ROOT / "notebooks" / "poetiq"
    if mode == PORTFOLIO_MODE:
        return ROOT / "notebooks" / "portfolio"
    if mode == RETRODICT_MODE:
        return ROOT / "notebooks" / "retrodict"
    raise ValueError(f"unsupported notebook mode: {mode}")


def _assert_generated_validation(
    notebook_path: Path,
    *,
    expected_mode: str,
    expected_seed: int | None,
) -> None:
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata", {}).get("ouro3", {}))
    if metadata.get("mode") != expected_mode:
        raise RuntimeError(
            f"refusing to push {notebook_path}: expected mode "
            f"{expected_mode}, found {metadata.get('mode')!r}"
        )
    if metadata.get("validation_seed") != expected_seed:
        raise RuntimeError(
            f"refusing to push {notebook_path}: expected seed "
            f"{expected_seed!r}, found {metadata.get('validation_seed')!r}"
        )


def generate_artifacts(
    *,
    validation_seed: int | None,
    mode: str = REFERENCE_MODE,
) -> Path:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported artifact mode: {mode}")
    seed_value = "unseeded" if validation_seed is None else str(validation_seed)
    notebooks_root = _notebooks_root(mode)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "package_source.py"),
            "--output",
            str(ROOT / "dist" / "source-dataset"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_notebooks.py"),
            "--output",
            str(notebooks_root),
            "--validation-seed",
            seed_value,
            "--mode",
            mode,
        ],
        check=True,
    )
    _assert_generated_validation(
        notebooks_root / "validation" / "validation.ipynb",
        expected_mode=mode,
        expected_seed=validation_seed,
    )
    return notebooks_root


def _validation_result_dir(
    seed: int | None,
    version: int,
    *,
    mode: str = REFERENCE_MODE,
) -> Path:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported result mode: {mode}")
    label = "unseeded" if seed is None else f"seed-{seed}"
    return ROOT / "results" / f"{mode}-{label}-v{version}"


def run_validation_kernel(
    *,
    seed: int | None,
    resume_version: int | None = None,
    mode: str = REFERENCE_MODE,
) -> tuple[int, Path, dict[str, Any]]:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported validation mode: {mode}")
    if resume_version is None:
        notebooks_root = generate_artifacts(
            validation_seed=seed,
            mode=mode,
        )
        validation_dir = notebooks_root / "validation"
        _assert_generated_validation(
            validation_dir / "validation.ipynb",
            expected_mode=mode,
            expected_seed=seed,
        )
        version = push_kernel(validation_dir)
        wait_kernel(VALIDATION_REF)
        output_dir = pull_kernel_output(
            VALIDATION_REF,
            _validation_result_dir(seed, version, mode=mode),
        )
    else:
        version = int(resume_version)
        if version < 1:
            raise ValueError("validation versions must be positive")
        output_dir = _validation_result_dir(seed, version, mode=mode)
        if not (output_dir / "validation_metrics.json").is_file():
            raise RuntimeError(
                "resuming an older kernel requires its exact cached metrics at "
                f"{output_dir}"
            )
    metrics_path = output_dir / "validation_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if mode == REFERENCE_MODE and seed is None:
        metrics = enforce_fidelity_gate(metrics_path)
    else:
        _validate_validation_artifact(
            metrics,
            expected_seed=seed,
            expected_mode=mode,
        )
    return version, metrics_path, metrics


def attach_running_validation_kernel(
    *,
    seed: int,
    version: int,
) -> tuple[int, Path, dict[str, Any]]:
    """Wait for and pull an already-pushed validation kernel version."""

    output_dir = _validation_result_dir(seed, version)
    metrics_path = output_dir / "validation_metrics.json"
    if not metrics_path.is_file():
        wait_kernel(VALIDATION_REF)
        pull_kernel_output(VALIDATION_REF, output_dir)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    _validate_reference_artifact(metrics, expected_seed=seed)
    return version, metrics_path, metrics


def _latest_reference_seed_score(seed: int) -> float | None:
    candidates: list[tuple[int, Path]] = []
    pattern = re.compile(rf"duck-reference-seed-{seed}-v(\d+)")
    for path in (ROOT / "results").glob(
        f"duck-reference-seed-{seed}-v*/validation_metrics.json"
    ):
        match = pattern.fullmatch(path.parent.name)
        if match:
            candidates.append((int(match.group(1)), path))
    for _version, path in sorted(candidates, reverse=True):
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
            _validate_reference_artifact(metrics, expected_seed=seed)
            return float(metrics["mean_engine_score"])
        except (KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
            continue
    return None


def _robust_recovery_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    telemetry = dict(metrics.get("telemetry") or {})
    diagnostics = dict(metrics.get("recovery_diagnostics") or {})
    recovery_games = sorted(
        game_id
        for game_id, value in diagnostics.items()
        if isinstance(value, dict) and list(value.get("recovered_levels") or [])
    )
    return {
        "recovery_count": int(telemetry.get("recovery_count", 0)),
        "recovery_successes": int(telemetry.get("recovery_successes", 0)),
        "recovery_resets": int(telemetry.get("recovery_resets", 0)),
        "prediction_matches": int(telemetry.get("prediction_matches", 0)),
        "prediction_mismatches": int(
            telemetry.get("prediction_mismatches", 0)
        ),
        "recovery_game_count": len(recovery_games),
        "recovery_games": recovery_games,
    }


def execute_robust_seed0_validation(
    *,
    resume_version: int | None = None,
) -> dict[str, Any]:
    """Publish and evaluate exactly one seed-0 robust candidate; never submit."""

    if resume_version is None:
        generate_artifacts(validation_seed=0, mode=ROBUST_MODE)
        source_output = publish_dataset(ROOT / "dist" / "source-dataset")
    else:
        source_output = f"reused {SOURCE_REF}"
    version, metrics_path, metrics = run_validation_kernel(
        seed=0,
        resume_version=resume_version,
        mode=ROBUST_MODE,
    )
    expected_config_hash = config_hash(ROBUST_MODE)
    if metrics.get("config_hash") != expected_config_hash:
        raise RuntimeError(
            "duck-robust config hash mismatch: "
            f"expected {expected_config_hash}, found {metrics.get('config_hash')!r}"
        )
    reference_score = _latest_reference_seed_score(0)
    candidate_score = float(metrics["mean_engine_score"])
    recovery = _robust_recovery_summary(metrics)
    record = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "duck-robust-seed-0-complete",
        "mode": ROBUST_MODE,
        "experiment": "duck-robust-seed-0",
        "git_sha": source_hash(),
        "config_hash": expected_config_hash,
        "source_manifest_sha256": source_manifest_hash(),
        "source_dataset": SOURCE_REF,
        "source_publish": source_output,
        "validation_kernel": VALIDATION_REF,
        "validation_version": version,
        "seed": 0,
        "metrics_path": str(metrics_path),
        "mean_engine_score": candidate_score,
        "mean_completed_levels": float(metrics["mean_completed_levels"]),
        "total_completed_levels": int(metrics["total_completed_levels"]),
        "infrastructure_failures": list(
            metrics.get("infrastructure_failures") or []
        ),
        "runtime_fingerprint": metrics["runtime_fingerprint"],
        "reference_seed_0_score": reference_score,
        "score_delta_vs_reference_seed_0": (
            candidate_score - reference_score
            if reference_score is not None
            else None
        ),
        "recovery": recovery,
        "decision": "review-required-no-automatic-submission",
    }
    write_robust_progress(record)
    return record


def execute_memory_seed0_candidate(
    *,
    submit: bool,
    resume_version: int | None = None,
    gpu_hours_remaining: float | None = None,
) -> dict[str, Any]:
    """Run one retained-reasoning candidate and submit only after its gate."""

    local_evidence = validate_memory_local_prerequisites()
    required_gpu_hours = 12.0 if resume_version is None else (9.0 if submit else 0.0)
    proven_gpu_hours = (
        ensure_gpu_hours_remaining(
            gpu_hours_remaining,
            required_hours=required_gpu_hours,
        )
        if required_gpu_hours
        else gpu_hours_remaining
    )
    if resume_version is None:
        generate_artifacts(validation_seed=0, mode=MEMORY_MODE)
        source_output = publish_dataset(ROOT / "dist" / "source-dataset")
    else:
        source_output = f"reused {SOURCE_REF}"

    version, metrics_path, _metrics = run_validation_kernel(
        seed=0,
        resume_version=resume_version,
        mode=MEMORY_MODE,
    )
    metrics = enforce_memory_gate(metrics_path)
    expected_config_hash = config_hash(MEMORY_MODE)
    if metrics.get("config_hash") != expected_config_hash:
        raise RuntimeError(
            "duck-memory config hash mismatch: "
            f"expected {expected_config_hash}, "
            f"found {metrics.get('config_hash')!r}"
        )

    telemetry = {
        str(key): int(value)
        for key, value in dict(metrics.get("telemetry") or {}).items()
    }
    record: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": MEMORY_MODE,
        "experiment": "duck-memory-v1",
        "git_sha": source_hash(),
        "config_hash": expected_config_hash,
        "source_manifest_sha256": str(
            metrics["runtime_fingerprint"]["source_manifest_sha256"]
        ),
        "source_dataset": SOURCE_REF,
        "source_publish": source_output,
        "validation_kernel": VALIDATION_REF,
        "validation_version": version,
        "validation_metrics_path": str(metrics_path),
        "validation_scores": {
            "mean_engine_score": float(metrics["mean_engine_score"]),
            "mean_completed_levels": float(metrics["mean_completed_levels"]),
            "total_completed_levels": int(metrics["total_completed_levels"]),
        },
        "memory_health": {
            key: telemetry.get(key, 0)
            for key in (
                "reasoning_template_verified",
                "reasoning_turns",
                "reasoning_chars",
                "reasoning_retained_turns",
                "reasoning_compacted_turns",
                "reasoning_accounted_turns",
                "reasoning_unaccounted_turns",
                "compaction_count",
                "compaction_retries",
                "compaction_failures",
                "emergency_trims",
                "context_evictions",
                "context_overflow_recoveries",
                "compaction_pre_tokens",
                "compaction_post_tokens",
                "compaction_compression_ratio_bps",
                "compaction_latency_ms",
            )
        },
        "local_evidence": local_evidence,
        "gpu_hours_remaining_before_launch": proven_gpu_hours,
        "public_gate": "passed",
        "visible_score_target": MEMORY_VISIBLE_SCORE_TARGET,
        "result": "validation-passed-not-submitted",
    }
    write_memory_progress(
        {
            **record,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "stage": "duck-memory-seed-0-validation-passed",
        }
    )
    if not submit:
        return record

    previous_refs = ensure_daily_quota()
    notebooks_root = generate_artifacts(
        validation_seed=0,
        mode=MEMORY_MODE,
    )
    packaged_manifest = source_manifest_hash()
    if packaged_manifest != record["source_manifest_sha256"]:
        raise RuntimeError(
            "duck-memory source changed after validation; refusing to submit "
            f"{packaged_manifest} instead of {record['source_manifest_sha256']}"
        )
    submission_version = push_kernel(notebooks_root / "submission")
    wait_kernel(SUBMISSION_REF)
    best = refresh_leaderboard_best()
    record.update(
        {
            "submission_kernel": SUBMISSION_REF,
            "submission_version": submission_version,
            "leaderboard_best_before": best,
            "result": "kernel-complete-not-submitted",
        }
    )
    message = (
        f"kaggle-v3 duck-memory-v1 git={record['git_sha'][:12]} "
        f"src={record['source_manifest_sha256'][:12]} "
        f"cfg={record['config_hash'][:12]} "
        f"val={metrics['mean_engine_score']:.4f} "
        "qwen3.6-27b-fp8 retained-reasoning+compaction"
    )
    submission_reference = submit_exact_kernel(
        kernel_version=submission_version,
        message=message,
    )
    submission_result = wait_submission(previous_refs=previous_refs)
    visible_text = str(submission_result.get("publicScore", "")).strip()
    try:
        visible_score = float(visible_text)
    except ValueError:
        visible_score = None
    record.update(
        {
            "submission_reference": submission_reference,
            "submission_result": submission_result,
            "visible_score": visible_score,
            "beat_previous_best": (
                visible_score is not None
                and visible_score > MEMORY_VISIBLE_SCORE_TARGET
            ),
            "result": "complete",
        }
    )
    append_ledger(record)
    write_memory_progress(
        {
            **record,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "stage": "duck-memory-submission-complete",
        }
    )
    return record


def execute_reasoning_seed0_candidate(
    *,
    submit: bool,
    resume_version: int | None = None,
    gpu_hours_remaining: float | None = None,
) -> dict[str, Any]:
    """Run the isolated reasoning-only candidate and optionally submit it."""

    local_evidence = validate_reasoning_local_prerequisites()
    required_gpu_hours = 12.0 if resume_version is None else (9.0 if submit else 0.0)
    proven_gpu_hours = (
        ensure_gpu_hours_remaining(
            gpu_hours_remaining,
            required_hours=required_gpu_hours,
        )
        if required_gpu_hours
        else gpu_hours_remaining
    )
    if resume_version is None:
        generate_artifacts(validation_seed=0, mode=REASONING_MODE)
        source_output = publish_dataset(ROOT / "dist" / "source-dataset")
    else:
        source_output = f"reused {SOURCE_REF}"

    version, metrics_path, _metrics = run_validation_kernel(
        seed=0,
        resume_version=resume_version,
        mode=REASONING_MODE,
    )
    metrics = enforce_reasoning_gate(metrics_path)
    expected_config_hash = config_hash(REASONING_MODE)
    if metrics.get("config_hash") != expected_config_hash:
        raise RuntimeError(
            "duck-reasoning config hash mismatch: "
            f"expected {expected_config_hash}, found {metrics.get('config_hash')!r}"
        )

    telemetry = {
        str(key): int(value)
        for key, value in dict(metrics.get("telemetry") or {}).items()
    }
    record: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": REASONING_MODE,
        "experiment": "duck-reasoning-v1",
        "git_sha": source_hash(),
        "config_hash": expected_config_hash,
        "source_manifest_sha256": str(
            metrics["runtime_fingerprint"]["source_manifest_sha256"]
        ),
        "source_dataset": SOURCE_REF,
        "source_publish": source_output,
        "validation_kernel": VALIDATION_REF,
        "validation_version": version,
        "validation_metrics_path": str(metrics_path),
        "validation_scores": {
            "mean_engine_score": float(metrics["mean_engine_score"]),
            "mean_completed_levels": float(metrics["mean_completed_levels"]),
            "total_completed_levels": int(metrics["total_completed_levels"]),
        },
        "reasoning_health": {
            key: telemetry.get(key, 0)
            for key in (
                "reasoning_template_verified",
                "reasoning_turns",
                "reasoning_chars",
                "reasoning_retained_turns",
                "reasoning_evicted_turns",
                "reasoning_unaccounted_turns",
                "compaction_count",
                "compaction_retries",
                "compaction_failures",
                "emergency_trims",
                "context_evictions",
                "context_overflow_recoveries",
            )
        },
        "local_evidence": local_evidence,
        "gpu_hours_remaining_before_launch": proven_gpu_hours,
        "public_gate": "passed",
        "visible_score_target": REASONING_VISIBLE_SCORE_TARGET,
        "result": "validation-passed-not-submitted",
    }
    write_reasoning_progress(
        {
            **record,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "stage": "duck-reasoning-seed-0-validation-passed",
        }
    )
    if not submit:
        return record

    previous_refs = ensure_daily_quota()
    notebooks_root = generate_artifacts(validation_seed=0, mode=REASONING_MODE)
    packaged_manifest = source_manifest_hash()
    if packaged_manifest != record["source_manifest_sha256"]:
        raise RuntimeError(
            "duck-reasoning source changed after validation; refusing to submit "
            f"{packaged_manifest} instead of {record['source_manifest_sha256']}"
        )
    submission_version = push_kernel(notebooks_root / "submission")
    wait_kernel(SUBMISSION_REF)
    best = refresh_leaderboard_best()
    record.update(
        {
            "submission_kernel": SUBMISSION_REF,
            "submission_version": submission_version,
            "leaderboard_best_before": best,
            "result": "kernel-complete-not-submitted",
        }
    )
    message = (
        f"kaggle-v3 duck-reasoning-v1 git={record['git_sha'][:12]} "
        f"src={record['source_manifest_sha256'][:12]} "
        f"cfg={record['config_hash'][:12]} "
        f"val={metrics['mean_engine_score']:.4f} "
        "qwen3.6-27b-fp8 retained-reasoning-only"
    )
    submission_reference = submit_exact_kernel(
        kernel_version=submission_version,
        message=message,
    )
    submission_result = wait_submission(previous_refs=previous_refs)
    visible_text = str(submission_result.get("publicScore", "")).strip()
    try:
        visible_score = float(visible_text)
    except ValueError:
        visible_score = None
    record.update(
        {
            "submission_reference": submission_reference,
            "submission_result": submission_result,
            "visible_score": visible_score,
            "beat_previous_best": (
                visible_score is not None
                and visible_score > REASONING_VISIBLE_SCORE_TARGET
            ),
            "result": "complete",
        }
    )
    append_ledger(record)
    write_reasoning_progress(
        {
            **record,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "stage": "duck-reasoning-submission-complete",
        }
    )
    return record


def execute_contract_seed0_validation(
    *,
    mode: str,
    resume_version: int | None = None,
) -> dict[str, Any]:
    """Run a public-only one-step contract ablation.

    The first contract-repair result was launched before the wire-preserving
    sandbox dataset was published.  This path deliberately records the
    corrected candidate as a separate artifact so the repair count and
    prediction telemetry are attributable to the actual adapter, rather than
    to an obsolete transport layer.
    """

    if mode not in {CONTRACT_MODE, CONTRACT_REPAIR_MODE}:
        raise ValueError(f"unsupported contract mode: {mode}")
    repair = mode == CONTRACT_REPAIR_MODE
    experiment = "duck-contract-repair-v1" if repair else "duck-contract-v1"
    local_smoke = (
        "results/duck-contract-repair-local-cn04-wirefix.json"
        if repair
        else "results/duck-contract-local-ft09-v2.json"
    )
    if resume_version is None:
        generate_artifacts(validation_seed=0, mode=mode)
        source_output = publish_dataset(ROOT / "dist" / "source-dataset")
    else:
        source_output = f"reused {SOURCE_REF}"
    version, metrics_path, metrics = run_validation_kernel(
        seed=0,
        resume_version=resume_version,
        mode=mode,
    )
    expected_config_hash = config_hash(mode)
    if metrics.get("config_hash") != expected_config_hash:
        raise RuntimeError(
            f"{mode} config hash mismatch: expected {expected_config_hash}, "
            f"found {metrics.get('config_hash')!r}"
        )
    if list(metrics.get("infrastructure_failures") or []):
        raise RuntimeError(
            f"{mode} public validation contains infrastructure failures: "
            + json.dumps(metrics["infrastructure_failures"], sort_keys=True)
        )
    telemetry = {
        str(key): int(value)
        for key, value in dict(metrics.get("telemetry") or {}).items()
        if isinstance(value, (int, float))
    }
    record: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": f"{mode}-seed-0-public-complete",
        "mode": mode,
        "experiment": experiment,
        "git_sha": source_hash(),
        "config_hash": expected_config_hash,
        "source_manifest_sha256": str(
            metrics["runtime_fingerprint"]["source_manifest_sha256"]
        ),
        "source_dataset": SOURCE_REF,
        "source_publish": source_output,
        "validation_kernel": VALIDATION_REF,
        "validation_version": version,
        "validation_metrics_path": str(metrics_path),
        "validation_scores": {
            "mean_engine_score": float(metrics["mean_engine_score"]),
            "mean_completed_levels": float(metrics["mean_completed_levels"]),
            "median_completed_levels": int(metrics["median_completed_levels"]),
            "total_completed_levels": int(metrics["total_completed_levels"]),
            "game_count": int(metrics["game_count"]),
        },
        "telemetry": {
            key: telemetry.get(key, 0)
            for key in (
                "deliberate_proposals",
                "contract_repairs",
                "contract_batch_truncations",
                "prediction_matches",
                "prediction_mismatches",
                "hypothesis_revisions",
                "context_evictions",
                "request_failures",
                "request_timeouts",
                "tool_call_parse_failures",
            )
        },
        "infrastructure_failures": list(metrics.get("infrastructure_failures") or []),
        "local_smoke": local_smoke,
        "hidden_submission": False,
        "result": "public-validation-complete-not-submitted",
    }
    write_contract_progress(record, repair=repair)
    return record


def execute_audit_seed0_validation(
    *,
    resume_version: int | None = None,
) -> dict[str, Any]:
    """Run the public-only sparse self-audit candidate.

    This lane intentionally has no hidden-submission path yet.  Its first
    public run is an attributable harness experiment; the result is recorded
    even when the score gate is not met so the next loop iteration can use the
    exact telemetry.
    """

    if resume_version is None:
        generate_artifacts(validation_seed=0, mode=AUDIT_MODE)
        source_output = publish_dataset(ROOT / "dist" / "source-dataset")
    else:
        source_output = f"reused {SOURCE_REF}"
    version, metrics_path, metrics = run_validation_kernel(
        seed=0,
        resume_version=resume_version,
        mode=AUDIT_MODE,
    )
    expected_config_hash = config_hash(AUDIT_MODE)
    if metrics.get("config_hash") != expected_config_hash:
        raise RuntimeError(
            "duck-audit config hash mismatch: "
            f"expected {expected_config_hash}, found {metrics.get('config_hash')!r}"
        )
    if list(metrics.get("infrastructure_failures") or []):
        raise RuntimeError(
            "duck-audit public validation contains infrastructure failures: "
            + json.dumps(metrics["infrastructure_failures"], sort_keys=True)
        )
    telemetry = {
        str(key): int(value)
        for key, value in dict(metrics.get("telemetry") or {}).items()
        if isinstance(value, (int, float))
    }
    record: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "duck-audit-seed-0-public-complete",
        "mode": AUDIT_MODE,
        "experiment": "duck-audit-v1",
        "git_sha": source_hash(),
        "config_hash": expected_config_hash,
        "source_manifest_sha256": str(
            metrics["runtime_fingerprint"]["source_manifest_sha256"]
        ),
        "source_dataset": SOURCE_REF,
        "source_publish": source_output,
        "validation_kernel": VALIDATION_REF,
        "validation_version": version,
        "validation_metrics_path": str(metrics_path),
        "validation_scores": {
            "mean_engine_score": float(metrics["mean_engine_score"]),
            "mean_completed_levels": float(metrics["mean_completed_levels"]),
            "median_completed_levels": int(metrics["median_completed_levels"]),
            "total_completed_levels": int(metrics["total_completed_levels"]),
            "game_count": int(metrics["game_count"]),
        },
        "telemetry": {
            key: telemetry.get(key, 0)
            for key in (
                "audit_trigger_count",
                "audit_repeat_triggers",
                "audit_no_change_triggers",
                "context_evictions",
                "request_failures",
                "request_timeouts",
                "tool_call_parse_failures",
            )
        },
        "infrastructure_failures": list(metrics.get("infrastructure_failures") or []),
        "local_smoke": "results/duck-audit-local-ft09-v2.json",
        "hidden_submission": False,
        "result": "public-validation-complete-not-submitted",
    }
    write_audit_progress(record)
    return record


def write_information_progress(record: dict[str, Any]) -> None:
    """Atomically checkpoint the information-acquisition candidate."""

    INFORMATION_PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    temporary = INFORMATION_PROGRESS.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(INFORMATION_PROGRESS)


def execute_information_seed0_validation(
    *,
    resume_version: int | None = None,
) -> dict[str, Any]:
    """Run the public-only targeted-information candidate."""

    if resume_version is None:
        generate_artifacts(validation_seed=0, mode=INFORMATION_MODE)
        source_output = publish_dataset(ROOT / "dist" / "source-dataset")
    else:
        source_output = f"reused {SOURCE_REF}"
    version, metrics_path, metrics = run_validation_kernel(
        seed=0,
        resume_version=resume_version,
        mode=INFORMATION_MODE,
    )
    expected_config_hash = config_hash(INFORMATION_MODE)
    if metrics.get("config_hash") != expected_config_hash:
        raise RuntimeError(
            "duck-information config hash mismatch: "
            f"expected {expected_config_hash}, found {metrics.get('config_hash')!r}"
        )
    if list(metrics.get("infrastructure_failures") or []):
        raise RuntimeError(
            "duck-information public validation contains infrastructure failures: "
            + json.dumps(metrics["infrastructure_failures"], sort_keys=True)
        )
    telemetry = {
        str(key): int(value)
        for key, value in dict(metrics.get("telemetry") or {}).items()
        if isinstance(value, (int, float))
    }
    record: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "duck-information-seed-0-public-complete",
        "mode": INFORMATION_MODE,
        "experiment": "duck-information-v1",
        "git_sha": source_hash(),
        "config_hash": expected_config_hash,
        "source_manifest_sha256": str(
            metrics["runtime_fingerprint"]["source_manifest_sha256"]
        ),
        "source_dataset": SOURCE_REF,
        "source_publish": source_output,
        "validation_kernel": VALIDATION_REF,
        "validation_version": version,
        "validation_metrics_path": str(metrics_path),
        "validation_scores": {
            "mean_engine_score": float(metrics["mean_engine_score"]),
            "mean_completed_levels": float(metrics["mean_completed_levels"]),
            "median_completed_levels": int(metrics["median_completed_levels"]),
            "total_completed_levels": int(metrics["total_completed_levels"]),
            "game_count": int(metrics["game_count"]),
        },
        "telemetry": {
            key: telemetry.get(key, 0)
            for key in (
                "information_trigger_count",
                "information_no_change_triggers",
                "context_evictions",
                "request_failures",
                "request_timeouts",
                "tool_call_parse_failures",
            )
        },
        "infrastructure_failures": list(metrics.get("infrastructure_failures") or []),
        "hidden_submission": False,
        "result": "public-validation-complete-not-submitted",
    }
    write_information_progress(record)
    return record


def write_hierarchy_progress(record: dict[str, Any]) -> None:
    """Atomically checkpoint the bounded candidate-search experiment."""

    HIERARCHY_PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    temporary = HIERARCHY_PROGRESS.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(HIERARCHY_PROGRESS)


def execute_hierarchy_seed0_validation(
    *,
    resume_version: int | None = None,
) -> dict[str, Any]:
    """Run the public-only bounded hierarchy candidate."""

    if resume_version is None:
        generate_artifacts(validation_seed=0, mode=HIERARCHY_MODE)
        source_output = publish_dataset(ROOT / "dist" / "source-dataset")
    else:
        source_output = f"reused {SOURCE_REF}"
    version, metrics_path, metrics = run_validation_kernel(
        seed=0,
        resume_version=resume_version,
        mode=HIERARCHY_MODE,
    )
    expected_config_hash = config_hash(HIERARCHY_MODE)
    if metrics.get("config_hash") != expected_config_hash:
        raise RuntimeError(
            "duck-hierarchy config hash mismatch: "
            f"expected {expected_config_hash}, found {metrics.get('config_hash')!r}"
        )
    if list(metrics.get("infrastructure_failures") or []):
        raise RuntimeError(
            "duck-hierarchy public validation contains infrastructure failures: "
            + json.dumps(metrics["infrastructure_failures"], sort_keys=True)
        )
    telemetry = {
        str(key): int(value)
        for key, value in dict(metrics.get("telemetry") or {}).items()
        if isinstance(value, (int, float))
    }
    record: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "duck-hierarchy-seed-0-public-complete",
        "mode": HIERARCHY_MODE,
        "experiment": "duck-hierarchy-v1",
        "git_sha": source_hash(),
        "config_hash": expected_config_hash,
        "source_manifest_sha256": str(
            metrics["runtime_fingerprint"]["source_manifest_sha256"]
        ),
        "source_dataset": SOURCE_REF,
        "source_publish": source_output,
        "validation_kernel": VALIDATION_REF,
        "validation_version": version,
        "validation_metrics_path": str(metrics_path),
        "validation_scores": {
            "mean_engine_score": float(metrics["mean_engine_score"]),
            "mean_completed_levels": float(metrics["mean_completed_levels"]),
            "median_completed_levels": int(metrics["median_completed_levels"]),
            "total_completed_levels": int(metrics["total_completed_levels"]),
            "game_count": int(metrics["game_count"]),
        },
        "telemetry": {
            key: telemetry.get(key, 0)
            for key in (
                "hierarchy_trigger_count",
                "hierarchy_no_change_triggers",
                "hierarchy_level_start_triggers",
                "context_evictions",
                "request_failures",
                "request_timeouts",
                "tool_call_parse_failures",
            )
        },
        "infrastructure_failures": list(metrics.get("infrastructure_failures") or []),
        "hidden_submission": False,
        "result": "public-validation-complete-not-submitted",
    }
    write_hierarchy_progress(record)
    return record


def write_diversity_progress(record: dict[str, Any]) -> None:
    """Atomically checkpoint the controlled-diversity experiment."""

    DIVERSITY_PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    temporary = DIVERSITY_PROGRESS.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(DIVERSITY_PROGRESS)


def execute_diversity_seed0_validation(
    *,
    resume_version: int | None = None,
) -> dict[str, Any]:
    """Run the public-only controlled-diversity candidate."""

    if resume_version is None:
        generate_artifacts(validation_seed=0, mode=DIVERSITY_MODE)
        source_output = publish_dataset(ROOT / "dist" / "source-dataset")
    else:
        source_output = f"reused {SOURCE_REF}"
    version, metrics_path, metrics = run_validation_kernel(
        seed=0,
        resume_version=resume_version,
        mode=DIVERSITY_MODE,
    )
    expected_config_hash = config_hash(DIVERSITY_MODE)
    if metrics.get("config_hash") != expected_config_hash:
        raise RuntimeError(
            "duck-diversity config hash mismatch: "
            f"expected {expected_config_hash}, found {metrics.get('config_hash')!r}"
        )
    if list(metrics.get("infrastructure_failures") or []):
        raise RuntimeError(
            "duck-diversity public validation contains infrastructure failures: "
            + json.dumps(metrics["infrastructure_failures"], sort_keys=True)
        )
    telemetry = {
        str(key): int(value)
        for key, value in dict(metrics.get("telemetry") or {}).items()
        if isinstance(value, (int, float))
    }
    record: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "duck-diversity-seed-0-public-complete",
        "mode": DIVERSITY_MODE,
        "experiment": "duck-diversity-v1",
        "git_sha": source_hash(),
        "config_hash": expected_config_hash,
        "source_manifest_sha256": str(
            metrics["runtime_fingerprint"]["source_manifest_sha256"]
        ),
        "source_dataset": SOURCE_REF,
        "source_publish": source_output,
        "validation_kernel": VALIDATION_REF,
        "validation_version": version,
        "validation_metrics_path": str(metrics_path),
        "validation_scores": {
            "mean_engine_score": float(metrics["mean_engine_score"]),
            "mean_completed_levels": float(metrics["mean_completed_levels"]),
            "median_completed_levels": int(metrics["median_completed_levels"]),
            "total_completed_levels": int(metrics["total_completed_levels"]),
            "game_count": int(metrics["game_count"]),
        },
        "telemetry": {
            key: telemetry.get(key, 0)
            for key in (
                "diversity_trigger_count",
                "diversity_no_change_triggers",
                "diversity_seed_uses",
                "context_evictions",
                "request_failures",
                "request_timeouts",
                "tool_call_parse_failures",
            )
        },
        "infrastructure_failures": list(metrics.get("infrastructure_failures") or []),
        "hidden_submission": False,
        "result": "public-validation-complete-not-submitted",
    }
    write_diversity_progress(record)
    return record


def execute_fidelity_validation(
    *,
    resume_version: int | None = None,
) -> dict[str, Any]:
    """Publish and run only the unseeded fidelity stage."""

    if resume_version is None:
        generate_artifacts(validation_seed=None)
        source_output = publish_dataset(ROOT / "dist" / "source-dataset")
    else:
        source_output = f"reused {SOURCE_REF}"
    version, metrics_path, metrics = run_validation_kernel(
        seed=None,
        resume_version=resume_version,
    )
    record = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "unseeded-fidelity-complete",
        "git_sha": source_hash(),
        "config_hash": config_hash(),
        "source_manifest_sha256": source_manifest_hash(),
        "source_dataset": SOURCE_REF,
        "source_publish": source_output,
        "validation_kernel": VALIDATION_REF,
        "fidelity_validation_version": version,
        "metrics_path": str(metrics_path),
        "mean_engine_score": metrics["mean_engine_score"],
        "infrastructure_failures": metrics["infrastructure_failures"],
        "runtime_fingerprint": metrics["runtime_fingerprint"],
        "telemetry": metrics.get("telemetry", {}),
    }
    write_reference_progress(record)
    return record


def execute_poetiq_candidate(
    *,
    submit: bool,
    seed_versions: dict[int, int] | None = None,
    gpu_hours_remaining: float | None = None,
) -> dict[str, Any]:
    """Run Poetiq seeds 0/1 and submit only after the robust two-seed gate."""

    local_evidence = validate_poetiq_local_prerequisites()
    cached = seed_versions or {}
    # The two independent 25-game public kernels each fit inside the audited
    # 2h12 public envelope. The current competition account reports that the
    # hidden gateway rerun is quota-neutral, so reserve only 4.5h for public
    # validation and do not reserve GPU hours for the later submission.
    required_gpu_hours = (
        POETIQ_PUBLIC_GPU_RESERVE_HOURS if len(cached) < 2 else 0.0
    )
    proven_gpu_hours = (
        ensure_gpu_hours_remaining(
            gpu_hours_remaining,
            required_hours=required_gpu_hours,
        )
        if required_gpu_hours
        else gpu_hours_remaining
    )

    source_output = f"reused {SOURCE_REF}"
    seed_metrics: list[dict[str, Any]] = []
    versions: dict[int, int] = {}
    paths: dict[int, str] = {}
    for seed in (0, 1):
        resume = cached.get(seed)
        if resume is None and seed == 0:
            generate_artifacts(validation_seed=0, mode=POETIQ_MODE)
            source_output = publish_dataset(ROOT / "dist" / "source-dataset")
        version, metrics_path, metrics = run_validation_kernel(
            seed=seed,
            resume_version=resume,
            mode=POETIQ_MODE,
        )
        versions[seed] = version
        paths[seed] = str(metrics_path)
        seed_metrics.append(metrics)
        write_poetiq_progress(
            {
                "schema_version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "stage": f"{POETIQ_MODE}-seed-{seed}-complete",
                "mode": POETIQ_MODE,
                "experiment": "duck-poetiq-v1",
                "seed_validation_versions": {str(key): value for key, value in versions.items()},
                "seed_validation_scores": {
                    str(index): float(item.get("mean_engine_score", 0.0))
                    for index, item in enumerate(seed_metrics)
                },
                "seed_metric_paths": paths,
                "source_dataset": SOURCE_REF,
            }
        )
        if seed == 0 and (
            int(metrics.get("total_completed_levels", 0)) < 18
            or int(metrics.get("nonzero_game_count", 0)) < 15
        ):
            raise RuntimeError(
                "duck-poetiq seed 0 failed the hard breadth/level floor; "
                "seed 1 was not started"
            )

    aggregate = aggregate_two_seed_runs(seed_metrics, expected_mode=POETIQ_MODE)
    aggregate_path = ROOT / "results" / "duck-poetiq-seeds-0-1.json"
    write_metrics(aggregate, aggregate_path)
    decision = evaluate_poetiq_promotion(
        seed_metrics,
        rehearsal_elapsed_s=float(local_evidence["rehearsal_elapsed_seconds"]),
        soft_deadline_s=SOFT_DEADLINE_S,
    )
    if not decision.passed:
        raise RuntimeError(
            "duck-poetiq promotion gate failed: "
            + "; ".join(decision.reasons)
        )

    record: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "duck-poetiq-two-seed-gate-passed",
        "mode": POETIQ_MODE,
        "experiment": "duck-poetiq-v1",
        "git_sha": source_hash(),
        "config_hash": config_hash(POETIQ_MODE),
        "source_manifest_sha256": source_manifest_hash(),
        "source_dataset": SOURCE_REF,
        "source_publish": source_output,
        "validation_kernel": VALIDATION_REF,
        "seed_validation_versions": {str(seed): version for seed, version in versions.items()},
        "seed_metric_paths": paths,
        "aggregate_metrics_path": str(aggregate_path),
        "validation_scores": {
            "mean_engine_score": float(aggregate["mean_engine_score"]),
            "trimmed_mean_engine_score": float(aggregate["trimmed_mean_engine_score"]),
            "seed_engine_scores": aggregate["seed_engine_scores"],
            "seed_completed_levels": aggregate["seed_completed_levels"],
            "seed_nonzero_game_counts": aggregate["seed_nonzero_game_counts"],
        },
        "local_evidence": local_evidence,
        "gpu_hours_remaining_before_launch": proven_gpu_hours,
        "public_gate": "passed",
        "result": "validation-passed-not-submitted",
    }
    if submit:
        previous_refs = ensure_daily_quota()
        notebooks_root = generate_artifacts(validation_seed=0, mode=POETIQ_MODE)
        packaged_manifest = source_manifest_hash()
        if packaged_manifest != record["source_manifest_sha256"]:
            raise RuntimeError(
                "duck-poetiq source changed after validation; refusing to submit "
                f"{packaged_manifest} instead of {record['source_manifest_sha256']}"
            )
        submission_version = push_kernel(notebooks_root / "submission")
        wait_kernel(SUBMISSION_REF)
        leaderboard_best = refresh_leaderboard_best()
        message = (
            f"kaggle-v3 duck-poetiq-v1 git={record['git_sha'][:12]} "
            f"src={record['source_manifest_sha256'][:12]} "
            f"cfg={record['config_hash'][:12]} "
            f"val={aggregate['mean_engine_score']:.4f} "
            "audit+information+hypotheses+verification+diversity"
        )
        submission_reference = submit_exact_kernel(
            kernel_version=submission_version,
            message=message,
        )
        submission_result = wait_submission(previous_refs=previous_refs)
        record.update(
            {
                "submission_kernel": SUBMISSION_REF,
                "submission_version": submission_version,
                "submission_reference": submission_reference,
                "submission_result": submission_result,
                "leaderboard_best_before": leaderboard_best,
                "result": "complete",
            }
        )
    write_poetiq_progress(record)
    append_ledger(record)
    return record


def execute_portfolio_candidate(
    *,
    submit: bool,
    seed_versions: dict[int, int] | None = None,
    gpu_hours_remaining: float | None = None,
) -> dict[str, Any]:
    """Run the routed public seeds and submit only the exact gated artifact."""

    local_evidence = validate_portfolio_local_prerequisites()
    router = PortfolioRouter.load()
    cached = seed_versions or {}
    required_gpu_hours = (
        PORTFOLIO_PUBLIC_GPU_RESERVE_HOURS if len(cached) < 2 else 0.0
    )
    proven_gpu_hours = (
        ensure_gpu_hours_remaining(
            gpu_hours_remaining,
            required_hours=required_gpu_hours,
        )
        if required_gpu_hours
        else gpu_hours_remaining
    )

    source_output = f"reused {SOURCE_REF}"
    seed_metrics: list[dict[str, Any]] = []
    versions: dict[int, int] = {}
    paths: dict[int, str] = {}
    source_manifests: set[str] = set()
    prompt_hashes: set[str] = set()
    for seed in (0, 1):
        resume = cached.get(seed)
        if resume is None and seed == 0:
            generate_artifacts(validation_seed=0, mode=PORTFOLIO_MODE)
            source_output = publish_dataset(ROOT / "dist" / "source-dataset")
        version, metrics_path, metrics = run_validation_kernel(
            seed=seed,
            resume_version=resume,
            mode=PORTFOLIO_MODE,
        )
        expected_config = HarnessConfig.portfolio(seed=seed).config_hash
        if metrics.get("config_hash") != expected_config:
            raise RuntimeError(
                f"duck-portfolio seed {seed} config hash mismatch: expected "
                f"{expected_config}, found {metrics.get('config_hash')!r}"
            )
        fingerprint = dict(metrics.get("runtime_fingerprint") or {})
        source_manifests.add(str(fingerprint.get("source_manifest_sha256", "")))
        prompt_hashes.add(str(metrics.get("prompt_sha256", "")))
        versions[seed] = version
        paths[seed] = str(metrics_path)
        seed_metrics.append(metrics)
        write_portfolio_progress(
            {
                "schema_version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "stage": f"{PORTFOLIO_MODE}-seed-{seed}-complete",
                "mode": PORTFOLIO_MODE,
                "experiment": "duck-portfolio-v1",
                "router_artifact_sha256": router.artifact_hash,
                "seed_validation_versions": {
                    str(key): value for key, value in versions.items()
                },
                "seed_validation_scores": {
                    str(index): float(item.get("mean_engine_score", 0.0))
                    for index, item in enumerate(seed_metrics)
                },
                "seed_metric_paths": paths,
                "source_dataset": SOURCE_REF,
            }
        )
        if seed == 0:
            seed0_failures: list[str] = []
            if float(metrics.get("mean_engine_score", 0.0)) < 2.5631:
                seed0_failures.append("mean engine score below 2.5631")
            if int(metrics.get("total_completed_levels", 0)) < 18:
                seed0_failures.append("completed levels below 18")
            if int(metrics.get("nonzero_game_count", 0)) < 15:
                seed0_failures.append("nonzero-game breadth below 15")
            if float(metrics.get("trimmed_mean_engine_score", 0.0)) <= 0.9370812:
                seed0_failures.append("top-three-trimmed mean did not exceed 0.9370812")
            if seed0_failures:
                raise RuntimeError(
                    "duck-portfolio seed 0 failed its hard gate; seed 1 was not "
                    "started: " + "; ".join(seed0_failures)
                )

    if len(source_manifests) != 1 or not next(iter(source_manifests), ""):
        raise RuntimeError("duck-portfolio seeds do not share one source manifest")
    if len(prompt_hashes) != 1 or not next(iter(prompt_hashes), ""):
        raise RuntimeError("duck-portfolio seeds do not share one prompt hash")

    aggregate = aggregate_two_seed_runs(
        seed_metrics,
        expected_mode=PORTFOLIO_MODE,
    )
    aggregate_path = ROOT / "results" / "duck-portfolio-seeds-0-1.json"
    write_metrics(aggregate, aggregate_path)
    decision = evaluate_portfolio_promotion(
        seed_metrics,
        rehearsal_elapsed_s=float(local_evidence["rehearsal_elapsed_seconds"]),
        soft_deadline_s=SOFT_DEADLINE_S,
    )
    if not decision.passed:
        raise RuntimeError(
            "duck-portfolio promotion gate failed: " + "; ".join(decision.reasons)
        )

    source_manifest = next(iter(source_manifests))
    record: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "duck-portfolio-two-seed-gate-passed",
        "mode": PORTFOLIO_MODE,
        "experiment": "duck-portfolio-v1",
        "git_sha": source_hash(),
        "config_hash": config_hash(PORTFOLIO_MODE),
        "seed_config_hashes": {
            str(seed): HarnessConfig.portfolio(seed=seed).config_hash
            for seed in (0, 1)
        },
        "prompt_sha256": next(iter(prompt_hashes)),
        "router_artifact_sha256": router.artifact_hash,
        "source_manifest_sha256": source_manifest,
        "source_dataset": SOURCE_REF,
        "source_publish": source_output,
        "validation_kernel": VALIDATION_REF,
        "seed_validation_versions": {
            str(seed): version for seed, version in versions.items()
        },
        "seed_metric_paths": paths,
        "aggregate_metrics_path": str(aggregate_path),
        "validation_scores": {
            "mean_engine_score": float(aggregate["mean_engine_score"]),
            "trimmed_mean_engine_score": float(
                aggregate["trimmed_mean_engine_score"]
            ),
            "seed_engine_scores": aggregate["seed_engine_scores"],
            "seed_completed_levels": aggregate["seed_completed_levels"],
            "seed_nonzero_game_counts": aggregate["seed_nonzero_game_counts"],
        },
        "local_evidence": local_evidence,
        "gpu_hours_remaining_before_launch": proven_gpu_hours,
        "hidden_gpu_quota_behavior": "account-confirmed-quota-neutral",
        "public_gate": "passed",
        "result": "validation-passed-not-submitted",
    }
    if submit:
        previous_refs = ensure_daily_quota()
        notebooks_root = generate_artifacts(
            validation_seed=0,
            mode=PORTFOLIO_MODE,
        )
        packaged_manifest = source_manifest_hash()
        packaged_identity = portfolio_packaged_identity()
        if packaged_manifest != source_manifest:
            raise RuntimeError(
                "duck-portfolio source changed after validation; refusing to submit "
                f"{packaged_manifest} instead of {source_manifest}"
            )
        if packaged_identity["router_artifact_sha256"] != router.artifact_hash:
            raise RuntimeError(
                "duck-portfolio router changed after validation; refusing to submit"
            )
        if packaged_identity["config_hash"] != record["config_hash"]:
            raise RuntimeError(
                "duck-portfolio config changed after validation; refusing to submit"
            )
        if packaged_identity["prompt_sha256"] != record["prompt_sha256"]:
            raise RuntimeError(
                "duck-portfolio prompt changed after validation; refusing to submit"
            )
        submission_version = push_kernel(notebooks_root / "submission")
        wait_kernel(SUBMISSION_REF)
        leaderboard_best = refresh_leaderboard_best()
        message = (
            f"kaggle-v3 duck-portfolio-v1 git={record['git_sha'][:12]} "
            f"src={source_manifest[:12]} cfg={record['config_hash'][:12]} "
            f"router={router.artifact_hash[:12]} "
            f"val={aggregate['mean_engine_score']:.4f} deterministic-router"
        )
        submission_reference = submit_exact_kernel(
            kernel_version=submission_version,
            message=message,
        )
        submission_result = wait_submission(previous_refs=previous_refs)
        record.update(
            {
                "submission_kernel": SUBMISSION_REF,
                "submission_version": submission_version,
                "submission_reference": submission_reference,
                "submission_result": submission_result,
                "leaderboard_best_before": leaderboard_best,
                "result": "complete",
            }
        )
    write_portfolio_progress(record)
    append_ledger(record)
    return record


def execute_retrodict_candidate(
    *,
    submit: bool,
    seed_versions: dict[int, int] | None = None,
    gpu_hours_remaining: float | None = None,
    experimental_public: bool = False,
) -> dict[str, Any]:
    """Run two public seeds and submit only the exact retrodictive artifact."""

    if experimental_public and submit:
        raise RuntimeError(
            "experimental duck-retrodict public runs can never submit"
        )
    local_evidence = validate_retrodict_local_prerequisites(
        require_offline_pass=not experimental_public
    )
    cached = seed_versions or {}
    required_gpu_hours = (
        RETRODICT_PUBLIC_GPU_RESERVE_HOURS if len(cached) < 2 else 0.0
    )
    proven_gpu_hours = (
        ensure_gpu_hours_remaining(
            gpu_hours_remaining,
            required_hours=required_gpu_hours,
        )
        if required_gpu_hours
        else gpu_hours_remaining
    )

    source_output = f"reused {SOURCE_REF}"
    seed_metrics: list[dict[str, Any]] = []
    versions: dict[int, int] = {}
    paths: dict[int, str] = {}
    source_manifests: set[str] = set()
    prompt_hashes: set[str] = set()
    for seed in (0, 1):
        resume = cached.get(seed)
        if resume is None and seed == 0:
            generate_artifacts(validation_seed=0, mode=RETRODICT_MODE)
            source_output = publish_dataset(ROOT / "dist" / "source-dataset")
        version, metrics_path, metrics = run_validation_kernel(
            seed=seed,
            resume_version=resume,
            mode=RETRODICT_MODE,
        )
        expected_config = HarnessConfig.retrodict(seed=seed).config_hash
        if metrics.get("config_hash") != expected_config:
            raise RuntimeError(
                f"duck-retrodict seed {seed} config hash mismatch: expected "
                f"{expected_config}, found {metrics.get('config_hash')!r}"
            )
        fingerprint = dict(metrics.get("runtime_fingerprint") or {})
        source_manifests.add(str(fingerprint.get("source_manifest_sha256", "")))
        prompt_hashes.add(str(metrics.get("prompt_sha256", "")))
        versions[seed] = version
        paths[seed] = str(metrics_path)
        seed_metrics.append(metrics)
        write_retrodict_progress(
            {
                "schema_version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "stage": f"{RETRODICT_MODE}-seed-{seed}-complete",
                "mode": RETRODICT_MODE,
                "experiment": "duck-retrodict-v1",
                "seed_validation_versions": {
                    str(key): value for key, value in versions.items()
                },
                "seed_validation_scores": {
                    str(index): float(item.get("mean_engine_score", 0.0))
                    for index, item in enumerate(seed_metrics)
                },
                "seed_metric_paths": paths,
                "source_dataset": SOURCE_REF,
            }
        )

    if len(source_manifests) != 1 or not next(iter(source_manifests), ""):
        raise RuntimeError("duck-retrodict seeds do not share one source manifest")
    if len(prompt_hashes) != 1 or not next(iter(prompt_hashes), ""):
        raise RuntimeError("duck-retrodict seeds do not share one prompt hash")

    aggregate = aggregate_two_seed_runs(
        seed_metrics,
        expected_mode=RETRODICT_MODE,
    )
    aggregate_path = ROOT / "results" / "duck-retrodict-seeds-0-1.json"
    write_metrics(aggregate, aggregate_path)
    leaderboard_best = max(1.86, refresh_leaderboard_best())
    decision = evaluate_retrodict_promotion(
        seed_metrics,
        offline_report=dict(local_evidence["offline"]),
        rehearsal_elapsed_s=float(local_evidence["rehearsal_elapsed_seconds"]),
        soft_deadline_s=SOFT_DEADLINE_S,
        leaderboard_target=leaderboard_best,
    )
    if not decision.passed and not experimental_public:
        raise RuntimeError(
            "duck-retrodict promotion gate failed: "
            + "; ".join(decision.reasons)
        )

    source_manifest = next(iter(source_manifests))
    record: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": (
            "duck-retrodict-two-seed-gate-passed"
            if decision.passed
            else "duck-retrodict-experimental-public-complete"
        ),
        "mode": RETRODICT_MODE,
        "experiment": "duck-retrodict-v1",
        "git_sha": source_hash(),
        "config_hash": config_hash(RETRODICT_MODE),
        "seed_config_hashes": {
            str(seed): HarnessConfig.retrodict(seed=seed).config_hash
            for seed in (0, 1)
        },
        "prompt_sha256": next(iter(prompt_hashes)),
        "source_manifest_sha256": source_manifest,
        "source_dataset": SOURCE_REF,
        "source_publish": source_output,
        "validation_kernel": VALIDATION_REF,
        "seed_validation_versions": {
            str(seed): version for seed, version in versions.items()
        },
        "seed_metric_paths": paths,
        "aggregate_metrics_path": str(aggregate_path),
        "validation_scores": {
            "mean_engine_score": float(aggregate["mean_engine_score"]),
            "trimmed_mean_engine_score": float(
                aggregate["trimmed_mean_engine_score"]
            ),
            "seed_engine_scores": aggregate["seed_engine_scores"],
            "seed_completed_levels": aggregate["seed_completed_levels"],
            "seed_nonzero_game_counts": aggregate["seed_nonzero_game_counts"],
        },
        "leaderboard_best_before": leaderboard_best,
        "winning_target": leaderboard_best + 0.01,
        "local_evidence": {
            key: value for key, value in local_evidence.items() if key != "offline"
        },
        "offline_gate": local_evidence["offline"].get("promotion"),
        "gpu_hours_remaining_before_launch": proven_gpu_hours,
        "hidden_gpu_quota_behavior": "account-confirmed-quota-neutral",
        "public_gate": "passed" if decision.passed else "failed",
        "public_gate_reasons": list(decision.reasons),
        "experimental_public": experimental_public,
        "result": (
            "validation-passed-not-submitted"
            if decision.passed
            else "experimental-public-not-promotable"
        ),
    }
    if submit:
        previous_refs = ensure_daily_quota()
        notebooks_root = generate_artifacts(
            validation_seed=0,
            mode=RETRODICT_MODE,
        )
        packaged_manifest = source_manifest_hash()
        packaged_identity = retrodict_packaged_identity()
        if packaged_manifest != source_manifest:
            raise RuntimeError(
                "duck-retrodict source changed after validation; refusing to submit"
            )
        if packaged_identity["config_hash"] != record["config_hash"]:
            raise RuntimeError(
                "duck-retrodict config changed after validation; refusing to submit"
            )
        if packaged_identity["prompt_sha256"] != record["prompt_sha256"]:
            raise RuntimeError(
                "duck-retrodict prompt changed after validation; refusing to submit"
            )
        submission_version = push_kernel(notebooks_root / "submission")
        wait_kernel(SUBMISSION_REF)
        message = (
            f"kaggle-v3 duck-retrodict-v1 git={record['git_sha'][:12]} "
            f"src={source_manifest[:12]} cfg={record['config_hash'][:12]} "
            f"val={aggregate['mean_engine_score']:.4f} typed-replay+verified-search"
        )
        submission_reference = submit_exact_kernel(
            kernel_version=submission_version,
            message=message,
        )
        submission_result = wait_submission(previous_refs=previous_refs)
        record.update(
            {
                "submission_kernel": SUBMISSION_REF,
                "submission_version": submission_version,
                "submission_reference": submission_reference,
                "submission_result": submission_result,
                "result": "complete",
            }
        )
    write_retrodict_progress(record)
    append_ledger(record)
    return record


def execute_pipeline(
    *,
    submit: bool,
    fidelity_version: int | None = None,
    seed_versions: dict[int, int] | None = None,
) -> dict[str, Any]:
    if fidelity_version is None:
        generate_artifacts(validation_seed=None)
        source_output = publish_dataset(ROOT / "dist" / "source-dataset")
    else:
        source_output = f"reused {SOURCE_REF}"

    fidelity_version, _fidelity_path, fidelity_metrics = run_validation_kernel(
        seed=None,
        resume_version=fidelity_version,
    )

    versions: dict[int, int] = {}
    seed_metric_paths: list[Path] = []
    seed_scores: dict[int, float] = {}
    for seed in range(5):
        resume = (seed_versions or {}).get(seed)
        version, path, seed_metrics = run_validation_kernel(
            seed=seed,
            resume_version=resume,
        )
        versions[seed] = version
        seed_metric_paths.append(path)
        seed_scores[seed] = float(seed_metrics["mean_engine_score"])
        write_reference_progress(
            {
                "schema_version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "stage": "seed-validation-in-progress",
                "git_sha": source_hash(),
                "config_hash": config_hash(),
                "source_manifest_sha256": source_manifest_hash(),
                "source_dataset": SOURCE_REF,
                "source_publish": source_output,
                "validation_kernel": VALIDATION_REF,
                "fidelity_validation_version": fidelity_version,
                "fidelity_validation_score": fidelity_metrics["mean_engine_score"],
                "completed_seed_versions": {
                    str(completed_seed): completed_version
                    for completed_seed, completed_version in versions.items()
                },
                "completed_seed_scores": {
                    str(completed_seed): completed_score
                    for completed_seed, completed_score in seed_scores.items()
                },
                "completed_seed_metrics": {
                    str(completed_seed): str(completed_path)
                    for completed_seed, completed_path in zip(
                        versions,
                        seed_metric_paths,
                        strict=True,
                    )
                },
                "next_seed": seed + 1 if seed < 4 else None,
            }
        )

    aggregate = aggregate_metric_files(seed_metric_paths)
    aggregate_path = ROOT / "results" / "duck-reference-seeds-0-4.json"
    write_metrics(aggregate, aggregate_path)
    metrics = enforce_reference_gate(aggregate_path)
    freeze_reference(
        aggregate_path,
        metrics,
        kernel_versions=versions,
    )
    write_reference_progress(
        {
            "schema_version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "stage": "five-seed-reference-complete",
            "git_sha": source_hash(),
            "config_hash": config_hash(),
            "source_manifest_sha256": source_manifest_hash(),
            "source_dataset": SOURCE_REF,
            "source_publish": source_output,
            "validation_kernel": VALIDATION_REF,
            "fidelity_validation_version": fidelity_version,
            "fidelity_validation_score": fidelity_metrics["mean_engine_score"],
            "seed_validation_versions": {
                str(seed): version for seed, version in versions.items()
            },
            "seed_validation_scores": {
                str(seed): score for seed, score in seed_scores.items()
            },
            "aggregate_metrics_path": str(aggregate_path),
            "mean_engine_score": metrics["mean_engine_score"],
            "infrastructure_failures": metrics["infrastructure_failures"],
        }
    )
    generate_artifacts(validation_seed=None)
    submission_version = push_kernel(ROOT / "notebooks" / "submission")
    wait_kernel(SUBMISSION_REF)
    best = refresh_leaderboard_best()
    target = max(1.86, best) + 0.01
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": source_hash(),
        "config_hash": config_hash(),
        "source_manifest_sha256": source_manifest_hash(),
        "experiment": metrics["experiment"],
        "source_dataset": SOURCE_REF,
        "source_publish": source_output,
        "validation_kernel": VALIDATION_REF,
        "fidelity_validation_version": fidelity_version,
        "fidelity_validation_score": fidelity_metrics["mean_engine_score"],
        "seed_validation_versions": {
            str(seed): version for seed, version in versions.items()
        },
        "validation_scores": {
            "mean_engine_score": metrics["mean_engine_score"],
            "seed_engine_scores": [
                item["mean_engine_score"] for item in metrics["seed_runs"]
            ],
            "mean_completed_levels": metrics["mean_completed_levels"],
            "seed_completed_levels": [
                item["mean_completed_levels"] for item in metrics["seed_runs"]
            ],
        },
        "submission_kernel": SUBMISSION_REF,
        "submission_version": submission_version,
        "leaderboard_best_before": best,
        "target": target,
        "result": "kernel-complete-not-submitted",
    }
    if submit:
        previous_refs = ensure_daily_quota()
        message = (
            f"kaggle-v3 {record['experiment']} git={record['git_sha'][:12]} "
            f"src={record['source_manifest_sha256'][:12]} "
            f"cfg={record['config_hash'][:12]} "
            f"val={metrics['mean_engine_score']:.4f} "
            "qwen3.6-27b-fp8/vllm0.19"
        )
        submission_reference = submit_exact_kernel(
            kernel_version=submission_version,
            message=message,
        )
        result = wait_submission(previous_refs=previous_refs)
        record.update(
            {
                "submission_reference": submission_reference,
                "submission_result": result,
                "result": "complete",
            }
        )
    append_ledger(record)
    return record


def execute_early_private_baseline(
    *,
    submit: bool,
    fidelity_version: int,
    seed0_version: int,
    seed1_version: int,
) -> dict[str, Any]:
    """Use two healthy seeded runs, skip seeds 2-4, and launch a hidden baseline."""

    fidelity_version, fidelity_path, fidelity_metrics = run_validation_kernel(
        seed=None,
        resume_version=fidelity_version,
    )
    seed0_version, seed0_path, seed0_metrics = run_validation_kernel(
        seed=0,
        resume_version=seed0_version,
    )
    seed1_version, seed1_path, seed1_metrics = attach_running_validation_kernel(
        seed=1,
        version=seed1_version,
    )

    seed_metrics = [seed0_metrics, seed1_metrics]
    prompt_hashes = {str(item["prompt_sha256"]) for item in seed_metrics}
    source_hashes = {
        str(item["runtime_fingerprint"]["source_manifest_sha256"])
        for item in seed_metrics
    }
    if len(prompt_hashes) != 1 or len(source_hashes) != 1:
        raise RuntimeError("early baseline seed artifacts do not share exact source and prompt hashes")
    early_mean = sum(float(item["mean_engine_score"]) for item in seed_metrics) / 2
    early_metrics = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "duck-reference-early-private-baseline",
        "reason": "conserve the weekly Kaggle GPU quota for the hidden 110-game run",
        "full_five_seed_gate_deferred": True,
        "fidelity_validation": {
            "version": fidelity_version,
            "metrics_path": str(fidelity_path),
            "mean_engine_score": fidelity_metrics["mean_engine_score"],
        },
        "seed_validation_versions": {
            "0": seed0_version,
            "1": seed1_version,
        },
        "seed_metric_paths": {
            "0": str(seed0_path),
            "1": str(seed1_path),
        },
        "seed_engine_scores": [
            seed0_metrics["mean_engine_score"],
            seed1_metrics["mean_engine_score"],
        ],
        "mean_engine_score": early_mean,
        "infrastructure_failures": [
            failure
            for item in seed_metrics
            for failure in item["infrastructure_failures"]
        ],
        "prompt_sha256": prompt_hashes.pop(),
        "source_manifest_sha256": source_hashes.pop(),
    }
    write_metrics(early_metrics, EARLY_BASELINE)
    write_reference_progress(
        {
            **early_metrics,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "stage": "early-private-baseline-validation-complete",
            "validation_kernel": VALIDATION_REF,
            "early_baseline_metrics_path": str(EARLY_BASELINE),
        }
    )

    previous_refs = ensure_daily_quota() if submit else set()
    generate_artifacts(validation_seed=None)
    submission_version = push_kernel(ROOT / "notebooks" / "submission")
    wait_kernel(SUBMISSION_REF)
    best = refresh_leaderboard_best()
    target = max(1.86, best) + 0.01
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": source_hash(),
        "config_hash": config_hash(),
        "source_manifest_sha256": source_manifest_hash(),
        "experiment": early_metrics["experiment"],
        "reason": early_metrics["reason"],
        "five_seed_gate_deferred": True,
        "validation_kernel": VALIDATION_REF,
        "fidelity_validation_version": fidelity_version,
        "fidelity_validation_score": fidelity_metrics["mean_engine_score"],
        "seed_validation_versions": early_metrics["seed_validation_versions"],
        "validation_scores": {
            "mean_engine_score": early_mean,
            "seed_engine_scores": early_metrics["seed_engine_scores"],
        },
        "submission_kernel": SUBMISSION_REF,
        "submission_version": submission_version,
        "leaderboard_best_before": best,
        "target": target,
        "result": "kernel-complete-not-submitted",
    }
    if submit:
        message = (
            f"kaggle-v3 early-private-baseline git={record['git_sha'][:12]} "
            f"src={record['source_manifest_sha256'][:12]} "
            f"cfg={record['config_hash'][:12]} "
            f"val2={early_mean:.4f} qwen3.6-27b-fp8/vllm0.19"
        )
        submission_reference = submit_exact_kernel(
            kernel_version=submission_version,
            message=message,
        )
        result = wait_submission(previous_refs=previous_refs)
        record.update(
            {
                "submission_reference": submission_reference,
                "submission_result": result,
                "result": "complete",
            }
        )
    append_ledger(record)
    write_reference_progress(
        {
            **record,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "stage": (
                "early-private-baseline-submission-complete"
                if submit
                else "early-private-baseline-kernel-complete"
            ),
        }
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        default=REFERENCE_MODE,
        help=(
            "artifact mode; Duck candidates use exactly one seed-0 validation"
        ),
    )
    parser.add_argument(
        "--seed0-only",
        action="store_true",
        help="publish and run exactly one seed-0 candidate validation",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="submit the exact gated kernel version (enforces the daily quota)",
    )
    parser.add_argument(
        "--experimental-public",
        action="store_true",
        help=(
            "run duck-retrodict public seeds for measurement even when its "
            "offline gate fails; this mode can never submit"
        ),
    )
    parser.add_argument(
        "--fidelity-only",
        action="store_true",
        help="publish and run only the unseeded 25-game fidelity stage",
    )
    parser.add_argument(
        "--fidelity-version",
        type=int,
        help="resume from a cached unseeded fidelity kernel version",
    )
    parser.add_argument(
        "--early-private-baseline",
        action="store_true",
        help="after seed 1, skip seeds 2-4 and launch the hidden baseline",
    )
    parser.add_argument(
        "--seed0-version",
        type=int,
        help="cached seed 0 kernel version for the early private baseline",
    )
    parser.add_argument(
        "--seed1-version",
        type=int,
        help="running or cached seed 1 kernel version for the early private baseline",
    )
    parser.add_argument(
        "--seed-versions",
        help="resume cached seed artifacts as 0:VERSION,1:VERSION,...,4:VERSION",
    )
    parser.add_argument(
        "--gpu-hours-remaining",
        type=float,
        help=(
            "verified weekly Kaggle GPU hours remaining; duck-poetiq and "
            "duck-portfolio/duck-retrodict require 4.5 hours for two public seeds, while "
            "the current hidden gateway rerun is quota-neutral"
        ),
    )
    args = parser.parse_args()
    seed_versions = None
    if args.seed_versions:
        seed_versions = {}
        for item in args.seed_versions.split(","):
            seed_text, version_text = item.split(":", 1)
            seed_versions[int(seed_text)] = int(version_text)
    if args.experimental_public and args.mode != RETRODICT_MODE:
        parser.error("--experimental-public requires --mode duck-retrodict")
    if args.mode == MEMORY_MODE:
        if not args.seed0_only:
            parser.error("duck-memory requires --seed0-only")
        if args.fidelity_only or args.early_private_baseline:
            parser.error(
                "duck-memory --seed0-only cannot run reference-only stages"
            )
        if args.fidelity_version is not None or args.seed_versions:
            parser.error(
                "duck-memory --seed0-only does not accept reference versions"
            )
        if args.seed1_version is not None:
            parser.error("duck-memory --seed0-only does not accept seed 1")
        result = execute_memory_seed0_candidate(
            submit=args.submit,
            resume_version=args.seed0_version,
            gpu_hours_remaining=args.gpu_hours_remaining,
        )
    elif args.mode == REASONING_MODE:
        if not args.seed0_only:
            parser.error("duck-reasoning requires --seed0-only")
        if args.fidelity_only or args.early_private_baseline:
            parser.error(
                "duck-reasoning --seed0-only cannot run reference-only stages"
            )
        if args.fidelity_version is not None or args.seed_versions:
            parser.error(
                "duck-reasoning --seed0-only does not accept reference versions"
            )
        if args.seed1_version is not None:
            parser.error("duck-reasoning --seed0-only does not accept seed 1")
        result = execute_reasoning_seed0_candidate(
            submit=args.submit,
            resume_version=args.seed0_version,
            gpu_hours_remaining=args.gpu_hours_remaining,
        )
    elif args.mode == ROBUST_MODE:
        if not args.seed0_only:
            parser.error("duck-robust requires --seed0-only")
        if args.submit:
            parser.error("duck-robust --seed0-only cannot submit")
        if args.fidelity_only or args.early_private_baseline:
            parser.error(
                "duck-robust --seed0-only cannot run reference-only stages"
            )
        if args.fidelity_version is not None or args.seed_versions:
            parser.error(
                "duck-robust --seed0-only does not accept reference versions"
            )
        if args.seed1_version is not None:
            parser.error("duck-robust --seed0-only does not accept seed 1")
        result = execute_robust_seed0_validation(
            resume_version=args.seed0_version,
        )
    elif args.mode in {CONTRACT_MODE, CONTRACT_REPAIR_MODE}:
        if not args.seed0_only:
            parser.error(f"{args.mode} requires --seed0-only")
        if args.submit:
            parser.error(f"{args.mode} is public-validation-only; hidden submission is not authorized")
        if args.fidelity_only or args.early_private_baseline:
            parser.error(f"{args.mode} --seed0-only cannot run reference-only stages")
        if args.fidelity_version is not None or args.seed_versions:
            parser.error(f"{args.mode} --seed0-only does not accept reference versions")
        if args.seed1_version is not None:
            parser.error(f"{args.mode} --seed0-only does not accept seed 1")
        result = execute_contract_seed0_validation(
            mode=args.mode,
            resume_version=args.seed0_version,
        )
    elif args.mode == AUDIT_MODE:
        if not args.seed0_only:
            parser.error("duck-audit requires --seed0-only")
        if args.submit:
            parser.error("duck-audit is public-validation-only; hidden submission is not authorized")
        if args.fidelity_only or args.early_private_baseline:
            parser.error("duck-audit --seed0-only cannot run reference-only stages")
        if args.fidelity_version is not None or args.seed_versions:
            parser.error("duck-audit --seed0-only does not accept reference versions")
        if args.seed1_version is not None:
            parser.error("duck-audit --seed0-only does not accept seed 1")
        result = execute_audit_seed0_validation(
            resume_version=args.seed0_version,
        )
    elif args.mode == INFORMATION_MODE:
        if not args.seed0_only:
            parser.error("duck-information requires --seed0-only")
        if args.submit:
            parser.error(
                "duck-information is public-validation-only; hidden submission is not authorized"
            )
        if args.fidelity_only or args.early_private_baseline:
            parser.error(
                "duck-information --seed0-only cannot run reference-only stages"
            )
        if args.fidelity_version is not None or args.seed_versions:
            parser.error(
                "duck-information --seed0-only does not accept reference versions"
            )
        if args.seed1_version is not None:
            parser.error("duck-information --seed0-only does not accept seed 1")
        result = execute_information_seed0_validation(
            resume_version=args.seed0_version,
        )
    elif args.mode == HIERARCHY_MODE:
        if not args.seed0_only:
            parser.error("duck-hierarchy requires --seed0-only")
        if args.submit:
            parser.error(
                "duck-hierarchy is public-validation-only; hidden submission is not authorized"
            )
        if args.fidelity_only or args.early_private_baseline:
            parser.error(
                "duck-hierarchy --seed0-only cannot run reference-only stages"
            )
        if args.fidelity_version is not None or args.seed_versions:
            parser.error(
                "duck-hierarchy --seed0-only does not accept reference versions"
            )
        if args.seed1_version is not None:
            parser.error("duck-hierarchy --seed0-only does not accept seed 1")
        result = execute_hierarchy_seed0_validation(
            resume_version=args.seed0_version,
        )
    elif args.mode == DIVERSITY_MODE:
        if not args.seed0_only:
            parser.error("duck-diversity requires --seed0-only")
        if args.submit:
            parser.error(
                "duck-diversity is public-validation-only; hidden submission is not authorized"
            )
        if args.fidelity_only or args.early_private_baseline:
            parser.error(
                "duck-diversity --seed0-only cannot run reference-only stages"
            )
        if args.fidelity_version is not None or args.seed_versions:
            parser.error(
                "duck-diversity --seed0-only does not accept reference versions"
            )
        if args.seed1_version is not None:
            parser.error("duck-diversity --seed0-only does not accept seed 1")
        result = execute_diversity_seed0_validation(
            resume_version=args.seed0_version,
        )
    elif args.mode == POETIQ_MODE:
        if args.seed0_only:
            parser.error("duck-poetiq uses independent public seeds 0 and 1")
        if args.fidelity_only or args.early_private_baseline:
            parser.error("duck-poetiq does not use the reference-only stages")
        if args.fidelity_version is not None or args.seed1_version is not None:
            parser.error("duck-poetiq resumes with --seed-versions=0:VERSION,1:VERSION")
        if seed_versions and any(seed not in {0, 1} for seed in seed_versions):
            parser.error("duck-poetiq accepts only seed versions 0 and 1")
        result = execute_poetiq_candidate(
            submit=args.submit,
            seed_versions=seed_versions,
            gpu_hours_remaining=args.gpu_hours_remaining,
        )
    elif args.mode == PORTFOLIO_MODE:
        if args.seed0_only:
            parser.error("duck-portfolio uses independent public seeds 0 and 1")
        if args.fidelity_only or args.early_private_baseline:
            parser.error("duck-portfolio does not use the reference-only stages")
        if args.fidelity_version is not None or args.seed1_version is not None:
            parser.error(
                "duck-portfolio resumes with --seed-versions=0:VERSION,1:VERSION"
            )
        if seed_versions and any(seed not in {0, 1} for seed in seed_versions):
            parser.error("duck-portfolio accepts only seed versions 0 and 1")
        result = execute_portfolio_candidate(
            submit=args.submit,
            seed_versions=seed_versions,
            gpu_hours_remaining=args.gpu_hours_remaining,
        )
    elif args.mode == RETRODICT_MODE:
        if args.seed0_only:
            parser.error("duck-retrodict uses independent public seeds 0 and 1")
        if args.fidelity_only or args.early_private_baseline:
            parser.error("duck-retrodict does not use the reference-only stages")
        if args.fidelity_version is not None or args.seed1_version is not None:
            parser.error(
                "duck-retrodict resumes with --seed-versions=0:VERSION,1:VERSION"
            )
        if seed_versions and any(seed not in {0, 1} for seed in seed_versions):
            parser.error("duck-retrodict accepts only seed versions 0 and 1")
        if args.experimental_public and args.submit:
            parser.error("--experimental-public cannot be combined with --submit")
        result = execute_retrodict_candidate(
            submit=args.submit,
            seed_versions=seed_versions,
            gpu_hours_remaining=args.gpu_hours_remaining,
            experimental_public=args.experimental_public,
        )
    elif args.seed0_only:
        parser.error("--seed0-only requires a Duck candidate mode")
    elif args.early_private_baseline:
        if (
            args.fidelity_version is None
            or args.seed0_version is None
            or args.seed1_version is None
        ):
            parser.error(
                "--early-private-baseline requires --fidelity-version, "
                "--seed0-version, and --seed1-version"
            )
        result = execute_early_private_baseline(
            submit=args.submit,
            fidelity_version=args.fidelity_version,
            seed0_version=args.seed0_version,
            seed1_version=args.seed1_version,
        )
    elif args.fidelity_only:
        result = execute_fidelity_validation(resume_version=args.fidelity_version)
    else:
        result = execute_pipeline(
            submit=args.submit,
            fidelity_version=args.fidelity_version,
            seed_versions=seed_versions,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
