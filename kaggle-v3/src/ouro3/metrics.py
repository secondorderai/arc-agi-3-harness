"""JSON metrics and infrastructure health summaries."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


INFRASTRUCTURE_FAILURE_STATES = {"crashed"}
INFRASTRUCTURE_FAILURE_MARKERS = (
    "model-load",
    "model load",
    "out of memory",
    "oom",
    "sandbox",
    "transport",
    "session crash",
)


@dataclass(frozen=True)
class GameMetric:
    game_id: str
    state: str
    levels_completed: int
    number_of_levels: int
    actions: int
    generated_tokens: int
    uncached_input_tokens: int
    wallclock_seconds: float
    final_score: float
    solver_note: str = ""
    telemetry: dict[str, int] | None = None
    trace: tuple[dict[str, Any], ...] = ()


def summarize_runs(
    runs: Iterable[Any],
    *,
    experiment: str,
    seed: int | None,
    config_hash: str,
    elapsed_seconds: float,
    mode: str = "",
    prompt_sha256: str = "",
    runtime_fingerprint: dict[str, Any] | None = None,
    aggregate_memory_telemetry_only: bool = False,
) -> dict[str, Any]:
    run_list = list(runs)
    rows = [
        GameMetric(
            game_id=str(run.game_id),
            state=str(run.state),
            levels_completed=int(run.levels_completed),
            number_of_levels=int(run.number_of_levels),
            actions=sum(int(value) for value in run.actions_per_level),
            generated_tokens=(
                sum(int(record.generated_tokens) for record in run.history)
                + int(getattr(run, "final_generated_tokens", 0) or 0)
            ),
            uncached_input_tokens=(
                sum(int(record.uncached_input_tokens) for record in run.history)
                + int(getattr(run, "final_uncached_input_tokens", 0) or 0)
            ),
            wallclock_seconds=round(
                float(getattr(run, "final_wallclock_seconds", 0.0) or 0.0),
                3,
            ),
            final_score=float(run.final_score or 0.0),
            solver_note=str(run.solver_note or ""),
            telemetry={
                str(key): int(value)
                for key, value in (
                    getattr(run, "solver_telemetry", {}) or {}
                ).items()
            },
            trace=tuple(
                {
                    "action": record.action.id.name,
                    "data": dict(record.action.data),
                    "generated_tokens": int(record.generated_tokens),
                    "wallclock_seconds": round(float(record.wallclock_seconds), 3),
                }
                for record in run.history
            ),
        )
        for run in run_list
    ]
    game_payloads = [asdict(row) for row in rows]
    if mode == "duck-memory" and aggregate_memory_telemetry_only:
        for payload in game_payloads:
            payload.pop("telemetry", None)
    recovery_diagnostics: dict[str, Any] = {}
    memory_diagnostics: dict[str, Any] = {}
    reasoning_diagnostics: dict[str, Any] = {}
    deliberate_diagnostics: dict[str, Any] = {}
    contract_diagnostics: dict[str, Any] = {}
    audit_diagnostics: dict[str, Any] = {}
    hierarchy_diagnostics: dict[str, Any] = {}
    diversity_diagnostics: dict[str, Any] = {}
    poetiq_diagnostics: dict[str, Any] = {}
    portfolio_diagnostics: dict[str, Any] = {}
    retrodict_diagnostics: dict[str, Any] = {}
    if mode in {
        "duck-robust",
        "duck-memory",
        "duck-reasoning",
        "duck-deliberate",
        "duck-contract",
        "duck-contract-repair",
        "duck-audit",
        "duck-information",
        "duck-hierarchy",
        "duck-diversity",
        "duck-poetiq",
        "duck-portfolio",
        "duck-retrodict",
    }:
        for run, payload in zip(run_list, game_payloads):
            diagnostics = getattr(run, "solver_diagnostics", None)
            if isinstance(diagnostics, dict):
                if mode == "duck-robust":
                    payload["recovery_diagnostics"] = diagnostics
                    recovery_diagnostics[str(run.game_id)] = diagnostics
                elif mode == "duck-memory" and not aggregate_memory_telemetry_only:
                    payload["memory_diagnostics"] = diagnostics
                    memory_diagnostics[str(run.game_id)] = diagnostics
                elif mode == "duck-reasoning":
                    payload["reasoning_diagnostics"] = diagnostics
                    reasoning_diagnostics[str(run.game_id)] = diagnostics
                elif mode == "duck-deliberate":
                    payload["deliberate_diagnostics"] = diagnostics
                    deliberate_diagnostics[str(run.game_id)] = diagnostics
                elif mode == "duck-contract":
                    payload["contract_diagnostics"] = diagnostics
                    contract_diagnostics[str(run.game_id)] = diagnostics
                elif mode == "duck-contract-repair":
                    payload["contract_diagnostics"] = diagnostics
                    contract_diagnostics[str(run.game_id)] = diagnostics
                elif mode == "duck-audit":
                    payload["audit_diagnostics"] = diagnostics
                    audit_diagnostics[str(run.game_id)] = diagnostics
                elif mode == "duck-information":
                    payload["information_diagnostics"] = diagnostics
                elif mode == "duck-hierarchy":
                    payload["hierarchy_diagnostics"] = diagnostics
                    hierarchy_diagnostics[str(run.game_id)] = diagnostics
                elif mode == "duck-diversity":
                    payload["diversity_diagnostics"] = diagnostics
                    diversity_diagnostics[str(run.game_id)] = diagnostics
                elif mode == "duck-poetiq":
                    payload["poetiq_diagnostics"] = diagnostics
                    poetiq_diagnostics[str(run.game_id)] = diagnostics
                elif mode == "duck-portfolio":
                    payload["portfolio_diagnostics"] = diagnostics
                    portfolio_diagnostics[str(run.game_id)] = diagnostics
                elif mode == "duck-retrodict":
                    payload["retrodict_diagnostics"] = diagnostics
                    retrodict_diagnostics[str(run.game_id)] = diagnostics
    levels = [row.levels_completed for row in rows]
    final_scores = [row.final_score for row in rows]
    ordered_scores = sorted(final_scores, reverse=True)
    total_score = sum(final_scores)
    trimmed_scores = ordered_scores[3:] if len(ordered_scores) > 3 else []
    failures = [
        row.game_id
        for row in rows
        if row.state in INFRASTRUCTURE_FAILURE_STATES
        or any(marker in row.solver_note.lower() for marker in INFRASTRUCTURE_FAILURE_MARKERS)
    ]
    telemetry: dict[str, int] = {}
    for row in rows:
        for key, value in (row.telemetry or {}).items():
            telemetry[key] = telemetry.get(key, 0) + int(value)
    return {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "mode": mode,
        "seed": seed,
        "config_hash": config_hash,
        "prompt_sha256": prompt_sha256,
        "runtime_fingerprint": dict(runtime_fingerprint or {}),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "game_count": len(rows),
        "mean_completed_levels": statistics.fmean(levels) if levels else 0.0,
        "median_completed_levels": statistics.median(levels) if levels else 0.0,
        "total_completed_levels": sum(levels),
        "mean_engine_score": statistics.fmean(final_scores) if final_scores else 0.0,
        "nonzero_game_count": sum(score > 0.0 for score in final_scores),
        "top_three_score_share": (
            sum(ordered_scores[:3]) / total_score if total_score > 0 else 0.0
        ),
        "trimmed_mean_engine_score": (
            statistics.fmean(trimmed_scores) if trimmed_scores else 0.0
        ),
        "total_actions": sum(row.actions for row in rows),
        "total_generated_tokens": sum(row.generated_tokens for row in rows),
        "total_uncached_input_tokens": sum(
            row.uncached_input_tokens for row in rows
        ),
        "telemetry": telemetry,
        "infrastructure_failures": failures,
        "games": game_payloads,
        **(
            {"recovery_diagnostics": recovery_diagnostics}
            if mode == "duck-robust"
            else {}
        ),
        **(
            {"memory_diagnostics": memory_diagnostics}
            if mode == "duck-memory" and not aggregate_memory_telemetry_only
            else {}
        ),
        **(
            {
                "memory_telemetry_scope": (
                    "aggregate-only"
                    if aggregate_memory_telemetry_only
                    else "validation-full-trace"
                )
            }
            if mode == "duck-memory"
            else {}
        ),
        **(
            {"reasoning_diagnostics": reasoning_diagnostics}
            if mode == "duck-reasoning"
            else {}
        ),
        **(
            {"deliberate_diagnostics": deliberate_diagnostics}
            if mode == "duck-deliberate"
            else {}
        ),
        **(
            {"contract_diagnostics": contract_diagnostics}
            if mode == "duck-contract"
            else {}
        ),
        **(
            {"contract_diagnostics": contract_diagnostics}
            if mode == "duck-contract-repair"
            else {}
        ),
        **(
            {"audit_diagnostics": audit_diagnostics}
            if mode == "duck-audit"
            else {}
        ),
        **(
            {
                "information_diagnostics": {
                    str(run.game_id): getattr(run, "solver_diagnostics", {})
                    for run in run_list
                    if isinstance(getattr(run, "solver_diagnostics", None), dict)
                }
            }
            if mode == "duck-information"
            else {}
        ),
        **(
            {"hierarchy_diagnostics": hierarchy_diagnostics}
            if mode == "duck-hierarchy"
            else {}
        ),
        **(
            {"diversity_diagnostics": diversity_diagnostics}
            if mode == "duck-diversity"
            else {}
        ),
        **(
            {"poetiq_diagnostics": poetiq_diagnostics}
            if mode == "duck-poetiq"
            else {}
        ),
        **(
            {"portfolio_diagnostics": portfolio_diagnostics}
            if mode == "duck-portfolio"
            else {}
        ),
        **(
            {"retrodict_diagnostics": retrodict_diagnostics}
            if mode == "duck-retrodict"
            else {}
        ),
    }


def aggregate_two_seed_runs(
    seed_runs: Iterable[dict[str, Any]],
    *,
    expected_game_count: int = 25,
    expected_mode: str = "duck-poetiq",
) -> dict[str, Any]:
    """Aggregate two independent public artifacts without merging sessions."""

    runs = list(seed_runs)
    seeds = sorted(int(run.get("seed")) for run in runs)
    if len(runs) != 2 or seeds != [0, 1]:
        raise ValueError("two-seed aggregation requires independent seeds 0 and 1")
    if any(int(run.get("game_count", 0)) != expected_game_count for run in runs):
        raise ValueError(f"every seed must contain {expected_game_count} games")
    if any(run.get("mode") != expected_mode for run in runs):
        raise ValueError(f"all seed artifacts must use {expected_mode} mode")
    failures = sorted(
        {
            str(value)
            for run in runs
            for value in run.get("infrastructure_failures", [])
        }
    )
    return {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": f"{expected_mode}-seeds-0-1",
        "mode": expected_mode,
        "seeds": [0, 1],
        "game_count_per_seed": expected_game_count,
        "total_session_count": expected_game_count * 2,
        "mean_engine_score": statistics.fmean(
            float(run.get("mean_engine_score", 0.0)) for run in runs
        ),
        "trimmed_mean_engine_score": statistics.fmean(
            float(run.get("trimmed_mean_engine_score", 0.0)) for run in runs
        ),
        "seed_engine_scores": [
            float(next(run for run in runs if int(run.get("seed")) == seed).get("mean_engine_score", 0.0))
            for seed in [0, 1]
        ],
        "seed_completed_levels": [
            int(next(run for run in runs if int(run.get("seed")) == seed).get("total_completed_levels", 0))
            for seed in [0, 1]
        ],
        "seed_nonzero_game_counts": [
            int(next(run for run in runs if int(run.get("seed")) == seed).get("nonzero_game_count", 0))
            for seed in [0, 1]
        ],
        "infrastructure_failures": failures,
        "seed_runs": runs,
    }


def aggregate_seed_runs(
    seed_runs: Iterable[dict[str, Any]],
    *,
    expected_game_count: int = 25,
) -> dict[str, Any]:
    """Aggregate five independent Duck kernel artifacts without merging sessions."""

    runs = list(seed_runs)
    seeds = [run.get("seed") for run in runs]
    if len(runs) != 5 or sorted(seeds) != [0, 1, 2, 3, 4]:
        raise ValueError("reference aggregation requires independent seeds 0-4")
    if any(int(run.get("game_count", 0)) != expected_game_count for run in runs):
        raise ValueError(
            f"every reference seed must contain {expected_game_count} games"
        )
    if any(run.get("mode") != "duck-reference" for run in runs):
        raise ValueError("all seed artifacts must use duck-reference mode")

    failures = [
        str(value)
        for run in runs
        for value in run.get("infrastructure_failures", [])
    ]
    scores = [float(run.get("mean_engine_score", 0.0)) for run in runs]
    level_means = [
        float(run.get("mean_completed_levels", 0.0)) for run in runs
    ]
    level_totals = [
        int(run.get("total_completed_levels", 0)) for run in runs
    ]
    elapsed = [float(run.get("elapsed_seconds", 0.0)) for run in runs]
    fingerprints = set()
    for run in runs:
        fingerprint = dict(run.get("runtime_fingerprint", {}))
        fingerprint.pop("seed", None)
        fingerprints.add(json.dumps(fingerprint, sort_keys=True))
    return {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "duck-reference-seeds-0-4",
        "mode": "duck-reference",
        "seeds": [0, 1, 2, 3, 4],
        "game_count_per_seed": expected_game_count,
        "total_session_count": expected_game_count * len(runs),
        "mean_engine_score": statistics.fmean(scores),
        "seed_engine_scores": scores,
        "mean_completed_levels": statistics.fmean(level_means),
        "median_completed_level_total": statistics.median(level_totals),
        "total_elapsed_seconds": sum(elapsed),
        "max_kernel_elapsed_seconds": max(elapsed),
        "infrastructure_failures": sorted(set(failures)),
        "runtime_fingerprint_consistent": len(fingerprints) == 1,
        "prompt_sha256": runs[0].get("prompt_sha256", ""),
        "prompt_fingerprint_consistent": len(
            {str(run.get("prompt_sha256", "")) for run in runs}
        )
        == 1,
        "config_hashes": {
            str(run["seed"]): str(run.get("config_hash", "")) for run in runs
        },
        "seed_runs": runs,
    }


def aggregate_metric_files(paths: Iterable[Path]) -> dict[str, Any]:
    return aggregate_seed_runs(
        json.loads(path.read_text(encoding="utf-8")) for path in paths
    )


def write_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
