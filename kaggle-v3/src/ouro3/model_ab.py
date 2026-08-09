"""Paired evaluation for retrodict actor-model challengers."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ModelABDecision:
    passed: bool
    reasons: tuple[str, ...]
    control_mean: float
    challenger_mean: float
    score_delta: float
    elapsed_delta_fraction: float
    generated_token_delta_fraction: float


def compare_retrodict_model_runs(
    control_runs: Iterable[dict[str, Any]],
    challenger_runs: Iterable[dict[str, Any]],
    *,
    maximum_score_regression: float = 0.02,
    minimum_score_lift: float = 0.10,
    minimum_elapsed_improvement: float = 0.15,
) -> ModelABDecision:
    """Promote a model only for a real score lift or safe throughput gain."""

    controls = sorted(control_runs, key=lambda run: int(run.get("seed", -1)))
    challengers = sorted(
        challenger_runs,
        key=lambda run: int(run.get("seed", -1)),
    )
    reasons: list[str] = []
    control_seeds = [int(run.get("seed", -1)) for run in controls]
    challenger_seeds = [int(run.get("seed", -1)) for run in challengers]
    if control_seeds != [0, 1] or challenger_seeds != [0, 1]:
        reasons.append("model A/B requires independent seeds 0 and 1 in both arms")
    for label, runs in (("control", controls), ("challenger", challengers)):
        if any(run.get("mode") != "duck-retrodict" for run in runs):
            reasons.append(f"{label} contains a non-retrodict artifact")
        if any(int(run.get("game_count", 0)) != 25 for run in runs):
            reasons.append(f"{label} must cover all 25 public games per seed")
        failures = [
            failure
            for run in runs
            for failure in list(run.get("infrastructure_failures") or [])
        ]
        if failures:
            reasons.append(
                f"{label} infrastructure failures: "
                f"{sorted(set(map(str, failures)))}"
            )
    control_games = _game_keys(controls)
    challenger_games = _game_keys(challengers)
    if len(control_games) != 50 or len(challenger_games) != 50:
        reasons.append("both arms must expose all 50 seed/game keys")
    elif control_games != challenger_games:
        reasons.append("control and challenger do not contain identical game keys")
    control_runtime = _runtime_signatures(controls)
    challenger_runtime = _runtime_signatures(challengers)
    if bool(control_runtime) != bool(challenger_runtime):
        reasons.append("one model arm is missing runtime fingerprints")
    elif control_runtime and control_runtime != challenger_runtime:
        reasons.append("control and challenger runtime signatures differ")

    control_mean = _mean(controls, "mean_engine_score")
    challenger_mean = _mean(challengers, "mean_engine_score")
    score_delta = challenger_mean - control_mean
    control_elapsed = _mean(controls, "elapsed_seconds")
    challenger_elapsed = _mean(challengers, "elapsed_seconds")
    elapsed_delta = _relative_improvement(control_elapsed, challenger_elapsed)
    control_tokens = _mean(controls, "total_generated_tokens")
    challenger_tokens = _mean(challengers, "total_generated_tokens")
    token_delta = _relative_improvement(control_tokens, challenger_tokens)

    if score_delta < -maximum_score_regression:
        reasons.append(
            f"challenger score regression {score_delta:.4f} exceeds "
            f"{maximum_score_regression:.4f}"
        )
    if score_delta < minimum_score_lift and elapsed_delta < minimum_elapsed_improvement:
        reasons.append(
            "challenger provides neither the required score lift nor elapsed-time gain"
        )
    return ModelABDecision(
        passed=not reasons,
        reasons=tuple(reasons),
        control_mean=control_mean,
        challenger_mean=challenger_mean,
        score_delta=score_delta,
        elapsed_delta_fraction=elapsed_delta,
        generated_token_delta_fraction=token_delta,
    )


def decision_payload(decision: ModelABDecision) -> dict[str, Any]:
    return asdict(decision)


def _mean(runs: list[dict[str, Any]], key: str) -> float:
    values = [float(run.get(key, 0.0)) for run in runs]
    return statistics.fmean(values) if values else 0.0


def _relative_improvement(control: float, challenger: float) -> float:
    return (control - challenger) / control if control > 0 else 0.0


def _game_keys(runs: list[dict[str, Any]]) -> set[tuple[int, str]]:
    return {
        (int(run.get("seed", -1)), str(game.get("game_id", "")))
        for run in runs
        for game in list(run.get("games") or [])
    }


def _runtime_signatures(runs: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    fields = (
        "gpu",
        "vllm",
        "torch",
        "flashinfer",
        "concurrency",
        "per_game_cap_s",
        "active_context",
    )
    return {
        tuple(dict(run.get("runtime_fingerprint") or {}).get(field) for field in fields)
        for run in runs
        if run.get("runtime_fingerprint")
    }
