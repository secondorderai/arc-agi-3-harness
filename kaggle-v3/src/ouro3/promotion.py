"""Multi-seed promotion gates for controlled v3 experiments."""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class PromotionDecision:
    passed: bool
    reasons: tuple[str, ...]
    reference_mean: float
    candidate_mean: float
    reference_median_total: float
    candidate_median_total: float


@dataclass(frozen=True)
class PoetiqPromotionDecision:
    passed: bool
    reasons: tuple[str, ...]
    baseline_mean: float
    candidate_mean: float
    baseline_trimmed_mean: float
    candidate_trimmed_mean: float


@dataclass(frozen=True)
class PortfolioPromotionDecision:
    passed: bool
    reasons: tuple[str, ...]
    candidate_mean: float
    candidate_trimmed_mean: float
    seed0_score: float
    seed0_trimmed_mean: float


@dataclass(frozen=True)
class RetrodictOfflinePromotionDecision:
    passed: bool
    reasons: tuple[str, ...]
    precision: float
    coverage: float
    latency_p95_ms: float


@dataclass(frozen=True)
class RetrodictPromotionDecision:
    passed: bool
    reasons: tuple[str, ...]
    offline_passed: bool
    candidate_mean: float
    candidate_trimmed_mean: float


def evaluate_promotion(
    reference_runs: Iterable[dict[str, Any]],
    candidate_runs: Iterable[dict[str, Any]],
    *,
    rehearsal_elapsed_s: float,
    soft_deadline_s: float,
    reference_minimum: float = 1.20,
) -> PromotionDecision:
    references = list(reference_runs)
    candidates = list(candidate_runs)
    reasons: list[str] = []
    if len(references) != 5 or len(candidates) != 5:
        reasons.append("TEST gate requires exactly five seeds for reference and candidate")

    reference_scores = [float(run.get("mean_engine_score", 0.0)) for run in references]
    candidate_scores = [float(run.get("mean_engine_score", 0.0)) for run in candidates]
    reference_mean = statistics.fmean(reference_scores) if reference_scores else 0.0
    candidate_mean = statistics.fmean(candidate_scores) if candidate_scores else 0.0
    reference_totals = [float(run.get("total_completed_levels", 0.0)) for run in references]
    candidate_totals = [float(run.get("total_completed_levels", 0.0)) for run in candidates]
    reference_median_total = statistics.median(reference_totals) if reference_totals else 0.0
    candidate_median_total = statistics.median(candidate_totals) if candidate_totals else 0.0

    if reference_mean < reference_minimum:
        reasons.append(
            f"reference engine-score mean {reference_mean:.3f} is below required "
            f"{reference_minimum:.2f}"
        )
    if candidate_mean - reference_mean < 0.10:
        reasons.append("candidate engine-score mean improvement is below 0.10")
    if candidate_median_total - reference_median_total < 1:
        reasons.append("candidate median completed-level total did not improve by at least one")
    if _has_median_game_regression(references, candidates):
        reasons.append("at least one TEST game lost a median completed level")
    failures = [
        value
        for run in [*references, *candidates]
        for value in run.get("infrastructure_failures", [])
    ]
    if failures:
        reasons.append(f"infrastructure failures present: {sorted(set(map(str, failures)))}")
    if rehearsal_elapsed_s >= soft_deadline_s:
        reasons.append("110-game rehearsal projects completion after the soft deadline")

    return PromotionDecision(
        passed=not reasons,
        reasons=tuple(reasons),
        reference_mean=reference_mean,
        candidate_mean=candidate_mean,
        reference_median_total=reference_median_total,
        candidate_median_total=candidate_median_total,
    )


def evaluate_poetiq_promotion(
    candidate_runs: Iterable[dict[str, Any]],
    *,
    rehearsal_elapsed_s: float,
    soft_deadline_s: float,
    baseline_runs: Iterable[dict[str, Any]] | None = None,
    baseline_mean: float = 1.3081755365847543,
    baseline_trimmed_mean: float = 0.606051301975289,
    minimum_mean_lift: float = 0.10,
    minimum_levels: tuple[int, int] = (18, 10),
    minimum_nonzero_games: tuple[int, int] = (15, 9),
) -> PoetiqPromotionDecision:
    """Gate the quota-aware two-seed composite Poetiq experiment."""

    candidates = sorted(list(candidate_runs), key=lambda run: int(run.get("seed", -1)))
    reasons: list[str] = []
    if [int(run.get("seed", -1)) for run in candidates] != [0, 1]:
        reasons.append("Poetiq gate requires independent seeds 0 and 1")
    if any(int(run.get("game_count", 0)) != 25 for run in candidates):
        reasons.append("each Poetiq seed must contain all 25 public games")

    if baseline_runs is not None:
        baselines = sorted(list(baseline_runs), key=lambda run: int(run.get("seed", -1)))
        if len(baselines) == 2 and [int(run.get("seed", -1)) for run in baselines] == [0, 1]:
            baseline_mean = statistics.fmean(
                float(run.get("mean_engine_score", 0.0)) for run in baselines
            )
            baseline_trimmed_mean = statistics.fmean(
                float(run.get("trimmed_mean_engine_score", 0.0)) for run in baselines
            )

    candidate_scores = [float(run.get("mean_engine_score", 0.0)) for run in candidates]
    candidate_trimmed = [
        float(run.get("trimmed_mean_engine_score", 0.0)) for run in candidates
    ]
    candidate_mean = statistics.fmean(candidate_scores) if candidate_scores else 0.0
    candidate_trimmed_mean = statistics.fmean(candidate_trimmed) if candidate_trimmed else 0.0

    if candidate_mean < baseline_mean + minimum_mean_lift:
        reasons.append(
            f"two-seed mean {candidate_mean:.4f} is below required {baseline_mean + minimum_mean_lift:.4f}"
        )
    if candidate_trimmed_mean <= baseline_trimmed_mean:
        reasons.append(
            f"trimmed mean {candidate_trimmed_mean:.4f} does not exceed baseline {baseline_trimmed_mean:.4f}"
        )
    for index, run in enumerate(candidates[:2]):
        levels = int(run.get("total_completed_levels", 0))
        breadth = int(run.get("nonzero_game_count", 0))
        if levels < minimum_levels[index]:
            reasons.append(f"seed {index} completed levels {levels} < {minimum_levels[index]}")
        if breadth < minimum_nonzero_games[index]:
            reasons.append(f"seed {index} nonzero games {breadth} < {minimum_nonzero_games[index]}")
    failures = [
        value
        for run in candidates
        for value in run.get("infrastructure_failures", [])
    ]
    if failures:
        reasons.append(f"infrastructure failures present: {sorted(set(map(str, failures)))}")
    if any(
        int(run.get("telemetry", {}).get("poetiq_stalled_yields", 0)) < 0
        for run in candidates
    ):
        reasons.append("invalid Poetiq telemetry")
    if rehearsal_elapsed_s >= soft_deadline_s:
        reasons.append("110-game rehearsal projects completion after the soft deadline")

    return PoetiqPromotionDecision(
        passed=not reasons,
        reasons=tuple(reasons),
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        baseline_trimmed_mean=baseline_trimmed_mean,
        candidate_trimmed_mean=candidate_trimmed_mean,
    )


def evaluate_portfolio_promotion(
    candidate_runs: Iterable[dict[str, Any]],
    *,
    rehearsal_elapsed_s: float,
    soft_deadline_s: float,
    seed0_minimum_score: float = 2.5631,
    seed0_minimum_trimmed_mean: float = 0.9370812,
    two_seed_minimum_mean: float = 1.4081755,
    two_seed_minimum_trimmed_mean: float = 0.6060513,
) -> PortfolioPromotionDecision:
    """Apply the exact public-to-hidden gate for duck-portfolio-v1."""

    runs = sorted(list(candidate_runs), key=lambda run: int(run.get("seed", -1)))
    reasons: list[str] = []
    if [int(run.get("seed", -1)) for run in runs] != [0, 1]:
        reasons.append("portfolio gate requires independent seeds 0 and 1")
    if any(int(run.get("game_count", 0)) != 25 for run in runs):
        reasons.append("each portfolio seed must contain all 25 public games")

    seed0 = runs[0] if runs else {}
    seed0_score = float(seed0.get("mean_engine_score", 0.0))
    seed0_trimmed = float(seed0.get("trimmed_mean_engine_score", 0.0))
    if seed0_score < seed0_minimum_score:
        reasons.append(
            f"seed 0 score {seed0_score:.4f} is below required {seed0_minimum_score:.4f}"
        )
    if seed0_trimmed <= seed0_minimum_trimmed_mean:
        reasons.append(
            f"seed 0 trimmed mean {seed0_trimmed:.4f} does not exceed "
            f"{seed0_minimum_trimmed_mean:.4f}"
        )

    scores = [float(run.get("mean_engine_score", 0.0)) for run in runs]
    trimmed = [float(run.get("trimmed_mean_engine_score", 0.0)) for run in runs]
    candidate_mean = statistics.fmean(scores) if scores else 0.0
    candidate_trimmed_mean = statistics.fmean(trimmed) if trimmed else 0.0
    if candidate_mean < two_seed_minimum_mean:
        reasons.append(
            f"two-seed mean {candidate_mean:.4f} is below required "
            f"{two_seed_minimum_mean:.4f}"
        )
    if candidate_trimmed_mean <= two_seed_minimum_trimmed_mean:
        reasons.append(
            f"two-seed trimmed mean {candidate_trimmed_mean:.4f} does not exceed "
            f"{two_seed_minimum_trimmed_mean:.4f}"
        )

    for index, (minimum_levels, minimum_breadth) in enumerate(((18, 15), (10, 9))):
        if index >= len(runs):
            continue
        levels = int(runs[index].get("total_completed_levels", 0))
        breadth = int(runs[index].get("nonzero_game_count", 0))
        if levels < minimum_levels:
            reasons.append(
                f"seed {index} completed levels {levels} < {minimum_levels}"
            )
        if breadth < minimum_breadth:
            reasons.append(
                f"seed {index} nonzero games {breadth} < {minimum_breadth}"
            )

    failures = [
        failure
        for run in runs
        for failure in list(run.get("infrastructure_failures") or [])
    ]
    if failures:
        reasons.append(
            f"infrastructure failures present: {sorted(set(map(str, failures)))}"
        )
    if rehearsal_elapsed_s >= soft_deadline_s:
        reasons.append("110-game rehearsal projects completion after the soft deadline")

    return PortfolioPromotionDecision(
        passed=not reasons,
        reasons=tuple(reasons),
        candidate_mean=candidate_mean,
        candidate_trimmed_mean=candidate_trimmed_mean,
        seed0_score=seed0_score,
        seed0_trimmed_mean=seed0_trimmed,
    )


def evaluate_retrodict_offline_promotion(
    report: dict[str, Any],
    *,
    minimum_precision: float = 0.95,
    minimum_coverage: float = 0.60,
    maximum_p95_ms: float = 5.0,
) -> RetrodictOfflinePromotionDecision:
    """Fail closed before spending any public or hidden gameplay budget."""

    typed = dict(report.get("typed") or {})
    precision = float(typed.get("precision", 0.0))
    coverage = float(typed.get("coverage", 0.0))
    latency = float(typed.get("latency_p95_ms", float("inf")))
    reasons: list[str] = []
    if int(report.get("holdout_count", 0)) < 10:
        reasons.append("offline holdout must contain at least ten transitions")
    if precision < minimum_precision:
        reasons.append(
            f"typed precision {precision:.4f} is below {minimum_precision:.4f}"
        )
    if coverage < minimum_coverage:
        reasons.append(
            f"typed coverage {coverage:.4f} is below {minimum_coverage:.4f}"
        )
    if latency > maximum_p95_ms:
        reasons.append(
            f"typed p95 latency {latency:.3f}ms exceeds {maximum_p95_ms:.3f}ms"
        )
    baseline = report.get("generated_python")
    if isinstance(baseline, dict):
        baseline_precision = float(baseline.get("precision", 0.0))
        if precision < baseline_precision:
            reasons.append("typed precision regresses generated-Python precision")
    return RetrodictOfflinePromotionDecision(
        passed=not reasons,
        reasons=tuple(reasons),
        precision=precision,
        coverage=coverage,
        latency_p95_ms=latency,
    )


def evaluate_retrodict_promotion(
    candidate_runs: Iterable[dict[str, Any]],
    *,
    offline_report: dict[str, Any],
    rehearsal_elapsed_s: float,
    soft_deadline_s: float,
    leaderboard_target: float = 1.86,
    minimum_margin: float = 0.01,
    minimum_trimmed_mean: float = 1.0,
) -> RetrodictPromotionDecision:
    """Require offline safety, two-seed breadth and a leaderboard-beating mean."""

    offline = evaluate_retrodict_offline_promotion(offline_report)
    runs = sorted(list(candidate_runs), key=lambda run: int(run.get("seed", -1)))
    reasons = list(offline.reasons)
    if [int(run.get("seed", -1)) for run in runs] != [0, 1]:
        reasons.append("retrodict gate requires independent seeds 0 and 1")
    if any(int(run.get("game_count", 0)) != 25 for run in runs):
        reasons.append("each retrodict seed must contain all 25 public games")
    scores = [float(run.get("mean_engine_score", 0.0)) for run in runs]
    trimmed = [float(run.get("trimmed_mean_engine_score", 0.0)) for run in runs]
    candidate_mean = statistics.fmean(scores) if scores else 0.0
    candidate_trimmed = statistics.fmean(trimmed) if trimmed else 0.0
    required_mean = leaderboard_target + minimum_margin
    if candidate_mean < required_mean:
        reasons.append(
            f"two-seed mean {candidate_mean:.4f} is below winning target "
            f"{required_mean:.4f}"
        )
    if candidate_trimmed < minimum_trimmed_mean:
        reasons.append(
            f"two-seed trimmed mean {candidate_trimmed:.4f} is below "
            f"{minimum_trimmed_mean:.4f}"
        )
    for seed, run in enumerate(runs[:2]):
        if int(run.get("nonzero_game_count", 0)) < 12:
            reasons.append(f"seed {seed} has fewer than 12 nonzero games")
    failures = [
        failure
        for run in runs
        for failure in list(run.get("infrastructure_failures") or [])
    ]
    if failures:
        reasons.append(
            f"infrastructure failures present: {sorted(set(map(str, failures)))}"
        )
    if rehearsal_elapsed_s >= soft_deadline_s:
        reasons.append("110-game rehearsal projects completion after the soft deadline")
    return RetrodictPromotionDecision(
        passed=not reasons,
        reasons=tuple(reasons),
        offline_passed=offline.passed,
        candidate_mean=candidate_mean,
        candidate_trimmed_mean=candidate_trimmed,
    )


def _has_median_game_regression(
    references: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> bool:
    reference_levels = _levels_by_game(references)
    candidate_levels = _levels_by_game(candidates)
    for game_id in sorted(set(reference_levels) & set(candidate_levels)):
        if statistics.median(candidate_levels[game_id]) < statistics.median(
            reference_levels[game_id]
        ):
            return True
    return False


def _levels_by_game(runs: list[dict[str, Any]]) -> dict[str, list[int]]:
    values: dict[str, list[int]] = defaultdict(list)
    for run in runs:
        for game in run.get("games", []):
            values[str(game.get("game_id"))].append(int(game.get("levels_completed", 0)))
    return values
