from __future__ import annotations

from ouro3.metrics import aggregate_two_seed_runs
from ouro3.promotion import evaluate_poetiq_promotion


def _run(
    seed: int,
    score: float,
    levels: int,
    breadth: int,
    trimmed: float = 0.7,
) -> dict[str, object]:
    return {
        "mode": "duck-poetiq",
        "seed": seed,
        "game_count": 25,
        "mean_engine_score": score,
        "trimmed_mean_engine_score": trimmed,
        "total_completed_levels": levels,
        "nonzero_game_count": breadth,
        "infrastructure_failures": [],
        "telemetry": {"poetiq_stalled_yields": 0},
    }


def test_two_seed_aggregate_preserves_seed_attribution() -> None:
    aggregate = aggregate_two_seed_runs(
        [_run(1, 1.5, 11, 10), _run(0, 1.6, 19, 16)]
    )
    assert aggregate["seeds"] == [0, 1]
    assert aggregate["seed_engine_scores"] == [1.6, 1.5]
    assert aggregate["seed_completed_levels"] == [19, 11]
    assert aggregate["seed_nonzero_game_counts"] == [16, 10]


def test_poetiq_promotion_requires_score_breadth_and_trimmed_lift() -> None:
    passing = evaluate_poetiq_promotion(
        [_run(0, 1.6, 19, 16), _run(1, 1.5, 11, 10)],
        rehearsal_elapsed_s=30_000,
        soft_deadline_s=31_200,
    )
    assert passing.passed
    failing = evaluate_poetiq_promotion(
        [_run(0, 1.6, 18, 15, 0.5), _run(1, 1.5, 10, 9, 0.5)],
        rehearsal_elapsed_s=30_000,
        soft_deadline_s=31_200,
    )
    assert not failing.passed
    assert any("trimmed mean" in reason for reason in failing.reasons)
