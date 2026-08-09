from __future__ import annotations

import pytest

from ouro3.metrics import aggregate_seed_runs


PRIOR_V4_SEED_SCORES = [
    0.2441408519813753,
    0.3704228586951751,
    0.4993197252289991,
    0.3083988198887378,
    0.5143624865121008,
]


def _seed_run(seed: int, score: float) -> dict:
    return {
        "mode": "duck-reference",
        "seed": seed,
        "game_count": 25,
        "mean_engine_score": score,
        "mean_completed_levels": 0.2,
        "total_completed_levels": 5,
        "elapsed_seconds": 7_900,
        "infrastructure_failures": [],
        "prompt_sha256": "a" * 64,
        "config_hash": f"{seed}" * 64,
        "runtime_fingerprint": {
            "mode": "duck-reference",
            "seed": seed,
            "prompt_sha256": "a" * 64,
        },
        "games": [],
    }


def test_regression_seed_four_score_is_not_the_prior_five_seed_mean() -> None:
    aggregate = aggregate_seed_runs(
        _seed_run(seed, score)
        for seed, score in enumerate(PRIOR_V4_SEED_SCORES)
    )

    assert PRIOR_V4_SEED_SCORES[4] == 0.5143624865121008
    assert aggregate["mean_engine_score"] == pytest.approx(
        0.3873289484612774
    )
    assert aggregate["mean_engine_score"] != PRIOR_V4_SEED_SCORES[4]


def test_reference_aggregation_requires_five_independent_25_game_artifacts() -> None:
    with pytest.raises(ValueError, match="seeds 0-4"):
        aggregate_seed_runs([_seed_run(0, 1.2)] * 5)

    invalid = [_seed_run(seed, 1.2) for seed in range(5)]
    invalid[3]["game_count"] = 125
    with pytest.raises(ValueError, match="25 games"):
        aggregate_seed_runs(invalid)
