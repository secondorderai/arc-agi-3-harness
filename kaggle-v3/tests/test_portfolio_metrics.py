from __future__ import annotations

from ouro3.metrics import aggregate_two_seed_runs
from ouro3.promotion import evaluate_portfolio_promotion
from ouro3.runner import _enforce_portfolio_aggregate_diagnostics


def _run(
    seed: int,
    score: float,
    levels: int,
    breadth: int,
    trimmed: float,
) -> dict[str, object]:
    return {
        "mode": "duck-portfolio",
        "seed": seed,
        "game_count": 25,
        "mean_engine_score": score,
        "trimmed_mean_engine_score": trimmed,
        "total_completed_levels": levels,
        "nonzero_game_count": breadth,
        "infrastructure_failures": [],
    }


def test_portfolio_two_seed_aggregate_preserves_seed_attribution() -> None:
    aggregate = aggregate_two_seed_runs(
        [
            _run(1, 1.1, 10, 9, 0.65),
            _run(0, 2.6, 18, 15, 0.95),
        ],
        expected_mode="duck-portfolio",
    )
    assert aggregate["seeds"] == [0, 1]
    assert aggregate["seed_engine_scores"] == [2.6, 1.1]
    assert aggregate["seed_completed_levels"] == [18, 10]
    assert aggregate["seed_nonzero_game_counts"] == [15, 9]


def test_portfolio_promotion_applies_seed0_and_two_seed_gates() -> None:
    passing = evaluate_portfolio_promotion(
        [
            _run(0, 2.6, 18, 15, 0.95),
            _run(1, 1.1, 10, 9, 0.65),
        ],
        rehearsal_elapsed_s=30_000,
        soft_deadline_s=31_200,
    )
    assert passing.passed
    assert passing.candidate_mean == 1.85
    assert passing.candidate_trimmed_mean == 0.8

    failing = evaluate_portfolio_promotion(
        [
            _run(0, 2.55, 18, 15, 0.93),
            _run(1, 0.2, 9, 8, 0.2),
        ],
        rehearsal_elapsed_s=31_200,
        soft_deadline_s=31_200,
    )
    assert not failing.passed
    text = "; ".join(failing.reasons)
    assert "seed 0 score" in text
    assert "seed 0 trimmed" in text
    assert "seed 1 completed levels" in text
    assert "soft deadline" in text


def test_hidden_portfolio_metrics_keep_only_aggregate_routing_telemetry() -> None:
    metrics = {
        "telemetry": {
            "portfolio_audit_actions": 17,
            "request_failures": 1,
        },
        "portfolio_diagnostics": {"hidden-game": {"feature_vector": {}}},
        "games": [
            {
                "telemetry": {
                    "portfolio_audit_actions": 17,
                    "request_failures": 1,
                },
                "portfolio_diagnostics": {
                    "feature_vector": {"symmetry_fraction": 0.5}
                },
            }
        ],
    }
    _enforce_portfolio_aggregate_diagnostics(metrics)
    assert metrics["telemetry"]["portfolio_audit_actions"] == 17
    assert "portfolio_diagnostics" not in metrics
    assert metrics["games"][0]["telemetry"] == {"request_failures": 1}
    assert "portfolio_diagnostics" not in metrics["games"][0]
    assert metrics["portfolio_diagnostics_scope"] == "aggregate-only"
