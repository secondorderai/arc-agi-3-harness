from __future__ import annotations

from ouro3.promotion import evaluate_promotion
from ouro3.config import HarnessConfig, RuntimeProfile
from ouro3.runner import make_solver
from ouro3.scheduler import GlobalScheduler, compute_submission_budget


def test_scheduler_allocates_by_remaining_waves_and_deadline() -> None:
    scheduler = GlobalScheduler(
        total_games=110,
        concurrency=28,
        soft_deadline_s=1_000,
        setup_teardown_reserve_s=100,
        started_at=100,
    )
    initial = scheduler.start_session(0, now=100)
    assert 200 < initial < 225
    scheduler.finish_session(0)
    later = scheduler.budget_for_new_session(now=550)
    assert later < initial
    assert scheduler.finished_count == 1


def test_submission_budget_fits_all_110_games_inside_nine_hours() -> None:
    budget = compute_submission_budget(
        total_games=110,
        concurrency=28,
        configured_game_cap_s=7_920,
        soft_deadline_s=31_200,
        setup_teardown_reserve_s=1_200,
    )

    assert budget.waves == 4
    assert budget.per_game_cap_s == 7_200
    assert budget.worst_case_gameplay_s == 28_800
    assert budget.worst_case_gameplay_s < budget.soft_deadline_s
    assert budget.worst_case_gameplay_s + budget.setup_teardown_reserve_s < 9 * 60 * 60


def test_submission_profile_applies_the_safe_cap_without_changing_public_defaults() -> None:
    public = HarnessConfig.audit(seed=0)
    budget = compute_submission_budget(
        total_games=110,
        concurrency=public.concurrency,
        configured_game_cap_s=public.reference_game_cap_s,
        soft_deadline_s=public.soft_deadline_s,
        setup_teardown_reserve_s=public.setup_teardown_reserve_s,
    )
    submission = public.with_overrides(
        profile=RuntimeProfile.KAGGLE_SUBMISSION,
        reference_game_cap_s=budget.per_game_cap_s,
    )

    assert public.reference_game_cap_s == 7_920
    assert submission.reference_game_cap_s == 7_200
    assert make_solver(submission).max_runtime_s_per_game == 7_200


def _metric(
    seed: int,
    level: int,
    *,
    engine_score: float,
    failure: bool = False,
) -> dict:
    return {
        "seed": seed,
        "mean_engine_score": engine_score,
        "mean_completed_levels": float(level),
        "total_completed_levels": level * 9,
        "infrastructure_failures": ["x"] if failure else [],
        "games": [{"game_id": f"g{i}", "levels_completed": level} for i in range(9)],
    }


def test_promotion_gate_passes_only_all_constraints() -> None:
    reference = [_metric(seed, 2, engine_score=1.25) for seed in range(5)]
    candidate = [_metric(seed, 3, engine_score=1.36) for seed in range(5)]
    decision = evaluate_promotion(
        reference,
        candidate,
        rehearsal_elapsed_s=20_000,
        soft_deadline_s=31_200,
    )
    assert decision.passed

    failed = evaluate_promotion(
        reference,
        [_metric(0, 3, engine_score=1.36, failure=True), *candidate[1:]],
        rehearsal_elapsed_s=40_000,
        soft_deadline_s=31_200,
    )
    assert not failed.passed
    assert any("infrastructure" in reason for reason in failed.reasons)
    assert any("soft deadline" in reason for reason in failed.reasons)


def test_promotion_uses_engine_score_not_completed_level_mean() -> None:
    reference = [_metric(seed, 2, engine_score=1.25) for seed in range(5)]
    candidate = [_metric(seed, 3, engine_score=1.30) for seed in range(5)]

    decision = evaluate_promotion(
        reference,
        candidate,
        rehearsal_elapsed_s=20_000,
        soft_deadline_s=31_200,
    )

    assert not decision.passed
    assert any("engine-score mean improvement" in reason for reason in decision.reasons)
