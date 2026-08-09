from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_pipeline():
    path = ROOT / "scripts" / "kaggle_pipeline.py"
    spec = importlib.util.spec_from_file_location("ouro3_kaggle_pipeline_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fingerprint(seed, mode="duck-reference"):
    value = {
        "mode": mode,
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
        "seed": seed,
        "prompt_sha256": "a" * 64,
        "tool_agent_source_sha256": "b" * 64,
        "solver_source_sha256": "c" * 64,
        "source_manifest_sha256": "d" * 64,
    }
    if mode == "duck-memory":
        value["memory"] = {
            "reasoning_history_field": "reasoning_content",
            "preserve_thinking": True,
            "reasoning_template_verified": True,
            "reasoning_template_sha256": "e" * 64,
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
    if mode == "duck-reasoning":
        value["reasoning"] = {
            "reasoning_history_field": "reasoning_content",
            "preserve_thinking": True,
            "reasoning_template_verified": True,
            "reasoning_template_sha256": "e" * 64,
            "history_policy": "stock-duck",
            "semantic_compaction": False,
            "auxiliary_model_calls": 0,
        }
    if mode == "duck-portfolio":
        from duck_portfolio.router import PortfolioRouter

        router = PortfolioRouter.load()
        value["portfolio"] = {
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
    if mode == "duck-retrodict":
        value["retrodict"] = {
            "persistent_evidence": True,
            "rule_language": "typed-host-owned-v1",
            "full_log_replay": True,
            "multi_ontology": ["color-4", "color-8", "color-4-all"],
            "automatic_batch_size": 1,
            "fallback_batch_limit": None,
            "max_rules": 256,
            "prediction_threshold": 0.9,
        }
    return value


def _seed_metric(seed, games, score=1.25, mode="duck-reference"):
    if len(games) != 25:
        template = games[0] if games else {
            "state": "gave_up",
            "levels_completed": 0,
            "actions": 0,
            "final_score": 0.0,
        }
        games = [
            {**template, "game_id": f"synthetic-{index}"}
            for index in range(25)
        ]
    metric = {
        "mode": mode,
        "seed": seed,
        "game_count": 25,
        "elapsed_seconds": 7_920,
        "runtime_fingerprint": _fingerprint(seed, mode),
        "prompt_sha256": "a" * 64,
        "config_hash": str(seed) * 64,
        "mean_engine_score": score,
        "mean_completed_levels": 2.0,
        "median_completed_levels": 2,
        "total_completed_levels": 2,
        "infrastructure_failures": [],
        "games": games,
    }
    if mode == "duck-robust":
        metric["recovery_diagnostics"] = {}
    if mode == "duck-memory":
        metric["memory_diagnostics"] = {}
        metric["telemetry"] = {
            "reasoning_template_verified": 25,
            "reasoning_turns": 120,
            "reasoning_chars": 50_000,
            "reasoning_retained_turns": 40,
            "reasoning_compacted_turns": 80,
            "reasoning_accounted_turns": 120,
            "reasoning_unaccounted_turns": 0,
            "compaction_count": 20,
            "compaction_retries": 0,
            "compaction_failures": 0,
            "emergency_trims": 0,
            "context_evictions": 0,
            "context_overflow_recoveries": 0,
            "compaction_pre_tokens": 500_000,
            "compaction_post_tokens": 200_000,
            "compaction_compression_ratio_bps": 4_000,
            "compaction_latency_ms": 50_000,
        }
    if mode == "duck-reasoning":
        metric["reasoning_diagnostics"] = {}
        metric["telemetry"] = {
            "reasoning_template_verified": 25,
            "reasoning_turns": 120,
            "reasoning_chars": 50_000,
            "reasoning_retained_turns": 40,
            "reasoning_evicted_turns": 80,
            "reasoning_unaccounted_turns": 0,
            "compaction_count": 0,
            "compaction_retries": 0,
            "compaction_failures": 0,
            "emergency_trims": 0,
        }
    if mode == "duck-portfolio":
        from duck_portfolio.router import PortfolioRouter

        router = PortfolioRouter.load()
        diagnostics = {
            "one_persistent_conversation": True,
            "parallel_model_trajectories": 0,
            "features": {},
            "route_events": [],
            "policy_action_counts": {
                "stock": 8,
                "audit": 0,
                "deliberate": 0,
                "contract-repair": 0,
            },
            "switch_count": 0,
            "router_artifact_sha256": router.artifact_hash,
        }
        metric["portfolio_diagnostics"] = {
            game["game_id"]: dict(diagnostics) for game in games
        }
    if mode == "duck-retrodict":
        metric["retrodict_diagnostics"] = {
            game["game_id"]: {
                "mode": "duck-retrodict",
                "world_model": {"certified_rules": 1},
            }
            for game in games
        }
    return metric


def test_passing_reference_is_frozen_without_action_traces(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pipeline = _load_pipeline()
    games = [
        {
            "game_id": "synthetic",
            "state": "finished",
            "levels_completed": 2,
            "actions": 4,
            "final_score": 1.0,
            "trace": [{"action": "UP"}],
        }
    ]
    seed_runs = [_seed_metric(seed, games) for seed in range(5)]
    metrics = {
        "experiment": "duck-reference-seeds-0-4",
        "mode": "duck-reference",
        "config_hashes": {str(seed): str(seed) * 64 for seed in range(5)},
        "prompt_sha256": "a" * 64,
        "max_kernel_elapsed_seconds": 7_920,
        "mean_engine_score": 1.25,
        "mean_completed_levels": 2.0,
        "infrastructure_failures": [],
        "runtime_fingerprint_consistent": True,
        "prompt_fingerprint_consistent": True,
        "seed_runs": seed_runs,
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    baseline = tmp_path / "reference.json"
    monkeypatch.setattr(pipeline, "REFERENCE_BASELINE", baseline)

    assert pipeline.enforce_reference_gate(path)["mean_engine_score"] == 1.25
    frozen = pipeline.freeze_reference(
        path,
        metrics,
        kernel_versions={seed: seed + 10 for seed in range(5)},
    )
    assert frozen["kernel_versions"]["4"] == 14
    assert frozen["mean_engine_score"] == 1.25
    assert "trace" not in frozen["seed_runs"][0]["games"][0]
    assert json.loads(baseline.read_text())["metrics_sha256"] == frozen["metrics_sha256"]


def test_submission_csv_and_waiter_select_newest_new_submission(monkeypatch) -> None:
    pipeline = _load_pipeline()
    old = (
        "ref,fileName,date,description,status,publicScore,privateScore\n"
        '10,submission.parquet,2026-07-20 09:15:43.650000,"old, quoted",'
        "SubmissionStatus.COMPLETE,0.09,\n"
    )
    pending = (
        old
        + "11,submission.parquet,2026-07-29 01:00:00.000000,new,"
        "SubmissionStatus.PENDING,,\n"
    )
    complete = pending.replace("SubmissionStatus.PENDING", "SubmissionStatus.COMPLETE")
    responses = iter((pending, complete))

    def fake_run_kaggle(args):
        return pipeline.CommandResult(tuple(args), next(responses))

    monkeypatch.setattr(pipeline, "run_kaggle", fake_run_kaggle)
    monkeypatch.setattr(pipeline.time, "sleep", lambda _seconds: None)

    rows = pipeline.parse_submission_rows(old)
    assert rows[0]["description"] == "old, quoted"
    assert pipeline.wait_submission(previous_refs={"10"}, timeout_s=5)["ref"] == "11"


def test_daily_quota_uses_kaggle_utc_submission_date(monkeypatch) -> None:
    pipeline = _load_pipeline()
    today = datetime.now(timezone.utc).date().isoformat()
    output = (
        "ref,fileName,date,description,status,publicScore,privateScore\n"
        f"12,submission.parquet,{today} 00:00:01.000000,today,"
        "SubmissionStatus.COMPLETE,0.10,\n"
    )
    monkeypatch.setattr(
        pipeline,
        "run_kaggle",
        lambda args: pipeline.CommandResult(tuple(args), output),
    )

    try:
        pipeline.ensure_daily_quota()
    except RuntimeError as exc:
        assert today in str(exc)
    else:
        raise AssertionError("same-day submission should exhaust the daily quota")


def test_unseeded_fidelity_gate_does_not_require_score_1_2(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pipeline = _load_pipeline()
    metrics = _seed_metric(None, [], score=0.2)
    metrics["game_count"] = 25
    metrics_path = tmp_path / "validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    assert pipeline.enforce_fidelity_gate(metrics_path)["mean_engine_score"] == 0.2


def test_resume_requires_exact_cached_seed_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pipeline = _load_pipeline()
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    output = pipeline._validation_result_dir(3, 17)
    output.mkdir(parents=True)
    path = output / "validation_metrics.json"
    path.write_text(json.dumps(_seed_metric(3, [])), encoding="utf-8")
    monkeypatch.setattr(
        pipeline,
        "push_kernel",
        lambda _path: (_ for _ in ()).throw(AssertionError("must not push")),
    )

    version, metrics_path, metrics = pipeline.run_validation_kernel(
        seed=3,
        resume_version=17,
    )
    assert version == 17
    assert metrics_path == path
    assert metrics["seed"] == 3


def test_fidelity_only_stops_before_seed_or_submission_kernels(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pipeline = _load_pipeline()
    progress = tmp_path / "progress.json"
    metrics_path = tmp_path / "validation_metrics.json"
    metrics = _seed_metric(None, [], score=0.7)
    monkeypatch.setattr(pipeline, "REFERENCE_PROGRESS", progress)
    monkeypatch.setattr(pipeline, "generate_artifacts", lambda **_kwargs: None)
    monkeypatch.setattr(pipeline, "publish_dataset", lambda _path: "published")
    monkeypatch.setattr(
        pipeline,
        "run_validation_kernel",
        lambda **_kwargs: (21, metrics_path, metrics),
    )
    monkeypatch.setattr(pipeline, "source_hash", lambda: "a" * 40)
    monkeypatch.setattr(pipeline, "config_hash", lambda: "b" * 64)
    monkeypatch.setattr(pipeline, "source_manifest_hash", lambda: "c" * 64)

    record = pipeline.execute_fidelity_validation()

    assert record["fidelity_validation_version"] == 21
    assert record["mean_engine_score"] == 0.7
    assert json.loads(progress.read_text())["stage"] == "unseeded-fidelity-complete"


def test_generated_validation_mode_is_checked_before_push(tmp_path: Path) -> None:
    pipeline = _load_pipeline()
    path = tmp_path / "validation.ipynb"
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "ouro3": {
                        "mode": "duck-reference",
                        "validation_seed": 0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    try:
        pipeline._assert_generated_validation(
            path,
            expected_mode="duck-robust",
            expected_seed=0,
        )
    except RuntimeError as exc:
        assert "refusing to push" in str(exc)
        assert "duck-robust" in str(exc)
    else:
        raise AssertionError("a reference notebook must not pass the robust check")


def test_robust_validation_requires_mode_and_recovery_diagnostics() -> None:
    pipeline = _load_pipeline()
    robust = _seed_metric(0, [], mode="duck-robust")
    pipeline._validate_validation_artifact(
        robust,
        expected_seed=0,
        expected_mode="duck-robust",
    )

    robust.pop("recovery_diagnostics")
    try:
        pipeline._validate_validation_artifact(
            robust,
            expected_seed=0,
            expected_mode="duck-robust",
        )
    except RuntimeError as exc:
        assert "missing recovery diagnostics" in str(exc)
    else:
        raise AssertionError("robust artifacts must include recovery diagnostics")


def test_robust_seed0_flow_is_isolated_and_never_submits(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pipeline = _load_pipeline()
    progress = tmp_path / "robust-progress.json"
    metrics_path = tmp_path / "validation_metrics.json"
    metrics = _seed_metric(0, [], score=1.9, mode="duck-robust")
    metrics["config_hash"] = "b" * 64
    metrics["mean_completed_levels"] = 2.5
    metrics["total_completed_levels"] = 62
    metrics["telemetry"] = {
        "recovery_count": 3,
        "recovery_successes": 1,
        "recovery_resets": 0,
        "prediction_matches": 4,
        "prediction_mismatches": 1,
    }
    metrics["recovery_diagnostics"] = {
        "game-a": {"recovered_levels": [1]},
        "game-b": {"recovered_levels": []},
    }
    generated = []
    validation_calls = []
    monkeypatch.setattr(pipeline, "ROBUST_PROGRESS", progress)
    monkeypatch.setattr(
        pipeline,
        "generate_artifacts",
        lambda **kwargs: generated.append(kwargs),
    )
    monkeypatch.setattr(pipeline, "publish_dataset", lambda _path: "published")

    def fake_run_validation_kernel(**kwargs):
        validation_calls.append(kwargs)
        return 12, metrics_path, metrics

    monkeypatch.setattr(
        pipeline,
        "run_validation_kernel",
        fake_run_validation_kernel,
    )
    monkeypatch.setattr(
        pipeline,
        "config_hash",
        lambda mode="duck-reference": (
            "b" * 64 if mode == "duck-robust" else "a" * 64
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_latest_reference_seed_score",
        lambda _seed: 1.78,
    )
    monkeypatch.setattr(pipeline, "source_hash", lambda: "c" * 40)
    monkeypatch.setattr(pipeline, "source_manifest_hash", lambda: "d" * 64)
    monkeypatch.setattr(
        pipeline,
        "submit_exact_kernel",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("seed-0 validation must never submit")
        ),
    )

    record = pipeline.execute_robust_seed0_validation()

    assert generated == [{"validation_seed": 0, "mode": "duck-robust"}]
    assert validation_calls == [
        {
            "seed": 0,
            "resume_version": None,
            "mode": "duck-robust",
        }
    ]
    assert record["validation_version"] == 12
    assert record["score_delta_vs_reference_seed_0"] == pytest.approx(0.12)
    assert record["recovery"]["recovery_successes"] == 1
    assert record["recovery"]["recovery_games"] == ["game-a"]
    assert record["decision"] == "review-required-no-automatic-submission"
    assert json.loads(progress.read_text())["mode"] == "duck-robust"


def test_pipeline_checkpoints_each_completed_seed_before_next_kernel(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pipeline = _load_pipeline()
    progress = tmp_path / "progress.json"
    fidelity_path = tmp_path / "fidelity.json"
    seed_zero_path = tmp_path / "seed-zero.json"
    fidelity_metrics = _seed_metric(None, [], score=1.7)
    seed_zero_metrics = _seed_metric(0, [], score=1.8)
    monkeypatch.setattr(pipeline, "REFERENCE_PROGRESS", progress)
    monkeypatch.setattr(pipeline, "source_hash", lambda: "a" * 40)
    monkeypatch.setattr(pipeline, "config_hash", lambda: "b" * 64)
    monkeypatch.setattr(pipeline, "source_manifest_hash", lambda: "c" * 64)

    def fake_run_validation_kernel(*, seed, resume_version=None):
        if seed is None:
            assert resume_version == 5
            return 5, fidelity_path, fidelity_metrics
        if seed == 0:
            assert resume_version == 6
            return 6, seed_zero_path, seed_zero_metrics
        raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(
        pipeline,
        "run_validation_kernel",
        fake_run_validation_kernel,
    )

    try:
        pipeline.execute_pipeline(
            submit=False,
            fidelity_version=5,
            seed_versions={0: 6},
        )
    except RuntimeError as exc:
        assert str(exc) == "synthetic interruption"
    else:
        raise AssertionError("the synthetic interruption should stop the pipeline")

    checkpoint = json.loads(progress.read_text())
    assert checkpoint["stage"] == "seed-validation-in-progress"
    assert checkpoint["completed_seed_versions"] == {"0": 6}
    assert checkpoint["completed_seed_scores"] == {"0": 1.8}
    assert checkpoint["completed_seed_metrics"] == {"0": str(seed_zero_path)}
    assert checkpoint["next_seed"] == 1


def test_early_private_baseline_skips_seeds_two_through_four_and_submits(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pipeline = _load_pipeline()
    fidelity_path = tmp_path / "fidelity.json"
    seed_zero_path = tmp_path / "seed-zero.json"
    seed_one_path = tmp_path / "seed-one.json"
    early_path = tmp_path / "early.json"
    progress_path = tmp_path / "progress.json"
    ledger_records = []
    validation_calls = []
    fidelity_metrics = _seed_metric(None, [], score=1.7)
    seed_zero_metrics = _seed_metric(0, [], score=1.8)
    seed_one_metrics = _seed_metric(1, [], score=1.6)
    monkeypatch.setattr(pipeline, "EARLY_BASELINE", early_path)
    monkeypatch.setattr(pipeline, "REFERENCE_PROGRESS", progress_path)

    def fake_run_validation_kernel(*, seed, resume_version=None):
        validation_calls.append((seed, resume_version))
        if seed is None:
            return 5, fidelity_path, fidelity_metrics
        if seed == 0:
            return 6, seed_zero_path, seed_zero_metrics
        raise AssertionError(f"unexpected seeded validation: {seed}")

    monkeypatch.setattr(
        pipeline,
        "run_validation_kernel",
        fake_run_validation_kernel,
    )
    monkeypatch.setattr(
        pipeline,
        "attach_running_validation_kernel",
        lambda **_kwargs: (7, seed_one_path, seed_one_metrics),
    )
    monkeypatch.setattr(pipeline, "ensure_daily_quota", lambda: {"old"})
    monkeypatch.setattr(pipeline, "generate_artifacts", lambda **_kwargs: None)
    monkeypatch.setattr(pipeline, "push_kernel", lambda _path: 31)
    monkeypatch.setattr(pipeline, "wait_kernel", lambda _ref: "complete")
    monkeypatch.setattr(pipeline, "refresh_leaderboard_best", lambda: 1.86)
    monkeypatch.setattr(pipeline, "source_hash", lambda: "a" * 40)
    monkeypatch.setattr(pipeline, "config_hash", lambda: "b" * 64)
    monkeypatch.setattr(pipeline, "source_manifest_hash", lambda: "d" * 64)
    monkeypatch.setattr(
        pipeline,
        "submit_exact_kernel",
        lambda **kwargs: f"submitted-v{kwargs['kernel_version']}",
    )
    monkeypatch.setattr(
        pipeline,
        "wait_submission",
        lambda **_kwargs: {"ref": "new", "status": "complete", "publicScore": "0.2"},
    )
    monkeypatch.setattr(pipeline, "append_ledger", ledger_records.append)

    record = pipeline.execute_early_private_baseline(
        submit=True,
        fidelity_version=5,
        seed0_version=6,
        seed1_version=7,
    )

    assert validation_calls == [(None, 5), (0, 6)]
    assert round(record["validation_scores"]["mean_engine_score"], 10) == 1.7
    assert record["seed_validation_versions"] == {"0": 6, "1": 7}
    assert record["submission_version"] == 31
    assert record["result"] == "complete"
    assert ledger_records == [record]
    assert json.loads(progress_path.read_text())["stage"] == (
        "early-private-baseline-submission-complete"
    )


def test_memory_gate_requires_score_reasoning_and_lossless_compaction(
    tmp_path: Path,
) -> None:
    pipeline = _load_pipeline()
    metrics = _seed_metric(0, [], score=1.25, mode="duck-memory")
    path = tmp_path / "memory.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    assert pipeline.enforce_memory_gate(path)["mean_engine_score"] == 1.25

    metrics["telemetry"]["emergency_trims"] = 1
    metrics["telemetry"]["context_evictions"] = 1
    path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(RuntimeError, match="emergency_trims=0"):
        pipeline.enforce_memory_gate(path)

    metrics = _seed_metric(0, [], score=1.25, mode="duck-memory")
    metrics["telemetry"]["reasoning_accounted_turns"] = 119
    metrics["telemetry"]["reasoning_unaccounted_turns"] = 1
    path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(RuntimeError, match="account for every reasoning-bearing"):
        pipeline.enforce_memory_gate(path)

    metrics = _seed_metric(0, [], score=1.19, mode="duck-memory")
    path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(RuntimeError, match="below 1.20"):
        pipeline.enforce_memory_gate(path)


def test_reasoning_gate_requires_verified_transport_without_compaction(
    tmp_path: Path,
) -> None:
    pipeline = _load_pipeline()
    metrics = _seed_metric(0, [], score=1.25, mode="duck-reasoning")
    path = tmp_path / "reasoning.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    assert pipeline.enforce_reasoning_gate(path)["mean_engine_score"] == 1.25

    metrics["telemetry"]["compaction_count"] = 1
    path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(RuntimeError, match="semantic compaction"):
        pipeline.enforce_reasoning_gate(path)


def test_memory_gpu_reserve_and_local_prerequisites(tmp_path: Path) -> None:
    pipeline = _load_pipeline()
    with pytest.raises(RuntimeError, match="the candidate requires at least 12.0"):
        pipeline.ensure_gpu_hours_remaining(11.9, required_hours=12.0)
    assert pipeline.ensure_gpu_hours_remaining(12.0, required_hours=12.0) == 12.0

    public_path = tmp_path / "public.json"
    rehearsal_path = tmp_path / "rehearsal.json"
    public_path.write_text(
        json.dumps(
            {
                "mode": "duck-memory",
                "game_count": 25,
                "mean_engine_score": 0.1,
                "infrastructure_failures": [],
            }
        ),
        encoding="utf-8",
    )
    rehearsal_path.write_text(
        json.dumps(
            {
                "game_count": 110,
                "unique_game_ids": 110,
                "gateway_transport": "competition-http",
                "infrastructure_failures": [],
            }
        ),
        encoding="utf-8",
    )
    evidence = pipeline.validate_memory_local_prerequisites(
        public_path=public_path,
        rehearsal_path=rehearsal_path,
    )
    assert evidence["rehearsal_unique_game_ids"] == 110


def test_gpu_reserve_can_query_authenticated_kaggle_quota(monkeypatch) -> None:
    pipeline = _load_pipeline()
    monkeypatch.delenv("OURO3_KAGGLE_GPU_HOURS_REMAINING", raising=False)
    monkeypatch.setattr(
        pipeline,
        "run_kaggle",
        lambda _args: type(
            "Result",
            (),
            {
                "stdout": json.dumps(
                    [
                        {"resource": "GPU", "remaining": "13.25h"},
                        {"resource": "TPU", "remaining": "20.00h"},
                    ]
                )
            },
        )(),
    )
    assert pipeline.ensure_gpu_hours_remaining(None, required_hours=13.0) == 13.25

    monkeypatch.setattr(
        pipeline,
        "run_kaggle",
        lambda _args: type("Result", (), {"stdout": "not-json"})(),
    )
    with pytest.raises(RuntimeError, match="remaining Kaggle GPU hours are unknown"):
        pipeline.ensure_gpu_hours_remaining(None, required_hours=13.0)


def test_poetiq_candidate_resume_runs_both_seeds_and_records_gate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pipeline = _load_pipeline()
    seed_metrics = {
        0: _seed_metric(0, [], score=1.6, mode="duck-poetiq"),
        1: _seed_metric(1, [], score=1.5, mode="duck-poetiq"),
    }
    seed_metrics[0].update(
        total_completed_levels=19,
        nonzero_game_count=16,
        trimmed_mean_engine_score=0.7,
        telemetry={"poetiq_stalled_yields": 0},
    )
    seed_metrics[1].update(
        total_completed_levels=11,
        nonzero_game_count=10,
        trimmed_mean_engine_score=0.7,
        telemetry={"poetiq_stalled_yields": 0},
    )
    metric_paths = {
        seed: tmp_path / f"seed-{seed}.json" for seed in (0, 1)
    }
    calls: list[tuple[int, int | None]] = []
    monkeypatch.setattr(
        pipeline,
        "validate_poetiq_local_prerequisites",
        lambda: {
            "local_public": "local",
            "rehearsal": "rehearsal",
            "rehearsal_elapsed_seconds": 30_000,
        },
    )
    monkeypatch.setattr(pipeline, "run_validation_kernel", lambda **kwargs: (
        calls.append((kwargs["seed"], kwargs["resume_version"])) or (
            kwargs["resume_version"],
            metric_paths[kwargs["seed"]],
            seed_metrics[kwargs["seed"]],
        )
    ))
    monkeypatch.setattr(pipeline, "source_hash", lambda: "a" * 40)
    monkeypatch.setattr(pipeline, "config_hash", lambda _mode: "b" * 64)
    monkeypatch.setattr(pipeline, "source_manifest_hash", lambda: "c" * 64)
    monkeypatch.setattr(pipeline, "append_ledger", lambda _record: None)
    progress_path = tmp_path / "poetiq-progress.json"
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(pipeline, "POETIQ_PROGRESS", progress_path)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline,
        "write_metrics",
        lambda metrics, path: path.write_text(json.dumps(metrics), encoding="utf-8"),
    )

    record = pipeline.execute_poetiq_candidate(
        submit=False,
        seed_versions={0: 41, 1: 42},
        gpu_hours_remaining=13.25,
    )

    assert calls == [(0, 41), (1, 42)]
    assert record["public_gate"] == "passed"
    assert record["seed_validation_versions"] == {"0": 41, "1": 42}
    assert record["validation_scores"]["mean_engine_score"] == 1.55
    assert json.loads(progress_path.read_text(encoding="utf-8"))["stage"] == (
        "duck-poetiq-two-seed-gate-passed"
    )


def test_portfolio_validation_requires_complete_route_diagnostics() -> None:
    pipeline = _load_pipeline()
    metrics = _seed_metric(0, [], score=2.6, mode="duck-portfolio")
    pipeline._validate_validation_artifact(
        metrics,
        expected_seed=0,
        expected_mode="duck-portfolio",
    )
    metrics["portfolio_diagnostics"].pop("synthetic-0")
    with pytest.raises(RuntimeError, match="not present for every game"):
        pipeline._validate_validation_artifact(
            metrics,
            expected_seed=0,
            expected_mode="duck-portfolio",
        )


def test_portfolio_packaged_identity_covers_config_prompt_and_router(
    monkeypatch,
) -> None:
    pipeline = _load_pipeline()
    monkeypatch.setenv("OURO3_HARNESS_MODE", "duck-reference")
    identity = pipeline.portfolio_packaged_identity()
    assert set(identity) == {
        "config_hash",
        "prompt_sha256",
        "router_artifact_sha256",
    }
    assert all(len(value) == 64 for value in identity.values())
    assert pipeline.os.environ["OURO3_HARNESS_MODE"] == "duck-reference"


def test_portfolio_candidate_resume_runs_gated_seeds_and_records_hashes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pipeline = _load_pipeline()
    from duck_portfolio.router import PortfolioRouter
    from ouro3.config import HarnessConfig

    seed_metrics = {
        0: _seed_metric(0, [], score=2.6, mode="duck-portfolio"),
        1: _seed_metric(1, [], score=1.1, mode="duck-portfolio"),
    }
    seed_metrics[0].update(
        config_hash=HarnessConfig.portfolio(seed=0).config_hash,
        total_completed_levels=18,
        nonzero_game_count=15,
        trimmed_mean_engine_score=0.95,
    )
    seed_metrics[1].update(
        config_hash=HarnessConfig.portfolio(seed=1).config_hash,
        total_completed_levels=10,
        nonzero_game_count=9,
        trimmed_mean_engine_score=0.65,
    )
    metric_paths = {
        seed: tmp_path / f"portfolio-seed-{seed}.json" for seed in (0, 1)
    }
    calls: list[tuple[int, int | None]] = []
    router = PortfolioRouter.load()
    monkeypatch.setattr(
        pipeline,
        "validate_portfolio_local_prerequisites",
        lambda: {
            "local_public": "local",
            "rehearsal": "rehearsal",
            "rehearsal_elapsed_seconds": 30_000,
            "router_artifact_sha256": router.artifact_hash,
            "cross_validation": router.cross_validation,
        },
    )
    monkeypatch.setattr(
        pipeline,
        "run_validation_kernel",
        lambda **kwargs: (
            calls.append((kwargs["seed"], kwargs["resume_version"]))
            or (
                kwargs["resume_version"],
                metric_paths[kwargs["seed"]],
                seed_metrics[kwargs["seed"]],
            )
        ),
    )
    monkeypatch.setattr(pipeline, "source_hash", lambda: "a" * 40)
    monkeypatch.setattr(pipeline, "config_hash", lambda _mode: "b" * 64)
    monkeypatch.setattr(pipeline, "append_ledger", lambda _record: None)
    (tmp_path / "results").mkdir()
    progress_path = tmp_path / "portfolio-progress.json"
    monkeypatch.setattr(pipeline, "PORTFOLIO_PROGRESS", progress_path)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)

    record = pipeline.execute_portfolio_candidate(
        submit=False,
        seed_versions={0: 51, 1: 52},
        gpu_hours_remaining=4.5,
    )

    assert calls == [(0, 51), (1, 52)]
    assert record["public_gate"] == "passed"
    assert record["router_artifact_sha256"] == router.artifact_hash
    assert record["source_manifest_sha256"] == "d" * 64
    assert record["seed_validation_versions"] == {"0": 51, "1": 52}
    assert record["validation_scores"]["mean_engine_score"] == 1.85
    assert json.loads(progress_path.read_text())["stage"] == (
        "duck-portfolio-two-seed-gate-passed"
    )


def test_portfolio_seed0_hard_gate_stops_before_seed1(monkeypatch) -> None:
    pipeline = _load_pipeline()
    from ouro3.config import HarnessConfig

    failed = _seed_metric(0, [], score=2.5, mode="duck-portfolio")
    failed.update(
        config_hash=HarnessConfig.portfolio(seed=0).config_hash,
        total_completed_levels=18,
        nonzero_game_count=15,
        trimmed_mean_engine_score=0.95,
    )
    calls: list[int] = []
    monkeypatch.setattr(
        pipeline,
        "validate_portfolio_local_prerequisites",
        lambda: {
            "rehearsal_elapsed_seconds": 1,
            "router_artifact_sha256": "a" * 64,
            "cross_validation": {"passed": True},
        },
    )
    monkeypatch.setattr(
        pipeline,
        "run_validation_kernel",
        lambda **kwargs: (
            calls.append(kwargs["seed"])
            or (51, Path("metrics.json"), failed)
        ),
    )
    monkeypatch.setattr(pipeline, "write_portfolio_progress", lambda _record: None)
    with pytest.raises(RuntimeError, match="seed 1 was not started"):
        pipeline.execute_portfolio_candidate(
            submit=False,
            seed_versions={0: 51, 1: 52},
            gpu_hours_remaining=4.5,
        )
    assert calls == [0]


def test_retrodict_validation_requires_complete_host_diagnostics() -> None:
    pipeline = _load_pipeline()
    metrics = _seed_metric(0, [], score=1.9, mode="duck-retrodict")
    pipeline._validate_validation_artifact(
        metrics,
        expected_seed=0,
        expected_mode="duck-retrodict",
    )
    metrics["retrodict_diagnostics"].pop("synthetic-0")
    with pytest.raises(RuntimeError, match="not present for every game"):
        pipeline._validate_validation_artifact(
            metrics,
            expected_seed=0,
            expected_mode="duck-retrodict",
        )


def test_retrodict_local_prerequisites_fail_closed_on_offline_replay(
    tmp_path: Path,
) -> None:
    pipeline = _load_pipeline()
    public_path = tmp_path / "public.json"
    rehearsal_path = tmp_path / "rehearsal.json"
    offline_path = tmp_path / "offline.json"
    public_path.write_text(
        json.dumps(_seed_metric(0, [], mode="duck-retrodict")),
        encoding="utf-8",
    )
    rehearsal_path.write_text(
        json.dumps(
            {
                "game_count": 110,
                "unique_game_ids": 110,
                "gateway_transport": "competition-http",
                "elapsed_seconds": 20_000,
                "infrastructure_failures": [],
            }
        ),
        encoding="utf-8",
    )
    offline_path.write_text(
        json.dumps(
            {
                "holdout_count": 10,
                "typed": {
                    "precision": 0.94,
                    "coverage": 0.60,
                    "latency_p95_ms": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="offline gate failed"):
        pipeline.validate_retrodict_local_prerequisites(
            public_path=public_path,
            rehearsal_path=rehearsal_path,
            offline_report_path=offline_path,
        )
    evidence = pipeline.validate_retrodict_local_prerequisites(
        public_path=public_path,
        rehearsal_path=rehearsal_path,
        offline_report_path=offline_path,
        require_offline_pass=False,
    )
    assert evidence["offline"]["typed"]["precision"] == 0.94
    public = json.loads(public_path.read_text(encoding="utf-8"))
    public["telemetry"] = {"request_failures": 1}
    public_path.write_text(json.dumps(public), encoding="utf-8")
    with pytest.raises(RuntimeError, match="actor request failures"):
        pipeline.validate_retrodict_local_prerequisites(
            public_path=public_path,
            rehearsal_path=rehearsal_path,
            offline_report_path=offline_path,
            require_offline_pass=False,
        )


def test_retrodict_candidate_resume_requires_winning_two_seed_gate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pipeline = _load_pipeline()
    from ouro3.config import HarnessConfig

    seed_metrics = {
        seed: _seed_metric(seed, [], score=1.9, mode="duck-retrodict")
        for seed in (0, 1)
    }
    for seed, metrics in seed_metrics.items():
        metrics.update(
            config_hash=HarnessConfig.retrodict(seed=seed).config_hash,
            total_completed_levels=16,
            nonzero_game_count=12,
            trimmed_mean_engine_score=1.05,
        )
    metric_paths = {
        seed: tmp_path / f"retrodict-seed-{seed}.json" for seed in (0, 1)
    }
    calls: list[tuple[int, int | None]] = []
    offline = {
        "holdout_count": 10,
        "typed": {
            "precision": 0.95,
            "coverage": 0.60,
            "latency_p95_ms": 1.0,
        },
        "promotion": {"passed": True},
    }
    monkeypatch.setattr(
        pipeline,
        "validate_retrodict_local_prerequisites",
        lambda **_kwargs: {
            "local_public": "local",
            "offline_report": "offline",
            "offline": offline,
            "rehearsal": "rehearsal",
            "rehearsal_elapsed_seconds": 20_000,
        },
    )
    monkeypatch.setattr(
        pipeline,
        "run_validation_kernel",
        lambda **kwargs: (
            calls.append((kwargs["seed"], kwargs["resume_version"]))
            or (
                kwargs["resume_version"],
                metric_paths[kwargs["seed"]],
                seed_metrics[kwargs["seed"]],
            )
        ),
    )
    monkeypatch.setattr(pipeline, "refresh_leaderboard_best", lambda: 1.86)
    monkeypatch.setattr(pipeline, "source_hash", lambda: "a" * 40)
    monkeypatch.setattr(pipeline, "config_hash", lambda _mode: "b" * 64)
    monkeypatch.setattr(pipeline, "append_ledger", lambda _record: None)
    (tmp_path / "results").mkdir()
    progress_path = tmp_path / "retrodict-progress.json"
    monkeypatch.setattr(pipeline, "RETRODICT_PROGRESS", progress_path)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)

    record = pipeline.execute_retrodict_candidate(
        submit=False,
        seed_versions={0: 61, 1: 62},
        gpu_hours_remaining=4.5,
    )

    assert calls == [(0, 61), (1, 62)]
    assert record["public_gate"] == "passed"
    assert record["seed_validation_versions"] == {"0": 61, "1": 62}
    assert record["validation_scores"]["mean_engine_score"] == 1.9
    assert record["winning_target"] == pytest.approx(1.87)
    assert json.loads(progress_path.read_text())["stage"] == (
        "duck-retrodict-two-seed-gate-passed"
    )


def test_retrodict_experimental_public_mode_can_never_submit() -> None:
    pipeline = _load_pipeline()
    with pytest.raises(RuntimeError, match="can never submit"):
        pipeline.execute_retrodict_candidate(
            submit=True,
            experimental_public=True,
        )


def test_memory_seed0_flow_submits_exact_gated_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pipeline = _load_pipeline()
    metrics_path = tmp_path / "validation_metrics.json"
    progress = tmp_path / "memory-progress.json"
    ledger = []
    metrics = _seed_metric(0, [], score=1.3, mode="duck-memory")
    metrics["config_hash"] = "c" * 64
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    notebook_root = tmp_path / "notebooks"
    (notebook_root / "submission").mkdir(parents=True)

    monkeypatch.setattr(pipeline, "MEMORY_PROGRESS", progress)
    monkeypatch.setattr(
        pipeline,
        "validate_memory_local_prerequisites",
        lambda: {"local_public": "ok", "rehearsal": "ok"},
    )
    monkeypatch.setattr(
        pipeline,
        "run_validation_kernel",
        lambda **kwargs: (42, metrics_path, metrics),
    )
    monkeypatch.setattr(
        pipeline,
        "generate_artifacts",
        lambda **kwargs: notebook_root,
    )
    monkeypatch.setattr(pipeline, "publish_dataset", lambda _path: "published")
    monkeypatch.setattr(
        pipeline,
        "config_hash",
        lambda mode="duck-reference": (
            "c" * 64 if mode == "duck-memory" else "a" * 64
        ),
    )
    monkeypatch.setattr(pipeline, "source_hash", lambda: "b" * 40)
    monkeypatch.setattr(pipeline, "source_manifest_hash", lambda: "d" * 64)
    monkeypatch.setattr(pipeline, "ensure_daily_quota", lambda: {"old"})
    monkeypatch.setattr(pipeline, "push_kernel", lambda _path: 52)
    monkeypatch.setattr(pipeline, "wait_kernel", lambda _ref: "complete")
    monkeypatch.setattr(pipeline, "refresh_leaderboard_best", lambda: 1.86)
    monkeypatch.setattr(
        pipeline,
        "submit_exact_kernel",
        lambda **kwargs: f"submitted-v{kwargs['kernel_version']}",
    )
    monkeypatch.setattr(
        pipeline,
        "wait_submission",
        lambda **_kwargs: {
            "ref": "new",
            "status": "complete",
            "publicScore": "0.81",
        },
    )
    monkeypatch.setattr(pipeline, "append_ledger", ledger.append)

    record = pipeline.execute_memory_seed0_candidate(
        submit=True,
        gpu_hours_remaining=12.0,
    )

    assert record["validation_version"] == 42
    assert record["submission_version"] == 52
    assert record["visible_score"] == 0.81
    assert record["beat_previous_best"] is True
    assert record["config_hash"] == "c" * 64
    assert ledger == [record]
    assert json.loads(progress.read_text())["stage"] == (
        "duck-memory-submission-complete"
    )


def test_audit_mode_is_public_only_and_has_stable_artifact_mapping() -> None:
    pipeline = _load_pipeline()
    assert pipeline.AUDIT_MODE in pipeline.SUPPORTED_MODES
    assert pipeline.config_hash(pipeline.AUDIT_MODE) == (
        "93c15bced9077217a513655e4fcd9a92fff4729ebe7004b66121989c783ce66c"
    )
    assert pipeline._notebooks_root(pipeline.AUDIT_MODE).name == "audit"


def test_contract_modes_are_public_only_and_record_wirefix_telemetry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pipeline = _load_pipeline()
    metrics = _seed_metric(0, [], score=1.15, mode="duck-contract-repair")
    metrics["config_hash"] = "z" * 64
    metrics["telemetry"] = {
        "deliberate_proposals": 10,
        "contract_repairs": 0,
        "prediction_matches": 8,
        "prediction_mismatches": 2,
        "context_evictions": 3,
        "request_failures": 1,
    }
    metrics_path = tmp_path / "validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    progress = tmp_path / "contract-progress.json"

    monkeypatch.setattr(pipeline, "CONTRACT_REPAIR_PROGRESS", progress)
    monkeypatch.setattr(
        pipeline,
        "run_validation_kernel",
        lambda **_kwargs: (19, metrics_path, metrics),
    )
    monkeypatch.setattr(
        pipeline,
        "config_hash",
        lambda mode="duck-reference": "z" * 64
        if mode == pipeline.CONTRACT_REPAIR_MODE
        else "a" * 64,
    )
    monkeypatch.setattr(pipeline, "source_hash", lambda: "b" * 40)

    record = pipeline.execute_contract_seed0_validation(
        mode=pipeline.CONTRACT_REPAIR_MODE,
        resume_version=12,
    )

    assert pipeline.CONTRACT_REPAIR_MODE in pipeline.SUPPORTED_MODES
    assert record["validation_version"] == 19
    assert record["telemetry"]["contract_repairs"] == 0
    assert record["telemetry"]["prediction_matches"] == 8
    assert record["hidden_submission"] is False
    assert json.loads(progress.read_text()) == record
