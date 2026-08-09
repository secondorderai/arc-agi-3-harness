"""Direct Arcade benchmark construction and execution."""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import time
import zlib
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, ContextManager, Sequence

import arc_agi
import taaf.benchmark
import taaf.competition_arcade
import taaf.game_api

from duck_reference.solver import DuckReferenceHarnessSolver
from duck_reasoning.solver import DuckReasoningHarnessSolver
from duck_deliberate.solver import DuckDeliberateHarnessSolver
from duck_contract.solver import DuckContractHarnessSolver
from duck_contract.repair_solver import DuckContractRepairHarnessSolver
from duck_audit.solver import DuckAuditHarnessSolver
from duck_information.solver import DuckInformationHarnessSolver
from duck_hierarchy.solver import DuckHierarchyHarnessSolver
from duck_diversity.solver import DuckDiversityHarnessSolver
from duck_poetiq.solver import DuckPoetiqHarnessSolver
from duck_portfolio.solver import DuckPortfolioHarnessSolver
from duck_retrodict.solver import DuckRetrodictHarnessSolver
from duck_robust.solver import DuckRobustHarnessSolver
from duck_memory.solver import DuckMemoryHarnessSolver
from inference.framework.kaggle import DUCK_HARNESS_PUBLIC_GAME_IDS
from inference.framework.solver import HarnessSolver
from ouro3.agent import ScriptedAnalyzer
from ouro3.config import HarnessConfig, RuntimeProfile
from ouro3.fingerprint import prompt_sha256, runtime_fingerprint
from ouro3.metrics import summarize_runs, write_metrics
from ouro3.mode import HarnessMode
from ouro3.solver import HybridHarnessSolver
from ouro3.scheduler import compute_submission_budget
from ouro3.splits import base_game_id, select_fold


def _write_memory_trace_bundle(
    *,
    job_dir: Path,
    output_path: Path,
) -> Path | None:
    """Expose validation reasoning traces as one visible compressed artifact."""

    trace_paths = sorted(job_dir.rglob("*_memory_trace.jsonl.gz"))
    if not trace_paths:
        return None
    bundle_path = output_path.with_name(
        f"{output_path.stem}-memory-trace.jsonl.gz"
    )
    with gzip.open(bundle_path, "wt", encoding="utf-8") as target:
        for trace_path in trace_paths:
            game_id = trace_path.name.split("_p", 1)[0]
            with gzip.open(trace_path, "rt", encoding="utf-8") as source:
                for line in source:
                    event = json.loads(line)
                    event["game_id"] = game_id
                    target.write(
                        json.dumps(
                            event,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                    target.write("\n")
    return bundle_path


def _write_retrodict_trace_bundle(
    *,
    job_dir: Path,
    output_path: Path,
) -> Path | None:
    """Combine compressed per-game transition evidence for offline replay."""

    trace_paths = sorted(job_dir.rglob("*_retrodict_trace.jsonl.gz"))
    if not trace_paths:
        return None
    bundle_path = output_path.with_name(
        f"{output_path.stem}-retrodict-trace.jsonl.gz"
    )
    with gzip.open(bundle_path, "wt", encoding="utf-8") as target:
        for trace_path in trace_paths:
            try:
                with gzip.open(trace_path, "rt", encoding="utf-8") as source:
                    for line in source:
                        target.write(line)
            except (EOFError, OSError, UnicodeError, zlib.error):
                # Transition traces are diagnostic evidence, never a reason to
                # discard completed gameplay metrics.  A corrupt concurrent or
                # interrupted member is skipped while valid members survive.
                continue
    return bundle_path


def _enforce_portfolio_aggregate_diagnostics(metrics: dict[str, Any]) -> None:
    """Remove per-game routing details while retaining aggregate telemetry.

    Hidden evaluation needs enough aggregate information to audit policy use,
    but it must not emit feature vectors, candidate scores, or policy-specific
    counters keyed by hidden game.
    """

    metrics.pop("portfolio_diagnostics", None)
    for game in metrics.get("games", []):
        if not isinstance(game, dict):
            continue
        game.pop("portfolio_diagnostics", None)
        telemetry = game.get("telemetry")
        if isinstance(telemetry, dict):
            game["telemetry"] = {
                key: value
                for key, value in telemetry.items()
                if not str(key).startswith("portfolio_")
            }
    metrics["portfolio_diagnostics_scope"] = "aggregate-only"


def available_public_game_ids(environments_dir: Path) -> list[str]:
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(environments_dir),
    )
    available = [str(item.game_id) for item in arcade.get_environments()]
    by_base = {base_game_id(game_id): game_id for game_id in available}
    ordered = [
        by_base[base_game_id(game_id)]
        for game_id in DUCK_HARNESS_PUBLIC_GAME_IDS
        if base_game_id(game_id) in by_base
    ]
    if len(ordered) != 25:
        raise RuntimeError(f"expected all 25 public games, found {len(ordered)}")
    return ordered


def select_game_ids(environments_dir: Path, *, fold: str) -> list[str]:
    wanted = set(select_fold(fold))
    return [
        game_id
        for game_id in available_public_game_ids(environments_dir)
        if base_game_id(game_id) in wanted
    ]


def build_offline_benchmark(
    *,
    environments_dir: Path,
    game_ids: Sequence[str],
    solver: HarnessSolver,
    label: str,
    clones: int | None = None,
) -> taaf.benchmark.Benchmark:
    spec = taaf.game_api.ArcadeSpec(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(environments_dir),
    )
    selected = list(game_ids)
    if clones is not None:
        if clones < 1:
            raise ValueError("clones must be positive")
        selected = [selected[index % len(selected)] for index in range(clones)]
    seen: dict[str, int] = {}
    games: list[taaf.game_api.GameAPI] = []
    for game_id in selected:
        occurrence = seen.get(game_id, 0)
        seen[game_id] = occurrence + 1
        external_id = None if occurrence == 0 else f"{base_game_id(game_id)}_{occurrence}"
        games.append(
            taaf.game_api.GameAPI(
                env_name=game_id,
                external_game_id=external_id,
                arcade_spec=spec,
            )
        )
    return taaf.benchmark.Benchmark(label=label, games=games, solver=solver, n_passes=1)


def build_competition_benchmark(
    *,
    solver: HarnessSolver,
    arcade_spec: taaf.game_api.ArcadeSpec,
    game_ids: Sequence[str],
    label: str,
) -> taaf.benchmark.Benchmark:
    games = [
        taaf.game_api.GameAPI(env_name=game_id, arcade_spec=arcade_spec)
        for game_id in game_ids
    ]
    return taaf.benchmark.Benchmark(label=label, games=games, solver=solver, n_passes=1)


def make_solver(
    config: HarnessConfig,
    *,
    max_actions: int | None = None,
    scripted: bool = False,
    seed_group_size: int = 0,
) -> HarnessSolver:
    analyzer_factory = None
    if scripted:
        analyzer_factory = lambda game, index: ScriptedAnalyzer(  # noqa: E731
            game.game_run.game_id if game.game_run is not None else str(index)
        )
    common: dict[str, Any] = {
        "model": config.model_id,
        "analyzer_timeout": config.analyzer_timeout_s,
        "max_actions_per_game": max_actions,
        "max_runtime_s_per_game": (
            config.local_game_cap_s
            if config.profile == RuntimeProfile.LOCAL_MLX
            else config.reference_game_cap_s
        ),
        "concurrency": (
            config.local_workers
            if config.profile == RuntimeProfile.LOCAL_MLX
            else config.concurrency
        ),
        "analyzer_factory": analyzer_factory,
    }
    if config.mode == HarnessMode.DUCK_REFERENCE:
        if scripted:
            raise ValueError("scripted analyzers belong to ouro-hybrid mode")
        if seed_group_size:
            raise ValueError("duck-reference runs exactly one seed per kernel")
        return DuckReferenceHarnessSolver(label="duck-reference", **common)
    if config.mode == HarnessMode.DUCK_ROBUST:
        if scripted:
            raise ValueError("scripted analyzers belong to ouro-hybrid mode")
        if seed_group_size:
            raise ValueError("duck-robust runs exactly one seed per kernel")
        return DuckRobustHarnessSolver(
            label="duck-robust",
            primary_seed=config.seed,
            **common,
        )
    if config.mode == HarnessMode.DUCK_MEMORY:
        if scripted:
            return HarnessSolver(label="duck-memory-scripted", **common)
        if seed_group_size:
            raise ValueError("duck-memory runs exactly one seed per kernel")
        return DuckMemoryHarnessSolver(
            label="duck-memory",
            primary_seed=config.seed,
            **common,
        )
    if config.mode == HarnessMode.DUCK_REASONING:
        if scripted:
            return HarnessSolver(label="duck-reasoning-scripted", **common)
        if seed_group_size:
            raise ValueError("duck-reasoning runs exactly one seed per kernel")
        return DuckReasoningHarnessSolver(
            label="duck-reasoning",
            primary_seed=config.seed,
            **common,
        )
    if config.mode == HarnessMode.DUCK_DELIBERATE:
        if scripted:
            return HarnessSolver(label="duck-deliberate-scripted", **common)
        if seed_group_size:
            raise ValueError("duck-deliberate runs exactly one seed per kernel")
        return DuckDeliberateHarnessSolver(
            label="duck-deliberate",
            primary_seed=config.seed,
            **common,
        )
    if config.mode == HarnessMode.DUCK_CONTRACT:
        if scripted:
            return HarnessSolver(label="duck-contract-scripted", **common)
        if seed_group_size:
            raise ValueError("duck-contract runs exactly one seed per kernel")
        return DuckContractHarnessSolver(
            label="duck-contract",
            primary_seed=config.seed,
            **common,
        )
    if config.mode == HarnessMode.DUCK_CONTRACT_REPAIR:
        if scripted:
            return HarnessSolver(label="duck-contract-repair-scripted", **common)
        if seed_group_size:
            raise ValueError("duck-contract-repair runs exactly one seed per kernel")
        return DuckContractRepairHarnessSolver(
            label="duck-contract-repair",
            primary_seed=config.seed,
            **common,
        )
    if config.mode == HarnessMode.DUCK_AUDIT:
        if scripted:
            return HarnessSolver(label="duck-audit-scripted", **common)
        if seed_group_size:
            raise ValueError("duck-audit runs exactly one seed per kernel")
        return DuckAuditHarnessSolver(
            label="duck-audit",
            primary_seed=config.seed,
            audit_repeat_threshold=config.audit_repeat_threshold,
            audit_no_change_threshold=config.audit_no_change_threshold,
            audit_max_triggers=config.audit_max_triggers,
            **common,
        )
    if config.mode == HarnessMode.DUCK_INFORMATION:
        if scripted:
            return HarnessSolver(label="duck-information-scripted", **common)
        if seed_group_size:
            raise ValueError("duck-information runs exactly one seed per kernel")
        return DuckInformationHarnessSolver(
            label="duck-information",
            primary_seed=config.seed,
            information_no_change_threshold=config.information_no_change_threshold,
            information_max_triggers=config.information_max_triggers,
            **common,
        )
    if config.mode == HarnessMode.DUCK_HIERARCHY:
        if scripted:
            return HarnessSolver(label="duck-hierarchy-scripted", **common)
        if seed_group_size:
            raise ValueError("duck-hierarchy runs exactly one seed per kernel")
        return DuckHierarchyHarnessSolver(
            label="duck-hierarchy",
            primary_seed=config.seed,
            hierarchy_no_change_threshold=config.hierarchy_no_change_threshold,
            hierarchy_max_triggers=config.hierarchy_max_triggers,
            **common,
        )
    if config.mode == HarnessMode.DUCK_DIVERSITY:
        if scripted:
            return HarnessSolver(label="duck-diversity-scripted", **common)
        if seed_group_size:
            raise ValueError("duck-diversity runs exactly one seed per kernel")
        return DuckDiversityHarnessSolver(
            label="duck-diversity",
            primary_seed=config.seed,
            diversity_no_change_threshold=config.diversity_no_change_threshold,
            diversity_max_triggers=config.diversity_max_triggers,
            diversity_seed_offset=config.diversity_seed_offset,
            **common,
        )
    if config.mode == HarnessMode.DUCK_POETIQ:
        if scripted:
            return HarnessSolver(label="duck-poetiq-scripted", **common)
        if seed_group_size:
            raise ValueError("duck-poetiq runs exactly one seed per kernel")
        return DuckPoetiqHarnessSolver(
            label="duck-poetiq",
            primary_seed=config.seed,
            poetiq_repeat_threshold=config.poetiq_repeat_threshold,
            poetiq_no_change_threshold=config.poetiq_no_change_threshold,
            poetiq_intervention_cooldown_actions=config.poetiq_intervention_cooldown_actions,
            poetiq_max_interventions_per_level=config.poetiq_max_interventions_per_level,
            poetiq_diversity_seed_offset=config.poetiq_diversity_seed_offset,
            poetiq_yield_min_actions=config.poetiq_yield_min_actions,
            poetiq_yield_min_elapsed_s=config.poetiq_yield_min_elapsed_s,
            poetiq_yield_window=config.poetiq_yield_window,
            poetiq_yield_max_changes=config.poetiq_yield_max_changes,
            **common,
        )
    if config.mode == HarnessMode.DUCK_PORTFOLIO:
        if scripted:
            return HarnessSolver(label="duck-portfolio-scripted", **common)
        if seed_group_size:
            raise ValueError("duck-portfolio runs exactly one seed per kernel")
        return DuckPortfolioHarnessSolver(
            label="duck-portfolio",
            primary_seed=config.seed,
            portfolio_warmup_actions=config.portfolio_warmup_actions,
            portfolio_switch_min_actions=config.portfolio_switch_min_actions,
            portfolio_switch_window=config.portfolio_switch_window,
            portfolio_switch_max_changes=config.portfolio_switch_max_changes,
            portfolio_switch_min_remaining_s=config.portfolio_switch_min_remaining_s,
            portfolio_score_clip=config.portfolio_score_clip,
            portfolio_ridge_alpha=config.portfolio_ridge_alpha,
            portfolio_uncertainty_penalty=config.portfolio_uncertainty_penalty,
            portfolio_stock_margin=config.portfolio_stock_margin,
            **common,
        )
    if config.mode == HarnessMode.DUCK_RETRODICT:
        if scripted:
            return HarnessSolver(label="duck-retrodict-scripted", **common)
        if seed_group_size:
            raise ValueError("duck-retrodict runs exactly one seed per kernel")
        return DuckRetrodictHarnessSolver(
            label="duck-retrodict",
            primary_seed=config.seed,
            failure_floor=config.model_failure_floor,
            retrodict_max_rules=config.retrodict_max_rules,
            retrodict_prediction_threshold=(
                config.retrodict_prediction_threshold
            ),
            **common,
        )
    return HybridHarnessSolver(
        label="ouro-hybrid",
        **common,
        failure_floor=config.model_failure_floor,
        seed_base=int(config.seed or 0),
        seed_group_size=seed_group_size,
        scheduler_soft_deadline_s=config.soft_deadline_s,
        scheduler_reserve_s=config.setup_teardown_reserve_s,
    )


async def run_benchmark(
    benchmark: taaf.benchmark.Benchmark,
    *,
    config: HarnessConfig,
    output_path: Path,
    minimal_diagnostics: bool = True,
) -> dict[str, Any]:
    started = time.monotonic()
    benchmark.job_dir = (
        output_path.parent / f".{output_path.stem}-{os.getpid()}-work"
    )
    soft_end = datetime.now() + timedelta(seconds=config.soft_deadline_s)
    await benchmark.run(
        soft_end_time=soft_end,
        minimal_diagnostics=minimal_diagnostics,
    )
    metrics = summarize_runs(
        benchmark.game_runs,
        experiment=config.experiment,
        seed=config.seed,
        config_hash=config.config_hash,
        elapsed_seconds=time.monotonic() - started,
        mode=config.mode.value,
        prompt_sha256=prompt_sha256(
            tool_output_tokens=config.python_output_tokens
        ),
        runtime_fingerprint=runtime_fingerprint(config),
        aggregate_memory_telemetry_only=(
            config.mode == HarnessMode.DUCK_MEMORY
            and os.getenv("TAAF_MINIMAL_DIAGNOSTICS", "").strip().lower()
            in {"1", "true", "yes", "on"}
        ),
    )
    if (
        config.mode == HarnessMode.DUCK_PORTFOLIO
        and os.getenv("TAAF_MINIMAL_DIAGNOSTICS", "").strip().lower()
        in {"1", "true", "yes", "on"}
    ):
        _enforce_portfolio_aggregate_diagnostics(metrics)
    if (
        config.mode == HarnessMode.DUCK_MEMORY
        and os.getenv("TAAF_MINIMAL_DIAGNOSTICS", "").strip().lower()
        not in {"1", "true", "yes", "on"}
    ):
        trace_bundle = _write_memory_trace_bundle(
            job_dir=benchmark.job_dir,
            output_path=output_path,
        )
        if trace_bundle is not None:
            metrics["memory_trace_bundle"] = str(trace_bundle)
    if (
        config.mode == HarnessMode.DUCK_RETRODICT
        and os.getenv("OURO3_RETRODICT_TRACE", "").strip().lower()
        in {"1", "true", "yes", "on"}
    ):
        trace_bundle = _write_retrodict_trace_bundle(
            job_dir=benchmark.job_dir,
            output_path=output_path,
        )
        if trace_bundle is not None:
            metrics["retrodict_trace_bundle"] = str(trace_bundle)
    write_metrics(metrics, output_path)
    return metrics


def run_public(
    *,
    config: HarnessConfig,
    environments_dir: Path,
    fold: str,
    output_path: Path,
    max_actions: int | None = None,
    scripted: bool = False,
) -> dict[str, Any]:
    config.apply_environment()
    ids = select_game_ids(environments_dir, fold=fold)
    solver = make_solver(config, max_actions=max_actions, scripted=scripted)
    benchmark = build_offline_benchmark(
        environments_dir=environments_dir,
        game_ids=ids,
        solver=solver,
        label=f"{config.experiment}-{fold}-seed-{config.seed}",
    )
    return asyncio.run(run_benchmark(benchmark, config=config, output_path=output_path))


def run_public_multiseed(
    *,
    config: HarnessConfig,
    environments_dir: Path,
    fold: str,
    output_path: Path,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict[str, Any]:
    """Run seed groups concurrently while preserving deterministic attribution."""

    if config.mode in {
        HarnessMode.DUCK_REFERENCE,
        HarnessMode.DUCK_ROBUST,
        HarnessMode.DUCK_MEMORY,
        HarnessMode.DUCK_REASONING,
        HarnessMode.DUCK_DELIBERATE,
        HarnessMode.DUCK_CONTRACT,
        HarnessMode.DUCK_CONTRACT_REPAIR,
        HarnessMode.DUCK_AUDIT,
        HarnessMode.DUCK_INFORMATION,
        HarnessMode.DUCK_HIERARCHY,
        HarnessMode.DUCK_DIVERSITY,
        HarnessMode.DUCK_POETIQ,
        HarnessMode.DUCK_PORTFOLIO,
        HarnessMode.DUCK_RETRODICT,
    }:
        raise ValueError(
            f"{config.mode.value} seeds must run as independent Kaggle kernel versions"
        )
    if not seeds:
        raise ValueError("at least one seed is required")
    expected = list(range(int(seeds[0]), int(seeds[0]) + len(seeds)))
    if list(seeds) != expected:
        raise ValueError("seeds must be a consecutive sequence")
    seeded_config = config.with_overrides(seed=int(seeds[0]))
    seeded_config.apply_environment()
    ids = select_game_ids(environments_dir, fold=fold)
    solver = make_solver(seeded_config, seed_group_size=len(ids))
    benchmark = build_offline_benchmark(
        environments_dir=environments_dir,
        game_ids=ids,
        solver=solver,
        label=f"{config.experiment}-{fold}-seeds-{seeds[0]}-{seeds[-1]}",
        clones=len(ids) * len(seeds),
    )
    aggregate = asyncio.run(
        run_benchmark(
            benchmark,
            config=seeded_config,
            output_path=output_path,
        )
    )
    seed_runs = []
    for seed_index, seed in enumerate(seeds):
        start = seed_index * len(ids)
        end = start + len(ids)
        seed_runs.append(
            summarize_runs(
                benchmark.game_runs[start:end],
                experiment=config.experiment,
                seed=int(seed),
                config_hash=config.with_overrides(seed=int(seed)).config_hash,
                elapsed_seconds=float(aggregate["elapsed_seconds"]),
                mode=config.mode.value,
                prompt_sha256=prompt_sha256(
                    tool_output_tokens=config.python_output_tokens
                ),
                runtime_fingerprint=runtime_fingerprint(
                    config.with_overrides(seed=int(seed))
                ),
            )
        )
    aggregate["seed_runs"] = seed_runs
    aggregate["seeds"] = list(seeds)
    write_metrics(aggregate, output_path)
    return aggregate


def run_110_rehearsal(
    *,
    config: HarnessConfig,
    environments_dir: Path,
    output_path: Path,
    max_actions: int = 1,
) -> dict[str, Any]:
    """Exercise competition HTTP transport with 110 unique cloned IDs."""

    config.apply_environment()
    public_ids = available_public_game_ids(environments_dir)
    server: ContextManager[Any] = taaf.competition_arcade.CompetitionArcadeServer(
        game_ids=public_ids,
        total_runs=110,
        environments_dir=str(environments_dir),
    )
    with server as running:
        cloned_ids = list(running.exposed_game_ids)
        if len(cloned_ids) != 110 or len(set(cloned_ids)) != 110:
            raise RuntimeError("competition rehearsal did not expose 110 unique game IDs")
        empty_environments = output_path.parent / ".competition-client-environments"
        empty_environments.mkdir(parents=True, exist_ok=True)
        client_spec = taaf.game_api.ArcadeSpec(
            operation_mode=arc_agi.OperationMode.COMPETITION,
            arc_base_url=running.base_url,
            environments_dir=str(empty_environments),
        )
        solver = make_solver(config, max_actions=max_actions, scripted=True)
        benchmark = build_competition_benchmark(
            solver=solver,
            arcade_spec=client_spec,
            game_ids=cloned_ids,
            label=f"{config.experiment}-competition-110",
        )
        metrics = asyncio.run(
            run_benchmark(
                benchmark,
                config=config,
                output_path=output_path,
                minimal_diagnostics=True,
            )
        )
    metrics["unique_game_ids"] = len(
        {str(game.get("game_id")) for game in metrics.get("games", [])}
    )
    metrics["gateway_transport"] = "competition-http"
    write_metrics(metrics, output_path)
    return metrics


def discover_hidden_gateway(
    *,
    timeout_s: float = 10 * 60,
    poll_interval_s: float = 5.0,
) -> tuple[taaf.game_api.ArcadeSpec, list[str]]:
    """Wait for and discover hidden game IDs without leaking their payloads."""

    base_url = os.getenv("ARC_BASE_URL", "http://gateway:8001").strip()
    empty_environments = Path(
        os.getenv("OURO3_EMPTY_ENVIRONMENTS", "/kaggle/working/ouro3-empty-environments")
    )
    empty_environments.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    last_problem = "gateway did not answer"
    while True:
        try:
            arcade = arc_agi.Arcade(
                operation_mode=arc_agi.OperationMode.COMPETITION,
                arc_base_url=base_url,
                environments_dir=str(empty_environments),
            )
            game_ids = [str(item.game_id) for item in arcade.get_environments()]
            if len(game_ids) == 110 and len(set(game_ids)) == 110:
                break
            last_problem = (
                "competition gateway must expose 110 unique games; "
                f"found {len(game_ids)}"
            )
        except Exception as exc:
            last_problem = f"{type(exc).__name__}: {exc}"
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"competition gateway was not ready before timeout: {last_problem}"
            )
        time.sleep(max(0.01, float(poll_interval_s)))

    return (
        taaf.game_api.ArcadeSpec(
            operation_mode=arc_agi.OperationMode.COMPETITION,
            arc_base_url=base_url,
            environments_dir=str(empty_environments),
        ),
        game_ids,
    )


def configure_hidden_submission_environment() -> None:
    """Match the gateway environment used by the published Duck notebook.

    The competition gateway does not expose the public anonymous-key endpoint.
    Without the fixed rerun key, ``arc_agi.Arcade`` tries that endpoint before
    it can discover any games, so the failure is unique to a real submission
    and cannot be caught by an offline public validation.
    """

    os.environ.setdefault("ARC_API_KEY", "test-key-123")
    os.environ.setdefault("ARC_BASE_URL", "http://gateway:8001/")
    os.environ.setdefault(
        "RECORDINGS_DIR",
        "/kaggle/working/server_recording",
    )
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["TAAF_RUN_AS_SUBMISSION"] = "1"
    os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1"


def run_hidden_submission(
    *,
    config: HarnessConfig,
    output_path: Path,
) -> dict[str, Any]:
    configure_hidden_submission_environment()
    spec, game_ids = discover_hidden_gateway()
    budget = compute_submission_budget(
        total_games=len(game_ids),
        concurrency=config.concurrency,
        configured_game_cap_s=config.reference_game_cap_s,
        soft_deadline_s=config.soft_deadline_s,
        setup_teardown_reserve_s=config.setup_teardown_reserve_s,
    )
    # Keep public validation byte/config compatible while shrinking only the
    # hidden submission cap. Four 28-game waves at 7,200 seconds fit inside
    # the 9-hour Kaggle envelope with setup and teardown headroom.
    submission_config = config.with_overrides(
        reference_game_cap_s=budget.per_game_cap_s,
    )
    submission_config.apply_environment()
    solver = make_solver(submission_config)
    benchmark = build_competition_benchmark(
        solver=solver,
        arcade_spec=spec,
        game_ids=game_ids,
        label=f"{submission_config.experiment}-hidden-110",
    )
    metrics = asyncio.run(
        run_benchmark(
            benchmark,
            config=submission_config,
            output_path=output_path,
            minimal_diagnostics=True,
        )
    )
    metrics["submission_budget"] = {
        "total_games": budget.total_games,
        "concurrency": budget.concurrency,
        "waves": budget.waves,
        "per_game_cap_s": budget.per_game_cap_s,
        "worst_case_gameplay_s": budget.worst_case_gameplay_s,
        "soft_deadline_s": budget.soft_deadline_s,
        "setup_teardown_reserve_s": budget.setup_teardown_reserve_s,
        "safety_fraction": budget.safety_fraction,
    }
    write_metrics(metrics, output_path)
    return metrics


def write_smoke_submission(path: Path, *, message: str) -> None:
    """Create the save-run placeholder expected by Kaggle code competitions."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame(
            [
                {
                    "row_id": "1_0",
                    "game_id": "1",
                    "end_of_game": True,
                    "score": 1,
                }
            ]
        ).to_parquet(path, index=False)
    except Exception:
        path.write_bytes(b"PAR1")


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
