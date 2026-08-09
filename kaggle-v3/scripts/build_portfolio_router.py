"""Build the transparent duck-portfolio-v1 deterministic router artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))

from duck_portfolio.router import (  # noqa: E402
    FEATURE_NAMES,
    POLICY_PRIORITY,
    PortfolioPolicy,
    PortfolioTransition,
    extract_portfolio_features,
)
from ouro3.perception import analyze_transition  # noqa: E402


SCORE_CLIP = 10.0
RIDGE_ALPHA = 10.0
UNCERTAINTY_PENALTY = 0.5
STOCK_MARGIN = 0.25
WARMUP_ACTIONS = 8

METRIC_RUNS = {
    PortfolioPolicy.STOCK: (
        "duck-reference-unseeded-v5",
        "duck-reference-seed-0-v6",
        "duck-reference-seed-1-v7",
    ),
    PortfolioPolicy.AUDIT: ("duck-audit-seed-0-v14",),
    PortfolioPolicy.DELIBERATE: ("duck-deliberate-seed-0-v11",),
    PortfolioPolicy.CONTRACT_REPAIR: (
        "duck-contract-repair-seed-0-v12",
        "duck-contract-repair-seed-0-v15",
    ),
}
STOCK_PREFIX_RUN = "duck-reference-seed-0-v6"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_game_id(value: str) -> str:
    return str(value).split("-", 1)[0]


def _metric_path(run_name: str) -> Path:
    path = ROOT / "results" / run_name / "validation_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing portfolio training metric: {path}")
    return path


def _event_paths(run_name: str) -> list[Path]:
    paths = sorted(
        (ROOT / "results" / run_name).glob("**/artifacts/*_events.jsonl")
    )
    if len(paths) != 25:
        raise RuntimeError(
            f"portfolio training requires 25 Stock event traces, found {len(paths)}"
        )
    return paths


def _load_targets() -> tuple[list[str], dict[PortfolioPolicy, list[float]], dict[str, str]]:
    per_policy: dict[PortfolioPolicy, list[dict[str, float]]] = {}
    provenance: dict[str, str] = {}
    game_ids: set[str] | None = None
    for policy, run_names in METRIC_RUNS.items():
        rows: list[dict[str, float]] = []
        for index, run_name in enumerate(run_names):
            path = _metric_path(run_name)
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = {
                _base_game_id(item["game_id"]): min(
                    SCORE_CLIP, max(0.0, float(item.get("final_score", 0.0)))
                )
                for item in payload.get("games", [])
            }
            if len(values) != 25:
                raise RuntimeError(f"{run_name} does not contain 25 unique games")
            rows.append(values)
            provenance[f"{policy.value}-metrics-{index}"] = _sha256(path)
            game_ids = set(values) if game_ids is None else game_ids & set(values)
        per_policy[policy] = rows
    ordered = sorted(game_ids or ())
    if len(ordered) != 25:
        raise RuntimeError("portfolio training artifacts do not share 25 games")
    targets = {
        policy: [
            statistics.median(row[game_id] for row in rows)
            for game_id in ordered
        ]
        for policy, rows in per_policy.items()
    }
    return ordered, targets, provenance


def _features_from_event_file(path: Path) -> dict[str, float]:
    initial_grid: Any = None
    previous_grid: Any = None
    transitions: list[PortfolioTransition] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            board = event.get("board")
            if initial_grid is None and board is not None:
                initial_grid = board
                previous_grid = board
            if event.get("type") != "action" or board is None or previous_grid is None:
                continue
            analysis = analyze_transition(previous_grid, board)
            transitions.append(
                PortfolioTransition(
                    action=str(event.get("action_display") or event.get("action_name") or ""),
                    before_grid=previous_grid,
                    after_grid=board,
                    gameplay_changed=bool(analysis["gameplay_changed"]),
                    hud_changed=bool(analysis["hud_changed"]),
                    changed_area=sum(
                        int(item.get("area", 0))
                        for item in analysis["changed_regions"]
                    ),
                )
            )
            previous_grid = board
            if len(transitions) >= WARMUP_ACTIONS:
                break
    if initial_grid is None:
        raise RuntimeError(f"event trace has no initial board: {path}")
    return extract_portfolio_features(initial_grid, transitions)


def _load_features(game_ids: Iterable[str]) -> tuple[np.ndarray, str]:
    by_game: dict[str, dict[str, float]] = {}
    trace_hashes: list[str] = []
    for path in _event_paths(STOCK_PREFIX_RUN):
        game_id = path.name.split("-", 1)[0]
        if game_id in by_game:
            raise RuntimeError(f"duplicate Stock event trace for {game_id}")
        by_game[game_id] = _features_from_event_file(path)
        trace_hashes.append(_sha256(path))
    ordered = list(game_ids)
    if set(by_game) != set(ordered):
        raise RuntimeError("Stock event traces do not match metric games")
    matrix = np.array(
        [[by_game[game_id][name] for name in FEATURE_NAMES] for game_id in ordered],
        dtype=float,
    )
    aggregate = hashlib.sha256("".join(sorted(trace_hashes)).encode("ascii")).hexdigest()
    return matrix, aggregate


def _normalization(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales < 1e-12, 1.0, scales)
    return means, scales, (matrix - means) / scales


def _fit(matrix: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    means, scales, normalized = _normalization(matrix)
    centered_target = target - target.mean()
    identity = np.eye(normalized.shape[1], dtype=float)
    coefficients = np.linalg.solve(
        normalized.T @ normalized + RIDGE_ALPHA * identity,
        normalized.T @ centered_target,
    )
    return float(target.mean()), coefficients, means, scales


def _predict(
    row: np.ndarray,
    intercept: float,
    coefficients: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> float:
    value = intercept + float(coefficients @ ((row - means) / scales))
    return min(SCORE_CLIP, max(0.0, value))


def _predict_relative(
    row: np.ndarray,
    intercept: float,
    coefficients: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> float:
    value = intercept + float(coefficients @ ((row - means) / scales))
    return min(SCORE_CLIP, max(-SCORE_CLIP, value))


def _loo_predictions(matrix: np.ndarray, targets: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for holdout in range(len(matrix)):
        keep = np.array([index != holdout for index in range(len(matrix))])
        model = _fit(matrix[keep], targets[keep])
        values.append(_predict(matrix[holdout], *model))
    return np.array(values, dtype=float)


def _loo_relative_predictions(matrix: np.ndarray, targets: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for holdout in range(len(matrix)):
        keep = np.array([index != holdout for index in range(len(matrix))])
        model = _fit(matrix[keep], targets[keep])
        values.append(_predict_relative(matrix[holdout], *model))
    return np.array(values, dtype=float)


def _holdout_safe_rmse(
    matrix: np.ndarray,
    targets: np.ndarray,
    holdout: int,
) -> float:
    """Estimate uncertainty without reading the held-out game's target.

    The outer holdout is removed first. An inner leave-one-out pass over the
    remaining games produces the RMSE used to penalize that outer prediction.
    This prevents the router's policy choice from learning anything from the
    score it is meant to predict.
    """

    keep = np.array([index != holdout for index in range(len(matrix))])
    inner_matrix = matrix[keep]
    inner_targets = targets[keep]
    inner_predictions = _loo_predictions(inner_matrix, inner_targets)
    return math.sqrt(float(np.mean((inner_predictions - inner_targets) ** 2)))


def _runtime_decision(
    row: np.ndarray,
    models: dict[PortfolioPolicy, tuple[float, np.ndarray, np.ndarray, np.ndarray]],
    rmse: dict[PortfolioPolicy, float],
    relative_models: dict[
        PortfolioPolicy, tuple[float, np.ndarray, np.ndarray, np.ndarray]
    ] | None = None,
    relative_rmse: dict[PortfolioPolicy, float] | None = None,
) -> tuple[PortfolioPolicy, dict[str, float], dict[str, float]]:
    """Apply the exact decision rule used by ``PortfolioRouter``.

    The production router fits one model per policy, applies one global
    uncertainty estimate per policy, and then applies the Stock margin.  The
    original CV evaluated a different rule: it used a holdout-specific RMSE
    and a prediction that had already been trained without the holdout.  That
    made the offline gate optimistic and, more importantly, meant its selected
    policy counts did not describe the artifact used at runtime.
    """

    raw: dict[str, float] = {}
    adjusted: dict[str, float] = {}
    for policy in POLICY_PRIORITY:
        prediction = _predict(row, *models[policy])
        raw[policy.value] = prediction
        adjusted[policy.value] = prediction - UNCERTAINTY_PENALTY * rmse[policy]
    if relative_models and relative_rmse:
        stock_adjusted = adjusted[PortfolioPolicy.STOCK.value]
        for policy in POLICY_PRIORITY:
            if policy == PortfolioPolicy.STOCK:
                continue
            relative = _predict_relative(row, *relative_models[policy])
            # Relative models are trained on candidate minus Stock.  The
            # lower confidence bound is a conservative score-regression
            # guardrail: a candidate must show generic, stock-relative lift
            # after its own uncertainty is paid for.
            relative_lower = relative - UNCERTAINTY_PENALTY * relative_rmse[policy]
            adjusted[policy.value] = min(
                adjusted[policy.value], stock_adjusted + relative_lower
            )
    priority = {policy: order for order, policy in enumerate(POLICY_PRIORITY)}
    ranked = sorted(
        POLICY_PRIORITY,
        key=lambda policy: (-adjusted[policy.value], priority[policy]),
    )
    choice = ranked[0]
    if (
        choice != PortfolioPolicy.STOCK
        and adjusted[choice.value]
        <= adjusted[PortfolioPolicy.STOCK.value] + STOCK_MARGIN
    ):
        choice = PortfolioPolicy.STOCK
    return choice, raw, adjusted


def _fit_runtime_models(
    matrix: np.ndarray,
    target_arrays: dict[PortfolioPolicy, np.ndarray],
) -> tuple[
    dict[PortfolioPolicy, tuple[float, np.ndarray, np.ndarray, np.ndarray]],
    dict[PortfolioPolicy, float],
]:
    """Fit the model and one uncertainty scalar exactly as production does."""

    models = {
        policy: _fit(matrix, target_arrays[policy])
        for policy in POLICY_PRIORITY
    }
    rmse: dict[PortfolioPolicy, float] = {}
    for policy in POLICY_PRIORITY:
        predictions = _loo_predictions(matrix, target_arrays[policy])
        rmse[policy] = math.sqrt(
            float(np.mean((predictions - target_arrays[policy]) ** 2))
        )
    return models, rmse


def _fit_relative_models(
    matrix: np.ndarray,
    target_arrays: dict[PortfolioPolicy, np.ndarray],
) -> tuple[
    dict[PortfolioPolicy, tuple[float, np.ndarray, np.ndarray, np.ndarray]],
    dict[PortfolioPolicy, float],
]:
    """Fit candidate-minus-Stock models for conservative routing."""

    stock = target_arrays[PortfolioPolicy.STOCK]
    relative_models: dict[
        PortfolioPolicy, tuple[float, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    relative_rmse: dict[PortfolioPolicy, float] = {}
    for policy in POLICY_PRIORITY:
        if policy == PortfolioPolicy.STOCK:
            continue
        relative_target = target_arrays[policy] - stock
        relative_models[policy] = _fit(matrix, relative_target)
        predictions = _loo_relative_predictions(matrix, relative_target)
        relative_rmse[policy] = math.sqrt(
            float(np.mean((predictions - relative_target) ** 2))
        )
    return relative_models, relative_rmse


def _runtime_parity_cv(
    matrix: np.ndarray,
    target_arrays: dict[PortfolioPolicy, np.ndarray],
) -> tuple[list[PortfolioPolicy], list[float]]:
    """Leave-one-game-out evaluation of the production scoring function.

    Every outer fold fits each candidate on the other games, computes its
    uncertainty from that training fold only, and routes the held-out feature
    through the same score/margin/tie-break rule as the live router.  The
    held-out target is read only after the policy has been selected.
    """

    selected: list[PortfolioPolicy] = []
    routed_scores: list[float] = []
    for holdout in range(len(matrix)):
        keep = np.array([index != holdout for index in range(len(matrix))])
        fold_targets = {
            policy: values[keep] for policy, values in target_arrays.items()
        }
        models, rmse = _fit_runtime_models(matrix[keep], fold_targets)
        relative_models, relative_rmse = _fit_relative_models(
            matrix[keep], fold_targets
        )
        choice, _raw, _adjusted = _runtime_decision(
            matrix[holdout], models, rmse, relative_models, relative_rmse
        )
        selected.append(choice)
        routed_scores.append(float(target_arrays[choice][holdout]))
    return selected, routed_scores


def build_payload() -> dict[str, Any]:
    game_ids, target_rows, provenance = _load_targets()
    matrix, trace_hash = _load_features(game_ids)
    target_arrays = {
        policy: np.array(values, dtype=float)
        for policy, values in target_rows.items()
    }
    loo = {
        policy: _loo_predictions(matrix, target)
        for policy, target in target_arrays.items()
    }
    rmse = {
        policy: math.sqrt(float(np.mean((loo[policy] - target_arrays[policy]) ** 2)))
        for policy in POLICY_PRIORITY
    }
    holdout_rmse = {
        policy: [
            _holdout_safe_rmse(matrix, target_arrays[policy], holdout)
            for holdout in range(len(matrix))
        ]
        for policy in POLICY_PRIORITY
    }
    selected: list[PortfolioPolicy] = []
    routed_scores: list[float] = []
    for index in range(len(matrix)):
        adjusted = {
            policy: float(
                loo[policy][index]
                - UNCERTAINTY_PENALTY * holdout_rmse[policy][index]
            )
            for policy in POLICY_PRIORITY
        }
        priority = {policy: order for order, policy in enumerate(POLICY_PRIORITY)}
        ranked = sorted(
            POLICY_PRIORITY,
            key=lambda policy: (-adjusted[policy], priority[policy]),
        )
        choice = ranked[0]
        if (
            choice != PortfolioPolicy.STOCK
            and adjusted[choice]
            <= adjusted[PortfolioPolicy.STOCK] + STOCK_MARGIN
        ):
            choice = PortfolioPolicy.STOCK
        selected.append(choice)
        routed_scores.append(float(target_arrays[choice][index]))

    stock_values = target_arrays[PortfolioPolicy.STOCK]
    routed_mean = statistics.fmean(routed_scores)
    stock_mean = float(np.mean(stock_values))
    routed_nonzero = sum(value > 0 for value in routed_scores)
    stock_nonzero = int(np.sum(stock_values > 0))
    selected_counts = Counter(policy.value for policy in selected)
    distinct_non_stock = sum(
        selected_counts.get(policy.value, 0) > 0
        for policy in POLICY_PRIORITY
        if policy != PortfolioPolicy.STOCK
    )
    cross_validation = {
        "method": "leave-one-game-out",
        "uncertainty_method": "outer-holdout-safe-inner-loo-rmse",
        "game_count": len(matrix),
        "routed_clipped_mean": routed_mean,
        "stock_clipped_mean": stock_mean,
        "mean_lift": routed_mean - stock_mean,
        "routed_nonzero_games": routed_nonzero,
        "stock_nonzero_games": stock_nonzero,
        "selected_policy_counts": dict(sorted(selected_counts.items())),
        "distinct_non_stock_policies": distinct_non_stock,
        "passed": bool(
            routed_mean >= stock_mean + 0.10
            and routed_nonzero >= stock_nonzero
            and distinct_non_stock >= 2
        ),
    }

    means, scales, _ = _normalization(matrix)
    models: dict[str, Any] = {}
    for policy in POLICY_PRIORITY:
        intercept, coefficients, _policy_means, _policy_scales = _fit(
            matrix, target_arrays[policy]
        )
        models[policy.value] = {
            "intercept": intercept,
            "coefficients": {
                name: float(coefficients[index])
                for index, name in enumerate(FEATURE_NAMES)
            },
            "loo_rmse": rmse[policy],
        }
    provenance["stock-prefix-events"] = trace_hash
    return {
        "schema_version": 1,
        "experiment": "duck-portfolio-v1",
        "candidate_order": [policy.value for policy in POLICY_PRIORITY],
        "feature_names": list(FEATURE_NAMES),
        "warmup_actions": WARMUP_ACTIONS,
        "score_clip": SCORE_CLIP,
        "ridge_alpha": RIDGE_ALPHA,
        "uncertainty_penalty": UNCERTAINTY_PENALTY,
        "stock_margin": STOCK_MARGIN,
        "feature_means": {
            name: float(means[index]) for index, name in enumerate(FEATURE_NAMES)
        },
        "feature_scales": {
            name: float(scales[index]) for index, name in enumerate(FEATURE_NAMES)
        },
        "models": models,
        "cross_validation": cross_validation,
        "training_artifact_sha256": dict(sorted(provenance.items())),
        "forbidden_runtime_inputs": [
            "game identifiers",
            "coordinates",
            "full-frame hashes",
            "public-game rules",
        ],
    }


def build_parity_payload() -> dict[str, Any]:
    """Build the next portfolio artifact with runtime/CV parity.

    ``build_payload`` remains the audited v1 artifact builder.  Keeping this
    candidate separate makes the comparison reproducible and prevents a
    failed iteration from silently changing the reference portfolio result.
    """

    game_ids, target_rows, provenance = _load_targets()
    matrix, trace_hash = _load_features(game_ids)
    target_arrays = {
        policy: np.array(values, dtype=float)
        for policy, values in target_rows.items()
    }
    selected, routed_scores = _runtime_parity_cv(matrix, target_arrays)
    stock_values = target_arrays[PortfolioPolicy.STOCK]
    selected_counts = Counter(policy.value for policy in selected)
    routed_mean = statistics.fmean(routed_scores)
    stock_mean = float(np.mean(stock_values))
    routed_nonzero = sum(value > 0 for value in routed_scores)
    stock_nonzero = int(np.sum(stock_values > 0))
    distinct_non_stock = sum(
        selected_counts.get(policy.value, 0) > 0
        for policy in POLICY_PRIORITY
        if policy != PortfolioPolicy.STOCK
    )
    models, rmse = _fit_runtime_models(matrix, target_arrays)
    relative_models, relative_rmse = _fit_relative_models(matrix, target_arrays)
    means, scales, _ = _normalization(matrix)
    model_payload: dict[str, Any] = {}
    for policy in POLICY_PRIORITY:
        intercept, coefficients, _policy_means, _policy_scales = models[policy]
        model_payload[policy.value] = {
            "intercept": intercept,
            "coefficients": {
                name: float(coefficients[index])
                for index, name in enumerate(FEATURE_NAMES)
            },
            "loo_rmse": rmse[policy],
        }
    relative_payload: dict[str, Any] = {}
    for policy in POLICY_PRIORITY:
        if policy == PortfolioPolicy.STOCK:
            continue
        intercept, coefficients, _policy_means, _policy_scales = relative_models[policy]
        relative_payload[policy.value] = {
            "intercept": intercept,
            "coefficients": {
                name: float(coefficients[index])
                for index, name in enumerate(FEATURE_NAMES)
            },
            "loo_rmse": relative_rmse[policy],
        }
    provenance["stock-prefix-events"] = trace_hash
    cross_validation = {
        "method": "leave-one-game-out-runtime-parity",
        "uncertainty_method": "training-fold-loo-rmse-global-per-policy",
        "game_count": len(matrix),
        "routed_clipped_mean": routed_mean,
        "stock_clipped_mean": stock_mean,
        "mean_lift": routed_mean - stock_mean,
        "routed_nonzero_games": routed_nonzero,
        "stock_nonzero_games": stock_nonzero,
        "selected_policy_counts": dict(sorted(selected_counts.items())),
        "distinct_non_stock_policies": distinct_non_stock,
        "passed": bool(
            routed_mean >= stock_mean + 0.10
            and routed_nonzero >= stock_nonzero
            and distinct_non_stock >= 2
        ),
    }
    return {
        "schema_version": 1,
        "experiment": "duck-portfolio-parity-v1",
        "candidate_order": [policy.value for policy in POLICY_PRIORITY],
        "feature_names": list(FEATURE_NAMES),
        "warmup_actions": WARMUP_ACTIONS,
        "score_clip": SCORE_CLIP,
        "ridge_alpha": RIDGE_ALPHA,
        "uncertainty_penalty": UNCERTAINTY_PENALTY,
        "stock_margin": STOCK_MARGIN,
        "relative_models": relative_payload,
        "relative_uncertainty_penalty": UNCERTAINTY_PENALTY,
        "relative_stock_margin": STOCK_MARGIN,
        "feature_means": {
            name: float(means[index]) for index, name in enumerate(FEATURE_NAMES)
        },
        "feature_scales": {
            name: float(scales[index]) for index, name in enumerate(FEATURE_NAMES)
        },
        "models": model_payload,
        "cross_validation": cross_validation,
        "training_artifact_sha256": dict(sorted(provenance.items())),
        "forbidden_runtime_inputs": [
            "game identifiers",
            "coordinates",
            "full-frame hashes",
            "public-game rules",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "src" / "duck_portfolio" / "router_model.json",
    )
    parser.add_argument(
        "--runtime-parity",
        action="store_true",
        help="build the isolated runtime/CV-parity candidate artifact",
    )
    args = parser.parse_args()
    payload = build_parity_payload() if args.runtime_parity else build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["cross_validation"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
