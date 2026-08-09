"""Transparent deterministic router for the Duck policy portfolio.

The runtime artifact contains only aggregate normalization statistics and
linear coefficients.  It deliberately cannot contain game identifiers,
coordinates, frame hashes, or public-game lookup rules.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

from ouro3.perception import normalize_grid, segment_grid


FEATURE_NAMES = (
    "non_background_color_count",
    "component_count",
    "repeated_shape_fraction",
    "symmetry_fraction",
    "mouse_action_fraction",
    "gameplay_change_fraction",
    "hud_only_change_fraction",
    "repeated_action_fraction",
    "changed_area_fraction",
)

FORBIDDEN_ARTIFACT_KEYS = {
    "game_id",
    "game_ids",
    "coordinates",
    "fixed_coordinates",
    "frame_hash",
    "frame_hashes",
    "board_hash",
    "public_rules",
}

ALLOWED_TOP_LEVEL_KEYS = {
    "schema_version",
    "experiment",
    "candidate_order",
    "feature_names",
    "warmup_actions",
    "score_clip",
    "ridge_alpha",
    "uncertainty_penalty",
    "stock_margin",
    "feature_means",
    "feature_scales",
    "models",
    "relative_models",
    "relative_uncertainty_penalty",
    "relative_stock_margin",
    "cross_validation",
    "training_artifact_sha256",
    "forbidden_runtime_inputs",
}
ALLOWED_MODEL_KEYS = {"intercept", "coefficients", "loo_rmse"}
ALLOWED_RELATIVE_MODEL_KEYS = {
    "intercept",
    "coefficients",
    "loo_rmse",
}
ALLOWED_CROSS_VALIDATION_KEYS = {
    "method",
    "uncertainty_method",
    "game_count",
    "routed_clipped_mean",
    "stock_clipped_mean",
    "mean_lift",
    "routed_nonzero_games",
    "stock_nonzero_games",
    "selected_policy_counts",
    "distinct_non_stock_policies",
    "passed",
}
EXPECTED_PROVENANCE_KEYS = {
    "stock-metrics-0",
    "stock-metrics-1",
    "stock-metrics-2",
    "audit-metrics-0",
    "deliberate-metrics-0",
    "contract-repair-metrics-0",
    "contract-repair-metrics-1",
    "stock-prefix-events",
}


class PortfolioPolicy(StrEnum):
    STOCK = "stock"
    AUDIT = "audit"
    DELIBERATE = "deliberate"
    CONTRACT_REPAIR = "contract-repair"


POLICY_PRIORITY = (
    PortfolioPolicy.STOCK,
    PortfolioPolicy.AUDIT,
    PortfolioPolicy.DELIBERATE,
    PortfolioPolicy.CONTRACT_REPAIR,
)


@dataclass(frozen=True)
class PortfolioTransition:
    action: str
    before_grid: Any
    after_grid: Any
    gameplay_changed: bool
    hud_changed: bool
    changed_area: int


@dataclass(frozen=True)
class PortfolioDecision:
    policy: PortfolioPolicy
    raw_scores: dict[str, float]
    adjusted_scores: dict[str, float]
    stock_fallback: bool
    confidence_margin: float


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _symmetry_fraction(grid: Sequence[Sequence[int]]) -> float:
    rows = len(grid)
    cols = max((len(row) for row in grid), default=0)
    if not rows or not cols:
        return 0.0
    horizontal_matches = 0
    vertical_matches = 0
    cells = rows * cols
    for row in range(rows):
        for col in range(cols):
            value = grid[row][col] if col < len(grid[row]) else None
            other_row = rows - row - 1
            other_col = cols - col - 1
            horizontal = (
                grid[other_row][col]
                if col < len(grid[other_row])
                else None
            )
            vertical = (
                grid[row][other_col]
                if other_col < len(grid[row])
                else None
            )
            horizontal_matches += int(value == horizontal)
            vertical_matches += int(value == vertical)
    return max(horizontal_matches, vertical_matches) / cells


def extract_portfolio_features(
    initial_grid: Any,
    transitions: Sequence[PortfolioTransition],
) -> dict[str, float]:
    """Return the fixed, generic nine-feature routing vector."""

    grid = normalize_grid(initial_grid)
    flat = [cell for row in grid for cell in row]
    rows = len(grid)
    cols = max((len(row) for row in grid), default=0)
    cell_count = max(1, rows * cols)
    background = Counter(flat).most_common(1)[0][0] if flat else 0
    colors = {cell for cell in flat if cell != background}
    components = segment_grid(grid)
    shape_counts = Counter(
        (int(item["color"]), str(item["shape_hash"])) for item in components
    )
    repeated = sum(
        1
        for item in components
        if shape_counts[(int(item["color"]), str(item["shape_hash"]))] > 1
    )

    items = list(transitions)
    transition_count = max(1, len(items))
    actions = [" ".join(item.action.upper().split()) for item in items]
    repeated_actions = sum(
        first == second for first, second in zip(actions, actions[1:])
    )
    changed_area = sum(max(0, int(item.changed_area)) for item in items)
    values = {
        "non_background_color_count": _clamp01(len(colors) / 15.0),
        "component_count": _clamp01(len(components) / 256.0),
        "repeated_shape_fraction": _clamp01(
            repeated / max(1, len(components))
        ),
        "symmetry_fraction": _clamp01(_symmetry_fraction(grid)),
        "mouse_action_fraction": _clamp01(
            sum(action.startswith("MOUSE") for action in actions)
            / transition_count
        ),
        "gameplay_change_fraction": _clamp01(
            sum(item.gameplay_changed for item in items) / transition_count
        ),
        "hud_only_change_fraction": _clamp01(
            sum(item.hud_changed and not item.gameplay_changed for item in items)
            / transition_count
        ),
        "repeated_action_fraction": _clamp01(
            repeated_actions / max(1, len(actions) - 1)
        ),
        "changed_area_fraction": _clamp01(
            changed_area / (transition_count * cell_count)
        ),
    }
    return {name: float(values[name]) for name in FEATURE_NAMES}


def _artifact_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_ARTIFACT_KEYS:
                raise ValueError(
                    f"portfolio router artifact contains forbidden key {key!r}"
                )
            _reject_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_keys(item)


def _validate_artifact_schema(payload: Mapping[str, Any]) -> None:
    """Constrain the artifact to aggregate linear-model fields only."""

    unknown = set(payload) - ALLOWED_TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            "portfolio router artifact contains unsupported fields: "
            + ", ".join(sorted(map(str, unknown)))
        )
    models = payload.get("models")
    if isinstance(models, Mapping):
        for policy, model in models.items():
            if not isinstance(model, Mapping):
                raise ValueError(f"portfolio model {policy!r} must be an object")
            unknown_model = set(model) - ALLOWED_MODEL_KEYS
            if unknown_model:
                raise ValueError(
                    f"portfolio model {policy!r} contains unsupported fields"
                )
            coefficients = model.get("coefficients")
            if not isinstance(coefficients, Mapping) or set(coefficients) != set(
                FEATURE_NAMES
            ):
                raise ValueError(
                    f"portfolio model {policy!r} coefficient schema mismatch"
                )
    relative_models = payload.get("relative_models")
    if relative_models is not None:
        if not isinstance(relative_models, Mapping):
            raise ValueError("portfolio relative models must be an object")
        expected_relative = {policy.value for policy in POLICY_PRIORITY if policy != PortfolioPolicy.STOCK}
        if set(relative_models) != expected_relative:
            raise ValueError("portfolio relative model candidate schema mismatch")
        for policy, model in relative_models.items():
            if not isinstance(model, Mapping):
                raise ValueError(f"portfolio relative model {policy!r} must be an object")
            unknown_relative = set(model) - ALLOWED_RELATIVE_MODEL_KEYS
            if unknown_relative:
                raise ValueError(
                    f"portfolio relative model {policy!r} contains unsupported fields"
                )
            coefficients = model.get("coefficients")
            if not isinstance(coefficients, Mapping) or set(coefficients) != set(FEATURE_NAMES):
                raise ValueError(
                    f"portfolio relative model {policy!r} coefficient schema mismatch"
                )
    cross_validation = payload.get("cross_validation")
    if cross_validation is not None:
        if not isinstance(cross_validation, Mapping):
            raise ValueError("portfolio cross-validation data must be an object")
        unknown_cv = set(cross_validation) - ALLOWED_CROSS_VALIDATION_KEYS
        if unknown_cv:
            raise ValueError(
                "portfolio cross-validation contains unsupported fields: "
                + ", ".join(sorted(map(str, unknown_cv)))
            )
    provenance = payload.get("training_artifact_sha256")
    if provenance is not None:
        if not isinstance(provenance, Mapping) or set(provenance) != (
            EXPECTED_PROVENANCE_KEYS
        ):
            raise ValueError("portfolio training provenance schema mismatch")
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in provenance.values()
        ):
            raise ValueError("portfolio training provenance contains an invalid SHA-256")


class PortfolioRouter:
    """Load and evaluate the committed deterministic linear router."""

    def __init__(self, payload: Mapping[str, Any], *, artifact_hash: str = "") -> None:
        self.payload = dict(payload)
        _reject_forbidden_keys(self.payload)
        _validate_artifact_schema(self.payload)
        if int(self.payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported portfolio router schema")
        if tuple(self.payload.get("feature_names") or ()) != FEATURE_NAMES:
            raise ValueError("portfolio router feature schema mismatch")
        candidates = tuple(self.payload.get("candidate_order") or ())
        if candidates != tuple(policy.value for policy in POLICY_PRIORITY):
            raise ValueError("portfolio router candidate order mismatch")
        self.score_clip = float(self.payload.get("score_clip", 10.0))
        self.uncertainty_penalty = float(
            self.payload.get("uncertainty_penalty", 0.5)
        )
        self.stock_margin = float(self.payload.get("stock_margin", 0.25))
        self.relative_uncertainty_penalty = float(
            self.payload.get("relative_uncertainty_penalty", 0.5)
        )
        self.relative_stock_margin = float(
            self.payload.get("relative_stock_margin", self.stock_margin)
        )
        self.means = {
            str(key): float(value)
            for key, value in dict(self.payload.get("feature_means") or {}).items()
        }
        self.scales = {
            str(key): max(1e-12, float(value))
            for key, value in dict(self.payload.get("feature_scales") or {}).items()
        }
        self.models = dict(self.payload.get("models") or {})
        self.relative_models = dict(self.payload.get("relative_models") or {})
        if set(self.means) != set(FEATURE_NAMES) or set(self.scales) != set(
            FEATURE_NAMES
        ):
            raise ValueError("portfolio router normalization schema mismatch")
        if set(self.models) != {policy.value for policy in POLICY_PRIORITY}:
            raise ValueError("portfolio router is missing a candidate model")
        self.artifact_hash = artifact_hash

    @classmethod
    def default_path(cls) -> Path:
        return Path(__file__).with_name("router_model.json")

    @classmethod
    def load(cls, path: Path | None = None) -> "PortfolioRouter":
        configured_path = os.environ.get("OURO3_PORTFOLIO_ROUTER_PATH", "").strip()
        model_path = path or (Path(configured_path) if configured_path else cls.default_path())
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("portfolio router artifact must be a JSON object")
        return cls(payload, artifact_hash=_artifact_hash(model_path))

    def standardized(self, features: Mapping[str, float]) -> dict[str, float]:
        if set(features) != set(FEATURE_NAMES):
            raise ValueError("portfolio feature vector does not match the schema")
        return {
            name: (float(features[name]) - self.means[name]) / self.scales[name]
            for name in FEATURE_NAMES
        }

    def score(self, features: Mapping[str, float]) -> tuple[dict[str, float], dict[str, float]]:
        normalized = self.standardized(features)
        raw: dict[str, float] = {}
        adjusted: dict[str, float] = {}
        for policy in POLICY_PRIORITY:
            model = dict(self.models[policy.value])
            coefficients = dict(model.get("coefficients") or {})
            prediction = float(model.get("intercept", 0.0)) + sum(
                float(coefficients[name]) * normalized[name]
                for name in FEATURE_NAMES
            )
            prediction = max(0.0, min(self.score_clip, prediction))
            raw[policy.value] = prediction
            adjusted[policy.value] = prediction - self.uncertainty_penalty * float(
                model.get("loo_rmse", 0.0)
            )
        if self.relative_models:
            stock_adjusted = adjusted[PortfolioPolicy.STOCK.value]
            for policy in POLICY_PRIORITY:
                if policy == PortfolioPolicy.STOCK:
                    continue
                model = dict(self.relative_models[policy.value])
                coefficients = dict(model.get("coefficients") or {})
                relative = float(model.get("intercept", 0.0)) + sum(
                    float(coefficients[name]) * normalized[name]
                    for name in FEATURE_NAMES
                )
                relative = max(-self.score_clip, min(self.score_clip, relative))
                relative_lower = relative - self.relative_uncertainty_penalty * float(
                    model.get("loo_rmse", 0.0)
                )
                adjusted[policy.value] = min(
                    adjusted[policy.value], stock_adjusted + relative_lower
                )
        return raw, adjusted

    @staticmethod
    def _ordered_scores(scores: Mapping[str, float]) -> list[PortfolioPolicy]:
        priority = {policy: index for index, policy in enumerate(POLICY_PRIORITY)}
        return sorted(
            POLICY_PRIORITY,
            key=lambda policy: (-float(scores[policy.value]), priority[policy]),
        )

    def decide(self, features: Mapping[str, float]) -> PortfolioDecision:
        raw, adjusted = self.score(features)
        ranked = self._ordered_scores(adjusted)
        best = ranked[0]
        stock_score = adjusted[PortfolioPolicy.STOCK.value]
        stock_fallback = False
        if best != PortfolioPolicy.STOCK and adjusted[best.value] <= stock_score + self.relative_stock_margin:
            best = PortfolioPolicy.STOCK
            stock_fallback = True
        second = max(
            (
                adjusted[policy.value]
                for policy in POLICY_PRIORITY
                if policy != best
            ),
            default=adjusted[best.value],
        )
        return PortfolioDecision(
            policy=best,
            raw_scores=raw,
            adjusted_scores=adjusted,
            stock_fallback=stock_fallback,
            confidence_margin=float(adjusted[best.value] - second),
        )

    def next_policy(
        self,
        adjusted_scores: Mapping[str, float],
        current: PortfolioPolicy,
    ) -> PortfolioPolicy:
        return next(
            policy
            for policy in self._ordered_scores(adjusted_scores)
            if policy != current
        )

    @property
    def cross_validation(self) -> dict[str, Any]:
        return dict(self.payload.get("cross_validation") or {})
