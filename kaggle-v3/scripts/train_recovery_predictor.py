"""Train the generic recovery signal from Duck's published benchmark JSON.

The script intentionally uses only the standard library and fields available
without opaque pickle artifacts. It writes a diagnostic model even when the
leave-one-game-out promotion target is not met; runtime recovery remains
conjunctively gated by repeated-state or contradiction evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

FEATURES = (
    "elapsed_minutes",
    "generated_tokens_10k",
    "request_action_ratio",
    "zero_token_ratio",
    "dominant_action_ratio",
    "action_switch_rate",
    "unique_action_ratio",
    "mouse_ratio",
    "prior_levels",
)
SOURCE_URL = (
    "https://raw.githubusercontent.com/Tufalabs/duck-harness/"
    "main/example-run/benchmark.json"
)


@dataclass(frozen=True)
class Sample:
    game: str
    values: tuple[float, ...]
    label: int


def _features(
    history: list[dict[str, Any]],
    *,
    prior_levels: int,
    level_started_s: float,
) -> tuple[float, ...]:
    names = [
        str(record.get("action", {}).get("id", ""))
        for record in history
    ]
    count = max(1, len(names))
    requests = sum(
        int(record.get("generated_tokens", 0) or 0) > 0
        for record in history
    )
    request_ratio = requests / count
    switches = sum(
        first != second for first, second in zip(names, names[1:])
    )
    return (
        max(
            0.0,
            float(history[-1].get("wallclock_seconds", 0.0))
            - level_started_s,
        )
        / 60.0,
        sum(
            max(0, int(record.get("generated_tokens", 0) or 0))
            for record in history
        )
        / 10_000.0,
        request_ratio,
        1.0 - request_ratio,
        max((names.count(name) for name in set(names)), default=0) / count,
        switches / max(1, len(names) - 1),
        min(1.0, len(set(names)) / 6.0),
        sum(name == "ACTION6" for name in names) / count,
        min(prior_levels, 4) / 4.0,
    )


def extract_samples(payload: dict[str, Any]) -> list[Sample]:
    samples: list[Sample] = []
    for run in payload.get("game_runs", []):
        history = list(run.get("history") or [])
        offset = 0
        level_started_s = 0.0
        levels_completed = int(run.get("levels_completed", 0) or 0)
        for level, action_count in enumerate(
            run.get("actions_per_level") or []
        ):
            count = max(0, int(action_count or 0))
            level_history = history[offset : offset + count]
            if len(level_history) >= 64:
                for checkpoint in range(64, len(level_history) + 1, 16):
                    samples.append(
                        Sample(
                            game=str(run.get("game_id", ""))[:4],
                            values=_features(
                                level_history[:checkpoint],
                                prior_levels=level,
                                level_started_s=level_started_s,
                            ),
                            label=int(level < levels_completed),
                        )
                    )
            if level_history:
                level_started_s = float(
                    level_history[-1].get("wallclock_seconds", 0.0)
                )
            offset += count
    return samples


def train(
    samples: Iterable[Sample],
    *,
    epochs: int = 500,
) -> tuple[list[float], list[float], list[float]]:
    rows = list(samples)
    columns = list(zip(*(sample.values for sample in rows)))
    means = [statistics.fmean(column) for column in columns]
    scales = [
        max(1e-6, statistics.pstdev(column)) for column in columns
    ]
    standardized = [
        [
            (value - mean) / scale
            for value, mean, scale in zip(sample.values, means, scales)
        ]
        for sample in rows
    ]
    weights = [0.0] * (len(FEATURES) + 1)
    for epoch in range(max(1, epochs)):
        gradient = [0.0] * len(weights)
        for values, sample in zip(standardized, rows):
            logit = weights[0] + sum(
                weight * value
                for weight, value in zip(weights[1:], values)
            )
            probability = 1.0 / (
                1.0 + math.exp(-max(-30.0, min(30.0, logit)))
            )
            error = probability - sample.label
            gradient[0] += error
            for index, value in enumerate(values, start=1):
                gradient[index] += error * value
        rate = 0.15 / (1.0 + epoch / max(1.0, epochs / 2))
        for index in range(len(weights)):
            regularization = 0.01 * weights[index] if index else 0.0
            weights[index] -= rate * (
                gradient[index] / len(rows) + regularization
            )
    return means, scales, weights


def _predict(
    sample: Sample,
    means: list[float],
    scales: list[float],
    weights: list[float],
) -> float:
    logit = weights[0] + sum(
        weight * ((value - mean) / scale)
        for weight, value, mean, scale in zip(
            weights[1:], sample.values, means, scales
        )
    )
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))


def leave_one_game_out_auc(samples: list[Sample]) -> float:
    predictions: list[tuple[float, int]] = []
    for game in sorted({sample.game for sample in samples}):
        training = [sample for sample in samples if sample.game != game]
        testing = [sample for sample in samples if sample.game == game]
        means, scales, weights = train(training)
        predictions.extend(
            (_predict(sample, means, scales, weights), sample.label)
            for sample in testing
        )
    positive = [value for value, label in predictions if label]
    negative = [value for value, label in predictions if not label]
    if not positive or not negative:
        return 0.5
    wins = sum(
        float(first > second) + 0.5 * float(first == second)
        for first in positive
        for second in negative
    )
    return wins / (len(positive) * len(negative))


def build_artifact(source: Path) -> dict[str, Any]:
    raw = source.read_bytes()
    payload = json.loads(raw)
    samples = extract_samples(payload)
    means, scales, weights = train(samples)
    auc = leave_one_game_out_auc(samples)
    return {
        "schema_version": 1,
        "model_type": "standardized_logistic_regression",
        "purpose": "Conservative trajectory-success signal for confidence-gated recovery",
        "source": {
            "url": SOURCE_URL,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "trajectory_count": len(payload.get("game_runs", [])),
            "game_count": len({sample.game for sample in samples}),
            "license": "MIT",
            "feature_checkpoint": (
                "Every 16 actions after action 64 within a level"
            ),
        },
        "features": list(FEATURES),
        "means": means,
        "scales": scales,
        "intercept": weights[0],
        "weights": weights[1:],
        "validation": {
            "method": "leave-one-game-out",
            "sample_count": len(samples),
            "positive_count": sum(sample.label for sample in samples),
            "auc": auc,
            "promotion_target_met": auc >= 0.70,
            "note": (
                "Published benchmark JSON lacks frame-change and "
                "hypothesis-consistency features. The model is only one "
                "input to a conjunctive recovery gate."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "src"
        / "ouro3"
        / "recovery_predictor.json",
    )
    args = parser.parse_args()
    artifact = build_artifact(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "auc": artifact["validation"]["auc"],
                "promotion_target_met": artifact["validation"][
                    "promotion_target_met"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
