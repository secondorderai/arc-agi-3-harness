"""Chronological evaluation for typed retrodictive world models."""

from __future__ import annotations

import base64
import gzip
import io
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

from ouro3.perception import normalize_grid
from ouro3.retrodict import RetrodictiveWorldModel


_ARC_RGB = tuple(
    tuple(round(channel * 255) for channel in color)
    for color in (
        (1.0, 1.0, 1.0),
        (0.8, 0.8, 0.8),
        (0.6, 0.6, 0.6),
        (0.4, 0.4, 0.4),
        (0.2, 0.2, 0.2),
        (0.0, 0.0, 0.0),
        (0.898, 0.227, 0.639),
        (1.0, 0.482, 0.8),
        (0.976, 0.235, 0.192),
        (0.118, 0.576, 1.0),
        (0.533, 0.847, 0.945),
        (1.0, 0.863, 0.0),
        (1.0, 0.522, 0.106),
        (0.573, 0.071, 0.192),
        (0.310, 0.800, 0.188),
        (0.639, 0.337, 0.839),
    )
)
_LEVEL_PATTERN = re.compile(r"Current state: step \d+, level (\d+)\.")
_ONE_ACTION_PATTERN = re.compile(
    r"The code executed 1 action in the previous sequence\.\s*"
    r"Executed actions: ([^\n]+?)\.\s*(?:\n|$)"
)
_MOUSE_PATTERN = re.compile(
    r"MOUSE\(row=(\d+), col=(\d+)\)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RecordedTransition:
    level: int
    action: dict[str, Any] | str
    before: tuple[tuple[int, ...], ...]
    after: tuple[tuple[int, ...], ...]
    payload: dict[str, Any]
    baseline_prediction: tuple[tuple[int, ...], ...] | None = None
    game_id: str = "game"
    index: int = 0


@dataclass(frozen=True)
class PredictionScore:
    name: str
    total: int
    predicted: int
    correct: int
    precision: float
    coverage: float
    accuracy: float
    latency_p95_ms: float


def evaluate_recorded_transitions(
    transitions: Sequence[RecordedTransition],
    *,
    train_fraction: float = 0.60,
    max_rules: int = 256,
    prediction_threshold: float = 0.90,
) -> dict[str, Any]:
    """Evaluate online predictions after a chronological calibration prefix."""

    if not transitions:
        raise ValueError("at least one recorded transition is required")
    if not 0.1 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.1, 1.0)")
    typed_predictions = 0
    typed_correct = 0
    baseline_predictions = 0
    baseline_correct = 0
    latencies_ms: list[float] = []
    calibration = 0
    holdout = 0
    final_diagnostics: dict[str, Any] = {}
    by_action: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "predicted": 0, "correct": 0}
    )
    by_source: dict[str, dict[str, int]] = defaultdict(
        lambda: {"predicted": 0, "correct": 0}
    )
    prediction_errors: list[dict[str, Any]] = []
    source_by_game: dict[str, list[RecordedTransition]] = {}
    for record in transitions:
        source_by_game.setdefault(record.game_id, []).append(record)
    by_game = {
        game_id: records
        for game_id, records in source_by_game.items()
        if len(records) >= 2
    }
    if not by_game:
        raise ValueError("at least one game needs two chronological transitions")
    for game_id, game_records in by_game.items():
        split = min(
            len(game_records) - 1,
            max(1, int(math.floor(len(game_records) * train_fraction))),
        )
        calibration += split
        holdout += len(game_records) - split
        world = RetrodictiveWorldModel(
            max_rules=max_rules,
            prediction_threshold=prediction_threshold,
        )
        for record in game_records[:split]:
            world.observe(
                level=record.level,
                action=record.action,
                before=record.before,
                after=record.after,
                payload=record.payload,
            )
        for record in game_records[split:]:
            action_name = (
                str(record.action.get("action", ""))
                if isinstance(record.action, Mapping)
                else str(record.action)
            )
            by_action[action_name]["total"] += 1
            started = time.perf_counter()
            prediction = world.predict(
                record.before,
                record.action,
                level=record.level,
            )
            latencies_ms.append((time.perf_counter() - started) * 1_000.0)
            if prediction is not None:
                typed_predictions += 1
                correct = prediction.grid == record.after
                typed_correct += int(correct)
                by_action[action_name]["predicted"] += 1
                by_action[action_name]["correct"] += int(correct)
                if any(rule_id.startswith("exact:") for rule_id in prediction.rule_ids):
                    source = "exact"
                else:
                    source = "+".join(
                        sorted(
                            {
                                world.rules[rule_id].kind
                                for rule_id in prediction.rule_ids
                                if rule_id in world.rules
                            }
                        )
                    ) or "typed"
                by_source[source]["predicted"] += 1
                by_source[source]["correct"] += int(correct)
                if not correct and len(prediction_errors) < 50:
                    prediction_errors.append(
                        {
                            "game_id": game_id,
                            "index": record.index,
                            "level": record.level,
                            "action": action_name,
                            "source": source,
                            "rule_ids": list(prediction.rule_ids),
                        }
                    )
            if record.baseline_prediction is not None:
                baseline_predictions += 1
                baseline_correct += int(record.baseline_prediction == record.after)
            # Online evaluation: the answer becomes evidence only after scoring it.
            world.observe(
                level=record.level,
                action=record.action,
                before=record.before,
                after=record.after,
                payload=record.payload,
            )
        final_diagnostics[game_id] = world.diagnostics()
    typed = _score(
        "typed-rules",
        total=holdout,
        predicted=typed_predictions,
        correct=typed_correct,
        latencies_ms=latencies_ms,
    )
    baseline = _score(
        "generated-python",
        total=holdout,
        predicted=baseline_predictions,
        correct=baseline_correct,
        latencies_ms=[],
    )
    return {
        "schema_version": 1,
        "protocol": "chronological-prefix-then-predict-before-observe",
        "game_count": len(by_game),
        "source_game_count": len(source_by_game),
        "skipped_game_ids": sorted(set(source_by_game) - set(by_game)),
        "transition_count": sum(len(records) for records in by_game.values()),
        "calibration_count": calibration,
        "holdout_count": holdout,
        "typed": asdict(typed),
        "generated_python": (
            asdict(baseline) if baseline_predictions else None
        ),
        "precision_delta_vs_generated_python": (
            typed.precision - baseline.precision
            if baseline_predictions
            else None
        ),
        "coverage_delta_vs_generated_python": (
            typed.coverage - baseline.coverage
            if baseline_predictions
            else None
        ),
        "typed_by_action": {
            action: {
                **counts,
                "precision": (
                    counts["correct"] / counts["predicted"]
                    if counts["predicted"]
                    else 0.0
                ),
                "coverage": counts["predicted"] / counts["total"],
            }
            for action, counts in sorted(by_action.items())
        },
        "typed_by_source": {
            source: {
                **counts,
                "precision": counts["correct"] / counts["predicted"],
            }
            for source, counts in sorted(by_source.items())
        },
        "prediction_errors": prediction_errors,
        "final_world_models": final_diagnostics,
    }


def load_recorded_transitions(paths: Iterable[Path]) -> list[RecordedTransition]:
    records: list[RecordedTransition] = []
    for path in paths:
        values = _load_values(path)
        records.extend(_extract_memory_message_transitions(values))
        for value in values:
            for raw in _extract_transition_dicts(value):
                records.append(_parse_transition(raw))
    if not records:
        raise ValueError("no transition records found in the supplied files")
    return records


def _extract_memory_message_transitions(
    values: Sequence[Any],
) -> list[RecordedTransition]:
    """Recover one-action transitions from compressed Stock Duck transcripts."""

    previous: dict[str, tuple[int, tuple[tuple[int, ...], ...]]] = {}
    indices: dict[str, int] = {}
    records: list[RecordedTransition] = []
    for value in values:
        if not isinstance(value, Mapping) or value.get("event") != "message":
            continue
        message = value.get("message")
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        text, image_url = _message_text_and_image(message.get("content"))
        level_match = _LEVEL_PATTERN.search(text)
        if level_match is None or not image_url:
            continue
        game_id = str(value.get("game_id", message.get("game_id", "game")) or "game")
        level = int(level_match.group(1))
        grid = _decode_arc_image(image_url)
        prior = previous.get(game_id)
        action_match = _ONE_ACTION_PATTERN.search(text)
        if prior is not None and action_match is not None:
            action = _parse_prompt_action(action_match.group(1))
            if action is not None:
                prior_level, prior_grid = prior
                index = indices.get(game_id, 0)
                records.append(
                    RecordedTransition(
                        level=prior_level,
                        action=action,
                        before=prior_grid,
                        after=grid,
                        payload={"level_completed": level != prior_level},
                        game_id=game_id,
                        index=index,
                    )
                )
                indices[game_id] = index + 1
        previous[game_id] = (level, grid)
    return records


def _message_text_and_image(content: Any) -> tuple[str, str]:
    if not isinstance(content, list):
        return "", ""
    text = ""
    image_url = ""
    for item in content:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") == "text":
            text = str(item.get("text", ""))
        elif item.get("type") == "image_url":
            image = item.get("image_url")
            if isinstance(image, Mapping):
                image_url = str(image.get("url", ""))
    return text, image_url


def _parse_prompt_action(raw: str) -> dict[str, Any] | None:
    action = raw.strip().upper()
    mouse = _MOUSE_PATTERN.fullmatch(action)
    if mouse:
        return {
            "action": "MOUSE",
            "row": int(mouse.group(1)),
            "col": int(mouse.group(2)),
        }
    if re.fullmatch(r"[A-Z][A-Z0-9_]*", action):
        return {"action": action}
    return None


def _decode_arc_image(image_url: str) -> tuple[tuple[int, ...], ...]:
    prefix = "data:image/png;base64,"
    if not image_url.startswith(prefix):
        raise ValueError("memory trace frame is not an inline PNG")
    image = Image.open(
        io.BytesIO(base64.b64decode(image_url[len(prefix) :]))
    ).convert("RGB")
    width, height = image.size
    if width < 64 or height < 64:
        raise ValueError("memory trace frame is smaller than 64x64")
    resized = image.resize((64, 64), resample=Image.Resampling.NEAREST)
    exact = {rgb: index for index, rgb in enumerate(_ARC_RGB)}
    cache: dict[tuple[int, int, int], int] = {}
    values: list[int] = []
    pixels = (
        resized.get_flattened_data()
        if hasattr(resized, "get_flattened_data")
        else resized.getdata()
    )
    for rgb in pixels:
        normalized_rgb = tuple(int(channel) for channel in rgb)
        color = exact.get(normalized_rgb)
        if color is None:
            color = cache.get(normalized_rgb)
        if color is None:
            color = min(
                range(16),
                key=lambda index: sum(
                    (normalized_rgb[channel] - _ARC_RGB[index][channel]) ** 2
                    for channel in range(3)
                ),
            )
            cache[normalized_rgb] = color
        values.append(color)
    return tuple(
        tuple(values[offset : offset + 64])
        for offset in range(0, 64 * 64, 64)
    )


def attach_generated_python_predictions(
    transitions: Sequence[RecordedTransition],
    paths: Iterable[Path],
) -> list[RecordedTransition]:
    """Attach old arbitrary-Python predictions by immutable game/index key."""

    predictions: dict[tuple[str, int], tuple[tuple[int, ...], ...]] = {}
    for path in paths:
        for value in _load_values(path):
            for raw in _extract_prediction_dicts(value):
                candidate = _baseline_grid(raw)
                if candidate is None:
                    continue
                key = (
                    str(raw.get("game_id", "game") or "game"),
                    int(raw.get("index", 0) or 0),
                )
                predictions[key] = candidate
    if not predictions:
        raise ValueError("no generated-Python predictions found")
    attached = [
        replace(
            record,
            baseline_prediction=predictions.get((record.game_id, record.index)),
        )
        for record in transitions
    ]
    if not any(record.baseline_prediction is not None for record in attached):
        raise ValueError("generated-Python predictions did not match trace keys")
    return attached


def _extract_prediction_dicts(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return [
            raw
            for item in value
            for raw in _extract_prediction_dicts(item)
        ]
    if not isinstance(value, Mapping):
        return []
    if _baseline_grid(value) is not None:
        return [value]
    output: list[Mapping[str, Any]] = []
    for key in ("timeline", "transitions", "predictions"):
        nested = value.get(key)
        if isinstance(nested, list):
            output.extend(_extract_prediction_dicts(nested))
    return output


def _score(
    name: str,
    *,
    total: int,
    predicted: int,
    correct: int,
    latencies_ms: Sequence[float],
) -> PredictionScore:
    ordered = sorted(float(value) for value in latencies_ms)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return PredictionScore(
        name=name,
        total=total,
        predicted=predicted,
        correct=correct,
        precision=correct / predicted if predicted else 0.0,
        coverage=predicted / total if total else 0.0,
        accuracy=correct / total if total else 0.0,
        latency_p95_ms=ordered[p95_index] if ordered else 0.0,
    )


def _load_values(path: Path) -> list[Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        text = handle.read()
    try:
        return [json.loads(text)]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _extract_transition_dicts(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    if not isinstance(value, Mapping):
        return []
    if "before" in value or "before_grid" in value:
        return [value]
    direct_trace = value.get("recorded_trace")
    if isinstance(direct_trace, Mapping):
        return _extract_transition_dicts(direct_trace)
    timeline = value.get("timeline") or value.get("transitions")
    if isinstance(timeline, list):
        return [item for item in timeline if isinstance(item, Mapping)]
    output: list[Mapping[str, Any]] = []
    diagnostics = value.get("retrodict_diagnostics")
    if isinstance(diagnostics, Mapping):
        for game_id, item in diagnostics.items():
            if not isinstance(item, Mapping):
                continue
            trace = item.get("recorded_trace")
            if isinstance(trace, Mapping):
                for raw in _extract_transition_dicts(trace):
                    tagged = dict(raw)
                    tagged.setdefault("game_id", str(game_id))
                    output.append(tagged)
    games = value.get("games")
    if isinstance(games, list):
        for game in games:
            if isinstance(game, Mapping):
                output.extend(_extract_transition_dicts(game))
    return output


def _parse_transition(raw: Mapping[str, Any]) -> RecordedTransition:
    before = normalize_grid(raw.get("before", raw.get("before_grid")))
    after = normalize_grid(raw.get("after", raw.get("after_grid")))
    if not before or not after:
        raise ValueError("transition is missing non-empty before/after grids")
    payload = dict(raw.get("payload") or {})
    for key in ("level_completed", "game_over", "run_complete"):
        if key in raw and key not in payload:
            payload[key] = raw[key]
    baseline = _baseline_grid(raw)
    return RecordedTransition(
        level=int(raw.get("level", 1) or 1),
        action=raw.get("action", ""),
        before=before,
        after=after,
        payload=payload,
        baseline_prediction=baseline,
        game_id=str(raw.get("game_id", "game") or "game"),
        index=int(raw.get("index", 0) or 0),
    )


def _baseline_grid(
    raw: Mapping[str, Any],
) -> tuple[tuple[int, ...], ...] | None:
    baseline_raw = raw.get(
        "generated_python_prediction",
        raw.get(
            "baseline_prediction",
            raw.get("predicted_after", raw.get("predicted_grid")),
        ),
    )
    if isinstance(baseline_raw, Mapping):
        baseline_raw = baseline_raw.get("grid")
    if baseline_raw is None:
        return None
    return normalize_grid(baseline_raw)
