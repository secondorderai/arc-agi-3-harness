from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from PIL import Image

from ouro3.promotion import (
    evaluate_retrodict_offline_promotion,
    evaluate_retrodict_promotion,
)
from ouro3.retrodict_eval import (
    RecordedTransition,
    attach_generated_python_predictions,
    evaluate_recorded_transitions,
    load_recorded_transitions,
)


def _translation_trace(count: int = 30) -> list[RecordedTransition]:
    records = []
    for index in range(count):
        col = index % 3
        before = tuple(
            tuple(1 if cell == col else 0 for cell in range(4))
            for _row in range(1)
        )
        after = tuple(
            tuple(1 if cell == col + 1 else 0 for cell in range(4))
            for _row in range(1)
        )
        records.append(
            RecordedTransition(
                level=1,
                action="RIGHT",
                before=before,
                after=after,
                payload={},
                baseline_prediction=after,
            )
        )
    return records


def test_chronological_evaluator_scores_typed_rules_and_baseline() -> None:
    report = evaluate_recorded_transitions(_translation_trace())
    assert report["protocol"] == "chronological-prefix-then-predict-before-observe"
    assert report["typed"]["precision"] == 1.0
    assert report["typed"]["coverage"] == 1.0
    assert report["generated_python"]["precision"] == 1.0
    decision = evaluate_retrodict_offline_promotion(report)
    assert decision.passed


def test_trace_loader_accepts_world_model_payload(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps(
            {
                "timeline": [
                    {
                        "level": 1,
                        "action": {"action": "RIGHT"},
                        "before": [[1, 0]],
                        "after": [[0, 1]],
                        "level_completed": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    records = load_recorded_transitions([path])
    assert len(records) == 1
    assert records[0].action == {"action": "RIGHT"}


def test_trace_loader_recovers_one_action_memory_messages(tmp_path: Path) -> None:
    palette = [(255, 255, 255), (204, 204, 204)]

    def image_url(marker_col: int) -> str:
        image = Image.new("RGB", (64, 64), palette[0])
        image.putpixel((marker_col, 0), palette[1])
        output = io.BytesIO()
        image.save(output, format="PNG")
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()

    def message(sequence: int, marker_col: int, action: str | None) -> dict:
        prefix = (
            "No previous sequence has been executed yet."
            if action is None
            else "The code executed 1 action in the previous sequence.\n"
            f"Executed actions: {action}.\nYou are still on the same level."
        )
        return {
            "event": "message",
            "sequence": sequence,
            "game_id": "g1",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prefix + "\nCurrent state: step 1, level 1.",
                    },
                    {"type": "image_url", "image_url": {"url": image_url(marker_col)}},
                ],
            },
        }

    path = tmp_path / "memory.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(value)
            for value in (
                message(1, 0, None),
                message(2, 1, "RIGHT"),
                message(3, 2, "MOUSE(row=4, col=5)"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    records = load_recorded_transitions([path])
    assert len(records) == 2
    assert records[0].action == {"action": "RIGHT"}
    assert records[1].action == {"action": "MOUSE", "row": 4, "col": 5}
    assert records[0].before[0][0] == 1
    assert records[0].after[0][1] == 1


def test_external_generated_python_predictions_attach_by_game_and_index(
    tmp_path: Path,
) -> None:
    transitions = _translation_trace(4)
    transitions = [
        RecordedTransition(
            **{
                **record.__dict__,
                "game_id": "g1",
                "index": index,
                "baseline_prediction": None,
            }
        )
        for index, record in enumerate(transitions)
    ]
    path = tmp_path / "python-predictions.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "game_id": "g1",
                    "index": record.index,
                    "predicted_grid": [list(row) for row in record.after],
                }
            )
            for record in transitions
        )
        + "\n",
        encoding="utf-8",
    )
    attached = attach_generated_python_predictions(transitions, [path])
    assert all(record.baseline_prediction == record.after for record in attached)


def test_winning_promotion_requires_offline_pass_two_seeds_and_deadline() -> None:
    offline = evaluate_recorded_transitions(_translation_trace())
    runs = [
        {
            "seed": seed,
            "game_count": 25,
            "mean_engine_score": 1.90,
            "trimmed_mean_engine_score": 1.05,
            "nonzero_game_count": 15,
            "infrastructure_failures": [],
        }
        for seed in (0, 1)
    ]
    assert evaluate_retrodict_promotion(
        runs,
        offline_report=offline,
        rehearsal_elapsed_s=20_000,
        soft_deadline_s=31_200,
    ).passed
    assert not evaluate_retrodict_promotion(
        runs,
        offline_report=offline,
        rehearsal_elapsed_s=32_000,
        soft_deadline_s=31_200,
    ).passed
