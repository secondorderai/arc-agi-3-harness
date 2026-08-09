from __future__ import annotations

import json
from pathlib import Path

import nbformat

from ouro3.config import HarnessConfig
from ouro3.model_ab import compare_retrodict_model_runs


def _run(seed: int, *, score: float, elapsed: float) -> dict[str, object]:
    return {
        "mode": "duck-retrodict",
        "seed": seed,
        "game_count": 25,
        "mean_engine_score": score,
        "elapsed_seconds": elapsed,
        "total_generated_tokens": 100_000,
        "infrastructure_failures": [],
        "games": [
            {"game_id": f"game-{index}"}
            for index in range(25)
        ],
    }


def test_challenger_config_requires_explicit_model_snapshot() -> None:
    config = HarnessConfig.retrodict_challenger(
        model_dataset="owner/qwen36-35b-a3b-snapshot",
        seed=0,
    )
    assert config.model_id == "vrfai/Qwen3.6-35B-A3B-FP8"
    assert config.model_dataset == "owner/qwen36-35b-a3b-snapshot"
    assert config.mode.value == "duck-retrodict"


def test_model_ab_accepts_score_neutral_faster_challenger() -> None:
    controls = [_run(seed, score=1.9, elapsed=10_000) for seed in (0, 1)]
    challengers = [_run(seed, score=1.89, elapsed=8_000) for seed in (0, 1)]
    decision = compare_retrodict_model_runs(controls, challengers)
    assert decision.passed
    assert decision.elapsed_delta_fraction == 0.2


def test_model_ab_rejects_fast_but_material_score_regression() -> None:
    controls = [_run(seed, score=1.9, elapsed=10_000) for seed in (0, 1)]
    challengers = [_run(seed, score=1.7, elapsed=7_000) for seed in (0, 1)]
    decision = compare_retrodict_model_runs(controls, challengers)
    assert not decision.passed
    assert any("score regression" in reason for reason in decision.reasons)


def test_notebook_builder_can_pin_challenger_model_and_dataset(
    tmp_path: Path,
) -> None:
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "build_notebooks.py"
    spec = importlib.util.spec_from_file_location("retrodict_builder", path)
    assert spec and spec.loader
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    dataset = "owner/qwen36-35b-a3b-snapshot"
    model_id = "vrfai/Qwen3.6-35B-A3B-FP8"
    builder.build(
        tmp_path,
        validation_seed=0,
        mode="duck-retrodict",
        model_id=model_id,
        model_dataset=dataset,
    )
    notebook = nbformat.read(
        tmp_path / "validation" / "validation.ipynb",
        as_version=4,
    )
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    metadata = json.loads(
        (tmp_path / "validation" / "kernel-metadata.json").read_text()
    )
    assert notebook.metadata["ouro3"]["model_id"] == model_id
    assert notebook.metadata["ouro3"]["model_dataset"] == dataset
    assert f"served_model_name='{model_id}'" in code
    assert dataset in metadata["dataset_sources"]
