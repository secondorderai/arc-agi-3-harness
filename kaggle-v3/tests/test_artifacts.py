from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_dataset_has_manifest_and_no_pickles(tmp_path: Path) -> None:
    package_source = _load_script("package_source.py")
    output = tmp_path / "source"
    manifest = package_source.build(output)
    assert manifest["file_count"] > 40
    assert not list(output.rglob("*.pkl"))
    assert "dataset-metadata.json" not in manifest["files"]
    for relative, metadata in manifest["files"].items():
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == metadata["sha256"]
    dataset_metadata = json.loads((output / "dataset-metadata.json").read_text())
    assert dataset_metadata["id"] == "kinwochan/ouroboros-arc-agi-3-v3-source"
    assert dataset_metadata["isPrivate"] is True
    oracle = json.loads(
        (output / "baselines" / "duck-public-oracle.json").read_text()
    )
    assert oracle["overall_mean_engine_score"] == 1.600203536751952
    assert (
        oracle["source_sha256"]
        == "2953fae7696bef0d6e824f7565904ca2badfb55cf256686a1be9a08997f05460"
    )
    router_path = output / "src" / "duck_portfolio" / "router_model.json"
    assert router_path.is_file()
    router = json.loads(router_path.read_text())
    assert router["cross_validation"]["passed"] is True
    assert router["candidate_order"] == [
        "stock",
        "audit",
        "deliberate",
        "contract-repair",
    ]
    pyproject = (output / "pyproject.toml").read_text()
    assert '"duck_portfolio*"' in pyproject
    assert '"duck_retrodict*"' in pyproject
    assert 'duck_portfolio = ["router_model.json", "router_model_parity.json"]' in pyproject


def test_notebooks_are_reader_facing_and_pin_rtx_metadata(tmp_path: Path) -> None:
    builder = _load_script("build_notebooks.py")
    builder.build(tmp_path)
    for kind in ("validation", "submission"):
        notebook = nbformat.read(tmp_path / kind / f"{kind}.ipynb", as_version=4)
        headings = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "markdown"
        )
        for heading in ("# ", "## Goal", "## Setup", "## Runtime checks", "## Run", "## Checks", "## Next steps"):
            assert heading in headings
        code = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        assert '"vllm": "0.19.0"' in code
        assert '"torch": "2.10.0"' in code
        assert '"flashinfer-python": "0.6.6"' in code
        assert '"flashinfer-cubin": "0.6.6"' in code
        assert '"arc-agi==0.9.8"' in code
        assert 'cuda_library_path = "/usr/local/nvidia/lib64"' in code
        assert "await asyncio.to_thread" in code
        assert "OURO3_NOTEBOOK_DRY_RUN" in code
        assert "run_public_multiseed" not in code
        assert '"OURO3_HARNESS_MODE": "duck-reference"' in code
        assert '"LOCAL_ANALYZER_TOOL_STEPS": "0"' in code
        assert '"LOCAL_ANALYZER_TIMEOUT": "900"' in code
        assert '"LOCAL_ANALYZER_YIELD_SECONDS": "60"' in code
        assert 'os.environ["TAAF_RUN_AS_SUBMISSION"]' in code
        assert 'os.environ["TAAF_MINIMAL_DIAGNOSTICS"]' in code
        assert 'os.environ.setdefault("ARC_API_KEY", "test-key-123")' in code
        assert 'os.environ.setdefault("ARC_BASE_URL", "http://gateway:8001/")' in code
        assert notebook.metadata["ouro3"]["mode"] == "duck-reference"
        assert notebook.metadata["ouro3"]["one_seed_per_kernel"] is True
        metadata = json.loads(
            (tmp_path / kind / "kernel-metadata.json").read_text()
        )
        assert metadata["enable_gpu"] is True
        assert metadata["enable_internet"] is False
        assert metadata["machine_shape"] == "NvidiaRtxPro6000"
        assert metadata["dataset_sources"] == [
            "kinwochan/ouroboros-arc-agi-3-v3-source",
            "driessmit1/arc3-vllm-h100-wheelhouse-v3",
            "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
        ]
        assert metadata["competition_sources"] == ["arc-prize-2026-arc-agi-3"]


def test_validation_notebook_embeds_one_seed_per_artifact(tmp_path: Path) -> None:
    builder = _load_script("build_notebooks.py")
    observed = []
    for seed in (None, 0, 1, 2, 3, 4):
        output = tmp_path / ("unseeded" if seed is None else f"seed-{seed}")
        builder.build(output, validation_seed=seed)
        notebook = nbformat.read(
            output / "validation" / "validation.ipynb",
            as_version=4,
        )
        code = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        observed.append(notebook.metadata["ouro3"]["validation_seed"])
        expected = "None" if seed is None else str(seed)
        assert f"VALIDATION_SEED = {expected}" in code
        assert "seeds=(0, 1, 2, 3, 4)" not in code
    assert observed == [None, 0, 1, 2, 3, 4]


def test_portfolio_notebooks_embed_exact_router_mode_and_submission_profile(
    tmp_path: Path,
) -> None:
    builder = _load_script("build_notebooks.py")
    builder.build(tmp_path, validation_seed=0, mode="duck-portfolio")
    validation = nbformat.read(
        tmp_path / "validation" / "validation.ipynb",
        as_version=4,
    )
    submission = nbformat.read(
        tmp_path / "submission" / "submission.ipynb",
        as_version=4,
    )
    validation_code = "\n".join(
        cell.source for cell in validation.cells if cell.cell_type == "code"
    )
    submission_code = "\n".join(
        cell.source for cell in submission.cells if cell.cell_type == "code"
    )
    assert validation.metadata["ouro3"]["mode"] == "duck-portfolio"
    assert submission.metadata["ouro3"]["mode"] == "duck-portfolio"
    assert "HarnessConfig.portfolio(seed=VALIDATION_SEED)" in validation_code
    assert "HarnessConfig.portfolio(seed=0)" in submission_code
    assert "RuntimeProfile.KAGGLE_SUBMISSION" in submission_code
    assert '"OURO3_HARNESS_MODE": "duck-portfolio"' in validation_code


def test_retrodict_notebooks_embed_isolated_mode_and_submission_profile(
    tmp_path: Path,
) -> None:
    builder = _load_script("build_notebooks.py")
    builder.build(tmp_path, validation_seed=0, mode="duck-retrodict")
    validation = nbformat.read(
        tmp_path / "validation" / "validation.ipynb",
        as_version=4,
    )
    submission = nbformat.read(
        tmp_path / "submission" / "submission.ipynb",
        as_version=4,
    )
    validation_code = "\n".join(
        cell.source for cell in validation.cells if cell.cell_type == "code"
    )
    submission_code = "\n".join(
        cell.source for cell in submission.cells if cell.cell_type == "code"
    )
    assert validation.metadata["ouro3"]["mode"] == "duck-retrodict"
    assert submission.metadata["ouro3"]["mode"] == "duck-retrodict"
    assert "HarnessConfig.retrodict(seed=VALIDATION_SEED)" in validation_code
    assert "HarnessConfig.retrodict(seed=0)" in submission_code
    assert "RuntimeProfile.KAGGLE_SUBMISSION" in submission_code
    assert '"OURO3_HARNESS_MODE": "duck-retrodict"' in validation_code


def test_generated_notebooks_execute_top_to_bottom_from_kaggle_archives(
    tmp_path: Path,
) -> None:
    package_source = _load_script("package_source.py")
    builder = _load_script("build_notebooks.py")
    validator = _load_script("validate_notebooks.py")
    source = tmp_path / "source"
    package_source.build(source)
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    for path in source.iterdir():
        if path.is_file() and path.name != "dataset-metadata.json":
            shutil.copy2(path, mounted / path.name)
    for folder in ("src", "configs", "baselines"):
        shutil.make_archive(str(mounted / folder), "zip", source / folder)

    notebooks = tmp_path / "notebooks"
    builder.build(notebooks)
    assert validator.validate(notebooks, source_root=mounted) == [
        notebooks / "validation" / "validation.ipynb",
        notebooks / "submission" / "submission.ipynb",
    ]


def test_memory_notebooks_fail_closed_on_reasoning_template_and_keep_config(
    tmp_path: Path,
) -> None:
    package_source = _load_script("package_source.py")
    builder = _load_script("build_notebooks.py")
    validator = _load_script("validate_notebooks.py")
    source = tmp_path / "source"
    package_source.build(source)
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    for path in source.iterdir():
        if path.is_file() and path.name != "dataset-metadata.json":
            shutil.copy2(path, mounted / path.name)
    for folder in ("src", "configs", "baselines"):
        shutil.make_archive(str(mounted / folder), "zip", source / folder)

    notebooks = tmp_path / "memory-notebooks"
    builder.build(
        notebooks,
        validation_seed=0,
        mode="duck-memory",
    )
    for kind in ("validation", "submission"):
        notebook = nbformat.read(
            notebooks / kind / f"{kind}.ipynb",
            as_version=4,
        )
        code = "\n".join(
            cell.source
            for cell in notebook.cells
            if cell.cell_type == "code"
        )
        assert notebook.metadata["ouro3"]["mode"] == "duck-memory"
        assert "render_and_verify_reasoning" in code
        assert "OURO3_REASONING_TEMPLATE_VERIFIED" in code
        assert "OURO3_REASONING_TEMPLATE_SHA256" in code
        assert '"OURO3_HARNESS_MODE": "duck-memory"' in code
        assert "HarnessConfig.memory(seed=0)" in code or (
            "HarnessConfig.memory(seed=VALIDATION_SEED)" in code
        )
        assert "duck-memory-v1-seed-0" not in code
    assert validator.validate(notebooks, source_root=mounted) == [
        notebooks / "validation" / "validation.ipynb",
        notebooks / "submission" / "submission.ipynb",
    ]


def test_reasoning_notebooks_use_stock_history_and_no_compactor(tmp_path: Path) -> None:
    builder = _load_script("build_notebooks.py")
    notebooks = tmp_path / "reasoning-notebooks"
    builder.build(notebooks, validation_seed=0, mode="duck-reasoning")
    for kind in ("validation", "submission"):
        notebook = nbformat.read(
            notebooks / kind / f"{kind}.ipynb",
            as_version=4,
        )
        code = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        assert notebook.metadata["ouro3"]["mode"] == "duck-reasoning"
        assert "render_and_verify_reasoning" in code
        assert '"OURO3_HARNESS_MODE": "duck-reasoning"' in code
        assert "HarnessConfig.reasoning(seed=0)" in code or (
            "HarnessConfig.reasoning(seed=VALIDATION_SEED)" in code
        )
        assert "compaction" not in code.lower()


def test_architecture_has_two_drawio_pages_and_attribution() -> None:
    tree = ET.parse(ROOT / "architecture.drawio")
    pages = tree.getroot().findall("diagram")
    assert [page.attrib["name"] for page in pages] == [
        "1 - Per-game execution lanes",
        "2 - Evaluation and Kaggle cycle",
    ]
    explanation = (ROOT / "HOW-IT-WORKS.md").read_text(encoding="utf-8")
    assert "Tufalabs/duck-harness" in explanation
    assert "MIT" in explanation
