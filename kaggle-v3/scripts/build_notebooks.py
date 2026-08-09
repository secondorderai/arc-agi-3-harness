"""Generate the RTX validation and hidden-submission notebooks from source."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "notebooks"
COMPETITION = "arc-prize-2026-arc-agi-3"
SOURCE_DATASET = "kinwochan/ouroboros-arc-agi-3-v3-source"
WHEELHOUSE_DATASET = "driessmit1/arc3-vllm-h100-wheelhouse-v3"
MODEL_DATASET = "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"
MODEL_ID = "vrfai/Qwen3.6-27B-FP8"
GPU = "NvidiaRtxPro6000"


BOOTSTRAP = textwrap.dedent(
    """
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    import sys
    import sysconfig
    import zipfile
    from pathlib import Path

    DRY_RUN = os.getenv("OURO3_NOTEBOOK_DRY_RUN", "").lower() in {"1", "true", "yes"}
    TRUE_SUBMISSION = os.getenv("KAGGLE_IS_COMPETITION_RERUN", "").lower() in {"1", "true", "yes"}
    WORKING_DIR = Path(os.getenv("OURO3_WORKING_DIR", "/kaggle/working" if Path("/kaggle").exists() else "./notebook-output"))
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["TAAF_RUN_AS_SUBMISSION"] = "1" if TRUE_SUBMISSION else "0"
    os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1" if TRUE_SUBMISSION else "0"
    os.environ["ONLY_RESET_LEVELS"] = "true"
    if TRUE_SUBMISSION:
        os.environ.setdefault("ARC_API_KEY", "test-key-123")
        os.environ.setdefault("ARC_BASE_URL", "http://gateway:8001/")
        os.environ.setdefault("RECORDINGS_DIR", str(WORKING_DIR / "server_recording"))
    SOURCE_REFS = [
        "kinwochan/ouroboros-arc-agi-3-v3-source",
        "driessmit1/arc3-vllm-h100-wheelhouse-v3",
        "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
    ]
    cuda_library_path = "/usr/local/nvidia/lib64"
    os.environ["LIBRARY_PATH"] = os.pathsep.join(
        entry
        for entry in [
            cuda_library_path,
            *os.environ.get("LIBRARY_PATH", "").split(os.pathsep),
        ]
        if entry
    )

    def dataset_candidates(ref):
        owner, slug = ref.split("/", 1)
        return [
            Path("/kaggle/input") / slug,
            Path("/kaggle/input/datasets") / owner / slug,
        ]

    explicit_source = os.getenv("OURO3_SOURCE_ROOT", "").strip()
    if explicit_source:
        RAW_BUNDLE_DIR = Path(explicit_source).resolve()
    else:
        markers = list(Path("/kaggle/input").rglob("OURO3_SOURCE_BUNDLE.json"))
        if not markers:
            raise RuntimeError("Ouroboros v3 source dataset marker was not found")
        RAW_BUNDLE_DIR = markers[0].parent
    if (RAW_BUNDLE_DIR / "src").is_dir():
        BUNDLE_DIR = RAW_BUNDLE_DIR
    else:
        BUNDLE_DIR = WORKING_DIR / "ouro3-source-bundle"
        if BUNDLE_DIR.exists():
            shutil.rmtree(BUNDLE_DIR)
        BUNDLE_DIR.mkdir(parents=True)
        for source in RAW_BUNDLE_DIR.iterdir():
            if source.is_file() and source.suffix != ".zip":
                shutil.copy2(source, BUNDLE_DIR / source.name)
        for folder_name in ("src", "configs", "baselines"):
            archive = RAW_BUNDLE_DIR / f"{folder_name}.zip"
            if not archive.is_file():
                raise RuntimeError(f"source dataset is missing {archive.name}")
            destination = BUNDLE_DIR / folder_name
            destination.mkdir()
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(destination)
    SOURCE_DIR = BUNDLE_DIR / "src"
    sys.path.insert(0, str(SOURCE_DIR))
    print(f"source bundle: {BUNDLE_DIR}")
    print(f"dry_run={DRY_RUN} competition_rerun={TRUE_SUBMISSION}")
    """
).strip()

VERIFY = textwrap.dedent(
    """
    manifest_path = BUNDLE_DIR / "manifest.sha256.json"
    os.environ["OURO3_SOURCE_MANIFEST_SHA256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, metadata in manifest["files"].items():
        path = BUNDLE_DIR / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != metadata["sha256"]:
            raise RuntimeError(f"source manifest mismatch: {relative}")
    if any(path.suffix.lower() in {".pkl", ".pickle"} for path in BUNDLE_DIR.rglob("*")):
        raise RuntimeError("opaque pickle found in v3 source dataset")
    print(f"verified {manifest['file_count']} source files")
    """
).strip()

ARC_RUNTIME = textwrap.dedent(
    """
    if not DRY_RUN:
        wheel_candidates = sorted(Path("/kaggle/input").rglob("arc_agi*.whl"))
        if not wheel_candidates:
            raise RuntimeError("competition arc_agi wheelhouse was not found")
        wheelhouse = wheel_candidates[0].parent
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-index",
                "--no-warn-conflicts",
                "--disable-pip-version-check",
                "--find-links",
                str(wheelhouse),
                "arc-agi==0.9.8",
            ],
            stdout=subprocess.DEVNULL,
        )
        import arc_agi
        import arcengine
        print(f"ARC runtime: arc_agi={getattr(arc_agi, '__version__', '?')} arcengine={getattr(arcengine, '__version__', '?')}")
    else:
        print("dry run: ARC runtime installation skipped")
    """
).strip()

MODEL_SETUP = textwrap.dedent(
    """
    SETUP_ENV_PATH = WORKING_DIR / "ouro3-setup-env.json"
    input_paths = {}
    for ref in SOURCE_REFS:
        if ref == SOURCE_REFS[0]:
            resolved = BUNDLE_DIR
        else:
            candidates = dataset_candidates(ref)
            resolved = next((path for path in candidates if path.exists()), candidates[0])
        input_paths[ref] = str(resolved)
    setup_env = {
        "TAAF_KAGGLE_INPUT_PATHS": json.dumps(input_paths, sort_keys=True),
        "TAAF_KAGGLE_DATASET_SOURCES": json.dumps(SOURCE_REFS),
    }
    SETUP_ENV_PATH.write_text(json.dumps(setup_env, indent=2), encoding="utf-8")
    os.environ.update(setup_env)

    if not DRY_RUN:
        from inference.framework.kaggle import DuckKaggleVllmConfig, duck_kaggle_setup_command

        os.environ.update(
            {
                "TAAF_KAGGLE_WORKING_DIR": str(WORKING_DIR),
                "TAAF_KAGGLE_SETUP_ENV": str(SETUP_ENV_PATH),
                "KAGGLE_GPU_TYPE": "rtx-pro-6000",
                "KAGGLE_GPU_COUNT": "1",
                "LOCAL_ANALYZER_CONTEXT_WINDOW": "32768",
                # Keep the notebook-side fallback aligned with the
                # HarnessConfig analyzer timeout passed to every ToolAgent.
                "LOCAL_ANALYZER_TIMEOUT": "900",
                "LOCAL_ANALYZER_MAX_OUTPUT": "0",
                "LOCAL_ANALYZER_TOOL_STEPS": "0",
                "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
                "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
                "LOCAL_ANALYZER_YIELD_SECONDS": "60",
                "LOCAL_ANALYZER_TEMPERATURE": "0.6",
                "LOCAL_ANALYZER_TOP_P": "0.95",
                "LOCAL_ANALYZER_TOP_K": "20",
                "LOCAL_ANALYZER_ENABLE_THINKING": "true",
                "MULTIMODAL_UPSCALE": "4",
                "MULTIMODAL_CONTEXT": "current_grid",
                "OURO3_HARNESS_MODE": "duck-reference",
                "ONLY_RESET_LEVELS": "true",
            }
        )
        command = duck_kaggle_setup_command(DuckKaggleVllmConfig())
        env = os.environ.copy()
        env["PYTHON"] = sys.executable
        subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)
        persisted = json.loads(SETUP_ENV_PATH.read_text(encoding="utf-8"))
        os.environ.update({str(key): str(value) for key, value in persisted.items()})
        for entry in reversed([value for value in os.environ.get("PYTHONPATH", "").split(os.pathsep) if value]):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        from importlib.metadata import version as package_version

        import torch
        import vllm
        runtime_versions = {
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "flashinfer-python": package_version("flashinfer-python"),
            "flashinfer-cubin": package_version("flashinfer-cubin"),
        }
        expected_versions = {
            "vllm": "0.19.0",
            "torch": "2.10.0",
            "flashinfer-python": "0.6.6",
            "flashinfer-cubin": "0.6.6",
        }
        mismatches = {
            name: {"expected": expected, "actual": runtime_versions[name]}
            for name, expected in expected_versions.items()
            if runtime_versions[name].split("+", 1)[0] != expected
        }
        if mismatches:
            raise RuntimeError(f"runtime pin mismatch: {json.dumps(mismatches, sort_keys=True)}")
        print(f"model smoke complete: {json.dumps(runtime_versions, sort_keys=True)}")
    else:
        print("dry run: GPU and Qwen smoke checks skipped")
    """
).strip()

TEARDOWN = textwrap.dedent(
    """
    if not DRY_RUN:
        from inference.framework.kaggle import duck_kaggle_teardown_command

        env = os.environ.copy()
        env["PYTHON"] = sys.executable
        subprocess.run(
            duck_kaggle_teardown_command(),
            shell=True,
            check=False,
            cwd=WORKING_DIR,
            env=env,
        )
    """
).strip()

REASONING_SMOKE = textwrap.dedent(
    """
    if not DRY_RUN:
        from transformers import AutoTokenizer

        from duck_memory.reasoning import render_and_verify_reasoning

        model_path = Path(input_paths[SOURCE_REFS[2]])
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        rendered_reasoning_smoke = render_and_verify_reasoning(
            tokenizer.apply_chat_template
        )
        os.environ["OURO3_REASONING_TEMPLATE_VERIFIED"] = "true"
        reasoning_template_sha256 = hashlib.sha256(
            rendered_reasoning_smoke.encode("utf-8")
        ).hexdigest()
        os.environ["OURO3_REASONING_TEMPLATE_SHA256"] = (
            reasoning_template_sha256
        )
        reasoning_smoke = {
            "verified": True,
            "model_path": str(model_path),
            "rendered_prompt_sha256": reasoning_template_sha256,
        }
        (WORKING_DIR / "reasoning-template-smoke.json").write_text(
            json.dumps(reasoning_smoke, indent=2),
            encoding="utf-8",
        )
        print("Qwen historical reasoning round-trip verified")
    else:
        print("dry run: Qwen historical reasoning smoke skipped")
    """
).strip()


def _validation_run(
    seed: int | None,
    *,
    mode: str,
    model_id: str = MODEL_ID,
    model_dataset: str = MODEL_DATASET,
) -> str:
    if mode == "duck-robust":
        constructor = "HarnessConfig.robust"
        experiment = "duck-robust"
        config_expression = (
            f"{constructor}(seed=VALIDATION_SEED).with_overrides("
            f'experiment=f"{experiment}-{{run_label}}")'
        )
    elif mode == "duck-memory":
        constructor = "HarnessConfig.memory"
        experiment = "duck-memory-v1"
        config_expression = f"{constructor}(seed=VALIDATION_SEED)"
    elif mode == "duck-reasoning":
        constructor = "HarnessConfig.reasoning"
        experiment = "duck-reasoning-v1"
        config_expression = f"{constructor}(seed=VALIDATION_SEED)"
    elif mode == "duck-deliberate":
        constructor = "HarnessConfig.deliberate"
        experiment = "duck-deliberate-v1"
        config_expression = f"{constructor}(seed=VALIDATION_SEED)"
    elif mode == "duck-contract":
        constructor = "HarnessConfig.contract"
        experiment = "duck-contract-v1"
        config_expression = f"{constructor}(seed=VALIDATION_SEED)"
    elif mode == "duck-contract-repair":
        constructor = "HarnessConfig.contract_repair"
        experiment = "duck-contract-repair-v1"
        config_expression = f"{constructor}(seed=VALIDATION_SEED)"
    elif mode == "duck-audit":
        constructor = "HarnessConfig.audit"
        experiment = "duck-audit-v1"
        config_expression = f"{constructor}(seed=VALIDATION_SEED)"
    elif mode == "duck-information":
        constructor = "HarnessConfig.information"
        experiment = "duck-information-v1"
        config_expression = f"{constructor}(seed=VALIDATION_SEED)"
    elif mode == "duck-hierarchy":
        constructor = "HarnessConfig.hierarchy"
        experiment = "duck-hierarchy-v1"
        config_expression = f"{constructor}(seed=VALIDATION_SEED)"
    elif mode == "duck-diversity":
        constructor = "HarnessConfig.diversity"
        experiment = "duck-diversity-v1"
        config_expression = f"{constructor}(seed=VALIDATION_SEED)"
    elif mode == "duck-poetiq":
        constructor = "HarnessConfig.poetiq"
        experiment = "duck-poetiq-v1"
        config_expression = f"{constructor}(seed=VALIDATION_SEED)"
    elif mode == "duck-portfolio":
        constructor = "HarnessConfig.portfolio"
        experiment = "duck-portfolio-v1"
        config_expression = f"{constructor}(seed=VALIDATION_SEED)"
    elif mode == "duck-retrodict":
        constructor = "HarnessConfig.retrodict"
        experiment = "duck-retrodict-v1"
        config_expression = f"{constructor}(seed=VALIDATION_SEED)"
    else:
        constructor = "HarnessConfig.reference"
        experiment = "duck-reference"
        config_expression = (
            f"{constructor}(seed=VALIDATION_SEED).with_overrides("
            f'experiment=f"{experiment}-{{run_label}}")'
        )
    if model_id != MODEL_ID or model_dataset != MODEL_DATASET:
        config_expression = (
            f"({config_expression}).with_overrides("
            f"model_id={model_id!r}, model_dataset={model_dataset!r})"
        )
    router_setup = (
        '            os.environ["OURO3_PORTFOLIO_ROUTER_PATH"] = str(\n'
        '                SOURCE_DIR / "duck_portfolio" / "router_model_parity.json"\n'
        "            )\n"
        if mode == "duck-portfolio"
        else ""
    )
    return textwrap.dedent(
        f"""
        import asyncio

        from ouro3.config import HarnessConfig
        from ouro3.runner import run_public

        output_path = WORKING_DIR / "validation_metrics.json"
        VALIDATION_SEED = {seed!r}
        run_label = "unseeded" if VALIDATION_SEED is None else f"seed-{{VALIDATION_SEED}}"
        if DRY_RUN:
            output_path.write_text(
                json.dumps(
                    {{
                        "dry_run": True,
                        "notebook": "validation",
                        "mode": {mode!r},
                        "seed": VALIDATION_SEED,
                        "run_label": run_label,
                    }},
                    indent=2,
                ),
                encoding="utf-8",
            )
            validation_metrics = json.loads(output_path.read_text())
        else:
            environment_dirs = [
                path
                for path in Path("/kaggle/input").rglob("environment_files")
                if path.is_dir()
            ]
            environment_dir = next(
                (
                    path
                    for path in environment_dirs
                    if len([child for child in path.iterdir() if child.is_dir()]) >= 25
                ),
                None,
            )
            if environment_dir is None:
                raise RuntimeError("competition public environment_files directory was not found")
{router_setup}            config = {config_expression}
            assert config.concurrency == 28
            assert config.reference_game_cap_s == 7920
            assert config.analyzer_timeout_s == 900
            assert config.context_window == 32768
            assert config.max_model_len == 65536
            validation_metrics = await asyncio.to_thread(
                run_public,
                config=config,
                environments_dir=environment_dir,
                fold="public",
                output_path=output_path,
            )
            assert validation_metrics["game_count"] == 25
            assert validation_metrics["seed"] == VALIDATION_SEED
            assert validation_metrics["mode"] == {mode!r}
        print(json.dumps(validation_metrics, indent=2, sort_keys=True))
        """
    ).strip()


def _submission_run(
    *,
    mode: str,
    model_id: str = MODEL_ID,
    model_dataset: str = MODEL_DATASET,
) -> str:
    if mode == "duck-robust":
        constructor = "HarnessConfig.robust(seed=0)"
        experiment = "duck-robust-hidden"
        config_expression = (
            f'{constructor}.with_overrides(experiment=os.getenv('
            f'"OURO3_EXPERIMENT", "{experiment}"), '
            "profile=RuntimeProfile.KAGGLE_SUBMISSION)"
        )
    elif mode == "duck-memory":
        constructor = "HarnessConfig.memory(seed=0)"
        experiment = "duck-memory-v1"
        config_expression = constructor
    elif mode == "duck-reasoning":
        constructor = "HarnessConfig.reasoning(seed=0)"
        experiment = "duck-reasoning-v1"
        config_expression = (
            f'{constructor}.with_overrides(experiment=os.getenv('
            f'"OURO3_EXPERIMENT", "{experiment}"), '
            "profile=RuntimeProfile.KAGGLE_SUBMISSION)"
        )
    elif mode == "duck-deliberate":
        constructor = "HarnessConfig.deliberate(seed=0)"
        experiment = "duck-deliberate-v1"
        config_expression = (
            f'{constructor}.with_overrides(experiment=os.getenv('
            f'"OURO3_EXPERIMENT", "{experiment}"), '
            "profile=RuntimeProfile.KAGGLE_SUBMISSION)"
        )
    elif mode == "duck-contract":
        constructor = "HarnessConfig.contract(seed=0)"
        experiment = "duck-contract-v1"
        config_expression = (
            f'{constructor}.with_overrides(experiment=os.getenv('
            f'"OURO3_EXPERIMENT", "{experiment}"), '
            "profile=RuntimeProfile.KAGGLE_SUBMISSION)"
        )
    elif mode == "duck-contract-repair":
        constructor = "HarnessConfig.contract_repair(seed=0)"
        experiment = "duck-contract-repair-v1"
        config_expression = (
            f'{constructor}.with_overrides(experiment=os.getenv('
            f'"OURO3_EXPERIMENT", "{experiment}"), '
            "profile=RuntimeProfile.KAGGLE_SUBMISSION)"
        )
    elif mode == "duck-audit":
        constructor = "HarnessConfig.audit(seed=0)"
        experiment = "duck-audit-v1"
        config_expression = (
            f'{constructor}.with_overrides(experiment=os.getenv('
            f'"OURO3_EXPERIMENT", "{experiment}"), '
            "profile=RuntimeProfile.KAGGLE_SUBMISSION)"
        )
    elif mode == "duck-information":
        constructor = "HarnessConfig.information(seed=0)"
        experiment = "duck-information-v1"
        config_expression = (
            f'{constructor}.with_overrides(experiment=os.getenv('
            f'"OURO3_EXPERIMENT", "{experiment}"), '
            "profile=RuntimeProfile.KAGGLE_SUBMISSION)"
        )
    elif mode == "duck-hierarchy":
        constructor = "HarnessConfig.hierarchy(seed=0)"
        experiment = "duck-hierarchy-v1"
        config_expression = (
            f'{constructor}.with_overrides(experiment=os.getenv('
            f'"OURO3_EXPERIMENT", "{experiment}"), '
            "profile=RuntimeProfile.KAGGLE_SUBMISSION)"
        )
    elif mode == "duck-diversity":
        constructor = "HarnessConfig.diversity(seed=0)"
        experiment = "duck-diversity-v1"
        config_expression = (
            f'{constructor}.with_overrides(experiment=os.getenv('
            f'"OURO3_EXPERIMENT", "{experiment}"), '
            "profile=RuntimeProfile.KAGGLE_SUBMISSION)"
        )
    elif mode == "duck-poetiq":
        constructor = "HarnessConfig.poetiq(seed=0)"
        experiment = "duck-poetiq-v1"
        config_expression = (
            f'{constructor}.with_overrides(experiment=os.getenv('
            f'"OURO3_EXPERIMENT", "{experiment}"), '
            "profile=RuntimeProfile.KAGGLE_SUBMISSION)"
        )
    elif mode == "duck-portfolio":
        constructor = "HarnessConfig.portfolio(seed=0)"
        experiment = "duck-portfolio-v1"
        config_expression = (
            f'{constructor}.with_overrides(experiment=os.getenv('
            f'"OURO3_EXPERIMENT", "{experiment}"), '
            "profile=RuntimeProfile.KAGGLE_SUBMISSION)"
        )
    elif mode == "duck-retrodict":
        constructor = "HarnessConfig.retrodict(seed=0)"
        experiment = "duck-retrodict-v1"
        config_expression = (
            f'{constructor}.with_overrides(experiment=os.getenv('
            f'"OURO3_EXPERIMENT", "{experiment}"), '
            "profile=RuntimeProfile.KAGGLE_SUBMISSION)"
        )
    else:
        constructor = "HarnessConfig.reference(seed=None)"
        experiment = "duck-reference-hidden"
        config_expression = (
            f'{constructor}.with_overrides(experiment=os.getenv('
            f'"OURO3_EXPERIMENT", "{experiment}"), '
            "profile=RuntimeProfile.KAGGLE_SUBMISSION)"
        )
    if model_id != MODEL_ID or model_dataset != MODEL_DATASET:
        config_expression = (
            f"({config_expression}).with_overrides("
            f"model_id={model_id!r}, model_dataset={model_dataset!r})"
        )
    router_setup = (
        'os.environ["OURO3_PORTFOLIO_ROUTER_PATH"] = str(\n'
        '    SOURCE_DIR / "duck_portfolio" / "router_model_parity.json"\n'
        ")\n"
        if mode == "duck-portfolio"
        else ""
    )
    template = textwrap.dedent(
        """
        import asyncio

        from ouro3.config import HarnessConfig, RuntimeProfile
        from ouro3.runner import run_hidden_submission, write_smoke_submission

        output_path = WORKING_DIR / "submission_metrics.json"
        __ROUTER_SETUP__config = __CONFIG_EXPRESSION__
        if DRY_RUN:
            output_path.write_text(json.dumps({"dry_run": True, "notebook": "submission"}, indent=2), encoding="utf-8")
            submission_metrics = json.loads(output_path.read_text())
        elif TRUE_SUBMISSION:
            submission_metrics = await asyncio.to_thread(
                run_hidden_submission,
                config=config,
                output_path=output_path,
            )
        else:
            write_smoke_submission(
                WORKING_DIR / "submission.parquet",
                message=f"save-run hardware/model smoke {config.config_hash[:12]}",
            )
            submission_metrics = {
                "smoke": True,
                "config_hash": config.config_hash,
                "model": config.model_id,
                "gpu": config.kaggle_gpu,
            }
            output_path.write_text(json.dumps(submission_metrics, indent=2), encoding="utf-8")
        print(json.dumps(submission_metrics, indent=2, sort_keys=True))
        """
    ).strip()
    return template.replace("__ROUTER_SETUP__", router_setup).replace(
        "__CONFIG_EXPRESSION__", config_expression
    )


def build_notebook(
    kind: str,
    *,
    validation_seed: int | None = None,
    mode: str = "duck-reference",
    model_id: str = MODEL_ID,
    model_dataset: str = MODEL_DATASET,
):
    if kind not in {"validation", "submission"}:
        raise ValueError(kind)
    if mode not in {"duck-reference", "duck-robust", "duck-memory", "duck-reasoning", "duck-deliberate", "duck-contract", "duck-contract-repair", "duck-audit", "duck-information", "duck-hierarchy", "duck-diversity", "duck-poetiq", "duck-portfolio", "duck-retrodict"}:
        raise ValueError(mode)
    title = (
        "ARC-AGI-3 kaggle-v3 RTX Public Validation"
        if kind == "validation"
        else "ARC-AGI-3 kaggle-v3 Hidden Submission"
    )
    run_source = (
        _validation_run(
            validation_seed,
            mode=mode,
            model_id=model_id,
            model_dataset=model_dataset,
        )
        if kind == "validation"
        else _submission_run(
            mode=mode,
            model_id=model_id,
            model_dataset=model_dataset,
        )
    )
    validation_label = (
        "unseeded fidelity"
        if validation_seed is None
        else f"seed {validation_seed}"
    )
    cells = [
        new_markdown_cell(
            f"# {title}\n\n"
            "Generated from the tracked v3 source. It verifies the source manifest, "
            "checks the pinned RTX/Qwen runtime, runs the direct Arcade harness, and "
            "writes machine-readable metrics to `/kaggle/working`."
        ),
        new_markdown_cell(
            "## Goal\n\n"
            + (
                f"Exercise all 25 public games once ({validation_label}) with the full Duck budget and retain full JSON action traces."
                if kind == "validation"
                else "Smoke-test on Save & Run, then play all 110 hidden gateway games only during the competition rerun."
            )
        ),
        new_markdown_cell("## Setup\n\nLocate and verify the private source dataset."),
        new_code_cell(BOOTSTRAP.replace(MODEL_DATASET, model_dataset)),
        new_code_cell(VERIFY),
        new_markdown_cell("## Runtime checks\n\nInstall the competition ARC wheel and start the pinned Qwen/vLLM stack."),
        new_code_cell(ARC_RUNTIME),
        new_code_cell(
            MODEL_SETUP.replace(
                '"OURO3_HARNESS_MODE": "duck-reference"',
                f'"OURO3_HARNESS_MODE": "{mode}"',
            ).replace(
                "DuckKaggleVllmConfig()",
                "DuckKaggleVllmConfig("
                f"model_dataset_source={model_dataset!r}, "
                f"served_model_name={model_id!r})",
            )
        ),
        *(
            [
                new_markdown_cell(
                    "## Reasoning retention check\n\n"
                    "Render a two-turn sentinel with the exact attached Qwen "
                    "tokenizer and fail before gameplay unless historical "
                    "private reasoning survives inside `<think>` tags."
                ),
                new_code_cell(REASONING_SMOKE),
            ]
            if mode in {"duck-memory", "duck-reasoning"}
            else []
        ),
        new_markdown_cell("## Run\n\nExecute the selected direct-Arcade workflow."),
        new_code_cell(
            "try:\n"
            + textwrap.indent(run_source, "    ")
            + "\nfinally:\n"
            + textwrap.indent(TEARDOWN, "    ")
        ),
        new_markdown_cell(
            "## Checks\n\n"
            "The JSON artifact includes per-game state, completed levels, engine score, "
            "action counts, action traces, infrastructure failures, seed, and config hash."
        ),
        new_code_cell(
            "artifacts = sorted(path.name for path in WORKING_DIR.iterdir() if path.is_file())\n"
            "print('working artifacts:', artifacts)\n"
            "assert any(name.endswith('_metrics.json') for name in artifacts)"
        ),
        new_markdown_cell(
            "## Next steps\n\n"
            "Pull the metrics, enforce the frozen promotion gate, and only then version "
            "the submission notebook or submit its exact kernel version."
        ),
    ]
    return new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "ouro3": {
                "mode": mode,
                "model_id": model_id,
                "model_dataset": model_dataset,
                "validation_seed": validation_seed,
                "one_seed_per_kernel": True,
            },
        },
    )


def kernel_metadata(kind: str, *, model_dataset: str = MODEL_DATASET) -> dict:
    is_validation = kind == "validation"
    slug = (
        "kinwochan/ouroboros-arc-agi-3-v3-validation"
        if is_validation
        else "kinwochan/ouroboros-arc-agi-3-v3"
    )
    return {
        "id": slug,
        "title": (
            "Ouroboros ARC-AGI-3 v3 validation"
            if is_validation
            else "Ouroboros ARC-AGI-3 v3"
        ),
        "code_file": f"{kind}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": GPU,
        "dataset_sources": [SOURCE_DATASET, WHEELHOUSE_DATASET, model_dataset],
        "kernel_sources": [],
        "competition_sources": [COMPETITION],
        "model_sources": [],
    }


def build(
    output: Path,
    *,
    validation_seed: int | None = None,
    mode: str = "duck-reference",
    model_id: str = MODEL_ID,
    model_dataset: str = MODEL_DATASET,
) -> None:
    if model_dataset.count("/") != 1:
        raise ValueError("model_dataset must be a Kaggle owner/slug reference")
    if not model_id.strip():
        raise ValueError("model_id must be non-empty")
    for kind in ("validation", "submission"):
        target = output / kind
        target.mkdir(parents=True, exist_ok=True)
        nbformat.write(
            build_notebook(
                kind,
                validation_seed=validation_seed,
                mode=mode,
                model_id=model_id,
                model_dataset=model_dataset,
            ),
            target / f"{kind}.ipynb",
        )
        (target / "kernel-metadata.json").write_text(
            json.dumps(
                kernel_metadata(kind, model_dataset=model_dataset),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validation-seed",
        default="unseeded",
        choices=("unseeded", "0", "1", "2", "3", "4"),
        help="embed exactly one validation seed; unseeded omits the model seed",
    )
    parser.add_argument(
        "--mode",
        choices=("duck-reference", "duck-robust", "duck-memory", "duck-reasoning", "duck-deliberate", "duck-contract", "duck-contract-repair", "duck-audit", "duck-information", "duck-hierarchy", "duck-diversity", "duck-poetiq", "duck-portfolio", "duck-retrodict"),
        default="duck-reference",
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-dataset", default=MODEL_DATASET)
    args = parser.parse_args()
    seed = None if args.validation_seed == "unseeded" else int(args.validation_seed)
    build(
        args.output,
        validation_seed=seed,
        mode=args.mode,
        model_id=args.model_id,
        model_dataset=args.model_dataset,
    )
    print(
        f"wrote {args.mode} validation ({args.validation_seed}) and submission "
        f"notebooks under {args.output}"
    )


if __name__ == "__main__":
    main()
