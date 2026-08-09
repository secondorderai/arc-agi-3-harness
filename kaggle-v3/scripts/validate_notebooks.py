"""Execute both generated notebooks top to bottom in fail-fast dry-run mode."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def validate(
    notebooks_dir: Path,
    *,
    source_root: Path | None = None,
) -> list[Path]:
    resolved_source = source_root or ROOT / "dist" / "source-dataset"
    executed: list[Path] = []
    for kind in ("validation", "submission"):
        path = notebooks_dir / kind / f"{kind}.ipynb"
        notebook = nbformat.read(path, as_version=4)
        with tempfile.TemporaryDirectory(prefix=f"ouro3-{kind}-") as work_dir:
            previous = {
                key: os.environ.get(key)
                for key in (
                    "OURO3_NOTEBOOK_DRY_RUN",
                    "OURO3_SOURCE_ROOT",
                    "OURO3_WORKING_DIR",
                )
            }
            os.environ.update(
                {
                    "OURO3_NOTEBOOK_DRY_RUN": "1",
                    "OURO3_SOURCE_ROOT": str(resolved_source.resolve()),
                    "OURO3_WORKING_DIR": work_dir,
                }
            )
            try:
                NotebookClient(
                    notebook,
                    timeout=120,
                    kernel_name="python3",
                    resources={"metadata": {"path": str(ROOT)}},
                ).execute()
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        executed.append(path)
    return executed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks-dir", type=Path, default=ROOT / "notebooks")
    args = parser.parse_args()
    for path in validate(args.notebooks_dir):
        print(f"validated {path}")


if __name__ == "__main__":
    main()
