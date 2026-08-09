"""Build a private Kaggle source dataset with a SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist" / "source-dataset"
MARKER = "OURO3_SOURCE_BUNDLE.json"


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "*.pkl",
            "*.pickle",
            "*.ipynb",
            ".git",
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(output: Path) -> dict:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    _copy_tree(ROOT / "src", output / "src")
    _copy_tree(ROOT / "configs", output / "configs")
    _copy_tree(ROOT / "baselines", output / "baselines")
    for name in (
        "pyproject.toml",
        "README.md",
        "HOW-IT-WORKS.md",
        "OPERATING-GUIDE.md",
        "POETIQ-HARNESS-RECOMMENDATIONS.md",
        "SCORECARD.md",
        "architecture.drawio",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
    ):
        source = ROOT / name
        if source.exists():
            shutil.copy2(source, output / name)
    marker = {
        "schema_version": 2,
        "harness": "ouroboros-arc-agi-3-kaggle-v3",
        "default_mode": "duck-reference",
        "source_layout": "src",
        "opaque_pickles_included": False,
    }
    (output / MARKER).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "title": "Ouroboros ARC-AGI-3 kaggle-v3 source",
        "id": "kinwochan/ouroboros-arc-agi-3-v3-source",
        "licenses": [{"name": "MIT"}],
        "isPrivate": True,
    }
    (output / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    files = {
        str(path.relative_to(output)): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path.name not in {"manifest.sha256.json", "dataset-metadata.json"}
    }
    manifest = {"algorithm": "sha256", "file_count": len(files), "files": files}
    (output / "manifest.sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args.output)
    print(f"wrote {args.output} ({manifest['file_count']} manifested files)")


if __name__ == "__main__":
    main()
