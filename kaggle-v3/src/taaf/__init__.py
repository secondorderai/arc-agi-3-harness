"""ARC-AGI-3 orchestration framework (TAAF), vendored under the MIT license.

v3 intentionally keeps package import lazy so the competition runtime does not
load optional plotting/video dependencies before gameplay starts.
"""

from __future__ import annotations

import importlib
from typing import Any

_SUBMODULES = {
    "benchmark",
    "competition_arcade",
    "deploy",
    "deploy_inline",
    "deploy_kaggle",
    "deploy_slurm",
    "diagnostics",
    "game",
    "game_api",
    "game_examples",
    "solver",
    "solver_examples",
    "standard_benchmarks",
    "support",
}

__all__ = sorted(_SUBMODULES)


def __getattr__(name: str) -> Any:
    if name not in _SUBMODULES:
        raise AttributeError(name)
    module = importlib.import_module(f"taaf.{name}")
    globals()[name] = module
    return module
