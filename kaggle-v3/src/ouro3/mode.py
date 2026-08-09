"""Execution-mode helpers shared by the vendored Duck core and v3 extensions."""

from __future__ import annotations

import os
from enum import StrEnum


class HarnessMode(StrEnum):
    """Stable execution paths exposed by the v3 harness."""

    DUCK_REFERENCE = "duck-reference"
    DUCK_ROBUST = "duck-robust"
    DUCK_MEMORY = "duck-memory"
    DUCK_REASONING = "duck-reasoning"
    DUCK_DELIBERATE = "duck-deliberate"
    DUCK_CONTRACT = "duck-contract"
    DUCK_CONTRACT_REPAIR = "duck-contract-repair"
    DUCK_AUDIT = "duck-audit"
    DUCK_INFORMATION = "duck-information"
    DUCK_HIERARCHY = "duck-hierarchy"
    DUCK_DIVERSITY = "duck-diversity"
    DUCK_POETIQ = "duck-poetiq"
    DUCK_PORTFOLIO = "duck-portfolio"
    DUCK_RETRODICT = "duck-retrodict"
    OURO_HYBRID = "ouro-hybrid"


def active_harness_mode() -> HarnessMode:
    raw = os.environ.get(
        "OURO3_HARNESS_MODE", HarnessMode.DUCK_REFERENCE.value
    ).strip()
    try:
        return HarnessMode(raw)
    except ValueError as exc:
        raise RuntimeError(f"unsupported OURO3_HARNESS_MODE: {raw!r}") from exc


def is_hybrid_mode() -> bool:
    return active_harness_mode() == HarnessMode.OURO_HYBRID
