"""Prediction checks for aborting brittle queued plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Verification:
    matched: bool
    reason: str = ""


def verify_prediction(expectation: Any, observation: dict[str, Any]) -> Verification:
    if expectation in (None, {}, ""):
        return Verification(True)
    if not isinstance(expectation, dict):
        return Verification(False, "expect must be an object")
    checks = {
        "board_changed": bool(observation.get("board_changed")),
        "gameplay_changed": bool(observation.get("gameplay_changed")),
        "hud_changed": bool(observation.get("hud_changed")),
        "level_completed": bool(observation.get("level_completed")),
        "game_over": bool(observation.get("game_over")),
        "run_complete": bool(observation.get("run_complete")),
    }
    for key, actual in checks.items():
        if key in expectation and bool(expectation[key]) != actual:
            return Verification(
                False,
                f"prediction mismatch for {key}: expected {bool(expectation[key])}, observed {actual}",
            )
    if "level" in expectation:
        try:
            expected_level = int(expectation["level"])
            actual_level = int(observation.get("level"))
        except (TypeError, ValueError):
            return Verification(False, "level expectation or observation was not an integer")
        if expected_level != actual_level:
            return Verification(
                False,
                f"prediction mismatch for level: expected {expected_level}, observed {actual_level}",
            )
    return Verification(True)
