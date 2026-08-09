"""Deterministic action floor used after model transport or reasoning failures."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Sequence

from ouro3.perception import segment_grid

_DIRECTION_ORDER = ("UP", "RIGHT", "DOWN", "LEFT")
_ENGINE_TO_MODEL = {
    "ACTION1": "UP",
    "ACTION2": "DOWN",
    "ACTION3": "LEFT",
    "ACTION4": "RIGHT",
    "ACTION5": "SPACE",
    "ACTION6": "MOUSE",
}


@dataclass
class DeterministicExplorer:
    game_key: str
    cursor: int = 0
    attempted_clicks: set[tuple[int, int]] = field(default_factory=set)

    def choose(
        self,
        *,
        grid: Sequence[Sequence[int]],
        valid_actions: Sequence[str],
    ) -> dict[str, Any]:
        names = [
            _ENGINE_TO_MODEL.get(str(action).upper(), str(action).upper())
            for action in valid_actions
        ]
        if "MOUSE" in names:
            candidates: list[tuple[int, int]] = []
            for obj in sorted(segment_grid(grid), key=lambda item: (-item["area"], item["id"])):
                row = int(round(float(obj["centroid"][0])))
                col = int(round(float(obj["centroid"][1])))
                if (row, col) not in self.attempted_clicks:
                    candidates.append((row, col))
            if candidates:
                index = self._stable_offset(len(candidates))
                row, col = candidates[index]
                self.attempted_clicks.add((row, col))
                return {"action": "MOUSE", "row": row, "col": col}

        ordered = [action for action in _DIRECTION_ORDER if action in names]
        ordered.extend(action for action in names if action not in ordered and action != "MOUSE")
        if not ordered:
            raise RuntimeError("deterministic explorer received no valid actions")
        action = ordered[self.cursor % len(ordered)]
        self.cursor += 1
        return {"action": action}

    def _stable_offset(self, modulo: int) -> int:
        digest = hashlib.sha256(f"{self.game_key}:{self.cursor}".encode("utf-8")).digest()
        self.cursor += 1
        return int.from_bytes(digest[:4], "big") % modulo
