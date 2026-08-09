"""TAAF solver adapter for the persistent retrodictive Duck lane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import taaf.game

from duck_retrodict.agent import DuckRetrodictToolAgent
from inference.framework.solver import HarnessSolver, _LocalServerRuntime
from ouro3.mode import HarnessMode, active_harness_mode


@dataclass
class DuckRetrodictHarnessSolver(HarnessSolver):
    """Stock Duck actor plus a host-owned typed world model."""

    label: str = "duck-retrodict"
    primary_seed: int | None = 0
    failure_floor: int = 3
    retrodict_max_rules: int = 256
    retrodict_prediction_threshold: float = 0.90

    def _make_analyzer(
        self,
        game: taaf.game.Game,
        index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> Any:
        if active_harness_mode() != HarnessMode.DUCK_RETRODICT:
            raise RuntimeError(
                "DuckRetrodictHarnessSolver requires duck-retrodict mode"
            )
        game_key = (
            game.game_run.game_id
            if game.game_run is not None
            else f"retrodict-{index}"
        )
        return DuckRetrodictToolAgent(
            game_key=game_key,
            failure_floor=self.failure_floor,
            max_rules=self.retrodict_max_rules,
            prediction_threshold=self.retrodict_prediction_threshold,
            seed=self.primary_seed,
            model=self.model,
            timeout=self.analyzer_timeout,
            save_request_logs=self.save_request_logs,
            api_key=(
                local_server.api_key
                if local_server is not None
                else self._local_server_api_key
            )
            or None,
            base_url=(
                local_server.base_url
                if local_server is not None
                else self._local_server_base_url
            )
            or None,
            provider="vllm" if local_server is not None else None,
        )
