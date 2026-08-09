"""Solver construction for the isolated retained-reasoning Duck lane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import taaf.game

from duck_reasoning.agent import DuckReasoningToolAgent
from inference.framework.solver import HarnessSolver, _LocalServerRuntime
from ouro3.mode import HarnessMode, active_harness_mode


@dataclass
class DuckReasoningHarnessSolver(HarnessSolver):
    """Stock Duck solver with only historical reasoning normalization."""

    label: str = "duck-reasoning"
    primary_seed: int | None = None

    def _make_analyzer(
        self,
        game: taaf.game.Game,
        index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> Any:
        del game, index
        if active_harness_mode() != HarnessMode.DUCK_REASONING:
            raise RuntimeError(
                "DuckReasoningHarnessSolver requires duck-reasoning mode"
            )
        return DuckReasoningToolAgent(
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
