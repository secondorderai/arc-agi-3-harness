"""Solver construction for the falsification-first Stock Duck lane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import taaf.game

from duck_deliberate.agent import DuckDeliberateToolAgent
from inference.framework.solver import HarnessSolver, _LocalServerRuntime
from ouro3.mode import HarnessMode, active_harness_mode


@dataclass
class DuckDeliberateHarnessSolver(HarnessSolver):
    """Stock runtime with the deliberate analyzer adapter."""

    label: str = "duck-deliberate"
    primary_seed: int | None = 0

    def _make_analyzer(
        self,
        game: taaf.game.Game,
        index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> Any:
        del game, index
        if active_harness_mode() != HarnessMode.DUCK_DELIBERATE:
            raise RuntimeError(
                "DuckDeliberateHarnessSolver requires duck-deliberate mode"
            )
        return DuckDeliberateToolAgent(
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
