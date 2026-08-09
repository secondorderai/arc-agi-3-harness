"""Solver construction for retained-reasoning Duck."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import taaf.game

from duck_memory.agent import DuckMemoryToolAgent
from inference.framework.solver import HarnessSolver, _LocalServerRuntime
from ouro3.mode import HarnessMode, active_harness_mode


@dataclass
class DuckMemoryHarnessSolver(HarnessSolver):
    """Stock Duck solver with only memory transport and compaction changed."""

    label: str = "duck-memory"
    primary_seed: int | None = None

    def _make_analyzer(
        self,
        game: taaf.game.Game,
        index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> Any:
        del game, index
        if active_harness_mode() != HarnessMode.DUCK_MEMORY:
            raise RuntimeError(
                "DuckMemoryHarnessSolver requires duck-memory mode"
            )
        return DuckMemoryToolAgent(
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
