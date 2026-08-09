"""Solver adapter for the sparse information-acquisition lane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import taaf.game

from duck_information.agent import DuckInformationToolAgent
from inference.framework.solver import HarnessSolver, _LocalServerRuntime
from ouro3.mode import HarnessMode, active_harness_mode


@dataclass
class DuckInformationHarnessSolver(HarnessSolver):
    """Stock Duck runtime with a sparse information request."""

    label: str = "duck-information"
    primary_seed: int | None = 0
    information_no_change_threshold: int = 2
    information_max_triggers: int = 8

    def _make_analyzer(
        self,
        game: taaf.game.Game,
        index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> Any:
        del game, index
        if active_harness_mode() != HarnessMode.DUCK_INFORMATION:
            raise RuntimeError(
                "DuckInformationHarnessSolver requires duck-information mode"
            )
        return DuckInformationToolAgent(
            seed=self.primary_seed,
            model=self.model,
            timeout=self.analyzer_timeout,
            save_request_logs=self.save_request_logs,
            information_no_change_threshold=self.information_no_change_threshold,
            information_max_triggers=self.information_max_triggers,
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
