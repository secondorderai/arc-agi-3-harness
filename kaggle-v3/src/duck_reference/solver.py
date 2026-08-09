"""Stock Duck HarnessSolver selected through an isolated import path."""

from __future__ import annotations

from typing import Any

import taaf.game

from duck_reference.agent import DuckReferenceToolAgent
from inference.framework.solver import HarnessSolver, _LocalServerRuntime
from ouro3.mode import HarnessMode, active_harness_mode


class DuckReferenceHarnessSolver(HarnessSolver):
    """Reference solver that cannot construct the Ouroboros hybrid agent."""

    label: str = "duck-reference"

    def _make_analyzer(
        self,
        game: taaf.game.Game,
        index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> Any:
        del game, index
        if active_harness_mode() != HarnessMode.DUCK_REFERENCE:
            raise RuntimeError(
                "DuckReferenceHarnessSolver requires duck-reference mode"
            )
        return DuckReferenceToolAgent(
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
