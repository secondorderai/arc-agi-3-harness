"""Solver construction for controlled diversity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import taaf.game

from duck_diversity.agent import DuckDiversityToolAgent
from inference.framework.solver import HarnessSolver, _LocalServerRuntime
from ouro3.mode import HarnessMode, active_harness_mode


@dataclass
class DuckDiversityHarnessSolver(HarnessSolver):
    label: str = "duck-diversity"
    primary_seed: int | None = 0
    diversity_no_change_threshold: int = 2
    diversity_max_triggers: int = 8
    diversity_seed_offset: int = 17

    def _make_analyzer(
        self,
        game: taaf.game.Game,
        index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> Any:
        del game, index
        if active_harness_mode() != HarnessMode.DUCK_DIVERSITY:
            raise RuntimeError(
                "DuckDiversityHarnessSolver requires duck-diversity mode"
            )
        return DuckDiversityToolAgent(
            seed=self.primary_seed,
            model=self.model,
            timeout=self.analyzer_timeout,
            save_request_logs=self.save_request_logs,
            diversity_no_change_threshold=self.diversity_no_change_threshold,
            diversity_max_triggers=self.diversity_max_triggers,
            diversity_seed_offset=self.diversity_seed_offset,
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
