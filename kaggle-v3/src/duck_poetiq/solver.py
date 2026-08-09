"""TAAF solver adapter for the composite Poetiq Stock Duck lane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import taaf.game

from duck_poetiq.agent import DuckPoetiqToolAgent
from inference.framework.solver import HarnessSolver, _LocalServerRuntime
from ouro3.mode import HarnessMode, active_harness_mode


@dataclass
class DuckPoetiqHarnessSolver(HarnessSolver):
    """Stock Duck runtime with one bounded, auditable intervention protocol."""

    label: str = "duck-poetiq"
    primary_seed: int | None = 0
    poetiq_repeat_threshold: int = 4
    poetiq_no_change_threshold: int = 3
    poetiq_intervention_cooldown_actions: int = 12
    poetiq_max_interventions_per_level: int = 2
    poetiq_diversity_seed_offset: int = 17
    poetiq_yield_min_actions: int = 64
    poetiq_yield_min_elapsed_s: float = 30 * 60
    poetiq_yield_window: int = 16
    poetiq_yield_max_changes: int = 0

    def _make_analyzer(
        self,
        game: taaf.game.Game,
        index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> Any:
        del game, index
        if active_harness_mode() != HarnessMode.DUCK_POETIQ:
            raise RuntimeError("DuckPoetiqHarnessSolver requires duck-poetiq mode")
        return DuckPoetiqToolAgent(
            primary_seed=self.primary_seed,
            model=self.model,
            timeout=self.analyzer_timeout,
            save_request_logs=self.save_request_logs,
            repeat_threshold=self.poetiq_repeat_threshold,
            no_change_threshold=self.poetiq_no_change_threshold,
            intervention_cooldown_actions=self.poetiq_intervention_cooldown_actions,
            max_interventions_per_level=self.poetiq_max_interventions_per_level,
            diversity_seed_offset=self.poetiq_diversity_seed_offset,
            yield_min_actions=self.poetiq_yield_min_actions,
            yield_min_elapsed_s=self.poetiq_yield_min_elapsed_s,
            yield_window=self.poetiq_yield_window,
            yield_max_changes=self.poetiq_yield_max_changes,
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
