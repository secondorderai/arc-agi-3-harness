"""Solver adapter for the sparse Stock Duck self-audit lane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import taaf.game

from duck_audit.agent import DuckAuditToolAgent
from inference.framework.solver import HarnessSolver, _LocalServerRuntime
from ouro3.mode import HarnessMode, active_harness_mode


@dataclass
class DuckAuditHarnessSolver(HarnessSolver):
    """Stock Duck runtime with a sparse, event-triggered audit prompt."""

    label: str = "duck-audit"
    primary_seed: int | None = 0
    audit_repeat_threshold: int = 3
    audit_no_change_threshold: int = 2
    audit_max_triggers: int = 8

    def _make_analyzer(
        self,
        game: taaf.game.Game,
        index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> Any:
        del game, index
        if active_harness_mode() != HarnessMode.DUCK_AUDIT:
            raise RuntimeError("DuckAuditHarnessSolver requires duck-audit mode")
        return DuckAuditToolAgent(
            seed=self.primary_seed,
            model=self.model,
            timeout=self.analyzer_timeout,
            save_request_logs=self.save_request_logs,
            audit_repeat_threshold=self.audit_repeat_threshold,
            audit_no_change_threshold=self.audit_no_change_threshold,
            audit_max_triggers=self.audit_max_triggers,
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
