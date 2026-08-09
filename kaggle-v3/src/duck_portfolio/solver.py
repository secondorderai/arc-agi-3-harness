"""Solver adapter for the deterministic Duck portfolio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import taaf.game

from duck_portfolio.agent import DuckPortfolioToolAgent
from duck_portfolio.router import PortfolioRouter
from inference.framework.solver import HarnessSolver, _LocalServerRuntime
from ouro3.mode import HarnessMode, active_harness_mode


@dataclass
class DuckPortfolioHarnessSolver(HarnessSolver):
    label: str = "duck-portfolio"
    primary_seed: int | None = 0
    portfolio_warmup_actions: int = 8
    portfolio_switch_min_actions: int = 64
    portfolio_switch_window: int = 16
    portfolio_switch_max_changes: int = 0
    portfolio_switch_min_remaining_s: float = 1800.0
    portfolio_score_clip: float = 10.0
    portfolio_ridge_alpha: float = 10.0
    portfolio_uncertainty_penalty: float = 0.5
    portfolio_stock_margin: float = 0.25

    def _make_analyzer(
        self,
        game: taaf.game.Game,
        index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> Any:
        del game, index
        if active_harness_mode() != HarnessMode.DUCK_PORTFOLIO:
            raise RuntimeError(
                "DuckPortfolioHarnessSolver requires duck-portfolio mode"
            )
        router = PortfolioRouter.load()
        expected = {
            "score_clip": self.portfolio_score_clip,
            "ridge_alpha": self.portfolio_ridge_alpha,
            "uncertainty_penalty": self.portfolio_uncertainty_penalty,
            "stock_margin": self.portfolio_stock_margin,
            "warmup_actions": self.portfolio_warmup_actions,
        }
        mismatches = {
            key: {"expected": value, "actual": router.payload.get(key)}
            for key, value in expected.items()
            if router.payload.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"duck-portfolio router/config mismatch: {mismatches}")
        return DuckPortfolioToolAgent(
            router=router,
            seed=self.primary_seed,
            model=self.model,
            timeout=self.analyzer_timeout,
            save_request_logs=self.save_request_logs,
            warmup_actions=self.portfolio_warmup_actions,
            switch_min_actions=self.portfolio_switch_min_actions,
            switch_window=self.portfolio_switch_window,
            switch_max_changes=self.portfolio_switch_max_changes,
            switch_min_remaining_s=self.portfolio_switch_min_remaining_s,
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
