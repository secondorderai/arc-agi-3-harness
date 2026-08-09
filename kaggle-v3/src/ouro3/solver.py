"""Deadline-aware solver using the hybrid v3 analyzer."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import taaf.game

from inference.framework.solver import HarnessSolver, _LocalServerRuntime
from ouro3.agent import HybridToolAgent
from ouro3.scheduler import GlobalScheduler


@dataclass
class HybridHarnessSolver(HarnessSolver):
    label: str = "kaggle-v3-hybrid"
    failure_floor: int = 3
    seed_base: int = 0
    seed_group_size: int = 0
    scheduler_soft_deadline_s: float = 8 * 60 * 60 + 40 * 60
    scheduler_reserve_s: float = 20 * 60
    _scheduler: GlobalScheduler | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _runtime_budgets: dict[int, float] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __deepcopy__(self, memo: dict[int, Any]) -> "HybridHarnessSolver":
        cls = type(self)
        new = cls.__new__(cls)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            if key == "_scheduler":
                object.__setattr__(new, key, None)
            elif key == "_runtime_budgets":
                object.__setattr__(new, key, {})
            elif key == "_stop_event":
                import threading

                object.__setattr__(new, key, threading.Event())
            elif key == "_worker_pool":
                object.__setattr__(new, key, None)
            elif key in {"_local_servers", "_local_server_original_env"}:
                object.__setattr__(new, key, [] if key == "_local_servers" else {})
            else:
                object.__setattr__(new, key, copy.deepcopy(value, memo))
        return new

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        self._scheduler = GlobalScheduler(
            total_games=len(games),
            concurrency=self.concurrency,
            soft_deadline_s=self.scheduler_soft_deadline_s,
            setup_teardown_reserve_s=self.scheduler_reserve_s,
        )
        await super()._run_games(games)

    def _play_one(
        self,
        game: taaf.game.Game,
        index: int,
        pass_index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> None:
        scheduler = self._scheduler
        if scheduler is not None:
            self._runtime_budgets[index] = scheduler.start_session(index)
        try:
            super()._play_one(game, index, pass_index, local_server)
        finally:
            if scheduler is not None:
                scheduler.finish_session(index)

    def game_runtime_budget_seconds(self, game_index: int) -> float | None:
        adaptive = self._runtime_budgets.get(game_index)
        configured = self.max_runtime_s_per_game
        if adaptive is None:
            return configured
        if configured is None:
            return adaptive
        return min(float(configured), adaptive)

    def _make_analyzer(
        self,
        game: taaf.game.Game,
        index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> Any:
        if self.analyzer_factory is not None:
            return self.analyzer_factory(game, index)
        game_key = game.game_run.game_id if game.game_run is not None else str(index)
        model_seed = self.seed_base
        if self.seed_group_size > 0:
            model_seed += index // self.seed_group_size
        return HybridToolAgent(
            game_key=game_key,
            failure_floor=self.failure_floor,
            seed=model_seed,
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
