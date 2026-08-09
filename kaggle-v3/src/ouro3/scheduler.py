"""Global deadline-aware session budget allocation."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field


@dataclass
class GlobalScheduler:
    total_games: int
    concurrency: int = 28
    soft_deadline_s: float = 8 * 60 * 60 + 40 * 60
    setup_teardown_reserve_s: float = 20 * 60
    minimum_game_budget_s: float = 30
    started_at: float = field(default_factory=time.monotonic)
    _active: set[int] = field(default_factory=set, init=False, repr=False)
    _finished: set[int] = field(default_factory=set, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.total_games < 1 or self.concurrency < 1:
            raise ValueError("total_games and concurrency must be positive")
        if self.setup_teardown_reserve_s >= self.soft_deadline_s:
            raise ValueError("reserve must be shorter than the soft deadline")

    @property
    def usable_runtime_s(self) -> float:
        return self.soft_deadline_s - self.setup_teardown_reserve_s

    def remaining_runtime_s(self, now: float | None = None) -> float:
        current = time.monotonic() if now is None else now
        return max(0.0, self.usable_runtime_s - (current - self.started_at))

    def start_session(self, index: int, *, now: float | None = None) -> float:
        with self._lock:
            self._active.add(index)
            return self._budget_locked(now=now)

    def budget_for_new_session(self, *, now: float | None = None) -> float:
        with self._lock:
            return self._budget_locked(now=now)

    def _budget_locked(self, *, now: float | None) -> float:
        remaining_games = max(1, self.total_games - len(self._finished))
        remaining = self.remaining_runtime_s(now)
        waves = max(1, math.ceil(remaining_games / self.concurrency))
        wave_budget = remaining / waves if remaining else self.minimum_game_budget_s
        return max(self.minimum_game_budget_s, wave_budget * 0.94)

    def finish_session(self, index: int) -> None:
        with self._lock:
            self._active.discard(index)
            self._finished.add(index)

    @property
    def finished_count(self) -> int:
        with self._lock:
            return len(self._finished)

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)


@dataclass(frozen=True)
class SubmissionBudget:
    """Worst-case wall-clock budget for the 110-game gateway run.

    Kaggle's nine-hour limit includes notebook setup and teardown.  The
    gameplay loop therefore needs a cap that fits every concurrency wave,
    rather than simply copying the 132-minute public-game cap.
    """

    total_games: int
    concurrency: int
    waves: int
    per_game_cap_s: int
    soft_deadline_s: float
    setup_teardown_reserve_s: float
    safety_fraction: float

    @property
    def worst_case_gameplay_s(self) -> int:
        return self.waves * self.per_game_cap_s


def compute_submission_budget(
    *,
    total_games: int,
    concurrency: int,
    configured_game_cap_s: float,
    soft_deadline_s: float,
    setup_teardown_reserve_s: float,
    safety_fraction: float = 0.96,
) -> SubmissionBudget:
    """Fit all game waves inside the nine-hour submission envelope.

    The 4% per-wave cushion covers launch skew and cancellation/teardown
    overhead.  With the audited defaults (110 games, 28 workers, 7,920-second
    public cap, 31,200-second soft deadline, and a 20-minute reserve), this
    returns a 7,200-second cap and 28,800 seconds of worst-case gameplay.
    """

    if total_games < 1 or concurrency < 1:
        raise ValueError("total_games and concurrency must be positive")
    if configured_game_cap_s <= 0 or soft_deadline_s <= 0:
        raise ValueError("game cap and soft deadline must be positive")
    if setup_teardown_reserve_s < 0 or setup_teardown_reserve_s >= soft_deadline_s:
        raise ValueError("setup/teardown reserve must be within the deadline")
    if not 0 < safety_fraction <= 1:
        raise ValueError("safety_fraction must be in (0, 1]")

    waves = max(1, math.ceil(total_games / concurrency))
    usable = soft_deadline_s - setup_teardown_reserve_s
    per_wave = usable / waves
    per_game_cap_s = max(
        30,
        min(
            int(configured_game_cap_s),
            math.floor(per_wave * safety_fraction),
        ),
    )
    return SubmissionBudget(
        total_games=total_games,
        concurrency=concurrency,
        waves=waves,
        per_game_cap_s=per_game_cap_s,
        soft_deadline_s=soft_deadline_s,
        setup_teardown_reserve_s=setup_teardown_reserve_s,
        safety_fraction=safety_fraction,
    )
