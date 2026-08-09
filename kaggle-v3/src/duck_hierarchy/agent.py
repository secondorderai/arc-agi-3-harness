"""Stock Duck with a bounded candidate hierarchy and cheap verifier.

The host only adds an instruction at generic no-progress boundaries.  The
model still uses the original Python tool and action transport; the addendum
asks it to compare a small number of alternatives against observed evidence
and to execute one low-risk probe.  No game-specific candidate names,
coordinates, or reset behavior are injected.
"""

from __future__ import annotations

from typing import Any

from inference.agent.runtime_state import HistoryEntry
from inference.agent.tool_agent import ToolAgent


HIERARCHY_USER_ADDENDUM = """\
Bounded candidate search trigger: {reason}.
Do not continue an unverified plan. Build at most three candidate mechanics or
goal explanations from the current frame and transition history. For each,
state one observable consequence and the evidence supporting it. Select the
candidate with the strongest evidence and test it with exactly one low-risk
action or inspection. Use the returned frame delta as the cheap verifier:
discard any candidate whose predicted consequence is contradicted, then keep
only the surviving candidate for the next turn. Do not batch speculative
actions, invent coordinates, or reset unless the environment reports game
over.
"""


class DuckHierarchyToolAgent(ToolAgent):
    """Stock Duck with sparse, bounded candidate-search prompts."""

    def __init__(
        self,
        *,
        hierarchy_no_change_threshold: int = 2,
        hierarchy_max_triggers: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hierarchy_no_change_threshold = max(
            2, int(hierarchy_no_change_threshold)
        )
        self.hierarchy_max_triggers = max(1, int(hierarchy_max_triggers))
        self._hierarchy_trigger_count = 0
        self._hierarchy_no_change_triggers = 0
        self._hierarchy_level_start_triggers = 0
        self._last_hierarchy_reason = ""

    def _same_level(self, history_entries: list[HistoryEntry]) -> bool:
        if len(history_entries) < 2:
            return True
        recent = history_entries[-(self.hierarchy_no_change_threshold + 2) :]
        last_level = recent[-1].frame.level
        return all(entry.frame.level == last_level for entry in recent)

    @staticmethod
    def _trailing_no_change_count(history_entries: list[HistoryEntry]) -> int:
        if len(history_entries) < 2:
            return 0
        count = 0
        for current, previous in zip(
            reversed(history_entries), reversed(history_entries[:-1])
        ):
            if current.frame.level != previous.frame.level:
                break
            if current.frame.grid != previous.frame.grid:
                break
            count += 1
        return count

    def _level_started(self, history_entries: list[HistoryEntry]) -> bool:
        if len(history_entries) < 2:
            return False
        return history_entries[-1].frame.level != history_entries[-2].frame.level

    def _hierarchy_reason(
        self, history_entries: list[HistoryEntry]
    ) -> str | None:
        if self._hierarchy_trigger_count >= self.hierarchy_max_triggers:
            return None
        if self._level_started(history_entries):
            self._hierarchy_level_start_triggers += 1
            return "a new level has started and its mechanics are not yet ranked"
        if not self._same_level(history_entries):
            return None
        no_change_count = self._trailing_no_change_count(history_entries)
        if no_change_count < self.hierarchy_no_change_threshold:
            return None
        self._hierarchy_no_change_triggers += 1
        return (
            f"the last {no_change_count} transitions left the gameplay frame unchanged"
        )

    def _build_user_prompt(self, *args: Any, **kwargs: Any) -> str:
        prompt = super()._build_user_prompt(*args, **kwargs)
        history_entries = kwargs.get("history_entries")
        if not isinstance(history_entries, list):
            history_entries = []
        reason = self._hierarchy_reason(history_entries)
        if reason is None:
            return prompt
        self._hierarchy_trigger_count += 1
        self._last_hierarchy_reason = reason
        return f"{prompt}\n\n{HIERARCHY_USER_ADDENDUM.format(reason=reason)}"

    @property
    def telemetry(self) -> dict[str, int]:
        value = dict(super().telemetry)
        value.update(
            {
                "hierarchy_trigger_count": self._hierarchy_trigger_count,
                "hierarchy_no_change_triggers": self._hierarchy_no_change_triggers,
                "hierarchy_level_start_triggers": self._hierarchy_level_start_triggers,
            }
        )
        return value

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "mode": "duck-hierarchy",
            "stock_system_prompt": True,
            "stock_tool_surface": True,
            "stock_history_policy": True,
            "protocol": "bounded candidates -> low-risk probe -> frame-delta verifier",
            "maximum_candidates": 3,
            "hierarchy_trigger_count": self._hierarchy_trigger_count,
            "hierarchy_no_change_triggers": self._hierarchy_no_change_triggers,
            "hierarchy_level_start_triggers": self._hierarchy_level_start_triggers,
            "hierarchy_no_change_threshold": self.hierarchy_no_change_threshold,
            "hierarchy_max_triggers": self.hierarchy_max_triggers,
            "last_hierarchy_reason": self._last_hierarchy_reason,
        }
