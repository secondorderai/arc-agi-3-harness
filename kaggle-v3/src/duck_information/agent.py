"""Stock Duck sidecar that requests one discriminating observation on stalls.

The sidecar leaves the Stock Duck system prompt, tool surface, history policy,
and sampling unchanged. It only adds a compact request after an unchanged
gameplay frame; the model decides whether the existing Python inspection tool
is useful before returning to the normal action loop.
"""

from __future__ import annotations

from typing import Any

from inference.agent.runtime_state import Frame, HistoryEntry
from inference.agent.tool_agent import ToolAgent


INFORMATION_USER_ADDENDUM = """\\
Targeted information request: {reason}.
Do not restate the whole plan. Identify the single observation that would best
distinguish the two leading explanations for the unchanged state. Use the
existing Python inspection tool to obtain only that observation when useful,
then choose exactly one low-risk action. Record what the observation ruled in
or out and stop after one short action sequence. Do not reset unless the
environment reports game over.
"""


class DuckInformationToolAgent(ToolAgent):
    """Stock Duck with a sparse, host-triggered information query."""

    def __init__(
        self,
        *,
        information_no_change_threshold: int = 2,
        information_max_triggers: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.information_no_change_threshold = max(
            2, int(information_no_change_threshold)
        )
        self.information_max_triggers = max(1, int(information_max_triggers))
        self._information_trigger_count = 0
        self._information_no_change_triggers = 0
        self._last_information_reason = ""

    def _same_level(self, history_entries: list[HistoryEntry]) -> bool:
        if len(history_entries) < 2:
            return True
        recent = history_entries[-(self.information_no_change_threshold + 2) :]
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

    def _information_reason(
        self, history_entries: list[HistoryEntry]
    ) -> str | None:
        if self._information_trigger_count >= self.information_max_triggers:
            return None
        if not self._same_level(history_entries):
            return None
        no_change_count = self._trailing_no_change_count(history_entries)
        if no_change_count < self.information_no_change_threshold:
            return None
        self._information_no_change_triggers += 1
        return (
            f"the last {no_change_count} transitions left the gameplay frame unchanged"
        )

    def _build_user_prompt(self, *args: Any, **kwargs: Any) -> str:
        prompt = super()._build_user_prompt(*args, **kwargs)
        history_entries = kwargs.get("history_entries")
        if not isinstance(history_entries, list):
            history_entries = []
        reason = self._information_reason(history_entries)
        if reason is None:
            return prompt
        self._information_trigger_count += 1
        self._last_information_reason = reason
        return f"{prompt}\n\n{INFORMATION_USER_ADDENDUM.format(reason=reason)}"

    @property
    def telemetry(self) -> dict[str, int]:
        value = dict(super().telemetry)
        value.update(
            {
                "information_trigger_count": self._information_trigger_count,
                "information_no_change_triggers": self._information_no_change_triggers,
            }
        )
        return value

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "mode": "duck-information",
            "stock_system_prompt": True,
            "stock_tool_surface": True,
            "stock_history_policy": True,
            "information_trigger_count": self._information_trigger_count,
            "information_no_change_triggers": self._information_no_change_triggers,
            "information_no_change_threshold": self.information_no_change_threshold,
            "information_max_triggers": self.information_max_triggers,
            "last_information_reason": self._last_information_reason,
        }
