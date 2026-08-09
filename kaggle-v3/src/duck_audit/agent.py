"""Stock Duck with an event-triggered self-audit prompt.

The sidecar deliberately leaves the Stock Duck system prompt, tool surface,
action normalization, history policy, and sampling untouched.  It appends a
small audit instruction only after a generic no-progress signal is observed
in the serialized runtime frames.  This keeps the Poetiq-inspired
self-monitoring loop sparse instead of paying an extra reasoning turn on
every action.
"""

from __future__ import annotations

from typing import Any

from inference.agent.runtime_state import Frame, HistoryEntry
from inference.agent.tool_agent import ToolAgent


AUDIT_USER_ADDENDUM = """\
Sparse self-audit trigger: {reason}.
Do not assume that the current plan is still useful. First inspect the newest
frame and the smallest relevant transition evidence. Choose exactly one
response: continue the current plan, perform one targeted inspection, or
replan from the observed state. Do not reset unless the environment itself
reports game over. Execute at most one short action sequence after this audit,
and stop if it reaches a terminal state.
"""


class DuckAuditToolAgent(ToolAgent):
    """Stock Duck with a sparse, host-triggered self-audit instruction."""

    def __init__(
        self,
        *,
        audit_repeat_threshold: int = 3,
        audit_no_change_threshold: int = 2,
        audit_max_triggers: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.audit_repeat_threshold = max(2, int(audit_repeat_threshold))
        self.audit_no_change_threshold = max(2, int(audit_no_change_threshold))
        self.audit_max_triggers = max(1, int(audit_max_triggers))
        self._audit_trigger_count = 0
        self._audit_repeat_triggers = 0
        self._audit_no_change_triggers = 0
        self._last_audit_reason = ""

    def _same_level(self, history_entries: list[HistoryEntry]) -> bool:
        if len(history_entries) < 2:
            return True
        recent_window = max(
            self.audit_repeat_threshold,
            self.audit_no_change_threshold,
        ) + 2
        recent = history_entries[-recent_window:]
        last_level = recent[-1].frame.level
        return all(entry.frame.level == last_level for entry in recent)

    @staticmethod
    def _trailing_repeat_count(history_entries: list[HistoryEntry]) -> int:
        actions = [entry.action.strip() for entry in history_entries if entry.action.strip()]
        if not actions:
            return 0
        last = actions[-1]
        count = 0
        for action in reversed(actions):
            if action != last:
                break
            count += 1
        return count

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

    def _audit_reason(self, history_entries: list[HistoryEntry]) -> str | None:
        if self._audit_trigger_count >= self.audit_max_triggers:
            return None
        if not self._same_level(history_entries):
            return None
        repeat_count = self._trailing_repeat_count(history_entries)
        if repeat_count >= self.audit_repeat_threshold:
            self._audit_repeat_triggers += 1
            return (
                f"the same action was repeated {repeat_count} consecutive times"
            )
        no_change_count = self._trailing_no_change_count(history_entries)
        if no_change_count >= self.audit_no_change_threshold:
            self._audit_no_change_triggers += 1
            return (
                f"the last {no_change_count} transitions left the gameplay frame unchanged"
            )
        return None

    def _build_user_prompt(self, *args: Any, **kwargs: Any) -> str:
        prompt = super()._build_user_prompt(*args, **kwargs)
        history_entries = kwargs.get("history_entries")
        if not isinstance(history_entries, list):
            history_entries = []
        reason = self._audit_reason(history_entries)
        if reason is None:
            return prompt
        self._audit_trigger_count += 1
        self._last_audit_reason = reason
        return f"{prompt}\n\n{AUDIT_USER_ADDENDUM.format(reason=reason)}"

    @property
    def telemetry(self) -> dict[str, int]:
        value = dict(super().telemetry)
        value.update(
            {
                "audit_trigger_count": self._audit_trigger_count,
                "audit_repeat_triggers": self._audit_repeat_triggers,
                "audit_no_change_triggers": self._audit_no_change_triggers,
            }
        )
        return value

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "mode": "duck-audit",
            "stock_system_prompt": True,
            "stock_tool_surface": True,
            "stock_history_policy": True,
            "audit_trigger_count": self._audit_trigger_count,
            "audit_repeat_triggers": self._audit_repeat_triggers,
            "audit_no_change_triggers": self._audit_no_change_triggers,
            "audit_repeat_threshold": self.audit_repeat_threshold,
            "audit_no_change_threshold": self.audit_no_change_threshold,
            "audit_max_triggers": self.audit_max_triggers,
            "last_audit_reason": self._last_audit_reason,
        }
