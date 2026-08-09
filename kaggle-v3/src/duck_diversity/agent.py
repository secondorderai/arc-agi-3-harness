"""Stock Duck with one alternate sampling path on a generic stall.

The alternate path is deliberately bounded: it changes only the request seed
for the triggered analysis turn, uses the same model/tools/history, and then
returns to the primary seed.  It is not an always-on ensemble and does not
issue a second request for the same action.
"""

from __future__ import annotations

from typing import Any

from inference.agent.runtime_state import HistoryEntry
from inference.agent.tool_agent import ToolAgent, _ChatCompletionResult


DIVERSITY_USER_ADDENDUM = """\
Controlled diversity trigger: {reason}.
Use this turn as an independent second sampling path over the same evidence,
not as an ensemble or a vote. Reinspect the newest frame, propose one action
that is supported by the strongest surviving explanation, and execute at most
one short action sequence. Compare the result with prior transitions before
continuing. Do not reset, invent coordinates, or repeat a contradicted plan.
"""


class DuckDiversityToolAgent(ToolAgent):
    """Stock Duck with a bounded alternate seed only after no progress."""

    def __init__(
        self,
        *,
        diversity_no_change_threshold: int = 2,
        diversity_max_triggers: int = 8,
        diversity_seed_offset: int = 17,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.diversity_no_change_threshold = max(
            2, int(diversity_no_change_threshold)
        )
        self.diversity_max_triggers = max(1, int(diversity_max_triggers))
        self.diversity_seed_offset = max(1, int(diversity_seed_offset))
        self._diversity_trigger_count = 0
        self._diversity_no_change_triggers = 0
        self._diversity_seed_uses = 0
        self._diversity_seed_override: int | None = None
        self._last_diversity_reason = ""

    def _same_level(self, history_entries: list[HistoryEntry]) -> bool:
        if len(history_entries) < 2:
            return True
        recent = history_entries[-(self.diversity_no_change_threshold + 2) :]
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

    def _diversity_reason(
        self, history_entries: list[HistoryEntry]
    ) -> str | None:
        if self._diversity_trigger_count >= self.diversity_max_triggers:
            return None
        if not self._same_level(history_entries):
            return None
        no_change_count = self._trailing_no_change_count(history_entries)
        if no_change_count < self.diversity_no_change_threshold:
            return None
        self._diversity_no_change_triggers += 1
        return (
            f"the last {no_change_count} transitions left the gameplay frame unchanged"
        )

    def _build_user_prompt(self, *args: Any, **kwargs: Any) -> str:
        prompt = super()._build_user_prompt(*args, **kwargs)
        history_entries = kwargs.get("history_entries")
        if not isinstance(history_entries, list):
            history_entries = []
        reason = self._diversity_reason(history_entries)
        if reason is None:
            self._diversity_seed_override = None
            return prompt
        self._diversity_trigger_count += 1
        self._last_diversity_reason = reason
        primary = self._seed if self._seed is not None else 0
        self._diversity_seed_override = int(primary) + self.diversity_seed_offset
        self._diversity_seed_uses += 1
        return f"{prompt}\n\n{DIVERSITY_USER_ADDENDUM.format(reason=reason)}"

    def _chat_completion(self, *args: Any, **kwargs: Any) -> _ChatCompletionResult:
        original_seed = self._seed
        if self._diversity_seed_override is not None:
            self._seed = self._diversity_seed_override
        try:
            return super()._chat_completion(*args, **kwargs)
        finally:
            self._seed = original_seed
            self._diversity_seed_override = None

    @property
    def telemetry(self) -> dict[str, int]:
        value = dict(super().telemetry)
        value.update(
            {
                "diversity_trigger_count": self._diversity_trigger_count,
                "diversity_no_change_triggers": self._diversity_no_change_triggers,
                "diversity_seed_uses": self._diversity_seed_uses,
            }
        )
        return value

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "mode": "duck-diversity",
            "stock_system_prompt": True,
            "stock_tool_surface": True,
            "stock_history_policy": True,
            "protocol": "stall trigger -> alternate seed -> single proposal -> observe",
            "diversity_trigger_count": self._diversity_trigger_count,
            "diversity_no_change_triggers": self._diversity_no_change_triggers,
            "diversity_seed_uses": self._diversity_seed_uses,
            "diversity_no_change_threshold": self.diversity_no_change_threshold,
            "diversity_max_triggers": self.diversity_max_triggers,
            "diversity_seed_offset": self.diversity_seed_offset,
            "second_model_request": False,
            "last_diversity_reason": self._last_diversity_reason,
        }
