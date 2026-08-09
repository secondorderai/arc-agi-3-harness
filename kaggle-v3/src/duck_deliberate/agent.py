"""Stock Duck with a narrow proposal, falsification, and verification loop.

The deliberate lane intentionally leaves Stock Duck's perception, history
budget, action execution, and sampling untouched.  It changes only the
instructions for choosing actions and enables the already audited, generic
``expect`` verifier in the solver.  No game-specific rules or fixed
coordinates are introduced.
"""

from __future__ import annotations

from typing import Any

from inference.agent.tool_agent import ToolAgent


DELIBERATE_SYSTEM_ADDENDUM = """\
Deliberate control protocol:
Before acting, state a small falsifiable hypothesis about the current level,
the intended goal, and the predicted observable effect of the next action.
Prefer the smallest experiment that distinguishes competing hypotheses. After
every action, compare the before and after evidence. If a prediction is
wrong, explicitly discard or revise the hypothesis before choosing another
action; do not continue a queued plan that depended on it. Use an ``expect``
object on each proposed action whenever you can predict a boolean state change
or level transition (for example ``{'board_changed': True}``). Keep the
proposal concise and spend tool time inspecting evidence rather than repeating
unchanged descriptions.
"""


DELIBERATE_USER_ADDENDUM = """\
Deliberate loop for this turn:
1. Hypothesis: name the current best explanation and one alternative.
2. Test: choose the smallest valid action or short batch that can falsify it.
3. Prediction: attach ``expect`` to every action when a board change,
   completion, game-over, or level outcome is predictable.
4. Revision: inspect the returned ``prediction_matched`` field and the frame
   delta. On a mismatch, stop planning, explain what was falsified, and revise
   the world/goal/action model before the next experiment.
Do not invent coordinates or rules that are not supported by the current
frame, history, or transition evidence.
"""


class DuckDeliberateToolAgent(ToolAgent):
    """Stock Duck with falsification-first prompts and generic verification."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._system_prompt = (
            f"{self._system_prompt}\n\n{DELIBERATE_SYSTEM_ADDENDUM}"
        )
        self._prediction_matches = 0
        self._prediction_mismatches = 0
        self._proposal_count = 0
        self._revision_count = 0

    @property
    def prediction_verification_enabled(self) -> bool:
        """Enable the solver's generic, game-agnostic expectation checker."""

        return True

    def _build_user_prompt(self, *args: Any, **kwargs: Any) -> str:
        return f"{super()._build_user_prompt(*args, **kwargs)}\n\n{DELIBERATE_USER_ADDENDUM}"

    def _normalize_python_actions(self, value: Any) -> list[dict[str, Any]]:
        normalized = super()._normalize_python_actions(value)
        if isinstance(value, dict):
            items = [value]
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            items = []
        for index, item in enumerate(items):
            if index >= len(normalized) or not isinstance(item, dict):
                continue
            if "expect" in item:
                normalized[index]["expect"] = item.get("expect")
                self._proposal_count += 1
        return normalized

    def _compact_action_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        compact = super()._compact_action_result(payload)
        if "prediction_matched" in payload:
            compact["prediction_matched"] = bool(payload["prediction_matched"])
        if payload.get("prediction_mismatch"):
            compact["prediction_mismatch"] = payload["prediction_mismatch"]
        return compact

    def register_prediction_match(self) -> None:
        self._prediction_matches += 1

    def register_prediction_mismatch(self, reason: str) -> None:
        del reason
        self._prediction_mismatches += 1
        self._revision_count += 1

    @property
    def telemetry(self) -> dict[str, int]:
        value = dict(super().telemetry)
        value.update(
            {
                "deliberate_proposals": self._proposal_count,
                "prediction_matches": self._prediction_matches,
                "prediction_mismatches": self._prediction_mismatches,
                "hypothesis_revisions": self._revision_count,
            }
        )
        return value

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "mode": "duck-deliberate",
            "protocol": "falsification-first hypothesis -> test -> verify -> revise",
            "stock_perception": True,
            "stock_history_policy": True,
            "generic_prediction_verification": True,
            "prediction_matches": self._prediction_matches,
            "prediction_mismatches": self._prediction_mismatches,
            "hypothesis_revisions": self._revision_count,
            "deliberate_proposals": self._proposal_count,
        }
