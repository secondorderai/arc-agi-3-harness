"""Stock Duck with an executable, falsifiable one-step action contract.

This is deliberately narrower than the retired robust and memory lanes.  It
keeps Duck's perception, history, sampling, and runtime unchanged, but makes
the proposal/verification interface mandatory and unambiguous.
"""

from __future__ import annotations

from typing import Any

from duck_deliberate.agent import DuckDeliberateToolAgent


CONTRACT_SYSTEM_ADDENDUM = """\
Executable action contract:
Every gameplay turn must execute exactly one action, never a batch.  The
action call must use this exact Python shape, replacing only the valid action
and the evidence-based expectation:

    result = action([{'action': 'RIGHT', 'expect': {'board_changed': True}}])

The `expect` value must be a non-empty object with one or more observable,
generic fields: `board_changed`, `gameplay_changed`, `hud_changed`,
`level_completed`, `game_over`, `run_complete`, or `level`.  State an
expectation only when the current frame and prior transitions make that
outcome testable.  Do not emit a bare action string, an action without
`expect`, or a multi-action list.  Read `result['action_result']` immediately,
then compare the returned evidence with the expectation before the next turn.
"""


CONTRACT_USER_ADDENDUM = """\
Executable one-step loop for this turn:
1. Inspect the newest frame and the last transition.
2. State one hypothesis and one small test that can distinguish it.
3. Execute exactly one action using this literal template:

   result = action([{'action': '<one valid action>', 'expect': {'board_changed': True}}])

   Replace the example action and expectation with evidence-based values.  Do
   not call `action` with a string, without `expect`, or with more than one
   action.  The returned object is the only authoritative outcome.
4. Inspect `result['action_result']['prediction_matched']`; if false, stop
   planning and revise the hypothesis on the next turn.

Keep the code compact and call the action from Python.  This protocol is
generic: it must work for movement, clicks, timers, animations, and level
transitions without assuming a game ID or fixed coordinate.
"""


class DuckContractToolAgent(DuckDeliberateToolAgent):
    """Duck-deliberate with mandatory one-step executable predictions."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._system_prompt = (
            f"{self._system_prompt}\n\n{CONTRACT_SYSTEM_ADDENDUM}"
        )

    def _build_user_prompt(self, *args: Any, **kwargs: Any) -> str:
        base = super()._build_user_prompt(*args, **kwargs)
        return f"{CONTRACT_USER_ADDENDUM}\n\n{base}\n\n{CONTRACT_USER_ADDENDUM}"

    def _normalize_python_actions(self, value: Any) -> list[dict[str, Any]]:
        # The parent preserves expect fields and counts accepted proposals. If
        # validation fails, roll back that count so telemetry measures only
        # executable proposals, not rejected retries.
        before = self._proposal_count
        normalized = super()._normalize_python_actions(value)
        if len(normalized) != 1:
            self._proposal_count = before
            raise ValueError(
                "duck-contract requires exactly one action with an expect object"
            )
        expectation = normalized[0].get("expect")
        allowed = {
            "board_changed",
            "gameplay_changed",
            "hud_changed",
            "level_completed",
            "game_over",
            "run_complete",
            "level",
        }
        if not isinstance(expectation, dict) or not expectation:
            self._proposal_count = before
            raise ValueError(
                "duck-contract requires a non-empty generic expect object"
            )
        if not set(expectation).issubset(allowed):
            self._proposal_count = before
            raise ValueError(
                "duck-contract expect fields must be generic observable fields"
            )
        return normalized

    @property
    def diagnostics(self) -> dict[str, Any]:
        value = dict(super().diagnostics)
        value.update(
            {
                "mode": "duck-contract",
                "protocol": "one-step executable action -> verify -> revise",
                "mandatory_expect": True,
                "maximum_action_batch_size": 1,
            }
        )
        return value
