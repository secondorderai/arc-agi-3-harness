"""Logged adapter repairs for the executable one-step contract."""

from __future__ import annotations

from typing import Any

from duck_deliberate.agent import DuckDeliberateToolAgent
from duck_contract.agent import CONTRACT_SYSTEM_ADDENDUM, CONTRACT_USER_ADDENDUM


REPAIR_SYSTEM_ADDENDUM = """\
Contract-repair lane:
Keep selecting exactly one evidence-based action. Prefer an explicit non-empty
`expect` object. If a generated action omits it, the harness will insert the
generic probe `{'board_changed': True}` and record `contract_repairs`; this is
not evidence that the model predicted correctly. If a generated list contains
several actions, only the first is executable and the truncation is recorded.
After the returned result, use the observed match or mismatch to revise the
hypothesis. Never rely on a repaired expectation as a game-specific rule.
"""


class DuckContractRepairToolAgent(DuckDeliberateToolAgent):
    """One-step contract with explicit, auditable syntax repairs."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._system_prompt = (
            f"{self._system_prompt}\n\n{CONTRACT_SYSTEM_ADDENDUM}\n\n"
            f"{REPAIR_SYSTEM_ADDENDUM}"
        )
        self._contract_repairs = 0
        self._contract_batch_truncations = 0

    def _build_user_prompt(self, *args: Any, **kwargs: Any) -> str:
        base = super()._build_user_prompt(*args, **kwargs)
        return (
            f"{REPAIR_SYSTEM_ADDENDUM}\n\n{CONTRACT_USER_ADDENDUM}\n\n"
            f"{base}\n\n{REPAIR_SYSTEM_ADDENDUM}"
        )

    def _normalize_python_actions(self, value: Any) -> list[dict[str, Any]]:
        before_proposals = self._proposal_count
        normalized = DuckDeliberateToolAgent._normalize_python_actions(self, value)
        if not normalized:
            raise ValueError("duck-contract-repair requires an action")
        if len(normalized) > 1:
            self._proposal_count = before_proposals
            self._contract_batch_truncations += len(normalized) - 1
            normalized = normalized[:1]

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
        repaired = (
            not isinstance(expectation, dict)
            or not expectation
            or not set(expectation).issubset(allowed)
        )
        if repaired:
            self._proposal_count = before_proposals
            normalized[0]["expect"] = {"board_changed": True}
            self._contract_repairs += 1
            self._proposal_count += 1
        return normalized

    @property
    def telemetry(self) -> dict[str, int]:
        value = dict(super().telemetry)
        value.update(
            {
                "contract_repairs": self._contract_repairs,
                "contract_batch_truncations": self._contract_batch_truncations,
            }
        )
        return value

    @property
    def diagnostics(self) -> dict[str, Any]:
        value = dict(super().diagnostics)
        value.update(
            {
                "mode": "duck-contract-repair",
                "protocol": "one-step action -> logged repair if needed -> verify -> revise",
                "mandatory_expect": False,
                "repair_expectation": {"board_changed": True},
                "contract_repairs": self._contract_repairs,
                "contract_batch_truncations": self._contract_batch_truncations,
                "maximum_action_batch_size": 1,
            }
        )
        return value
