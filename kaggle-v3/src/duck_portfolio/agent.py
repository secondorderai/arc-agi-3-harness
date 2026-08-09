"""One-conversation Stock Duck agent with deterministic policy routing."""

from __future__ import annotations

from typing import Any

from duck_audit.agent import AUDIT_USER_ADDENDUM
from duck_contract.agent import CONTRACT_SYSTEM_ADDENDUM, CONTRACT_USER_ADDENDUM
from duck_contract.repair_agent import REPAIR_SYSTEM_ADDENDUM
from duck_deliberate.agent import DELIBERATE_SYSTEM_ADDENDUM, DELIBERATE_USER_ADDENDUM
from duck_portfolio.router import (
    PortfolioDecision,
    PortfolioPolicy,
    PortfolioRouter,
    PortfolioTransition,
    extract_portfolio_features,
)
from inference.agent.runtime_state import HistoryEntry
from inference.agent.tool_agent import ToolAgent


PORTFOLIO_SELECTION_NOTICE = """\
Deterministic portfolio route: {policy}. The host selected this generic policy
from coarse visual structure and the first {warmup_actions} observed action
effects. Continue from the existing conversation and world model. Do not infer
or guess a game identity from the routing decision.
"""

PORTFOLIO_SWITCH_NOTICE = """\
Portfolio stall switch: {previous} -> {policy}. No level progress occurred and
the latest {window} gameplay transitions were unchanged. Preserve verified
knowledge, discard only the stalled plan, and continue without resetting.
"""


class DuckPortfolioToolAgent(ToolAgent):
    """Route among Stock-derived policies without replacing the chat agent."""

    def __init__(
        self,
        *,
        router: PortfolioRouter | None = None,
        warmup_actions: int = 8,
        switch_min_actions: int = 64,
        switch_window: int = 16,
        switch_max_changes: int = 0,
        switch_min_remaining_s: float = 1800.0,
        audit_repeat_threshold: int = 3,
        audit_no_change_threshold: int = 2,
        audit_max_triggers: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.router = router or PortfolioRouter.load()
        self.warmup_actions = max(1, int(warmup_actions))
        self.switch_min_actions = max(1, int(switch_min_actions))
        self.switch_window = max(1, int(switch_window))
        self.switch_max_changes = max(0, int(switch_max_changes))
        self.switch_min_remaining_s = max(0.0, float(switch_min_remaining_s))
        self.audit_repeat_threshold = max(2, int(audit_repeat_threshold))
        self.audit_no_change_threshold = max(2, int(audit_no_change_threshold))
        self.audit_max_triggers = max(1, int(audit_max_triggers))
        self._base_system_prompt = self._system_prompt
        self._policy = PortfolioPolicy.STOCK
        self._initial_policy: PortfolioPolicy | None = None
        self._pending_policy: PortfolioPolicy | None = None
        self._pending_notice = ""
        self._selection_action: int | None = None
        self._switch_count = 0
        self._level_progress_seen = False
        self._initial_grid: Any = None
        self._transitions: list[PortfolioTransition] = []
        self._decision: PortfolioDecision | None = None
        self._features: dict[str, float] = {}
        self._route_events: list[dict[str, Any]] = []
        self._audit_trigger_count = 0
        self._audit_repeat_triggers = 0
        self._audit_no_change_triggers = 0
        self._last_audit_reason = ""
        self._prediction_matches = 0
        self._prediction_mismatches = 0
        self._proposal_count = 0
        self._revision_count = 0
        self._contract_repairs = 0
        self._contract_batch_truncations = 0
        self._policy_action_counts = {policy.value: 0 for policy in PortfolioPolicy}
        self._stock_fallbacks = 0

    @property
    def augmented_features_enabled(self) -> bool:
        return False

    @property
    def verified_actions_enabled(self) -> bool:
        return self._policy in {
            PortfolioPolicy.DELIBERATE,
            PortfolioPolicy.CONTRACT_REPAIR,
        }

    @property
    def prediction_verification_enabled(self) -> bool:
        return self.verified_actions_enabled

    @property
    def maximum_action_batch_size(self) -> int | None:
        # The router decision is applied at the next ordinary model turn.  Do
        # not allow a second action(...) call in the same Python tool program
        # to spill past the exact eight-action Stock warm-up (or a queued
        # switch) before that turn can receive the policy notice.
        if self._pending_policy is not None:
            return 0
        if self._initial_policy is None:
            return max(1, self.warmup_actions - len(self._transitions))
        if self._policy == PortfolioPolicy.CONTRACT_REPAIR:
            return 1
        return None

    def _system_prompt_for_policy(self, policy: PortfolioPolicy) -> str:
        if policy == PortfolioPolicy.DELIBERATE:
            return f"{self._base_system_prompt}\n\n{DELIBERATE_SYSTEM_ADDENDUM}"
        if policy == PortfolioPolicy.CONTRACT_REPAIR:
            return (
                f"{self._base_system_prompt}\n\n{DELIBERATE_SYSTEM_ADDENDUM}\n\n"
                f"{CONTRACT_SYSTEM_ADDENDUM}\n\n{REPAIR_SYSTEM_ADDENDUM}"
            )
        return self._base_system_prompt

    def _queue_policy(
        self,
        policy: PortfolioPolicy,
        *,
        action_num: int,
        event: str,
        notice: str,
    ) -> None:
        self._pending_policy = policy
        self._pending_notice = notice
        self._route_events.append(
            {
                "event": event,
                "action_num": int(action_num),
                "from": self._policy.value,
                "to": policy.value,
            }
        )

    def _activate_pending_policy(self) -> None:
        if self._pending_policy is None:
            return
        self._policy = self._pending_policy
        self._system_prompt = self._system_prompt_for_policy(self._policy)
        self._pending_policy = None

    @staticmethod
    def _trailing_repeat_count(history_entries: list[HistoryEntry]) -> int:
        actions = [entry.action.strip() for entry in history_entries if entry.action.strip()]
        if not actions:
            return 0
        count = 0
        for action in reversed(actions):
            if action != actions[-1]:
                break
            count += 1
        return count

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
    def _trailing_no_change_count(history_entries: list[HistoryEntry]) -> int:
        count = 0
        for current, previous in zip(
            reversed(history_entries), reversed(history_entries[:-1])
        ):
            if current.frame.level != previous.frame.level or current.frame.grid != previous.frame.grid:
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
            return f"the same action was repeated {repeat_count} consecutive times"
        no_change_count = self._trailing_no_change_count(history_entries)
        if no_change_count >= self.audit_no_change_threshold:
            self._audit_no_change_triggers += 1
            return f"the last {no_change_count} transitions left the gameplay frame unchanged"
        return None

    def _build_user_prompt(self, *args: Any, **kwargs: Any) -> str:
        self._activate_pending_policy()
        prompt = super()._build_user_prompt(*args, **kwargs)
        notice = self._pending_notice
        self._pending_notice = ""
        if self._policy == PortfolioPolicy.CONTRACT_REPAIR:
            # Preserve the standalone Contract Repair policy's prompt order;
            # its historical scores are one of the router's training targets.
            return "\n\n".join(
                [
                    REPAIR_SYSTEM_ADDENDUM,
                    CONTRACT_USER_ADDENDUM,
                    prompt,
                    *([notice] if notice else []),
                    DELIBERATE_USER_ADDENDUM,
                    REPAIR_SYSTEM_ADDENDUM,
                ]
            )
        additions: list[str] = []
        if notice:
            additions.append(notice)
        history_entries = kwargs.get("history_entries")
        if not isinstance(history_entries, list):
            history_entries = []
        if self._policy == PortfolioPolicy.AUDIT:
            reason = self._audit_reason(history_entries)
            if reason is not None:
                self._audit_trigger_count += 1
                self._last_audit_reason = reason
                additions.append(AUDIT_USER_ADDENDUM.format(reason=reason))
        elif self._policy == PortfolioPolicy.DELIBERATE:
            additions.append(DELIBERATE_USER_ADDENDUM)
        elif self._policy == PortfolioPolicy.CONTRACT_REPAIR:
            additions.extend(
                [DELIBERATE_USER_ADDENDUM, CONTRACT_USER_ADDENDUM, REPAIR_SYSTEM_ADDENDUM]
            )
        return "\n\n".join([prompt, *additions])

    def _normalize_python_actions(self, value: Any) -> list[dict[str, Any]]:
        before_proposals = self._proposal_count
        normalized = super()._normalize_python_actions(value)
        if self._policy not in {
            PortfolioPolicy.DELIBERATE,
            PortfolioPolicy.CONTRACT_REPAIR,
        }:
            return normalized
        items = [value] if isinstance(value, dict) else list(value) if isinstance(value, (list, tuple)) else []
        for index, item in enumerate(items):
            if index < len(normalized) and isinstance(item, dict) and "expect" in item:
                normalized[index]["expect"] = item.get("expect")
                self._proposal_count += 1
        if self._policy != PortfolioPolicy.CONTRACT_REPAIR:
            return normalized
        if len(normalized) > 1:
            self._proposal_count = before_proposals
            self._contract_batch_truncations += len(normalized) - 1
            normalized = normalized[:1]
        allowed = {
            "board_changed",
            "gameplay_changed",
            "hud_changed",
            "level_completed",
            "game_over",
            "run_complete",
            "level",
        }
        expectation = normalized[0].get("expect")
        if (
            not isinstance(expectation, dict)
            or not expectation
            or not set(expectation).issubset(allowed)
        ):
            self._proposal_count = before_proposals
            normalized[0]["expect"] = {"board_changed": True}
            self._contract_repairs += 1
            self._proposal_count += 1
        return normalized

    def _compact_action_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        compact = super()._compact_action_result(payload)
        if self.verified_actions_enabled and "prediction_matched" in payload:
            compact["prediction_matched"] = bool(payload["prediction_matched"])
        if payload.get("prediction_mismatch"):
            compact["prediction_mismatch"] = payload["prediction_mismatch"]
        return compact

    def register_prediction_match(self) -> None:
        self._prediction_matches += 1

    def register_prediction_mismatch(self, reason: str = "") -> None:
        del reason
        self._prediction_mismatches += 1
        self._revision_count += 1

    def observe_transition(
        self,
        *,
        action: str,
        before_grid: Any,
        after_grid: Any,
        payload: dict[str, Any],
    ) -> None:
        if self._initial_grid is None:
            self._initial_grid = before_grid
        changed_area = sum(
            max(0, int(item.get("area", 0)))
            for item in list(payload.get("changed_regions") or [])
            if isinstance(item, dict)
        )
        transition = PortfolioTransition(
            action=str(action),
            before_grid=before_grid,
            after_grid=after_grid,
            gameplay_changed=bool(payload.get("gameplay_changed")),
            hud_changed=bool(payload.get("hud_changed")),
            changed_area=changed_area,
        )
        self._transitions.append(transition)
        self._policy_action_counts[self._policy.value] += 1
        action_num = int(payload.get("action_num", len(self._transitions)))
        if bool(payload.get("level_completed")):
            self._level_progress_seen = True
            if self._initial_policy is None:
                self._initial_policy = PortfolioPolicy.STOCK
                self._selection_action = action_num
                self._route_events.append(
                    {
                        "event": "warmup-progress-lock",
                        "action_num": action_num,
                        "from": "stock",
                        "to": "stock",
                    }
                )
            return

        if self._initial_policy is None and len(self._transitions) >= self.warmup_actions:
            self._features = extract_portfolio_features(
                self._initial_grid,
                self._transitions[: self.warmup_actions],
            )
            self._decision = self.router.decide(self._features)
            self._initial_policy = self._decision.policy
            self._selection_action = action_num
            self._stock_fallbacks += int(self._decision.stock_fallback)
            self._queue_policy(
                self._decision.policy,
                action_num=action_num,
                event="selection",
                notice=(
                    ""
                    if self._decision.policy == PortfolioPolicy.STOCK
                    else PORTFOLIO_SELECTION_NOTICE.format(
                        policy=self._decision.policy.value,
                        warmup_actions=self.warmup_actions,
                    )
                ),
            )
            return

        if (
            self._initial_policy is None
            or self._decision is None
            or self._switch_count
            or self._level_progress_seen
            or self._selection_action is None
            or action_num - self._selection_action < self.switch_min_actions
        ):
            return
        recent = self._transitions[-self.switch_window :]
        if len(recent) < self.switch_window:
            return
        if sum(item.gameplay_changed for item in recent) > self.switch_max_changes:
            return
        remaining = payload.get("time_remaining_seconds")
        try:
            remaining_s = float(remaining)
        except (TypeError, ValueError):
            return
        if remaining_s < self.switch_min_remaining_s:
            return
        next_policy = self.router.next_policy(
            self._decision.adjusted_scores,
            self._policy,
        )
        previous = self._policy
        self._switch_count = 1
        self._queue_policy(
            next_policy,
            action_num=action_num,
            event="stall-switch",
            notice=PORTFOLIO_SWITCH_NOTICE.format(
                previous=previous.value,
                policy=next_policy.value,
                window=self.switch_window,
            ),
        )

    @property
    def telemetry(self) -> dict[str, int]:
        value = dict(super().telemetry)
        value.update(
            {
                "portfolio_selections": int(self._initial_policy is not None),
                "portfolio_stock_fallbacks": self._stock_fallbacks,
                "portfolio_switches": self._switch_count,
                "portfolio_audit_triggers": self._audit_trigger_count,
                "portfolio_prediction_matches": self._prediction_matches,
                "portfolio_prediction_mismatches": self._prediction_mismatches,
                "portfolio_hypothesis_revisions": self._revision_count,
                "portfolio_contract_repairs": self._contract_repairs,
                "portfolio_contract_batch_truncations": self._contract_batch_truncations,
                **{
                    f"portfolio_{policy.value.replace('-', '_')}_actions": count
                    for policy, count in (
                        (PortfolioPolicy(key), count)
                        for key, count in self._policy_action_counts.items()
                    )
                },
            }
        )
        return value

    @property
    def diagnostics(self) -> dict[str, Any]:
        decision = self._decision
        return {
            "mode": "duck-portfolio",
            "one_persistent_conversation": True,
            "parallel_model_trajectories": 0,
            "candidate_order": [policy.value for policy in PortfolioPolicy],
            "warmup_actions": self.warmup_actions,
            "initial_policy": self._initial_policy.value if self._initial_policy else None,
            "active_policy": self._policy.value,
            "selection_action": self._selection_action,
            "features": {key: round(value, 8) for key, value in self._features.items()},
            "raw_scores": dict(decision.raw_scores) if decision else {},
            "adjusted_scores": dict(decision.adjusted_scores) if decision else {},
            "confidence_margin": decision.confidence_margin if decision else 0.0,
            "stock_fallback": bool(decision and decision.stock_fallback),
            "relative_guardrail_enabled": bool(
                getattr(self.router, "relative_models", {})
            ),
            "relative_uncertainty_penalty": float(
                getattr(self.router, "relative_uncertainty_penalty", 0.0)
            ),
            "relative_stock_margin": float(
                getattr(self.router, "relative_stock_margin", 0.0)
            ),
            "router_artifact_sha256": self.router.artifact_hash,
            "route_events": list(self._route_events),
            "switch_count": self._switch_count,
            "level_progress_seen": self._level_progress_seen,
            "policy_action_counts": dict(self._policy_action_counts),
            "audit": {
                "trigger_count": self._audit_trigger_count,
                "repeat_triggers": self._audit_repeat_triggers,
                "no_change_triggers": self._audit_no_change_triggers,
                "last_reason": self._last_audit_reason,
            },
            "verification": {
                "proposals": self._proposal_count,
                "matches": self._prediction_matches,
                "mismatches": self._prediction_mismatches,
                "revisions": self._revision_count,
                "contract_repairs": self._contract_repairs,
                "contract_batch_truncations": self._contract_batch_truncations,
            },
        }
