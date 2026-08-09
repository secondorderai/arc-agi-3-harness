"""Stock Duck actor controlled by a persistent, typed transition verifier."""

from __future__ import annotations

import gzip
import json
import os
from typing import Any

from inference.agent.runtime_state import Frame, load_runtime_state
from inference.agent.tool_agent import AnalyzerTurnResult
from ouro3.agent import HybridToolAgent, _state_path_from_call
from ouro3.retrodict import RetrodictiveWorldModel, action_key, normalize_action


RETRODICT_SYSTEM_ADDENDUM = """
The host maintains the authoritative transition log and a bounded typed rule
version space. Host-certified rules have been replayed against every observed
transition; prose hypotheses have not. Prefer the supplied exact/certified
plan when present. When no plan is certified, retain the stock actor policy;
use a recommended probe only when it distinguishes multiple live outcomes.
Never describe model-written Python or a verbal hypothesis as certified.
""".strip()


class DuckRetrodictToolAgent(HybridToolAgent):
    """A single stock actor with host-owned evidence, replay and CPU search."""

    prediction_verification_enabled = True

    def __init__(
        self,
        *,
        game_key: str,
        failure_floor: int = 3,
        max_rules: int = 256,
        prediction_threshold: float = 0.90,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            game_key=game_key,
            failure_floor=failure_floor,
            **kwargs,
        )
        self.world_model = RetrodictiveWorldModel(
            max_rules=max_rules,
            prediction_threshold=prediction_threshold,
        )
        self._system_prompt = (
            f"{self._system_prompt}\n\n{RETRODICT_SYSTEM_ADDENDUM}"
        )
        self._host_plan_actions = 0
        self._host_plan_failures = 0
        self._host_probe_suggestions = 0
        self._last_level = 1

    @property
    def augmented_features_enabled(self) -> bool:
        return True

    @property
    def verified_actions_enabled(self) -> bool:
        return True

    @property
    def maximum_action_batch_size(self) -> int | None:
        # Certified host plans are executed one action at a time in analyze().
        # The fallback actor stays stock-compatible until such a plan exists;
        # an always-on one-action cap was both slower and materially weaker.
        return None

    def observe_transition(
        self,
        *,
        action: str,
        before_grid: Any,
        after_grid: Any,
        payload: dict[str, Any],
    ) -> None:
        level = int(payload.get("level", self._last_level) or self._last_level)
        if payload.get("level_completed"):
            level = max(1, level - 1)
        record = self.world_model.observe(
            level=level,
            action=action,
            before=before_grid,
            after=after_grid,
            payload=payload,
        )
        self._append_transition_trace(record)
        self._last_level = int(payload.get("level", level) or level)
        if self.world_model.last_prediction_mismatch:
            super().register_prediction_mismatch(
                self.world_model.last_prediction_mismatch
            )

    def analyze(self, *args: Any, **kwargs: Any) -> AnalyzerTurnResult | None:
        state_path = _state_path_from_call(args, kwargs)
        step_env = kwargs.get("step_env")
        valid_actions = [str(item) for item in kwargs.get("valid_actions") or []]
        if state_path is not None and callable(step_env) and valid_actions:
            current_frame, _history = load_runtime_state(state_path)
            if current_frame is not None:
                plan = self.world_model.plan(
                    current_frame.grid,
                    level=current_frame.level,
                    valid_actions=valid_actions,
                )
                if plan is not None and plan.actions:
                    action = dict(plan.actions[0])
                    if _legal_action(action, valid_actions):
                        payload = step_env(action)
                        executed = bool(
                            payload.get("executed")
                            if isinstance(payload, dict)
                            else False
                        )
                        if executed:
                            self._host_plan_actions += 1
                            self.consecutive_failures = 0
                            return AnalyzerTurnResult(
                                step_executed=True,
                                retryable_failure=False,
                                reasoning=(
                                    f"host {plan.source} executed one revalidated "
                                    f"action: {action_key(action)}"
                                ),
                            )
                        self._host_plan_failures += 1
        return super().analyze(*args, **kwargs)

    def _build_user_prompt(self, *args: Any, **kwargs: Any) -> str:
        prompt = super()._build_user_prompt(*args, **kwargs)
        current_frame = kwargs.get("current_frame")
        valid_actions = [str(item) for item in kwargs.get("valid_actions") or []]
        if not isinstance(current_frame, Frame):
            return prompt
        plan = self.world_model.plan(
            current_frame.grid,
            level=current_frame.level,
            valid_actions=valid_actions,
            max_depth=12,
            max_expanded=1_000,
        )
        probe = self.world_model.select_probe(
            current_frame.grid,
            valid_actions,
            level=current_frame.level,
        )
        useful_probe = (
            probe is not None
            and probe.predicted_outcomes >= 2
            and probe.information_gain > 0.0
        )
        if useful_probe:
            self._host_probe_suggestions += 1
        if plan is None and not useful_probe:
            return prompt
        facts = {
            "authoritative": True,
            "diagnostics": self.world_model.diagnostics(),
            "certified_plan": (
                {
                    "source": plan.source,
                    "confidence": plan.confidence,
                    "actions": list(plan.actions[:8]),
                }
                if plan is not None
                else None
            ),
            "recommended_probe": (
                {
                    "action": probe.action,
                    "information_gain": round(probe.information_gain, 6),
                    "risk": round(probe.risk, 6),
                    "novelty": round(probe.novelty, 6),
                    "predicted_outcomes": probe.predicted_outcomes,
                }
                if useful_probe and probe is not None
                else None
            ),
        }
        return (
            f"{prompt}\n\nHost retrodictive verifier (authoritative JSON):\n"
            f"{json.dumps(facts, sort_keys=True, separators=(',', ':'))}\n"
            "If certified_plan is null, treat recommended_probe as a suggestion "
            "and otherwise retain the stock actor policy."
        )

    @property
    def telemetry(self) -> dict[str, int]:
        value = dict(super().telemetry)
        diagnostics = self.world_model.diagnostics()
        value.update(
            {
                "retrodict_host_plan_actions": self._host_plan_actions,
                "retrodict_host_plan_failures": self._host_plan_failures,
                "retrodict_probe_suggestions": self._host_probe_suggestions,
                "retrodict_transitions": int(diagnostics["transitions"]),
                "retrodict_certified_rules": int(
                    diagnostics["certified_rules"]
                ),
                "retrodict_prediction_attempts": int(
                    diagnostics["prediction_attempts"]
                ),
                "retrodict_prediction_matches": int(
                    diagnostics["prediction_matches"]
                ),
                "retrodict_prediction_mismatches": int(
                    diagnostics["prediction_mismatches"]
                ),
                "retrodict_alias_conflicts": int(
                    diagnostics["alias_conflicts"]
                ),
            }
        )
        return value

    @property
    def diagnostics(self) -> dict[str, Any]:
        value = {
            "mode": "duck-retrodict",
            "stock_actor": True,
            "persistent_host_evidence": True,
            "host_owned_typed_rules": True,
            "automatic_execution_policy": "one exact-or-certified action then replan",
            "world_model": self.world_model.diagnostics(),
            "host_plan_actions": self._host_plan_actions,
            "host_plan_failures": self._host_plan_failures,
            "host_probe_suggestions": self._host_probe_suggestions,
        }
        return value

    def _append_transition_trace(self, record: Any) -> None:
        enabled = os.environ.get(
            "OURO3_RETRODICT_TRACE", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not enabled or self._session_runtime_dir is None:
            return
        safe_game_key = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in self.game_key
        )[:96] or "game"
        path = self._session_runtime_dir / (
            f"{safe_game_key}_retrodict_trace.jsonl.gz"
        )
        payload = {
            "game_id": self.game_key,
            "index": int(record.index),
            "level": int(record.level),
            "action": dict(record.action),
            "before": [list(row) for row in record.before],
            "after": [list(row) for row in record.after],
            "level_completed": bool(record.level_completed),
            "game_over": bool(record.game_over),
            "run_complete": bool(record.run_complete),
        }
        with gzip.open(path, "at", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")))
            handle.write("\n")


def _legal_action(action: dict[str, Any], valid_actions: list[str]) -> bool:
    normalized = normalize_action(action)
    allowed = {normalize_action(item).get("action") for item in valid_actions}
    name = normalized.get("action")
    if name not in allowed or name == "RESET":
        return False
    if name == "MOUSE":
        return int(normalized.get("row", -1)) >= 0 and int(
            normalized.get("col", -1)
        ) >= 0
    return True
