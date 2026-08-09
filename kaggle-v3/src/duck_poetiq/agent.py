"""One compact Poetiq-inspired protocol on top of Stock Duck.

The protocol is deliberately host-triggered.  Normal turns keep Duck's tool,
history, and batching behavior; only a generic stall asks the model to audit,
rank a bounded set of hypotheses, acquire one discriminating observation, and
execute one verifiable probe.  The second failed probe uses one alternate
sampling seed and then the agent returns to the primary path.
"""

from __future__ import annotations

from typing import Any

from inference.agent.runtime_state import Frame, HistoryEntry
from inference.agent.tool_agent import ToolAgent, _ChatCompletionResult


POETIQ_SYSTEM_ADDENDUM = """

Poetiq intervention protocol (compact, persistent, and generic): maintain at
most three candidate mechanics or goal explanations, each with evidence and
contradictions.  Before continuing a stalled plan, audit the newest
transition, identify the smallest observation that distinguishes the leading
 candidates, and use the Python tool to inspect only that evidence.  Choose
 one low-risk probe and optionally attach one falsifiable `expect` object to
 its action.  Treat the next frame as a cheap verifier: discard contradicted
 candidates, preserve the best-supported explanation, and update the plan.
Do not invent game-specific rules, coordinates, or predictions, and do not
repair a failed prediction automatically.
""".strip()


POETIQ_INTERVENTION_ADDENDUM = """

Poetiq intervention trigger: {reason}.
Use this single intervention turn to audit the newest transition, retain at most three
candidate mechanics or goals with evidence and contradictions,
identify the one smallest discriminating observation, and execute exactly one
low-risk probe.  Optionally attach one falsifiable `expect` object to that
action.  Use the returned frame delta as the verifier, discard contradicted
candidates, and preserve only the strongest explanation.  Do not batch
speculative actions, invent coordinates, or reset unless the environment
reports game over.
""".strip()


class DuckPoetiqToolAgent(ToolAgent):
    """Stock Duck with bounded Poetiq interventions and no extra model call."""

    prediction_verification_enabled = True

    def __init__(
        self,
        *,
        primary_seed: int | None = 0,
        repeat_threshold: int = 4,
        no_change_threshold: int = 3,
        intervention_cooldown_actions: int = 12,
        max_interventions_per_level: int = 2,
        diversity_seed_offset: int = 17,
        yield_min_actions: int = 64,
        yield_min_elapsed_s: float = 30 * 60,
        yield_window: int = 16,
        yield_max_changes: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(seed=primary_seed, **kwargs)
        self._system_prompt = f"{self._system_prompt}\n\n{POETIQ_SYSTEM_ADDENDUM}"
        self.primary_seed = primary_seed
        self.repeat_threshold = max(2, int(repeat_threshold))
        self.no_change_threshold = max(2, int(no_change_threshold))
        self.intervention_cooldown_actions = max(
            0, int(intervention_cooldown_actions)
        )
        self.max_interventions_per_level = max(1, int(max_interventions_per_level))
        self.diversity_seed_offset = max(1, int(diversity_seed_offset))
        self.yield_min_actions = max(1, int(yield_min_actions))
        self.yield_min_elapsed_s = max(0.0, float(yield_min_elapsed_s))
        self.yield_window = max(1, int(yield_window))
        self.yield_max_changes = max(0, int(yield_max_changes))

        self._current_level: int | None = None
        self._interventions_used = 0
        self._failed_interventions = 0
        self._active_intervention = False
        self._active_attempt = 0
        self._active_seed: int | None = None
        self._active_action_count: int | None = None
        self._last_intervention_action_count = -10**9
        self._last_intervention_reason = ""
        self._active_prediction_supplied = False
        self._active_prediction_mismatch = False
        self._intervention_outcome_pending = False
        self._stall_yielded = False
        self._last_observation_changed = False
        self._gameplay_change_history: list[bool] = []
        self._intervention_events: list[dict[str, Any]] = []
        self._telemetry = {
            "poetiq_intervention_triggers": 0,
            "poetiq_primary_interventions": 0,
            "poetiq_diverse_retries": 0,
            "poetiq_information_requests": 0,
            "poetiq_candidate_sets": 0,
            "poetiq_prediction_supplied": 0,
            "poetiq_prediction_omissions": 0,
            "poetiq_intervention_successes": 0,
            "poetiq_failed_interventions": 0,
            "poetiq_cooldown_suppressions": 0,
            "poetiq_stalled_yields": 0,
            "poetiq_actions_saved_estimate": 0,
            "prediction_matches": 0,
            "prediction_mismatches": 0,
        }

    @property
    def augmented_features_enabled(self) -> bool:
        # Keep the Stock Duck frame/tool surface.  Poetiq only needs the
        # generic transition fields already returned by the shared executor.
        return False

    @property
    def verified_actions_enabled(self) -> bool:
        return True

    @property
    def maximum_action_batch_size(self) -> int | None:
        return 1 if self._active_intervention else None

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

    def _trailing_no_change_count(self, history_entries: list[HistoryEntry]) -> int:
        # Prefer the executor's gameplay-only classification so HUD animation
        # does not hide a genuine stall.  The frame comparison is a fallback
        # for direct/unit callers that have not supplied transition callbacks.
        if self._gameplay_change_history:
            count = 0
            for changed in reversed(self._gameplay_change_history):
                if changed:
                    break
                count += 1
            return count
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

    def _sync_level(self, history_entries: list[HistoryEntry]) -> int | None:
        if not history_entries:
            return self._current_level
        level = int(history_entries[-1].frame.level)
        if self._current_level is None:
            self._current_level = level
        elif level != self._current_level:
            self._current_level = level
            self._interventions_used = 0
            self._failed_interventions = 0
            self._active_intervention = False
            self._active_attempt = 0
            self._active_seed = None
            self._active_action_count = None
            self._gameplay_change_history.clear()
        return self._current_level

    def _same_level(self, history_entries: list[HistoryEntry]) -> bool:
        if len(history_entries) < 2:
            return True
        recent = history_entries[-(max(self.repeat_threshold, self.no_change_threshold) + 2) :]
        level = recent[-1].frame.level
        return all(entry.frame.level == level for entry in recent)

    def _trigger_reason(self, history_entries: list[HistoryEntry], action_num: int) -> str | None:
        self._sync_level(history_entries)
        if self._active_intervention:
            return self._last_intervention_reason
        if self._interventions_used >= self.max_interventions_per_level:
            return None
        if action_num - self._last_intervention_action_count < self.intervention_cooldown_actions:
            if self._same_level(history_entries):
                self._telemetry["poetiq_cooldown_suppressions"] += 1
            return None
        if not self._same_level(history_entries):
            return None
        repeat_count = self._trailing_repeat_count(history_entries)
        if repeat_count >= self.repeat_threshold:
            return f"the same action repeated {repeat_count} consecutive times"
        no_change_count = self._trailing_no_change_count(history_entries)
        if no_change_count >= self.no_change_threshold:
            return f"the last {no_change_count} transitions left gameplay unchanged"
        return None

    def _start_intervention(self, *, reason: str, action_num: int) -> None:
        self._active_intervention = True
        self._interventions_used += 1
        self._active_attempt = self._interventions_used
        primary = self.primary_seed if self.primary_seed is not None else 0
        self._active_seed = primary if self._active_attempt == 1 else primary + self.diversity_seed_offset
        self._active_action_count = int(action_num)
        self._last_intervention_action_count = int(action_num)
        self._last_intervention_reason = reason
        self._active_prediction_supplied = False
        self._active_prediction_mismatch = False
        self._intervention_outcome_pending = False
        self._telemetry["poetiq_intervention_triggers"] += 1
        self._telemetry["poetiq_information_requests"] += 1
        self._telemetry["poetiq_candidate_sets"] += 1
        if self._active_attempt == 1:
            self._telemetry["poetiq_primary_interventions"] += 1
        else:
            self._telemetry["poetiq_diverse_retries"] += 1
        self._intervention_events.append(
            {
                "attempt": self._active_attempt,
                "seed": self._active_seed,
                "reason": reason,
                "action_num": int(action_num),
                "outcome": "pending",
            }
        )

    def _build_user_prompt(self, *args: Any, **kwargs: Any) -> str:
        prompt = super()._build_user_prompt(*args, **kwargs)
        action_num = int(args[0]) if args else int(kwargs.get("action_num", 0) or 0)
        history_entries = kwargs.get("history_entries")
        if not isinstance(history_entries, list):
            history_entries = []
        reason = self._trigger_reason(history_entries, action_num)
        if reason is None:
            return prompt
        if not self._active_intervention:
            self._start_intervention(reason=reason, action_num=action_num)
        return f"{prompt}\n\n{POETIQ_INTERVENTION_ADDENDUM.format(reason=reason)}"

    def _chat_completion(self, *args: Any, **kwargs: Any) -> _ChatCompletionResult:
        original_seed = self._seed
        if self._active_intervention:
            self._seed = self._active_seed
        try:
            return super()._chat_completion(*args, **kwargs)
        finally:
            self._seed = original_seed

    def _update_summarized_knowledge_from_assistant(self, content: str) -> None:
        super()._update_summarized_knowledge_from_assistant(content)
        lowered = str(content or "").lower()
        if "candidate" in lowered or "hypothesis" in lowered:
            self._telemetry["poetiq_candidate_sets"] += 1

    def register_prediction(self, expectation: Any) -> None:
        if not self._active_intervention:
            return
        supplied = expectation not in (None, {}, "")
        self._active_prediction_supplied = self._active_prediction_supplied or supplied
        key = "poetiq_prediction_supplied" if supplied else "poetiq_prediction_omissions"
        self._telemetry[key] += 1

    def register_prediction_match(self) -> None:
        self._telemetry["prediction_matches"] += 1

    def register_prediction_mismatch(self, reason: str = "") -> None:
        del reason
        self._active_prediction_mismatch = True
        self._telemetry["prediction_mismatches"] += 1

    def finish_intervention(self, payload: dict[str, Any]) -> None:
        if not self._active_intervention or not payload.get("executed"):
            return
        gameplay_changed = bool(payload.get("gameplay_changed", payload.get("board_changed")))
        level_completed = bool(payload.get("level_completed"))
        success = (level_completed or gameplay_changed) and not self._active_prediction_mismatch
        if success:
            self._telemetry["poetiq_intervention_successes"] += 1
            self._last_observation_changed = True
        else:
            self._failed_interventions += 1
            self._telemetry["poetiq_failed_interventions"] += 1
            self._last_observation_changed = False
        if self._intervention_events:
            event = self._intervention_events[-1]
            event.update(
                {
                    "outcome": "success" if success else "failure",
                    "gameplay_changed": gameplay_changed,
                    "level_completed": level_completed,
                    "prediction_mismatch": self._active_prediction_mismatch,
                }
            )
        self._active_intervention = False
        self._active_action_count = None
        self._active_seed = None
        self._active_prediction_supplied = False
        self._active_prediction_mismatch = False
        self._intervention_outcome_pending = False

    def observe_transition(
        self,
        *,
        action: str,
        before_grid: Any,
        after_grid: Any,
        payload: dict[str, Any],
    ) -> None:
        """Receive generic transition facts without changing Duck's tool surface.

        The shared executor only computes gameplay-versus-HUD deltas when an
        analyzer opts into this hook.  Poetiq needs those facts for intervention
        verification and the stalled-yield guard, but deliberately does not
        expose Ouroboros's richer perception objects to the model.
        """

        del action, before_grid, after_grid
        changed = bool(
            payload.get("gameplay_changed", payload.get("board_changed", False))
        )
        self._last_observation_changed = changed
        self._gameplay_change_history.append(changed)
        del self._gameplay_change_history[:-128]

    def should_yield_stalled_game(
        self,
        *,
        action_count: int,
        levels_completed: int,
        elapsed_seconds: float,
        gameplay_change_history: list[bool],
    ) -> bool:
        if self._stall_yielded or levels_completed > 0:
            return False
        if self._failed_interventions < self.max_interventions_per_level:
            return False
        if action_count < self.yield_min_actions or elapsed_seconds < self.yield_min_elapsed_s:
            return False
        if sum(bool(value) for value in gameplay_change_history[-self.yield_window :]) > self.yield_max_changes:
            return False
        self._stall_yielded = True
        self._telemetry["poetiq_stalled_yields"] += 1
        return True

    @property
    def poetiq_stalled_yielded(self) -> bool:
        return self._stall_yielded

    @property
    def telemetry(self) -> dict[str, int]:
        value = dict(super().telemetry)
        value.update(self._telemetry)
        return value

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "mode": "duck-poetiq",
            "stock_system_prompt": False,
            "stock_tool_surface": True,
            "stock_history_policy": True,
            "protocol": "audit -> information -> <=3 candidates -> one verified probe -> optional alternate seed",
            "repeat_threshold": self.repeat_threshold,
            "no_change_threshold": self.no_change_threshold,
            "intervention_cooldown_actions": self.intervention_cooldown_actions,
            "max_interventions_per_level": self.max_interventions_per_level,
            "diversity_seed_offset": self.diversity_seed_offset,
            "yield_min_actions": self.yield_min_actions,
            "yield_min_elapsed_s": self.yield_min_elapsed_s,
            "yield_window": self.yield_window,
            "yield_max_changes": self.yield_max_changes,
            "interventions_used_current_level": self._interventions_used,
            "failed_interventions_current_level": self._failed_interventions,
            "stalled_yielded": self._stall_yielded,
            "active_attempt": self._active_attempt,
            "active_seed": self._active_seed,
            "intervention_events": [dict(event) for event in self._intervention_events],
        }
