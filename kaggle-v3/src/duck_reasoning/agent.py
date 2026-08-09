"""Stock Duck reasoning-only ablation.

This mode deliberately delegates history budgeting, tool execution, prompt
construction, and action policy to the audited Stock Duck ``ToolAgent``.  Its
only behavioral change is normalizing vLLM's parsed ``reasoning`` field to the
``reasoning_content`` field consumed by the Qwen3.6 chat template before a
historical message is sent again.  In particular, there is no semantic
compactor, extra model call, summary injection, or fallback planner here.
"""

from __future__ import annotations

import json
import os
from typing import Any

from duck_memory.reasoning import normalize_reasoning_history, reasoning_text
from inference.agent.tool_agent import ToolAgent


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class DuckReasoningToolAgent(ToolAgent):
    """Stock Duck with only Qwen reasoning-history transport corrected."""

    preserve_thinking_history = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._reasoning_seen: set[str] = set()
        self._reasoning_turns = 0
        self._reasoning_chars = 0

    def _ensure_session(self, state_path: Any) -> None:
        previous_runtime_dir = self._session_runtime_dir
        super()._ensure_session(state_path)
        if previous_runtime_dir != self._session_runtime_dir:
            self._reasoning_seen.clear()
            self._reasoning_turns = 0
            self._reasoning_chars = 0

    def _normalize_and_count(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        normalized = normalize_reasoning_history(messages)
        for message in normalized:
            private_reasoning = reasoning_text(message)
            if not private_reasoning:
                continue
            identity = json.dumps(
                message,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            if identity in self._reasoning_seen:
                continue
            self._reasoning_seen.add(identity)
            self._reasoning_turns += 1
            self._reasoning_chars += len(private_reasoning)
        return normalized

    def _trim_messages_for_context(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        preserve_recent: int = 1,
        extra_safety_tokens: int = 0,
    ) -> list[dict[str, Any]]:
        # Normalization happens immediately before every request and before
        # Stock Duck's persistent-history eviction.  This preserves the stock
        # 30-turn/context behavior while fixing only the wire-field mismatch.
        normalized = self._normalize_and_count(messages)
        return super()._trim_messages_for_context(
            normalized,
            tools=tools,
            preserve_recent=preserve_recent,
            extra_safety_tokens=extra_safety_tokens,
        )

    @property
    def telemetry(self) -> dict[str, int]:
        value = dict(super().telemetry)
        retained_reasoning_turns = sum(
            1 for message in self._history_messages if reasoning_text(message)
        )
        value.update(
            {
                "reasoning_template_verified": int(
                    _env_bool("OURO3_REASONING_TEMPLATE_VERIFIED")
                ),
                "reasoning_turns": self._reasoning_turns,
                "reasoning_chars": self._reasoning_chars,
                "reasoning_retained_turns": retained_reasoning_turns,
                "reasoning_evicted_turns": max(
                    0, self._reasoning_turns - retained_reasoning_turns
                ),
                # There is no summary or auxiliary memory path in this mode.
                "reasoning_unaccounted_turns": 0,
                "compaction_count": 0,
                "compaction_retries": 0,
                "compaction_failures": 0,
                "emergency_trims": 0,
            }
        )
        return value

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "mode": "duck-reasoning",
            "reasoning_adapter": "vllm.reasoning->qwen.reasoning_content",
            "reasoning_template_verified": _env_bool(
                "OURO3_REASONING_TEMPLATE_VERIFIED"
            ),
            "history_policy": "stock-duck",
            "semantic_compaction": False,
            "auxiliary_model_calls": 0,
        }
