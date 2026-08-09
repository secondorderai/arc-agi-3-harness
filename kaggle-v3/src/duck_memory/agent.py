"""Stock Duck with Qwen reasoning round-tripping and semantic compaction."""

from __future__ import annotations

import gzip
import json
import os
import time
import threading
from pathlib import Path
from typing import Any

import requests

from duck_memory.memory import (
    CompactionMemory,
    CompactionPartition,
    CompactionSettings,
    compaction_prompt,
    covered_action_range,
    covered_message_count,
    memory_message,
    partition_for_compaction,
    smaller_retry_prefix,
    validate_summary,
)
from duck_memory.reasoning import normalize_reasoning_history, reasoning_text
from inference.agent.tool_agent import (
    ToolAgent,
    _ChatCompletionResult,
    _get_env_float,
    _get_env_int,
)
from inference.utils.openai_compat import build_chat_payload


_COMPACTION_SEMAPHORE = threading.BoundedSemaphore(4)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class DuckMemoryToolAgent(ToolAgent):
    """Stock Duck behavior with lossless recent history and compacted old memory."""

    preserve_thinking_history = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.settings = CompactionSettings(
            trigger_tokens=_get_env_int("OURO3_COMPACTION_TRIGGER_TOKENS", 24_576),
            target_tokens=_get_env_int("OURO3_COMPACTION_TARGET_TOKENS", 16_384),
            recent_assistant_turns=_get_env_int(
                "OURO3_COMPACTION_RECENT_ASSISTANT_TURNS", 8
            ),
            max_output_tokens=_get_env_int(
                "OURO3_COMPACTION_MAX_OUTPUT_TOKENS", 2_048
            ),
            timeout_seconds=_get_env_float(
                "OURO3_COMPACTION_TIMEOUT_SECONDS", 300.0
            ),
            temperature=_get_env_float("OURO3_COMPACTION_TEMPERATURE", 0.2),
            top_p=_get_env_float("OURO3_COMPACTION_TOP_P", 0.9),
            top_k=_get_env_int("OURO3_COMPACTION_TOP_K", 20),
            max_concurrency=_get_env_int("OURO3_COMPACTION_MAX_CONCURRENCY", 4),
        ).validate()
        if self.settings.max_concurrency != 4:
            raise ValueError(
                "duck-memory v1 requires exactly four concurrent compactions"
            )
        self._compaction_generation = 0
        self._compaction_count = 0
        self._compaction_retries = 0
        self._compaction_failures = 0
        self._emergency_trims = 0
        self._reasoning_turns = 0
        self._reasoning_chars = 0
        self._reasoning_compacted_turns = 0
        self._compaction_input_tokens = 0
        self._compaction_output_tokens = 0
        self._compaction_pre_tokens = 0
        self._compaction_post_tokens = 0
        self._compaction_latency_ms = 0
        self._trace_active_history: list[str] = []
        self._trace_sequence = 0
        self._trace_path: Path | None = None
        self._last_summary: dict[str, Any] | None = None
        self.memory = CompactionMemory()

    @property
    def telemetry(self) -> dict[str, int]:
        value = dict(super().telemetry)
        retained_reasoning_turns = sum(
            1 for message in self._history_messages if reasoning_text(message)
        )
        accounted_reasoning_turns = (
            self._reasoning_compacted_turns + retained_reasoning_turns
        )
        value.update(
            {
                "reasoning_template_verified": int(
                    _env_bool("OURO3_REASONING_TEMPLATE_VERIFIED")
                ),
                "reasoning_turns": self._reasoning_turns,
                "reasoning_chars": self._reasoning_chars,
                "reasoning_retained_turns": retained_reasoning_turns,
                "reasoning_compacted_turns": self._reasoning_compacted_turns,
                "reasoning_accounted_turns": accounted_reasoning_turns,
                "reasoning_unaccounted_turns": max(
                    0, self._reasoning_turns - accounted_reasoning_turns
                ),
                "compaction_count": self._compaction_count,
                "compaction_retries": self._compaction_retries,
                "compaction_failures": self._compaction_failures,
                "emergency_trims": self._emergency_trims,
                "compaction_input_tokens": self._compaction_input_tokens,
                "compaction_output_tokens": self._compaction_output_tokens,
                "compaction_pre_tokens": self._compaction_pre_tokens,
                "compaction_post_tokens": self._compaction_post_tokens,
                "compaction_latency_ms": self._compaction_latency_ms,
                "compaction_compression_ratio_bps": (
                    int(
                        10_000
                        * self._compaction_post_tokens
                        / self._compaction_pre_tokens
                    )
                    if self._compaction_pre_tokens
                    else 0
                ),
            }
        )
        return value

    @property
    def diagnostics(self) -> dict[str, Any]:
        base = {
            "reasoning_adapter": "vllm.reasoning->qwen.reasoning_content",
            "reasoning_template_verified": _env_bool(
                "OURO3_REASONING_TEMPLATE_VERIFIED"
            ),
            "settings": {
                "trigger_tokens": self.settings.trigger_tokens,
                "target_tokens": self.settings.target_tokens,
                "recent_assistant_turns": self.settings.recent_assistant_turns,
                "max_output_tokens": self.settings.max_output_tokens,
                "timeout_seconds": self.settings.timeout_seconds,
                "temperature": self.settings.temperature,
                "top_p": self.settings.top_p,
                "top_k": self.settings.top_k,
                "max_concurrency": self.settings.max_concurrency,
            },
        }
        if not self._trace_enabled():
            return {**base, "aggregate_only": True}
        return {
            **base,
            "aggregate_only": False,
            "compaction_memory": self.memory.diagnostics(),
        }

    def _ensure_session(self, state_path: Path) -> None:
        super()._ensure_session(state_path)
        if self._trace_path is None:
            self._trace_path = state_path.with_name(
                f"{state_path.stem}_memory_trace.jsonl.gz"
            )
            self.memory.validation_trace_path = str(self._trace_path)

    def _trace_enabled(self) -> bool:
        return not _env_bool("TAAF_MINIMAL_DIAGNOSTICS")

    def _append_trace(self, event: dict[str, Any]) -> None:
        if not self._trace_enabled() or self._trace_path is None:
            return
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(self._trace_path, "at", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

    def _trace_new_messages(self, messages: list[dict[str, Any]]) -> None:
        encoded_messages = [
            json.dumps(
                message, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            for message in messages
        ]
        common_prefix = 0
        for previous, current in zip(
            self._trace_active_history,
            encoded_messages,
        ):
            if previous != current:
                break
            common_prefix += 1
        for message in messages[common_prefix:]:
            private_reasoning = reasoning_text(message)
            if private_reasoning:
                self._reasoning_turns += 1
                self._reasoning_chars += len(private_reasoning)
            self._trace_sequence += 1
            self._append_trace(
                {
                    "event": "message",
                    "sequence": self._trace_sequence,
                    "message": message,
                }
            )
        self._trace_active_history = encoded_messages

    def _set_trace_active_history(
        self,
        messages: list[dict[str, Any]],
    ) -> None:
        self._trace_active_history = [
            json.dumps(
                message,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for message in messages
        ]
        self.memory.update_active(
            messages,
            recent_assistant_turns=self.settings.recent_assistant_turns,
        )

    def _compaction_completion(
        self,
        messages: list[dict[str, Any]],
    ) -> _ChatCompletionResult:
        payload = build_chat_payload(
            provider=self._model.provider,
            model=self._model.model_id,
            messages=messages,
            max_tokens=self.settings.max_output_tokens,
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            top_k=self.settings.top_k,
            thinking=False,
            preserve_thinking=False,
            seed=self._seed,
        )
        self._request_count += 1
        started = time.monotonic()
        try:
            with _COMPACTION_SEMAPHORE:
                response = requests.post(
                    f"{self._model.base_url.rstrip('/')}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.settings.timeout_seconds,
                )
            response.raise_for_status()
            body = response.json()
            choices = body.get("choices") or []
            if not choices:
                raise requests.RequestException(
                    "compaction server returned no choices"
                )
            choice = choices[0]
            result = _ChatCompletionResult(
                message=choice.get("message") or {},
                finish_reason=str(choice.get("finish_reason") or ""),
                usage=body.get("usage"),
            )
            self._accumulate_usage_tokens(result.usage)
            usage = result.usage or {}
            self._compaction_input_tokens += int(
                usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            )
            self._compaction_output_tokens += int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )
            return result
        finally:
            self._compaction_latency_ms += int(
                (time.monotonic() - started) * 1_000
            )

    def _summarize_prefix(
        self,
        prefix: tuple[dict[str, Any], ...],
    ) -> dict[str, Any]:
        result = self._compaction_completion(compaction_prompt(prefix))
        content = result.message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("compaction response did not contain summary content")
        return validate_summary(content)

    def _compact_partition(
        self,
        partition: CompactionPartition,
        *,
        pre_tokens: int,
    ) -> list[dict[str, Any]] | None:
        prefix = partition.prefix
        remainder: tuple[dict[str, Any], ...] = ()
        try:
            summary = self._summarize_prefix(prefix)
        except Exception as first_error:
            self._compaction_retries += 1
            retry_prefix, remainder = smaller_retry_prefix(prefix)
            self._append_trace(
                {
                    "event": "compaction_retry",
                    "error": f"{type(first_error).__name__}: {first_error}",
                    "original_messages": len(prefix),
                    "retry_messages": len(retry_prefix),
                }
            )
            try:
                summary = self._summarize_prefix(retry_prefix)
                prefix = retry_prefix
            except Exception as second_error:
                self._compaction_failures += 1
                if isinstance(first_error, requests.RequestException):
                    self._request_failures += 1
                if isinstance(second_error, requests.RequestException):
                    self._request_failures += 1
                if isinstance(first_error, requests.Timeout):
                    self._request_timeouts += 1
                if isinstance(second_error, requests.Timeout):
                    self._request_timeouts += 1
                self._append_trace(
                    {
                        "event": "compaction_failed",
                        "first_error": (
                            f"{type(first_error).__name__}: {first_error}"
                        ),
                        "second_error": (
                            f"{type(second_error).__name__}: {second_error}"
                        ),
                    }
                )
                return None

        self._compaction_generation += 1
        self._compaction_count += 1
        compacted_reasoning = sum(
            1 for message in prefix if reasoning_text(message)
        )
        self._reasoning_compacted_turns += compacted_reasoning
        action_range = covered_action_range(prefix)
        represented_messages = covered_message_count(prefix)
        compacted_memory = memory_message(
            summary,
            generation=self._compaction_generation,
            covered_messages=represented_messages,
            action_range=action_range,
        )
        self._last_summary = {
            **summary,
            "_meta": {
                "generation": self._compaction_generation,
                "covered_messages": represented_messages,
                "covered_action_range": action_range,
            },
        }
        self.memory.record_compaction(
            summary,
            generation=self._compaction_generation,
            covered_messages=represented_messages,
            action_range=action_range,
        )
        output = [
            compacted_memory,
            *normalize_reasoning_history(list(remainder)),
            *normalize_reasoning_history(list(partition.suffix)),
        ]
        self._append_trace(
            {
                "event": "compaction_succeeded",
                "generation": self._compaction_generation,
                "pre_tokens": pre_tokens,
                "covered_messages": represented_messages,
                "compacted_reasoning_turns": compacted_reasoning,
                "retained_assistant_turns": partition.retained_assistant_turns,
                "summary": self._last_summary,
            }
        )
        return output

    def _emergency_trim_to_target(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        target_tokens: int,
        preserve_recent: int,
    ) -> list[dict[str, Any]]:
        system_message = messages[0]
        history = list(messages[1:])
        while (
            history
            and self._estimate_request_input_tokens(
                [system_message, *history], tools=tools
            )
            > target_tokens
        ):
            if not self._drop_oldest_history_block(
                history, preserve_recent=max(1, preserve_recent)
            ):
                break
            self._context_evictions += 1
            self._emergency_trims += 1
        return [system_message, *self._drop_until_first_user_message(history)]

    def _partition_for_target(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        target_tokens: int,
    ) -> CompactionPartition:
        history = messages[1:]
        fallback = CompactionPartition((), tuple(history), 0)
        for recent_turns in range(
            self.settings.recent_assistant_turns,
            1,
            -1,
        ):
            candidate = partition_for_compaction(
                history,
                recent_assistant_turns=recent_turns,
            )
            if not candidate.prefix:
                continue
            fallback = candidate
            summary_placeholder = {
                "role": "user",
                "content": (
                    "[COMPACTED GAME MEMORY]\n"
                    + ("x" * self.settings.max_output_tokens * 4)
                ),
            }
            projected = self._estimate_request_input_tokens(
                [messages[0], summary_placeholder, *candidate.suffix],
                tools=tools,
            )
            if projected <= target_tokens:
                return candidate
        return fallback

    def _trim_messages_for_context(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        preserve_recent: int = 1,
        extra_safety_tokens: int = 0,
    ) -> list[dict[str, Any]]:
        if not messages:
            return []
        normalized = normalize_reasoning_history(messages)
        self._trace_new_messages(normalized[1:])
        target = max(
            1_024,
            self.settings.target_tokens - max(0, extra_safety_tokens),
        )
        trigger = min(
            self.settings.trigger_tokens,
            self._context_budget_tokens - max(0, extra_safety_tokens),
        )
        pre_tokens = self._estimate_request_input_tokens(
            normalized, tools=tools
        )
        if pre_tokens < trigger:
            self._set_trace_active_history(normalized[1:])
            return normalized

        self._compaction_pre_tokens += pre_tokens
        partition = self._partition_for_target(
            normalized,
            tools=tools,
            target_tokens=target,
        )
        if not partition.prefix:
            output = self._emergency_trim_to_target(
                normalized,
                tools=tools,
                target_tokens=target,
                preserve_recent=preserve_recent,
            )
            self._compaction_post_tokens += self._estimate_request_input_tokens(
                output, tools=tools
            )
            self._set_trace_active_history(output[1:])
            return output

        compacted_history = self._compact_partition(
            partition,
            pre_tokens=pre_tokens,
        )
        if compacted_history is None:
            output = self._emergency_trim_to_target(
                normalized,
                tools=tools,
                target_tokens=target,
                preserve_recent=preserve_recent,
            )
        else:
            output = [normalized[0], *compacted_history]
            if (
                self._estimate_request_input_tokens(output, tools=tools)
                > trigger
            ):
                output = self._emergency_trim_to_target(
                    output,
                    tools=tools,
                    target_tokens=target,
                    preserve_recent=preserve_recent,
                )
        post_tokens = self._estimate_request_input_tokens(output, tools=tools)
        self._compaction_post_tokens += post_tokens
        self._set_trace_active_history(output[1:])
        return output

    def _force_reduce_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        preserve_recent: int = 1,
    ) -> list[dict[str, Any]]:
        return self._emergency_trim_to_target(
            normalize_reasoning_history(messages),
            tools=None,
            target_tokens=max(1_024, self.settings.target_tokens - 512),
            preserve_recent=preserve_recent,
        )

    def _persistent_history_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        normalized = normalize_reasoning_history(messages)
        trimmed = self._trim_messages_for_context(normalized, tools=tools)
        history = self._drop_until_first_user_message(trimmed[1:])
        self._trace_new_messages(history)
        self._set_trace_active_history(history)
        return history
