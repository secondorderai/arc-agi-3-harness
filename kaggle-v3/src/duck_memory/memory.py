"""Structured semantic compaction for the Duck retained-reasoning lane."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

from duck_memory.reasoning import normalize_reasoning_history, reasoning_text


MEMORY_MARKER = "[COMPACTED GAME MEMORY]"
SUMMARY_FIELDS = (
    "level_state",
    "mechanics",
    "action_effects",
    "objects_and_coordinates",
    "goal_hypotheses",
    "contradictions",
    "successful_experiments",
    "failed_experiments",
    "current_plan",
    "open_questions",
    "cross_level_knowledge",
)
LIST_SUMMARY_FIELDS = SUMMARY_FIELDS[1:]
_ACTION_NUMBER_RE = re.compile(r"\b(?:action|step)\s*[#:]?\s*(\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class CompactionSettings:
    trigger_tokens: int = 24_576
    target_tokens: int = 16_384
    recent_assistant_turns: int = 8
    max_output_tokens: int = 2_048
    timeout_seconds: float = 300.0
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 20
    max_concurrency: int = 4

    def validate(self) -> "CompactionSettings":
        if self.trigger_tokens <= self.target_tokens:
            raise ValueError("compaction trigger must exceed its target")
        if self.recent_assistant_turns < 2:
            raise ValueError("compaction must retain at least two assistant turns")
        if self.max_output_tokens < 256:
            raise ValueError("compaction output budget is too small")
        if self.timeout_seconds <= 0 or self.max_concurrency < 1:
            raise ValueError("compaction timeout and concurrency must be positive")
        return self


@dataclass(frozen=True)
class CompactionPartition:
    prefix: tuple[dict[str, Any], ...]
    suffix: tuple[dict[str, Any], ...]
    retained_assistant_turns: int


@dataclass
class CompactionMemory:
    """Auditable state replacing Stock Duck's destructive turn cap."""

    summary: dict[str, Any] | None = None
    covered_action_range: list[int] | None = None
    covered_messages: int = 0
    generation: int = 0
    uncompacted_raw_messages: tuple[dict[str, Any], ...] = field(
        default_factory=tuple
    )
    latest_assistant_turns: tuple[dict[str, Any], ...] = field(
        default_factory=tuple
    )
    validation_trace_path: str = ""

    def update_active(
        self,
        history: list[dict[str, Any]],
        *,
        recent_assistant_turns: int,
    ) -> None:
        raw = [
            copy.deepcopy(message)
            for message in normalize_reasoning_history(history)
            if not is_memory_message(message)
        ]
        assistants = [
            message
            for message in raw
            if str(message.get("role", "")).strip() == "assistant"
        ]
        self.uncompacted_raw_messages = tuple(raw)
        self.latest_assistant_turns = tuple(
            copy.deepcopy(message)
            for message in assistants[-max(1, recent_assistant_turns) :]
        )

    def record_compaction(
        self,
        summary: dict[str, Any],
        *,
        generation: int,
        covered_messages: int,
        action_range: list[int] | None,
    ) -> None:
        self.summary = copy.deepcopy(summary)
        self.generation = int(generation)
        self.covered_messages = int(covered_messages)
        self.covered_action_range = (
            list(action_range) if action_range is not None else None
        )

    def diagnostics(self) -> dict[str, Any]:
        raw_reasoning_turns = sum(
            1
            for message in self.uncompacted_raw_messages
            if reasoning_text(message)
        )
        return {
            "generation": self.generation,
            "summary": copy.deepcopy(self.summary),
            "covered_action_range": (
                list(self.covered_action_range)
                if self.covered_action_range is not None
                else None
            ),
            "covered_messages": self.covered_messages,
            "uncompacted_raw_messages": len(self.uncompacted_raw_messages),
            "uncompacted_reasoning_turns": raw_reasoning_turns,
            "latest_assistant_turns": len(self.latest_assistant_turns),
            "validation_trace_path": self.validation_trace_path,
        }


def is_memory_message(message: dict[str, Any]) -> bool:
    return (
        str(message.get("role", "")).strip() == "user"
        and str(message.get("content", "")).startswith(MEMORY_MARKER)
    )


def _history_blocks(
    history: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group at user boundaries so tool-call/result pairs are never split."""

    blocks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in history:
        role = str(message.get("role", "")).strip()
        if role == "user" and current:
            blocks.append(current)
            current = []
        current.append(message)
    if current:
        blocks.append(current)
    return blocks


def partition_for_compaction(
    history: list[dict[str, Any]],
    *,
    recent_assistant_turns: int,
) -> CompactionPartition:
    """Select an old complete prefix and a recent verbatim suffix."""

    normalized = normalize_reasoning_history(history)
    blocks = _history_blocks(normalized)
    if len(blocks) < 2:
        return CompactionPartition((), tuple(normalized), 0)

    retained = 0
    suffix_start = len(blocks) - 1
    for index in range(len(blocks) - 1, -1, -1):
        candidate_count = sum(
            1
            for message in blocks[index]
            if str(message.get("role", "")).strip() == "assistant"
        )
        if retained and retained + candidate_count > recent_assistant_turns:
            break
        suffix_start = index
        retained += candidate_count
        if retained >= recent_assistant_turns:
            break

    if suffix_start <= 0:
        return CompactionPartition((), tuple(normalized), retained)
    prefix = tuple(
        copy.deepcopy(message)
        for block in blocks[:suffix_start]
        for message in block
    )
    suffix = tuple(
        copy.deepcopy(message)
        for block in blocks[suffix_start:]
        for message in block
    )
    return CompactionPartition(prefix, suffix, retained)


def smaller_retry_prefix(
    prefix: tuple[dict[str, Any], ...],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Split the oldest prefix at a safe user-message boundary."""

    blocks = _history_blocks(list(prefix))
    if len(blocks) < 2:
        return prefix, ()
    split = max(1, len(blocks) // 2)
    compact = tuple(message for block in blocks[:split] for message in block)
    remainder = tuple(message for block in blocks[split:] for message in block)
    return compact, remainder


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif item.get("type") in {"image", "image_url"}:
                parts.append("[historical image omitted from compaction input]")
        return "\n".join(parts)
    return ""


def serialize_compaction_source(messages: tuple[dict[str, Any], ...]) -> str:
    """Serialize history without carrying old image bytes into the summary call."""

    rows: list[dict[str, Any]] = []
    for message in messages:
        row: dict[str, Any] = {
            "role": str(message.get("role", "")),
            "content": _message_text(message),
        }
        private_reasoning = reasoning_text(message)
        if private_reasoning:
            row["private_reasoning"] = private_reasoning
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            row["tool_calls"] = tool_calls
        if message.get("tool_call_id"):
            row["tool_call_id"] = str(message["tool_call_id"])
        rows.append(row)
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


def compaction_prompt(messages: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    schema = {
        "level_state": "concise string",
        **{name: ["concise evidence-preserving strings"] for name in LIST_SUMMARY_FIELDS},
    }
    return [
        {
            "role": "system",
            "content": (
                "You compact a game-playing agent's private reasoning and action "
                "history. Preserve verified facts, uncertainty, causal evidence, "
                "coordinates, failed attempts, and the live plan. Do not invent "
                "game rules. Return one JSON object only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Summarize the following completed history into this exact JSON "
                f"shape:\n{json.dumps(schema, separators=(',', ':'))}\n"
                "For goal hypotheses, include confidence and supporting/opposing "
                "evidence in each string. Keep concrete coordinates when they "
                "matter. Merge any earlier compacted memory found in the input.\n"
                f"HISTORY:\n{serialize_compaction_source(messages)}"
            ),
        },
    ]


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = str(text).strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    if start < 0:
        raise ValueError("compaction response did not contain a JSON object")
    value, _end = json.JSONDecoder().raw_decode(stripped[start:])
    if not isinstance(value, dict):
        raise ValueError("compaction response must be a JSON object")
    return value


def validate_summary(text: str) -> dict[str, Any]:
    raw = _extract_json_object(text)
    missing = [field for field in SUMMARY_FIELDS if field not in raw]
    if missing:
        raise ValueError(f"compaction summary is missing fields: {missing}")
    level_state = raw["level_state"]
    if not isinstance(level_state, str):
        raise ValueError("compaction level_state must be a string")
    normalized: dict[str, Any] = {"level_state": level_state.strip()}
    for field in LIST_SUMMARY_FIELDS:
        value = raw[field]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"compaction {field} must be a list of strings")
        normalized[field] = [item.strip() for item in value if item.strip()]
    return normalized


def covered_action_range(messages: tuple[dict[str, Any], ...]) -> list[int] | None:
    values: list[int] = []
    for message in messages:
        if is_memory_message(message):
            try:
                memory_payload = _extract_json_object(_message_text(message))
                previous_range = dict(memory_payload.get("_meta") or {}).get(
                    "covered_action_range"
                )
                if (
                    isinstance(previous_range, list)
                    and len(previous_range) == 2
                ):
                    values.extend(int(value) for value in previous_range)
            except (TypeError, ValueError):
                pass
        for match in _ACTION_NUMBER_RE.finditer(_message_text(message)):
            values.append(int(match.group(1)))
    return [min(values), max(values)] if values else None


def covered_message_count(messages: tuple[dict[str, Any], ...]) -> int:
    """Count raw history represented by a possibly nested compacted prefix."""

    count = 0
    for message in messages:
        if not is_memory_message(message):
            count += 1
            continue
        try:
            memory_payload = _extract_json_object(_message_text(message))
            previous_count = dict(memory_payload.get("_meta") or {}).get(
                "covered_messages"
            )
            count += max(1, int(previous_count))
        except (TypeError, ValueError):
            count += 1
    return count


def memory_message(
    summary: dict[str, Any],
    *,
    generation: int,
    covered_messages: int,
    action_range: list[int] | None,
) -> dict[str, Any]:
    payload = copy.deepcopy(summary)
    payload["_meta"] = {
        "generation": int(generation),
        "covered_messages": int(covered_messages),
        "covered_action_range": action_range,
    }
    return {
        "role": "user",
        "content": (
            f"{MEMORY_MARKER}\n"
            "This is a model-generated semantic replacement for older raw "
            "reasoning, observations, actions, and tool results. Treat claims "
            "according to their recorded evidence and uncertainty.\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        ),
    }
