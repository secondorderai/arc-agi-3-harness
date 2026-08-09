"""Qwen reasoning-history compatibility helpers.

vLLM exposes parsed Qwen reasoning as ``message.reasoning``.  The Qwen3.6
chat template bundled with the model snapshot renders historical thinking
from ``message.reasoning_content``.  This module owns that output-to-input
adapter so the stock Duck path remains untouched.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any


REASONING_SENTINEL = "OURO3_REASONING_ROUNDTRIP_SENTINEL_7F3A"


def reasoning_text(message: dict[str, Any]) -> str:
    """Return the canonical private reasoning text from either API spelling."""

    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_reasoning_message(message: dict[str, Any]) -> dict[str, Any]:
    """Copy one message into the Qwen3.6 historical-input wire shape."""

    normalized = copy.deepcopy(message)
    if str(normalized.get("role", "")).strip() != "assistant":
        return normalized
    # Stock Duck emits reasoning-only assistant turns with ``content=None``.
    # Some OpenAI-compatible servers reject that historical wire shape even
    # though Qwen treats it as empty visible content.
    if normalized.get("content") is None:
        normalized["content"] = ""
    value = reasoning_text(normalized)
    normalized.pop("reasoning", None)
    if value:
        normalized["reasoning_content"] = value
    elif "reasoning_content" in normalized:
        normalized["reasoning_content"] = ""
    return normalized


def normalize_reasoning_history(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [normalize_reasoning_message(message) for message in messages]


def sentinel_messages() -> list[dict[str, Any]]:
    """Two-turn history used to prove the exact tokenizer retains thinking."""

    return [
        {"role": "user", "content": "Remember the private marker."},
        {
            "role": "assistant",
            "reasoning_content": REASONING_SENTINEL,
            "content": "Marker stored.",
        },
        {"role": "user", "content": "Continue."},
    ]


def assert_reasoning_sentinel_rendered(
    rendered_prompt: str,
    *,
    sentinel: str = REASONING_SENTINEL,
) -> None:
    """Fail closed unless historical private reasoning is inside think tags."""

    rendered = str(rendered_prompt)
    if rendered.count(sentinel) != 1:
        raise RuntimeError(
            "Qwen reasoning retention smoke failed: the historical sentinel "
            "was not rendered exactly once"
        )
    before, after = rendered.split(sentinel, 1)
    think_start = before.rfind("<think>")
    think_end = after.find("</think>")
    if think_start < 0 or think_end < 0:
        raise RuntimeError(
            "Qwen reasoning retention smoke failed: the sentinel was not "
            "inside a historical <think> block"
        )


def render_and_verify_reasoning(
    apply_chat_template: Callable[..., Any],
) -> str:
    """Render the sentinel with the two supported Transformers call shapes."""

    attempts = (
        {
            "preserve_thinking": True,
            "enable_thinking": True,
        },
        {
            "chat_template_kwargs": {
                "preserve_thinking": True,
                "enable_thinking": True,
            }
        },
    )
    errors: list[str] = []
    for template_kwargs in attempts:
        try:
            rendered = apply_chat_template(
                sentinel_messages(),
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            assert_reasoning_sentinel_rendered(str(rendered))
            return str(rendered)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    raise RuntimeError(
        "Qwen reasoning retention smoke failed for every supported tokenizer "
        f"call shape: {' | '.join(errors)}"
    )
