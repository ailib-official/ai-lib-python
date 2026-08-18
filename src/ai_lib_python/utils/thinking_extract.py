"""OpenAI-compatible thinking / reasoning field extraction (ALP-RSN-001).

中文：从 delta/message 结构化字段提取思考文本；单一别名表，供 Driver 与
event_map 共用（GOV-007 / X-RUNTIME-MIRROR of ALR-RSN-001）。
"""

from __future__ import annotations

from typing import Any

# Wire keys observed across OpenAI-compatible reasoners / proxies.
# Order is preference when multiple keys appear on the same object.
OPENAI_COMPAT_THINKING_KEYS: tuple[str, ...] = (
    "reasoning_content",
    "reasoning",
    "thinking",
    "thought",
    "reasoning_text",
    "analysis",
)


def first_nonempty_string_field(obj: Any, keys: tuple[str, ...] | list[str]) -> str | None:
    """First non-empty string among ``keys`` on a mapping."""
    if not isinstance(obj, dict):
        return None
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def thinking_from_openai_compat_delta(frame: dict[str, Any]) -> str | None:
    """Thinking text from ``choices[0].delta.*`` (streaming)."""
    choices = frame.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    delta = first.get("delta")
    return first_nonempty_string_field(delta, OPENAI_COMPAT_THINKING_KEYS)


def thinking_from_openai_compat_message(frame: dict[str, Any]) -> str | None:
    """Thinking text from ``choices[0].message.*`` (non-streaming)."""
    choices = frame.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    return first_nonempty_string_field(message, OPENAI_COMPAT_THINKING_KEYS)
