"""Anthropic Messages API 驱动 — 实现 Anthropic 特有的请求/响应格式转换。

Anthropic Messages API driver. Key differences from OpenAI:
- System messages are a top-level ``system`` parameter, not part of ``messages``.
- Content uses typed blocks: ``[{"type": "text", "text": "..."}]``.
- Streaming uses ``event: content_block_delta`` with ``delta.text``.
- Response uses ``content[0].text`` instead of ``choices[0].message.content``.
- ``max_tokens`` is required, not optional.
"""

from __future__ import annotations

import json
from typing import Any

from ai_lib_python.drivers import (
    DriverRequest,
    DriverResponse,
    ProviderDriver,
    UsageInfo,
)
from ai_lib_python.protocol.v2.capabilities import Capability
from ai_lib_python.protocol.v2.manifest import ApiStyle
from ai_lib_python.types.events import StreamingEvent
from ai_lib_python.types.message import Message

_DEFAULT_MAX_TOKENS = 4096

# Anthropic stop_reason → AI-Protocol normalized finish_reason
_STOP_REASON_MAP: dict[str, str] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


class AnthropicDriver(ProviderDriver):
    """Anthropic Messages API driver."""

    def __init__(
        self,
        provider_id: str,
        capabilities: list[Capability] | None = None,
    ) -> None:
        self._provider_id = provider_id
        self._capabilities = capabilities or []

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def api_style(self) -> ApiStyle:
        return ApiStyle.ANTHROPIC_MESSAGES

    def build_request(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> DriverRequest:
        system_text, msgs = self._split_system(messages)

        body: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "max_tokens": max_tokens or _DEFAULT_MAX_TOKENS,
            "stream": stream,
        }
        if system_text:
            body["system"] = system_text
        if temperature is not None:
            body["temperature"] = temperature
        if extra:
            body.update(extra)

        headers = {"anthropic-version": "2023-06-01"}
        return DriverRequest(body=body, stream=stream, headers=headers)

    def parse_response(self, body: dict[str, Any]) -> DriverResponse:
        # content: [{type: "text", text: "..."}]
        content_blocks = body.get("content", [])
        text = next(
            (b["text"] for b in content_blocks if b.get("type") == "text"),
            None,
        )
        # Normalize stop_reason
        raw_reason = body.get("stop_reason", "")
        finish_reason = _STOP_REASON_MAP.get(raw_reason, raw_reason) or None

        usage_raw = body.get("usage")
        usage = None
        if usage_raw:
            inp = usage_raw.get("input_tokens", 0)
            out = usage_raw.get("output_tokens", 0)
            usage = UsageInfo(prompt_tokens=inp, completion_tokens=out, total_tokens=inp + out)

        tool_calls = [b for b in content_blocks if b.get("type") == "tool_use"]
        thinking_parts = [
            b.get("thinking", "")
            for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "thinking" and b.get("thinking")
        ]
        thinking = "".join(thinking_parts) if thinking_parts else None

        return DriverResponse(
            content=text,
            thinking=thinking,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=tool_calls,
            raw=body,
        )

    def parse_stream_event(self, data: str) -> list[StreamingEvent]:
        stripped = data.strip()
        if not stripped:
            return []

        chunk = json.loads(stripped)
        event_type = chunk.get("type", "")

        if event_type == "content_block_delta":
            delta = chunk.get("delta", {})
            if text := delta.get("text"):
                seq = chunk.get("index")
                return [StreamingEvent.content_delta(text, sequence_id=seq)]
            if thinking := delta.get("thinking"):
                return [StreamingEvent.thinking_delta(thinking)]
            return []

        if event_type == "message_delta":
            reason = chunk.get("delta", {}).get("stop_reason")
            if reason:
                return [StreamingEvent.stream_end(_STOP_REASON_MAP.get(reason, reason))]
            return []

        if event_type == "message_stop":
            return [StreamingEvent.stream_end("stop")]

        if event_type == "error":
            return [StreamingEvent.stream_error(chunk.get("error"))]

        return []

    def supported_capabilities(self) -> list[Capability]:
        return list(self._capabilities)

    def is_stream_done(self, _data: str) -> bool:
        # Anthropic signals done via event type, not a sentinel string.
        return False

    # -- internal helpers ------------------------------------------------

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Extract system text and convert remaining messages to Anthropic format."""
        system_parts: list[str] = []
        msgs: list[dict[str, Any]] = []

        for m in messages:
            role = m.role if isinstance(m.role, str) else m.role.value
            if role == "system":
                if isinstance(m.content, str):
                    system_parts.append(m.content)
                continue

            if role == "tool":
                # Anthropic: tool results as user message with tool_result block
                tool_id = getattr(m, "tool_call_id", None)
                if tool_id and isinstance(m.content, str):
                    msgs.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": m.content,
                                }
                            ],
                        }
                    )
                continue

            if isinstance(m.content, str):
                content: Any = [{"type": "text", "text": m.content}]
            else:
                content = [b.model_dump(by_alias=True) for b in m.content]

            msgs.append({"role": role, "content": content})

        system_text = "\n\n".join(system_parts) if system_parts else None
        return system_text, msgs
