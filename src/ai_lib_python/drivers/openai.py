"""OpenAI 兼容驱动 — 适用于 OpenAI、DeepSeek、Moonshot、Zhipu 等同风格厂商。

OpenAI-compatible driver. Works for any provider that follows the OpenAI
chat completions API format.
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


class OpenAiDriver(ProviderDriver):
    """OpenAI-compatible chat completions driver."""

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
        return ApiStyle.OPENAI_COMPATIBLE

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
        oai_messages = [self._format_message(m) for m in messages]

        body: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "stream": stream,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if extra:
            body.update(extra)

        return DriverRequest(body=body, stream=stream)

    def parse_response(self, body: dict[str, Any]) -> DriverResponse:
        choices = body.get("choices", [])
        first = choices[0] if choices else {}
        msg = first.get("message", {})
        usage_raw = body.get("usage")

        return DriverResponse(
            content=msg.get("content"),
            finish_reason=first.get("finish_reason"),
            usage=UsageInfo(
                prompt_tokens=usage_raw.get("prompt_tokens", 0),
                completion_tokens=usage_raw.get("completion_tokens", 0),
                total_tokens=usage_raw.get("total_tokens", 0),
            )
            if usage_raw
            else None,
            tool_calls=msg.get("tool_calls", []),
            raw=body,
        )

    def parse_stream_event(self, data: str) -> StreamingEvent | None:
        stripped = data.strip()
        if not stripped or self.is_stream_done(data):
            return None

        chunk = json.loads(stripped)
        choices = chunk.get("choices", [])
        if not choices:
            return None

        delta = choices[0].get("delta", {})

        if content := delta.get("content"):
            return StreamingEvent.content_delta(content)

        if reason := choices[0].get("finish_reason"):
            return StreamingEvent.stream_end(reason)

        return None

    def supported_capabilities(self) -> list[Capability]:
        return list(self._capabilities)

    def is_stream_done(self, data: str) -> bool:
        return data.strip() == "[DONE]"

    # -- internal helpers ------------------------------------------------

    @staticmethod
    def _format_message(m: Message) -> dict[str, Any]:
        """Convert a unified Message to OpenAI chat format."""
        # role is stored as str because model uses use_enum_values=True
        role = m.role if isinstance(m.role, str) else m.role.value
        if isinstance(m.content, str):
            out: dict[str, Any] = {"role": role, "content": m.content}
        else:
            # list[ContentBlock] → OpenAI content array
            blocks = []
            for b in m.content:
                if b.type == "text":
                    blocks.append({"type": "text", "text": b.text})
                elif b.type == "image":
                    blocks.append(b.model_dump(by_alias=True))
                else:
                    blocks.append(b.model_dump(by_alias=True))
            out = {"role": role, "content": blocks}
        if role == "tool" and getattr(m, "tool_call_id", None):
            out["tool_call_id"] = m.tool_call_id
        return out
