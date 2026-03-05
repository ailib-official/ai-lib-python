"""Gemini Generate API 驱动 — 实现 Google Gemini 特有的请求/响应格式转换。

Google Gemini generateContent API driver. Key differences:
- Uses ``contents`` instead of ``messages``, with ``parts`` instead of ``content``.
- Roles: ``user`` and ``model`` (not ``assistant``). System uses ``system_instruction``.
- ``generationConfig`` wraps temperature, max_tokens (→ ``maxOutputTokens``), etc.
- Response: ``candidates[0].content.parts[0].text``.
- Streaming uses NDJSON with the same structure (each line is a full response).
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

# Gemini finishReason → AI-Protocol normalized finish_reason
_FINISH_MAP: dict[str, str] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
}


class GeminiDriver(ProviderDriver):
    """Google Gemini generateContent API driver."""

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
        return ApiStyle.GEMINI_GENERATE

    def build_request(
        self,
        messages: list[Message],
        _model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> DriverRequest:
        sys_instruction, contents = self._split_messages(messages)

        body: dict[str, Any] = {"contents": contents}
        if sys_instruction:
            body["system_instruction"] = sys_instruction

        gen_config: dict[str, Any] = {}
        if temperature is not None:
            gen_config["temperature"] = temperature
        if max_tokens is not None:
            gen_config["maxOutputTokens"] = max_tokens
        if gen_config:
            body["generationConfig"] = gen_config

        if extra:
            body.update(extra)

        return DriverRequest(body=body, stream=stream)

    def parse_response(self, body: dict[str, Any]) -> DriverResponse:
        candidates = body.get("candidates", [])
        first = candidates[0] if candidates else {}
        parts = first.get("content", {}).get("parts", [])

        text = next((p["text"] for p in parts if "text" in p), None)

        raw_reason = first.get("finishReason", "")
        finish_reason = _FINISH_MAP.get(raw_reason, raw_reason.lower()) or None

        usage_meta = body.get("usageMetadata")
        usage = None
        if usage_meta:
            usage = UsageInfo(
                prompt_tokens=usage_meta.get("promptTokenCount", 0),
                completion_tokens=usage_meta.get("candidatesTokenCount", 0),
                total_tokens=usage_meta.get("totalTokenCount", 0),
            )

        tool_calls = [p for p in parts if "functionCall" in p]

        return DriverResponse(
            content=text,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=tool_calls,
            raw=body,
        )

    def parse_stream_event(self, data: str) -> StreamingEvent | None:
        stripped = data.strip()
        if not stripped:
            return None

        chunk = json.loads(stripped)

        if error := chunk.get("error"):
            return StreamingEvent.stream_error(error)

        candidates = chunk.get("candidates", [])
        if not candidates:
            return None

        first = candidates[0]
        parts = first.get("content", {}).get("parts", [])

        if parts and (text := parts[0].get("text")):
            return StreamingEvent.content_delta(text)

        reason = first.get("finishReason")
        if reason:
            return StreamingEvent.stream_end(_FINISH_MAP.get(reason, reason.lower()))

        return None

    def supported_capabilities(self) -> list[Capability]:
        return list(self._capabilities)

    def is_stream_done(self, _data: str) -> bool:
        # Gemini uses NDJSON; stream ends when connection closes.
        return False

    # -- internal helpers ------------------------------------------------

    @staticmethod
    def _split_messages(
        messages: list[Message],
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Separate system instructions from conversation contents."""
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []

        for m in messages:
            role = m.role if isinstance(m.role, str) else m.role.value

            if role == "system":
                if isinstance(m.content, str):
                    system_parts.append(m.content)
                continue

            if role == "tool":
                # Gemini: function_response with name (tool_call_id) and response
                tool_id = getattr(m, "tool_call_id", None)
                if tool_id and isinstance(m.content, str):
                    contents.append({
                        "role": "user",
                        "parts": [{"functionResponse": {"name": tool_id, "response": {"result": m.content}}}],
                    })
                continue

            gemini_role = "model" if role == "assistant" else "user"

            if isinstance(m.content, str):
                parts: list[dict[str, Any]] = [{"text": m.content}]
            else:
                parts = [{"text": b.text} if b.type == "text" else b.model_dump(by_alias=True) for b in m.content]

            contents.append({"role": gemini_role, "parts": parts})

        sys_instruction = None
        if system_parts:
            sys_instruction = {"parts": [{"text": "\n\n".join(system_parts)}]}

        return sys_instruction, contents
