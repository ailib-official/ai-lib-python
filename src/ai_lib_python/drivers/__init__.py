"""Provider 驱动抽象层 — 通过 ABC 实现多厂商 API 适配的动态分发。

Provider driver abstraction layer implementing the ProviderContract specification.
Uses abstract base class + factory for runtime polymorphism, enabling the same
client code to work with OpenAI, Anthropic, Gemini, and any compatible provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ai_lib_python.protocol.v2.capabilities import Capability
from ai_lib_python.protocol.v2.manifest import ApiStyle
from ai_lib_python.types.events import StreamingEvent
from ai_lib_python.types.message import Message


@dataclass
class DriverRequest:
    """Unified HTTP request representation for provider communication."""

    url: str = ""
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] = field(default_factory=dict)
    stream: bool = False


@dataclass
class DriverResponse:
    """Unified chat response from provider."""

    content: str | None = None
    thinking: str | None = None
    finish_reason: str | None = None
    usage: UsageInfo | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageInfo:
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ProviderDriver(ABC):
    """Core abstract class for provider-specific API adaptation.

    Each provider API style (OpenAI, Anthropic, Gemini) has a concrete
    implementation. The runtime selects the correct driver based on the
    manifest's ``api_style`` or ``provider_contract``.
    """

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique provider identifier (matches manifest ``id``)."""

    @property
    @abstractmethod
    def api_style(self) -> ApiStyle:
        """API style this driver implements."""

    @abstractmethod
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
        """Build a provider-specific HTTP request from unified parameters."""

    @abstractmethod
    def parse_response(self, body: dict[str, Any]) -> DriverResponse:
        """Parse a non-streaming response into unified format."""

    @abstractmethod
    def parse_stream_event(self, data: str) -> list[StreamingEvent]:
        """Parse streaming frame(s) from raw SSE/NDJSON data.

        Returns zero or more events (same-frame thinking+content may yield both).
        """

    @abstractmethod
    def supported_capabilities(self) -> list[Capability]:
        """Get the list of capabilities this driver supports."""

    @abstractmethod
    def is_stream_done(self, data: str) -> bool:
        """Check if the done signal has been received in streaming."""


# ---------------------------------------------------------------------------
# Concrete drivers (imported lazily to avoid circular deps)
# ---------------------------------------------------------------------------

from ai_lib_python.drivers.anthropic import AnthropicDriver  # noqa: E402
from ai_lib_python.drivers.gemini import GeminiDriver  # noqa: E402
from ai_lib_python.drivers.openai import OpenAiDriver  # noqa: E402


def create_driver(
    api_style: ApiStyle,
    provider_id: str,
    capabilities: list[Capability] | None = None,
) -> ProviderDriver:
    """Factory: create the appropriate driver from an API style.

    ``Custom`` falls back to OpenAI-compatible, which covers most
    providers that follow the OpenAI chat completions format (DeepSeek,
    Moonshot, Zhipu, etc.).
    """
    caps = capabilities or []
    match api_style:
        case ApiStyle.OPENAI_COMPATIBLE | ApiStyle.CUSTOM:
            return OpenAiDriver(provider_id=provider_id, capabilities=caps)
        case ApiStyle.ANTHROPIC_MESSAGES:
            return AnthropicDriver(provider_id=provider_id, capabilities=caps)
        case ApiStyle.GEMINI_GENERATE:
            return GeminiDriver(provider_id=provider_id, capabilities=caps)
        case _:
            return OpenAiDriver(provider_id=provider_id, capabilities=caps)


__all__ = [
    "AnthropicDriver",
    "DriverRequest",
    "DriverResponse",
    "GeminiDriver",
    "OpenAiDriver",
    "ProviderDriver",
    "UsageInfo",
    "create_driver",
]
