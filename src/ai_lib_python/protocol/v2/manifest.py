"""V2 三环清单数据模型 — 建模 Ring1 核心骨架 / Ring2 能力映射 / Ring3 高级扩展。

V2 three-ring manifest data model for AI-Protocol.

Models the concentric circle structure:
- Ring 1: Core Skeleton (id, version, auth, endpoints, models)
- Ring 2: Capability Mapping (streaming, multimodal, computer_use, mcp)
- Ring 3: Advanced Extensions (context_policy, service, parameters, tags)
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai_lib_python.protocol.v2.capabilities import CapabilitiesV2


class ApiStyle(str, Enum):
    """Provider API integration style — determines which driver to use."""

    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    GEMINI_GENERATE = "gemini_generate"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Ring 1: Core Skeleton
# ---------------------------------------------------------------------------


class AuthConfigV2(BaseModel):
    """Authentication configuration."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(default="bearer")
    token_env: str | None = None
    header_name: str | None = None


class EndpointV2(BaseModel):
    """Endpoint configuration."""

    model_config = ConfigDict(extra="allow")

    base_url: str
    chat: str | dict[str, Any] = Field(default="/v1/chat/completions")
    models: str | None = None
    embeddings: str | dict[str, Any] | None = None
    protocol: str = "https"
    timeout_ms: int = 10_000


class ModelDef(BaseModel):
    """Model definition within a provider manifest."""

    model_config = ConfigDict(extra="allow")

    id: str
    display_name: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    pricing: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Ring 2: Capability Mapping
# ---------------------------------------------------------------------------


class StreamingV2(BaseModel):
    """Streaming configuration."""

    model_config = ConfigDict(extra="allow")

    decoder: str | dict[str, Any] = "sse"
    event_map: dict[str, Any] | list[dict[str, Any]] = Field(default_factory=dict)
    accumulator: dict[str, Any] = Field(default_factory=dict)
    candidates: dict[str, Any] | None = None


class McpConfig(BaseModel):
    """MCP (Model Context Protocol) capability configuration."""

    model_config = ConfigDict(extra="allow")

    supported: bool = False
    protocol_version: str | None = None
    transport: list[str] = Field(default_factory=list)
    max_servers: int | None = None
    timeout_ms: int | None = None
    error_handling: dict[str, Any] | None = None


class ComputerUseConfig(BaseModel):
    """Computer Use capability configuration."""

    model_config = ConfigDict(extra="allow")

    supported: bool = False
    actions: list[str] | dict[str, Any] = Field(default_factory=list)
    coordinate_system: str | None = None
    max_actions_per_turn: int | None = None
    screenshot_format: str | None = None


class MultimodalConfig(BaseModel):
    """Extended multimodal capability configuration."""

    model_config = ConfigDict(extra="allow")

    # Latest V2 schema (nested input/output)
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    omni_mode: dict[str, Any] | None = None

    # Legacy flattened compatibility fields
    input_modalities: list[str] = Field(default_factory=lambda: ["text"])
    output_modalities: list[str] = Field(default_factory=lambda: ["text"])
    max_image_size_mb: float | None = None
    supported_image_formats: list[str] = Field(default_factory=list)
    audio_formats: list[str] = Field(default_factory=list)
    video_formats: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Ring 3: Advanced Extensions
# ---------------------------------------------------------------------------


class ParameterDef(BaseModel):
    """Parameter definition for model parameters."""

    model_config = ConfigDict(extra="allow")

    type: str = "number"
    min: float | None = Field(default=None, alias="min")
    max: float | None = Field(default=None, alias="max")
    default: Any = None


class ErrorClassificationV2(BaseModel):
    """Error classification rules."""

    model_config = ConfigDict(extra="allow")

    retryable: list[int] = Field(default_factory=lambda: [429, 500, 502, 503])
    fatal: list[int] = Field(default_factory=lambda: [401, 403])
    rate_limit_header: str | None = None


class ContextPolicy(BaseModel):
    """V2 context management policy."""

    model_config = ConfigDict(extra="allow")

    strategy: str = "sliding_window"
    token_budget: dict[str, Any] | None = None
    sliding_window: dict[str, Any] | None = None
    summarization: dict[str, Any] | None = None
    overflow_handling: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Root Manifest
# ---------------------------------------------------------------------------

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?")


class ManifestV2(BaseModel):
    """V2 three-ring provider manifest — the central configuration unit.

    Supports auto-promotion from V1 manifests: when ``protocol_version``
    starts with ``"1."``, the manifest is loaded with V1 defaults and the
    capabilities are promoted via :meth:`CapabilitiesV2.from_raw`.
    """

    model_config = ConfigDict(extra="allow")

    # --- Ring 1: Core Skeleton ------------------------------------------
    id: str = Field(description="Provider identifier (e.g., 'openai')")
    name: str = Field(default="", description="Human-readable name")
    protocol_version: str = Field(default="2.0")
    api_style: ApiStyle = Field(default=ApiStyle.OPENAI_COMPATIBLE)

    auth: AuthConfigV2 = Field(default_factory=AuthConfigV2)
    endpoint: EndpointV2 | None = None
    endpoints: EndpointV2 | dict[str, Any] | None = None
    models: list[ModelDef] = Field(default_factory=list)

    # --- Ring 2: Capability Mapping -------------------------------------
    capabilities: CapabilitiesV2 = Field(default_factory=CapabilitiesV2)
    streaming: StreamingV2 | None = None
    multimodal: MultimodalConfig | None = None
    computer_use: ComputerUseConfig | None = None
    mcp: McpConfig | None = None

    # --- Ring 3: Advanced Extensions ------------------------------------
    parameters: dict[str, ParameterDef] = Field(default_factory=dict)
    error_classification: ErrorClassificationV2 | None = None
    context_policy: ContextPolicy | None = None
    service: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)

    # ---- derived helpers -----------------------------------------------

    @property
    def is_v2(self) -> bool:
        """Whether this manifest uses V2 protocol (version >= 2.0)."""
        return self.protocol_semver >= (2, 0, 0)

    @property
    def protocol_semver(self) -> tuple[int, int, int]:
        """Parse ``protocol_version`` into a ``(major, minor, patch)`` tuple."""
        m = _SEMVER_RE.match(self.protocol_version)
        if not m:
            return (0, 0, 0)
        return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))

    @property
    def base_url(self) -> str:
        """Shortcut for the endpoint base URL."""
        if self.endpoint:
            return self.endpoint.base_url
        if isinstance(self.endpoints, EndpointV2):
            return self.endpoints.base_url
        if isinstance(self.endpoints, dict):
            maybe_base_url = self.endpoints.get("base_url")
            if isinstance(maybe_base_url, str):
                return maybe_base_url
        return ""

    @property
    def chat_path(self) -> str:
        """Shortcut for the chat endpoint path."""
        if self.endpoint:
            chat = self.endpoint.chat
            if isinstance(chat, dict):
                path = chat.get("path")
                if isinstance(path, str):
                    return path
            return chat if isinstance(chat, str) else "/v1/chat/completions"
        if isinstance(self.endpoints, EndpointV2):
            chat = self.endpoints.chat
            if isinstance(chat, dict):
                path = chat.get("path")
                if isinstance(path, str):
                    return path
            return chat if isinstance(chat, str) else "/v1/chat/completions"
        if isinstance(self.endpoints, dict):
            maybe_chat = self.endpoints.get("chat")
            if isinstance(maybe_chat, str):
                return maybe_chat
            if isinstance(maybe_chat, dict):
                path = maybe_chat.get("path")
                if isinstance(path, str):
                    return path
        return "/v1/chat/completions"

    def has_capability(self, cap: Any) -> bool:
        """Check if a capability is declared."""
        from ai_lib_python.protocol.v2.capabilities import Capability

        if isinstance(cap, str):
            cap = Capability(cap)
        return self.capabilities.has_capability(cap)

    def mcp_client_supported(self) -> bool:
        """Whether MCP client is supported by this provider."""
        return self.mcp is not None and self.mcp.supported

    def computer_use_supported(self) -> bool:
        """Whether computer use is supported by this provider."""
        return self.computer_use is not None and self.computer_use.supported

    def detect_api_style(self) -> ApiStyle:
        """Detect API style from provider id if not explicitly set."""
        if self.api_style != ApiStyle.OPENAI_COMPATIBLE:
            return self.api_style

        provider_lower = self.id.lower()
        if "anthropic" in provider_lower or "claude" in provider_lower:
            return ApiStyle.ANTHROPIC_MESSAGES
        if "google" in provider_lower or "gemini" in provider_lower:
            return ApiStyle.GEMINI_GENERATE
        return ApiStyle.OPENAI_COMPATIBLE
