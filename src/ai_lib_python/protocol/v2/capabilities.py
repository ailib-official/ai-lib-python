"""V2 能力声明系统 — 支持 required/optional 分离和 feature_flags 精细控制。

V2 capability declaration system with structured required/optional separation,
feature flags, and capability-to-module mapping for runtime loading.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Capability(str, Enum):
    """Standard capability identifiers aligned with ``schemas/v2/capabilities.json``."""

    TEXT = "text"
    STREAMING = "streaming"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"
    TOOLS = "tools"
    PARALLEL_TOOLS = "parallel_tools"
    AGENTIC = "agentic"
    REASONING = "reasoning"
    EMBEDDINGS = "embeddings"
    STRUCTURED_OUTPUT = "structured_output"
    BATCH = "batch"
    IMAGE_GENERATION = "image_generation"
    STT = "stt"
    TTS = "tts"
    RERANKING = "reranking"
    COMPUTER_USE = "computer_use"
    MCP_CLIENT = "mcp_client"
    MCP_SERVER = "mcp_server"

    # --- mapping helpers ------------------------------------------------

    @property
    def extra_name(self) -> str | None:
        """Map capability to the pip extra name (None = always available)."""
        return _CAP_TO_EXTRA.get(self)

    @property
    def is_extra_gated(self) -> bool:
        """Whether this capability requires a pip extra to be installed."""
        return self.extra_name is not None

    @property
    def module_path(self) -> str:
        """Runtime sub-module path this capability maps to."""
        return _CAP_TO_MODULE[self]


# Core capabilities are always loaded; no extra needed.
_CAP_TO_EXTRA: dict[Capability, str] = {
    Capability.VISION: "vision",
    Capability.AUDIO: "audio",
    Capability.VIDEO: "audio",  # shares the same extra
    Capability.EMBEDDINGS: "embeddings",
    Capability.BATCH: "batch",
    Capability.STT: "stt",
    Capability.TTS: "tts",
    Capability.RERANKING: "reranking",
    Capability.COMPUTER_USE: "computer_use",
    Capability.MCP_CLIENT: "mcp",
    Capability.MCP_SERVER: "mcp",
    Capability.REASONING: "reasoning",
}

_CAP_TO_MODULE: dict[Capability, str] = {
    Capability.TEXT: "core",
    Capability.STREAMING: "streaming",
    Capability.VISION: "multimodal.vision",
    Capability.AUDIO: "multimodal.audio",
    Capability.VIDEO: "multimodal.video",
    Capability.TOOLS: "tools",
    Capability.PARALLEL_TOOLS: "tools.parallel",
    Capability.AGENTIC: "agentic",
    Capability.REASONING: "reasoning",
    Capability.EMBEDDINGS: "embeddings",
    Capability.STRUCTURED_OUTPUT: "structured",
    Capability.BATCH: "batch",
    Capability.IMAGE_GENERATION: "generation.image",
    Capability.STT: "stt",
    Capability.TTS: "tts",
    Capability.RERANKING: "rerank",
    Capability.COMPUTER_USE: "computer_use",
    Capability.MCP_CLIENT: "mcp.client",
    Capability.MCP_SERVER: "mcp.server",
}


class FeatureFlags(BaseModel):
    """Fine-grained feature toggles within capabilities."""

    model_config = ConfigDict(extra="allow")

    structured_output: bool = False
    parallel_tool_calls: bool = False
    extended_thinking: bool = False
    streaming_usage: bool = False
    system_messages: bool = False
    image_generation: bool = False


class LegacyCapabilities(BaseModel):
    """V1 legacy boolean capability flags — backward compatible."""

    model_config = ConfigDict(extra="allow")

    streaming: bool = False
    tools: bool = False
    vision: bool = False
    agentic: bool = False
    reasoning: bool = False
    parallel_tools: bool = False

    def to_capabilities(self) -> list[Capability]:
        """Promote to a flat list of :class:`Capability` values."""
        caps = [Capability.TEXT]
        if self.streaming:
            caps.append(Capability.STREAMING)
        if self.tools:
            caps.append(Capability.TOOLS)
        if self.vision:
            caps.append(Capability.VISION)
        if self.agentic:
            caps.append(Capability.AGENTIC)
        if self.reasoning:
            caps.append(Capability.REASONING)
        if self.parallel_tools:
            caps.append(Capability.PARALLEL_TOOLS)
        return caps


class CapabilitiesV2(BaseModel):
    """V2 structured capability declaration with required/optional separation.

    Supports both the V2 structured format::

        required: [text, streaming]
        optional: [vision, mcp_client]
        feature_flags: {structured_output: true}

    and the V1 legacy flat format::

        streaming: true
        tools: true
        vision: false

    Use :meth:`promote_to_v2` to upgrade legacy declarations.
    """

    model_config = ConfigDict(extra="allow")

    required: list[Capability] = Field(default_factory=lambda: [Capability.TEXT])
    optional: list[Capability] = Field(default_factory=list)
    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)

    @field_validator("required", "optional", mode="before")
    @classmethod
    def _normalize_capability_aliases(cls, value: Any) -> Any:
        """Normalize known legacy aliases before enum validation."""
        if not isinstance(value, list):
            return value
        alias_map = {
            "rerank": "reranking",
        }
        normalized: list[Any] = []
        for item in value:
            if isinstance(item, str):
                normalized.append(alias_map.get(item, item))
            else:
                normalized.append(item)
        return normalized

    # ---- query helpers -------------------------------------------------

    def all_capabilities(self) -> list[Capability]:
        """Return the union of required + optional capabilities."""
        return list(dict.fromkeys(self.required + self.optional))

    def required_capabilities(self) -> list[Capability]:
        """Return only the required capabilities."""
        return list(self.required)

    def has_capability(self, cap: Capability) -> bool:
        """Check if *cap* is declared (required or optional)."""
        return cap in self.required or cap in self.optional

    # ---- V1 auto-promotion ---------------------------------------------

    @classmethod
    def from_legacy(cls, legacy: LegacyCapabilities) -> CapabilitiesV2:
        """Auto-promote V1 legacy capabilities to V2 structured format."""
        required = [Capability.TEXT]
        optional: list[Capability] = []

        if legacy.streaming:
            required.append(Capability.STREAMING)
        if legacy.tools:
            optional.append(Capability.TOOLS)
        if legacy.vision:
            optional.append(Capability.VISION)
        if legacy.agentic:
            optional.append(Capability.AGENTIC)
        if legacy.reasoning:
            optional.append(Capability.REASONING)
        if legacy.parallel_tools:
            optional.append(Capability.PARALLEL_TOOLS)

        return cls(required=required, optional=optional)

    @classmethod
    def from_raw(cls, data: dict[str, Any]) -> CapabilitiesV2:
        """Parse from a raw dict, auto-detecting V1 vs V2 format.

        V2 format has a ``required`` key; anything else is treated as V1.
        """
        if "required" in data:
            return cls.model_validate(data)
        return cls.from_legacy(LegacyCapabilities.model_validate(data))
