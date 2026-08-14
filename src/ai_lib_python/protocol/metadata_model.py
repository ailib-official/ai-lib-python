"""Experimental model-level generative capability facts (ME-001 / PT-GEN-001).

从 ``metadata.models`` 读取 Experimental 生成式能力；省略字段 = unknown，不得当成 false。
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class CapabilityKnown(str, Enum):
    """Tri-state support: omitted / unknown must not act like false."""

    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"

    @classmethod
    def from_option(cls, value: bool | None) -> CapabilityKnown:
        if value is True:
            return cls.YES
        if value is False:
            return cls.NO
        return cls.UNKNOWN

    def or_provider(self, provider_allows: bool) -> bool:
        """Prefer known model fact; otherwise fall back to provider-level boolean."""
        if self is CapabilityKnown.YES:
            return True
        if self is CapabilityKnown.NO:
            return False
        return provider_allows


GENERATIVE_KEYS = ("image_generation", "speech_to_text", "text_to_speech")


def supports_generative_capability(entry: dict[str, Any] | None, key: str) -> CapabilityKnown:
    """Experimental (PT-GEN-001): per-model generative capability fact."""
    if not isinstance(entry, dict):
        return CapabilityKnown.UNKNOWN
    caps = entry.get("model_capabilities")
    if not isinstance(caps, dict):
        return CapabilityKnown.UNKNOWN
    if key not in GENERATIVE_KEYS:
        return CapabilityKnown.UNKNOWN
    raw = caps.get(key, None)
    if raw is True:
        return CapabilityKnown.YES
    if raw is False:
        return CapabilityKnown.NO
    return CapabilityKnown.UNKNOWN


def model_entry_from_extra(extra: dict[str, Any] | None, model_id: str) -> dict[str, Any] | None:
    """Look up ``metadata.models.<model_id>`` from a flattened extra map."""
    if not extra:
        return None
    metadata = extra.get("metadata")
    if not isinstance(metadata, dict):
        return None
    models = metadata.get("models")
    if not isinstance(models, dict):
        return None
    entry = models.get(model_id)
    return entry if isinstance(entry, dict) else None
