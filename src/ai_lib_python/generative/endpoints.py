"""Resolve PT-GEN L-Exec maps from ``endpoints.<key>``.

从 manifest ``endpoints.<key>`` 解析生成式路径；缺映射 fail-closed。
"""

from __future__ import annotations

from typing import Any

from ai_lib_python.errors import ValidationError
from ai_lib_python.protocol.manifest import ProtocolManifest
from ai_lib_python.protocol.metadata_model import GENERATIVE_KEYS

KEY_IMAGE_GENERATION = "image_generation"
KEY_SPEECH_TO_TEXT = "speech_to_text"
KEY_TEXT_TO_SPEECH = "text_to_speech"


def adapter_name(endpoint: dict[str, Any]) -> str:
    """Adapter from L-Exec map; missing adapter defaults to openai (ALR-GEN-002)."""
    raw = endpoint.get("adapter")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return "openai"


def resolve_generative_endpoint(manifest: ProtocolManifest, key: str) -> dict[str, Any]:
    """Resolve ``endpoints.<key>`` (required for generative ops)."""
    if key not in GENERATIVE_KEYS:
        raise ValidationError(
            f"unknown generative capability `{key}`; expected one of {', '.join(GENERATIVE_KEYS)}"
        )
    eps = manifest.endpoints or {}
    ep = eps.get(key)
    if not isinstance(ep, dict):
        raise ValidationError(f"manifest endpoints.{key} missing; declare PT-GEN-002 L-Exec map")
    path = ep.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValidationError(f"manifest endpoints.{key} missing; declare PT-GEN-002 L-Exec map")
    return ep


def require_generative_endpoint(manifest: ProtocolManifest, model: str, key: str) -> dict[str, Any]:
    """Gate + resolve: capability must be known-true; endpoint must exist."""
    if not manifest.supports_generative_for_model(model, key):
        raise ValidationError(
            f"model `{model}` does not declare model_capabilities.{key}=true "
            "(omit≠false fail-closed)"
        )
    return resolve_generative_endpoint(manifest, key)
