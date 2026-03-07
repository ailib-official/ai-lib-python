"""Latest generative manifest consumption tests for ai-lib-python."""

from __future__ import annotations

from pathlib import Path

import yaml

from ai_lib_python.multimodal import Modality, MultimodalCapabilities
from ai_lib_python.protocol.v2.manifest import ApiStyle, ManifestV2


def _resolve_ai_protocol_root() -> Path:
    candidates = [
        Path.cwd() / "../ai-protocol",
        Path.cwd() / "../../ai-protocol",
        Path("d:/ai-protocol"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise AssertionError("Unable to locate ai-protocol root")


def test_consume_latest_v2_generative_manifests() -> None:
    root = _resolve_ai_protocol_root()
    providers = ("google", "deepseek", "qwen", "doubao")

    for provider in providers:
        path = root / "v2" / "providers" / f"{provider}.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        manifest = ManifestV2.model_validate(raw)

        assert manifest.is_v2, f"{provider} should parse as V2 manifest"
        assert manifest.id == provider
        assert manifest.endpoint is not None, f"{provider} should expose endpoint field"
        assert manifest.base_url

        if provider == "google":
            assert manifest.detect_api_style() == ApiStyle.GEMINI_GENERATE
        else:
            assert manifest.detect_api_style() == ApiStyle.OPENAI_COMPATIBLE

        assert manifest.multimodal is not None
        caps = MultimodalCapabilities.from_config(
            manifest.multimodal.model_dump(exclude_none=True)
        )

        assert caps.supports_input(Modality.TEXT)
        assert caps.supports_output(Modality.TEXT)
        if provider in {"google", "qwen"}:
            assert caps.supports_input(Modality.VIDEO)

        output = manifest.multimodal.output or {}
        video_out = output.get("video") or {}
        assert video_out.get("supported", False) is False
