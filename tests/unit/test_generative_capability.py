"""ALP-GEN-001: generative capability gate + L-Exec body parity with Rust.

生成式能力 omit≠false；L-Exec 缺失 fail-closed；adapter 决定请求体（非 if provider）。
"""

from __future__ import annotations

import pytest

from ai_lib_python.errors import ValidationError
from ai_lib_python.generative.endpoints import require_generative_endpoint
from ai_lib_python.generative.image import dashscope_image_body, openai_image_body
from ai_lib_python.generative.types import ImageGenerationRequest
from ai_lib_python.protocol.manifest import ProtocolManifest


def _openai_manifest() -> ProtocolManifest:
    return ProtocolManifest.model_validate(
        {
            "id": "openai",
            "protocol_version": "2.0",
            "status": "stable",
            "endpoint": {"base_url": "https://api.openai.com/v1"},
            "capabilities": ["text", "image_generation"],
            "endpoints": {
                "image_generation": {
                    "path": "/images/generations",
                    "method": "POST",
                    "adapter": "openai",
                }
            },
            "metadata": {
                "models": {
                    "gpt-image-1": {"model_capabilities": {"image_generation": True}},
                    "gpt-4o": {"context_window": 128000},
                }
            },
        }
    )


def _qwen_manifest() -> ProtocolManifest:
    return ProtocolManifest.model_validate(
        {
            "id": "qwen",
            "protocol_version": "2.0",
            "status": "stable",
            "endpoint": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
            "capabilities": ["text", "image_generation"],
            "endpoints": {
                "image_generation": {
                    "path": (
                        "https://dashscope.aliyuncs.com/api/v1/services/"
                        "aigc/multimodal-generation/generation"
                    ),
                    "method": "POST",
                    "adapter": "dashscope",
                }
            },
            "metadata": {
                "models": {
                    "qwen-image-plus": {"model_capabilities": {"image_generation": True}},
                }
            },
        }
    )


def test_omit_capability_fail_closed() -> None:
    m = _openai_manifest()
    assert m.supports_generative_for_model("gpt-image-1", "image_generation") is True
    assert m.supports_generative_for_model("gpt-4o", "image_generation") is False
    with pytest.raises(ValidationError, match="omit"):
        require_generative_endpoint(m, "gpt-4o", "image_generation")


def test_missing_lexec_fail_closed() -> None:
    m = ProtocolManifest.model_validate(
        {
            "id": "genprov",
            "protocol_version": "2.0",
            "status": "stable",
            "endpoint": {"base_url": "https://example.com/v1"},
            "capabilities": ["text"],
            "metadata": {
                "models": {
                    "img-1": {"model_capabilities": {"image_generation": True}},
                }
            },
        }
    )
    assert m.supports_generative_for_model("img-1", "image_generation") is True
    with pytest.raises(ValidationError, match=r"endpoints\.image_generation"):
        require_generative_endpoint(m, "img-1", "image_generation")


def test_openai_and_dashscope_bodies_differ() -> None:
    req = ImageGenerationRequest(model="m", prompt="a cat")
    oai = openai_image_body(req)
    ds = dashscope_image_body(req)
    assert oai["prompt"] == "a cat"
    assert "input" not in oai
    assert "prompt" not in ds
    assert ds["input"]["messages"][0]["content"][0]["text"] == "a cat"


def test_require_openai_and_qwen_image_endpoints() -> None:
    openai = _openai_manifest()
    ep = require_generative_endpoint(openai, "gpt-image-1", "image_generation")
    assert ep["path"] == "/images/generations"
    qwen = _qwen_manifest()
    ep = require_generative_endpoint(qwen, "qwen-image-plus", "image_generation")
    assert str(ep["path"]).startswith("https://")
    with pytest.raises(ValidationError):
        require_generative_endpoint(qwen, "missing", "image_generation")
