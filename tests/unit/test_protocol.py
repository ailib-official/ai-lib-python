"""Tests for protocol module."""

import json
import os
from pathlib import Path

import pytest

from ai_lib_python.protocol import (
    CapabilitiesConfig,
    DecoderConfig,
    EndpointConfig,
    ProtocolLoader,
    ProtocolManifest,
    ProtocolValidator,
)


class TestProtocolManifest:
    """Tests for ProtocolManifest model."""

    def test_minimal_manifest(self) -> None:
        """Test creating minimal manifest."""
        data = {
            "id": "test-provider",
            "endpoint": {
                "base_url": "https://api.example.com",
            },
        }
        manifest = ProtocolManifest.model_validate(data)
        assert manifest.id == "test-provider"
        assert manifest.endpoint.base_url == "https://api.example.com"

    def test_full_manifest(self) -> None:
        """Test creating full manifest with all fields."""
        data = {
            "id": "openai",
            "protocol_version": "1.5",
            "name": "OpenAI",
            "endpoint": {
                "base_url": "https://api.openai.com/v1",
                "protocol": "https",
                "timeout_ms": 10000,
            },
            "capabilities": {
                "streaming": True,
                "tools": True,
                "vision": True,
            },
            "streaming": {
                "decoder": {
                    "format": "sse",
                    "delimiter": "\n\n",
                    "prefix": "data: ",
                    "done_signal": "[DONE]",
                },
            },
            "parameter_mappings": {
                "temperature": "temperature",
                "max_tokens": "max_tokens",
            },
        }
        manifest = ProtocolManifest.model_validate(data)
        assert manifest.id == "openai"
        assert manifest.protocol_version == "1.5"
        assert manifest.capabilities.streaming is True
        assert manifest.capabilities.tools is True

    def test_supports_streaming(self) -> None:
        """Test streaming support check."""
        manifest = ProtocolManifest(
            id="test",
            endpoint=EndpointConfig(base_url="https://example.com"),
            capabilities=CapabilitiesConfig(streaming=True),
        )
        assert manifest.supports_streaming() is True

    def test_supports_tools(self) -> None:
        """Test tools support check."""
        manifest = ProtocolManifest(
            id="test",
            endpoint=EndpointConfig(base_url="https://example.com"),
            capabilities=CapabilitiesConfig(tools=True),
        )
        assert manifest.supports_tools() is True

    def test_get_parameter_name(self) -> None:
        """Test parameter name mapping."""
        manifest = ProtocolManifest(
            id="test",
            endpoint=EndpointConfig(base_url="https://example.com"),
            parameter_mappings={"max_tokens": "max_completion_tokens"},
        )
        assert manifest.get_parameter_name("max_tokens") == "max_completion_tokens"
        assert manifest.get_parameter_name("temperature") == "temperature"

    def test_extra_fields_allowed(self) -> None:
        """Test that extra fields are allowed."""
        data = {
            "id": "test",
            "endpoint": {"base_url": "https://example.com"},
            "custom_field": "custom_value",
        }
        manifest = ProtocolManifest.model_validate(data)
        assert manifest.id == "test"


class TestDecoderConfig:
    """Tests for DecoderConfig model."""

    def test_default_values(self) -> None:
        """Test default decoder config values."""
        config = DecoderConfig()
        assert config.format == "sse"
        assert config.delimiter == "\n\n"
        assert config.prefix == "data: "
        assert config.done_signal == "[DONE]"

    def test_custom_values(self) -> None:
        """Test custom decoder config values."""
        config = DecoderConfig(
            format="json_lines",
            delimiter="\n",
            prefix="",
            done_signal="",
        )
        assert config.format == "json_lines"
        assert config.delimiter == "\n"


class TestProtocolValidator:
    """Tests for ProtocolValidator."""

    @staticmethod
    def _load_real_manifest() -> dict:
        """Load a real manifest from ai-protocol checkout if available."""
        protocol_dir = os.getenv("AI_PROTOCOL_DIR")
        if protocol_dir:
            manifest_path = Path(protocol_dir) / "dist" / "v1" / "providers" / "openai.json"
            if manifest_path.exists():
                return json.loads(manifest_path.read_text(encoding="utf-8"))

        local_manifest = Path(__file__).resolve().parents[3] / "ai-protocol" / "dist" / "v1" / "providers" / "openai.json"
        if local_manifest.exists():
            return json.loads(local_manifest.read_text(encoding="utf-8"))

        pytest.skip("ai-protocol manifest not available")

    def test_validate_valid_manifest(self) -> None:
        """Test validating a valid manifest."""
        validator = ProtocolValidator()
        data = self._load_real_manifest()
        result = validator.validate(data)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_missing_id(self) -> None:
        """Test validating manifest without ID."""
        validator = ProtocolValidator()
        data = {
            "endpoint": {
                "base_url": "https://api.example.com",
            },
        }
        result = validator.validate(data)
        assert result.valid is False
        assert len(result.errors) > 0

    def test_validate_missing_endpoint(self) -> None:
        """Test validating manifest without endpoint."""
        validator = ProtocolValidator()
        data = {"id": "test"}
        result = validator.validate(data)
        assert result.valid is False

    def test_validate_or_raise(self) -> None:
        """Test validate_or_raise raises on invalid."""
        from ai_lib_python.errors import ProtocolError

        validator = ProtocolValidator()
        with pytest.raises(ProtocolError):
            validator.validate_or_raise({"invalid": "data"})

    def test_is_valid(self) -> None:
        """Test is_valid convenience method."""
        validator = ProtocolValidator()
        assert validator.is_valid(self._load_real_manifest())
        assert not validator.is_valid({})


class TestProtocolLoader:
    """Tests for ProtocolLoader."""

    def test_loader_initialization(self) -> None:
        """Test loader initialization."""
        loader = ProtocolLoader(
            base_path="/custom/path",
            fallback_to_github=False,
            cache_enabled=True,
        )
        assert loader._fallback_to_github is False
        assert loader._cache_enabled is True

    def test_register_provider(self) -> None:
        """Test registering a custom provider."""
        loader = ProtocolLoader()
        manifest = loader.register_provider({
            "id": "custom-provider",
            "endpoint": {"base_url": "https://custom.api.com"},
        })
        assert manifest.id == "custom-provider"
        assert "provider:custom-provider" in loader._cache

    def test_clear_cache(self) -> None:
        """Test clearing cache."""
        loader = ProtocolLoader()
        loader.register_provider({
            "id": "test",
            "endpoint": {"base_url": "https://example.com"},
        })
        assert len(loader._cache) > 0
        loader.clear_cache()
        assert len(loader._cache) == 0

    def test_invalidate_specific_key(self) -> None:
        """Test invalidating specific cache key."""
        loader = ProtocolLoader()
        loader.register_provider({
            "id": "test",
            "endpoint": {"base_url": "https://example.com"},
        })
        loader.invalidate("provider:test")
        assert "provider:test" not in loader._cache
