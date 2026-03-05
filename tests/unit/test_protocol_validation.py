"""Tests for protocol validation."""


from ai_lib_python.protocol import (
    SUPPORTED_PROTOCOL_VERSIONS,
    validate_manifest,
    validate_protocol_version,
    validate_streaming_config,
)


class TestProtocolVersionValidation:
    """Tests for protocol version validation."""

    def test_supported_versions(self) -> None:
        """Test list of supported versions."""
        assert "1.0" in SUPPORTED_PROTOCOL_VERSIONS
        assert "1.1" in SUPPORTED_PROTOCOL_VERSIONS
        assert "1.5" in SUPPORTED_PROTOCOL_VERSIONS

    def test_valid_version(self) -> None:
        """Test validation of supported version."""
        manifest = {"protocol_version": "1.5"}
        result = validate_protocol_version(manifest)
        assert result.valid is True

    def test_unsupported_version(self) -> None:
        """Test validation of unsupported version."""
        manifest = {"protocol_version": "99.0"}
        result = validate_protocol_version(manifest)
        assert result.valid is False
        assert any("Unsupported protocol version" in e for e in result.errors)

    def test_default_version(self) -> None:
        """Test default version when not specified."""
        manifest = {}
        result = validate_protocol_version(manifest)
        # Default 1.0 should generate warning but be valid
        assert result.valid is True

    def test_deprecated_version_warning(self) -> None:
        """Test warning for deprecated version."""
        manifest = {"protocol_version": "1.0"}
        result = validate_protocol_version(manifest)
        assert result.valid is True
        assert any("deprecated" in w.lower() for w in result.warnings)

    def test_custom_supported_versions(self) -> None:
        """Test with custom supported versions list."""
        manifest = {"protocol_version": "3.0"}
        result = validate_protocol_version(manifest, supported_versions=["3.0", "3.1"])
        assert result.valid is True


class TestStreamingConfigValidation:
    """Tests for streaming configuration validation."""

    def test_no_streaming_config(self) -> None:
        """Test manifest without streaming."""
        manifest = {"id": "test"}
        result = validate_streaming_config(manifest, strict=False)
        assert result.valid is True

    def test_streaming_capability_no_config_non_strict(self) -> None:
        """Test streaming capability without config (non-strict)."""
        manifest = {
            "id": "test",
            "capabilities": {"streaming": True},
        }
        result = validate_streaming_config(manifest, strict=False)
        assert result.valid is True
        assert len(result.warnings) > 0

    def test_streaming_capability_no_config_strict(self) -> None:
        """Test streaming capability without config (strict)."""
        manifest = {
            "id": "test",
            "capabilities": {"streaming": True},
        }
        result = validate_streaming_config(manifest, strict=True)
        assert result.valid is False

    def test_valid_streaming_config(self) -> None:
        """Test valid streaming configuration."""
        manifest = {
            "id": "test",
            "capabilities": {"streaming": True},
            "streaming": {
                "decoder": {"format": "sse"},
                "content_path": "choices[0].delta.content",
            },
        }
        result = validate_streaming_config(manifest, strict=True)
        assert result.valid is True

    def test_missing_decoder_strict(self) -> None:
        """Test missing decoder in strict mode."""
        manifest = {
            "id": "test",
            "capabilities": {"streaming": True},
            "streaming": {
                "content_path": "choices[0].delta.content",
            },
        }
        result = validate_streaming_config(manifest, strict=True)
        assert result.valid is False
        assert any("decoder" in e.lower() for e in result.errors)

    def test_empty_decoder_format_strict(self) -> None:
        """Test empty decoder format in strict mode."""
        manifest = {
            "id": "test",
            "capabilities": {"streaming": True},
            "streaming": {
                "decoder": {"format": ""},
                "content_path": "test",
            },
        }
        result = validate_streaming_config(manifest, strict=True)
        assert result.valid is False

    def test_missing_content_path_strict(self) -> None:
        """Test missing content_path when event_map is empty."""
        manifest = {
            "id": "test",
            "capabilities": {"streaming": True},
            "streaming": {
                "decoder": {"format": "sse"},
                "event_map": [],
            },
        }
        result = validate_streaming_config(manifest, strict=True)
        assert result.valid is False
        assert any("content_path" in e for e in result.errors)

    def test_missing_tool_call_path_with_tools(self) -> None:
        """Test missing tool_call_path when tools are supported."""
        manifest = {
            "id": "test",
            "capabilities": {"streaming": True, "tools": True},
            "streaming": {
                "decoder": {"format": "sse"},
                "content_path": "test",
            },
        }
        result = validate_streaming_config(manifest, strict=True)
        assert result.valid is False
        assert any("tool_call_path" in e for e in result.errors)

    def test_with_event_map(self) -> None:
        """Test with explicit event_map (no paths required)."""
        manifest = {
            "id": "test",
            "capabilities": {"streaming": True},
            "streaming": {
                "decoder": {"format": "sse"},
                "event_map": [{"match": "content", "emit": "content"}],
            },
        }
        result = validate_streaming_config(manifest, strict=True)
        assert result.valid is True


class TestValidateManifest:
    """Tests for comprehensive manifest validation."""

    def test_valid_manifest(self) -> None:
        """Test validating a complete valid manifest."""
        manifest = {
            "id": "test-provider",
            "protocol_version": "1.5",
            "endpoint": {"base_url": "https://api.example.com"},
        }
        result = validate_manifest(manifest, strict_streaming=False)
        assert result.valid is True

    def test_invalid_version_fails(self) -> None:
        """Test invalid version causes failure."""
        manifest = {
            "id": "test",
            "protocol_version": "99.0",
        }
        result = validate_manifest(manifest)
        assert result.valid is False

    def test_skip_version_check(self) -> None:
        """Test skipping version check."""
        manifest = {
            "id": "test",
            "protocol_version": "99.0",
        }
        result = validate_manifest(manifest, check_version=False)
        assert result.valid is True

    def test_combined_errors(self) -> None:
        """Test combined errors from multiple validations."""
        manifest = {
            "id": "test",
            "protocol_version": "99.0",
            "capabilities": {"streaming": True},
        }
        result = validate_manifest(manifest, strict_streaming=True)
        # Should have both version error and streaming error
        assert result.valid is False
        assert len(result.errors) >= 2
