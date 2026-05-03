"""Tests for transport module."""

import os
from unittest.mock import patch

import ai_lib_python.transport.auth as auth_module
from ai_lib_python.protocol.manifest import AuthConfig, EndpointConfig, ProtocolManifest
from ai_lib_python.transport.auth import (
    build_auth_metadata,
    diagnostic_text,
    get_auth_header,
    resolve_api_key,
    resolve_credential,
)


class TestResolveApiKey:
    """Tests for API key resolution."""

    def test_explicit_key(self) -> None:
        """Test explicit API key takes precedence."""
        key = resolve_api_key("openai", explicit_key="sk-explicit")
        assert key == "sk-explicit"

    def test_env_variable_standard(self) -> None:
        """Test standard environment variable."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env"}):
            key = resolve_api_key("openai")
            assert key == "sk-env"

    def test_env_variable_from_manifest(self) -> None:
        """Test environment variable from manifest config."""
        manifest = ProtocolManifest(
            id="custom",
            endpoint=EndpointConfig(base_url="https://example.com"),
            auth=AuthConfig(type="bearer", token_env="CUSTOM_TOKEN"),
        )
        with patch.dict(os.environ, {"CUSTOM_TOKEN": "sk-custom"}):
            key = resolve_api_key("custom", manifest=manifest)
            assert key == "sk-custom"

    def test_endpoint_auth_wins_over_top_level_auth(self) -> None:
        """Test V2 endpoint.auth is the single source when both auth blocks diverge."""
        manifest = ProtocolManifest(
            id="dualauth",
            endpoint=EndpointConfig(
                base_url="https://example.com",
                auth=AuthConfig(type="bearer", token_env="DUALAUTH_API_TOKEN"),
            ),
            auth=AuthConfig(type="api_key", key_env="DUALAUTH_LEGACY_KEY"),
        )
        with patch.dict(
            os.environ,
            {
                "DUALAUTH_API_TOKEN": "endpoint-token",
                "DUALAUTH_LEGACY_KEY": "legacy-token",
            },
            clear=True,
        ):
            resolved = resolve_credential("dualauth", manifest)
        assert resolved.secret == "endpoint-token"
        assert resolved.source_kind.value == "manifest_env"
        assert resolved.source_name == "DUALAUTH_API_TOKEN"
        assert resolved.required_envs == ["DUALAUTH_API_TOKEN"]
        assert "DUALAUTH_LEGACY_KEY" in diagnostic_text(resolved, manifest)

    def test_legacy_env_does_not_resolve_when_endpoint_auth_diverges(self) -> None:
        """Test shadowed top-level auth env does not silently resolve."""
        manifest = ProtocolManifest(
            id="dualauth",
            endpoint=EndpointConfig(
                base_url="https://example.com",
                auth=AuthConfig(type="bearer", token_env="DUALAUTH_API_TOKEN"),
            ),
            auth=AuthConfig(type="api_key", key_env="DUALAUTH_LEGACY_KEY"),
        )
        with patch.dict(os.environ, {"DUALAUTH_LEGACY_KEY": "legacy-token"}, clear=True):
            resolved = resolve_credential("dualauth", manifest, allow_keyring=False)
        assert resolved.secret is None
        assert resolved.source_kind.value == "none"
        assert resolved.required_envs == ["DUALAUTH_API_TOKEN"]

    def test_no_key_found(self) -> None:
        """Test when no key is found."""
        with patch.dict(os.environ, {}, clear=True):
            key = resolve_api_key("nonexistent")
            assert key is None

    def test_missing_keyring_package_degrades(self) -> None:
        """Test optional keyring import degrades without failing credential resolution."""
        with patch.object(auth_module, "_keyring", None), patch.dict(os.environ, {}, clear=True):
            resolved = resolve_credential("openai", allow_keyring=True)
        assert resolved.secret is None
        assert resolved.source_kind.value == "none"


class TestGetAuthHeader:
    """Tests for auth header generation."""

    def test_bearer_auth(self) -> None:
        """Test bearer authentication header."""
        manifest = ProtocolManifest(
            id="test",
            endpoint=EndpointConfig(base_url="https://example.com"),
            auth=AuthConfig(type="bearer"),
        )
        headers = get_auth_header("test", manifest, api_key="sk-test")
        assert headers == {"Authorization": "Bearer sk-test"}

    def test_api_key_auth(self) -> None:
        """Test API key authentication header."""
        manifest = ProtocolManifest(
            id="test",
            endpoint=EndpointConfig(base_url="https://example.com"),
            auth=AuthConfig(type="api_key"),
        )
        headers = get_auth_header("test", manifest, api_key="key-123")
        assert headers == {"X-API-Key": "key-123"}

    def test_custom_header_name(self) -> None:
        """Test custom header name."""
        manifest = ProtocolManifest(
            id="test",
            endpoint=EndpointConfig(base_url="https://example.com"),
            auth=AuthConfig(type="bearer", header_name="X-Custom-Auth"),
        )
        headers = get_auth_header("test", manifest, api_key="sk-test")
        assert headers == {"X-Custom-Auth": "Bearer sk-test"}

    def test_custom_header_alias(self) -> None:
        """Test V2 `header` alias parses into header_name."""
        manifest = ProtocolManifest.model_validate(
            {
                "id": "test",
                "endpoint": {"base_url": "https://example.com"},
                "auth": {"type": "custom_header", "header": "X-Provider-Token"},
            }
        )
        headers = get_auth_header("test", manifest, api_key="sk-test")
        assert headers == {"X-Provider-Token": "sk-test"}

    def test_query_param_auth(self) -> None:
        """Test query_param auth attachment."""
        manifest = ProtocolManifest(
            id="queryauth",
            endpoint=EndpointConfig(base_url="https://example.com"),
            auth=AuthConfig(type="query_param", key_env="QUERYAUTH_API_KEY", param_name="api_key"),
        )
        resolved = resolve_credential("queryauth", manifest, explicit_credential="qp-secret")
        headers, query_params = build_auth_metadata(manifest, resolved)
        assert headers == {}
        assert query_params == {"api_key": "qp-secret"}

    def test_redacted_repr(self) -> None:
        """Test resolved credential repr never leaks secret."""
        resolved = resolve_credential("test", explicit_credential="super-secret")
        debug = repr(resolved)
        assert "<redacted>" in debug
        assert "super-secret" not in debug

    def test_no_key(self) -> None:
        """Test when no key is available."""
        with patch.dict(os.environ, {}, clear=True):
            headers = get_auth_header("nonexistent")
            assert headers == {}
