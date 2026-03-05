"""
Integration tests for protocol loading and configuration.

Tests protocol manifest loading, YAML configurations, and provider protocols.
"""

import pytest

from ai_lib_python.client import AiClient
from ai_lib_python.types.message import Message


class TestProtocolLoading:
    """Tests for protocol loading functionality."""

    @pytest.mark.asyncio
    async def test_load_openai_protocol(self, httpx_mock) -> None:
        """Test loading OpenAI provider protocol."""
        from tests.integration.conftest import setup_mock_openai_response

        setup_mock_openai_response(httpx_mock, content="OpenAI protocol loaded")

        # Client should load protocol from model ID
        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")

        # Verify protocol loaded correctly
        assert client._model_id == "openai/gpt-4o"
        assert client._manifest is not None

        # Request should use OpenAI style
        response = await client.chat().messages([Message.user("Test")]).execute()
        assert response.content == "OpenAI protocol loaded"

    @pytest.mark.asyncio
    async def test_load_anthropic_protocol(self, httpx_mock) -> None:
        """Test loading Anthropic provider protocol."""
        from tests.integration.conftest import setup_mock_anthropic_response

        setup_mock_anthropic_response(httpx_mock, content="Anthropic protocol loaded")

        client = await AiClient.create("anthropic/claude-3-5-sonnet", api_key="sk-ant-test")

        assert client._model_id == "anthropic/claude-3-5-sonnet"
        assert client._manifest is not None

        response = await client.chat().messages([Message.user("Test")]).execute()
        assert response.content == "Anthropic protocol loaded"

    @pytest.mark.asyncio
    async def test_custom_base_url(self, httpx_mock) -> None:
        """Test custom base URL configuration."""

        custom_url = "https://custom.api.example.com/v1/chat/completions"

        httpx_mock.add_response(
            url=custom_url,
            method="POST",
            json=[{
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1699012345,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Custom URL response"},
                    "finish_reason": "stop",
                }],
            }],
        )

        client = await AiClient.create(
            "openai/gpt-4o",
            api_key="sk-test",
            base_url="https://custom.api.example.com/v1",
        )

        response = await client.chat().messages([Message.user("Test")]).execute()
        assert response.content == "Custom URL response"

    @pytest.mark.asyncio
    async def test_custom_timeout(self, httpx_mock) -> None:
        """Test custom timeout configuration."""
        from tests.integration.conftest import setup_mock_openai_response

        setup_mock_openai_response(httpx_mock, content="Response")

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test", timeout=60.0)

        response = await client.chat().messages([Message.user("Test")]).execute()
        assert response.content == "Response"

    @pytest.mark.asyncio
    async def test_protocol_headers(self, httpx_mock) -> None:
        """Test protocol-specific headers."""
        from tests.integration.conftest import setup_mock_openai_response

        setup_mock_openai_response(httpx_mock, content="Headers applied")

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")

        # The request should have proper headers (Authorization, Content-Type, etc.)
        response = await client.chat().messages([Message.user("Test")]).execute()
        assert response.content == "Headers applied"


class TestBuilderConfiguration:
    """Tests for client builder configuration."""

    @pytest.mark.asyncio
    async def test_builder_with_all_options(self, httpx_mock) -> None:
        """Test builder with all configuration options."""
        from tests.integration.conftest import setup_mock_openai_response

        setup_mock_openai_response(httpx_mock, content="Full config")

        from ai_lib_python.client import AiClientBuilder
        client = await (
            AiClientBuilder()
            .model("openai/gpt-4o")
            .api_key("sk-test")
            .base_url("https://api.openai.com/v1")
            .timeout(30.0)
            .max_inflight(10)
            .with_fallbacks(["openai/gpt-3.5-turbo"])
            .production_ready()
            .build()
        )

        assert client._model_id == "openai/gpt-4o"
        assert client._fallbacks == ["openai/gpt-3.5-turbo"]
        assert client._executor is not None  # production_ready adds resilience

        response = await client.chat().messages([Message.user("Test")]).execute()
        assert response.content == "Full config"

    @pytest.mark.asyncio
    async def test_builder_multiple_api_keys(self, httpx_mock) -> None:
        """Test builder with API keys for multiple models."""
        from tests.integration.conftest import (
            setup_mock_anthropic_response,
            setup_mock_openai_response,
        )

        setup_mock_openai_response(httpx_mock, content="OpenAI")
        setup_mock_anthropic_response(httpx_mock, content="Anthropic")

        from ai_lib_python.client import AiClientBuilder
        client = await (
            AiClientBuilder()
            .model("openai/gpt-4o")
            .api_key("sk-openai")
            .with_fallbacks(["anthropic/claude-3-5-sonnet"])
            .api_key_for("anthropic/claude-3-5-sonnet", "sk-anthropic")
            .build()
        )

        response = await client.chat().messages([Message.user("Test")]).execute()
        assert response.content == "OpenAI"


class TestCustomProtocol:
    """Tests for custom protocol configurations."""

    @pytest.mark.asyncio
    async def test_custom_model_in_protocol(self, httpx_mock) -> None:
        """Test using a custom model ID."""
        from tests.integration.conftest import setup_mock_openai_response

        setup_mock_openai_response(httpx_mock, content="Custom model response")

        # Custom model ID should still use OpenAI protocol if prefixed
        client = await AiClient.create("openai/custom-model-123", api_key="sk-test")

        assert client._model_id == "openai/custom-model-123"

        response = await client.chat().messages([Message.user("Test")]).execute()
        assert response.content == "Custom model response"


class TestProtocolCompatibility:
    """Tests for protocol feature compatibility."""

    @pytest.mark.asyncio
    async def test_streaming_support(self, httpx_mock) -> None:
        """Test streaming is supported by protocol."""
        from tests.integration.conftest import setup_mock_openai_response

        setup_mock_openai_response(httpx_mock, content="Streamed", stream=True)

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")

        events = []
        async for event in client.chat().messages([Message.user("Test")]).stream():
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_tool_calling_support(self, httpx_mock) -> None:
        """Test tool calling is supported by protocol."""
        from ai_lib_python.types.tool import ToolDefinition
        from tests.integration.conftest import mock_openai_tool_call_response

        tool = ToolDefinition.from_function(
            name="test_tool",
            description="Test",
            parameters={
                "type": "object",
                "properties": {"arg": {"type": "string"}},
                "required": ["arg"],
            },
        )

        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            method="POST",
            json=[mock_openai_tool_call_response(tool_name="test_tool", tool_args={"arg": "test"})],
        )

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")
        response = await (
            client.chat()
            .messages([Message.user("Use tool")])
            .tools([tool])
            .execute()
        )

        assert response.has_tool_calls

    @pytest.mark.asyncio
    async def test_system_message_support(self, httpx_mock) -> None:
        """Test system messages are supported."""
        from tests.integration.conftest import setup_mock_openai_response

        setup_mock_openai_response(httpx_mock, content="System message received")

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")
        response = await (
            client.chat()
            .system("You are a helpful assistant")
            .user("Hello")
            .execute()
        )

        assert response.content == "System message received"
