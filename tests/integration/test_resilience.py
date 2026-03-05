"""
Integration tests for error handling and resilience.

Tests error scenarios, retries, and fallback mechanisms.
"""

import pytest

from ai_lib_python.client import AiClient
from ai_lib_python.errors import TransportError
from ai_lib_python.types.message import Message


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_api_key_missing(self, _httpx_mock) -> None:
        """Test error when API key is missing."""
        with pytest.raises(ValueError, match="API key required"):
            await AiClient.create("openai/gpt-4o")

    @pytest.mark.asyncio
    async def test_invalid_model(self, httpx_mock) -> None:
        """Test error with invalid model ID."""

        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            method="POST",
            status_code=404,
            json={"error": {"message": "Model not found"}},
        )

        client = await AiClient.create("openai/invalid-model", api_key="sk-test")

        with pytest.raises(TransportError):
            await client.chat().messages([Message.user("Test")]).execute()

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, httpx_mock) -> None:
        """Test handling of rate limit errors."""

        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            method="POST",
            status_code=429,
            json={"error": {"message": "Rate limit exceeded"}},
        )

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")

        with pytest.raises(TransportError):
            await client.chat().messages([Message.user("Test")]).execute()

    @pytest.mark.asyncio
    async def test_authentication_error(self, httpx_mock) -> None:
        """Test handling of authentication errors."""

        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            method="POST",
            status_code=401,
            json={"error": {"message": "Invalid API key"}},
        )

        client = await AiClient.create("openai/gpt-4o", api_key="invalid-key")

        with pytest.raises(TransportError):
            await client.chat().messages([Message.user("Test")]).execute()

    @pytest.mark.asyncio
    async def test_server_error(self, httpx_mock) -> None:
        """Test handling of server errors."""

        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            method="POST",
            status_code=500,
            json={"error": {"message": "Internal server error"}},
        )

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")

        with pytest.raises(TransportError):
            await client.chat().messages([Message.user("Test")]).execute()

    @pytest.mark.asyncio
    async def test_timeout_error(self, httpx_mock) -> None:
        """Test handling of timeout errors."""
        from httpx import TimeoutException

        httpx_mock.add_exception(
            TimeoutException("Request timed out"),
            url="https://api.openai.com/v1/chat/completions",
            method="POST",
        )

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test", timeout=0.1)

        with pytest.raises(TransportError):
            await client.chat().messages([Message.user("Test")]).execute()

    @pytest.mark.asyncio
    async def test_network_error(self, httpx_mock) -> None:
        """Test handling of network errors."""
        from httpx import ConnectError

        httpx_mock.add_exception(
            ConnectError("Connection failed"),
            url="https://api.openai.com/v1/chat/completions",
            method="POST",
        )

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")

        with pytest.raises(TransportError):
            await client.chat().messages([Message.user("Test")]).execute()


class TestRetryMechanism:
    """Tests for retry mechanism."""

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self, httpx_mock) -> None:
        """Test retry on transient server error."""
        # First call fails, second succeeds

        from tests.integration.conftest import setup_mock_openai_response

        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            method="POST",
            status_code=503,
            json={"error": {"message": "Service unavailable"}},
        )

        setup_mock_openai_response(httpx_mock, content="Success after retry")

        # Build client with retry
        from ai_lib_python.client import AiClientBuilder
        client = await (
            AiClientBuilder()
            .model("openai/gpt-4o")
            .api_key("sk-test")
            .retry(max_attempts=3, backoff=0.1)
            .build()
        )

        response = await client.chat().messages([Message.user("Test")]).execute()
        assert response.content == "Success after retry"

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, httpx_mock) -> None:
        """Test that all retries are exhausted."""

        # All calls fail (matching retry attempts)
        for _ in range(3):  # 1 initial + 2 retries
            httpx_mock.add_response(
                url="https://api.openai.com/v1/chat/completions",
                method="POST",
                status_code=503,
                json={"error": {"message": "Service unavailable"}},
            )

        from ai_lib_python.client import AiClientBuilder
        client = await (
            AiClientBuilder()
            .model("openai/gpt-4o")
            .api_key("sk-test")
            .retry(max_attempts=2, backoff=0.1)
            .build()
        )

        with pytest.raises(TransportError):
            await client.chat().messages([Message.user("Test")]).execute()


class TestFallbackMechanism:
    """Tests for fallback mechanism."""

    @pytest.mark.asyncio
    async def test_fallback_to_secondary_model(self, httpx_mock) -> None:
        """Test fallback to secondary model on failure."""

        from tests.integration.conftest import setup_mock_anthropic_response

        # Primary model fails
        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            method="POST",
            status_code=503,
            json={"error": {"message": "Service unavailable"}},
        )

        # Fallback model succeeds
        setup_mock_anthropic_response(httpx_mock, content="Response from fallback")

        from ai_lib_python.client import AiClientBuilder
        client = await (
            AiClientBuilder()
            .model("openai/gpt-4o")
            .api_key("sk-test")
            .with_fallbacks(["anthropic/claude-3-5-sonnet"])
            .api_key_for("anthropic/claude-3-5-sonnet", "sk-ant-test")
            .build()
        )

        response = await client.chat().messages([Message.user("Test")]).execute()
        assert response.content == "Response from fallback"

    @pytest.mark.asyncio
    async def test_all_fallbacks_fail(self, httpx_mock) -> None:
        """Test that all fallbacks are tried before giving up."""

        # All models fail
        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            method="POST",
            status_code=503,
            json={"error": {"message": "Service unavailable"}},
        )
        httpx_mock.add_response(
            url="https://api.anthropic.com/v1/messages",
            method="POST",
            status_code=503,
            json={"error": {"message": "Service unavailable"}},
        )

        from ai_lib_python.client import AiClientBuilder
        client = await (
            AiClientBuilder()
            .model("openai/gpt-4o")
            .api_key("sk-test")
            .with_fallbacks(["anthropic/claude-3-5-sonnet"])
            .api_key_for("anthropic/claude-3-5-sonnet", "sk-ant-test")
            .build()
        )

        with pytest.raises(TransportError):
            await client.chat().messages([Message.user("Test")]).execute()


class TestCircuitBreaker:
    """Tests for circuit breaker mechanism."""

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self, httpx_mock) -> None:
        """Test circuit opens after consecutive failures."""

        # Multiple failures
        for _ in range(5):
            httpx_mock.add_response(
                url="https://api.openai.com/v1/chat/completions",
                method="POST",
                status_code=503,
                json={"error": {"message": "Service unavailable"}},
            )

        from ai_lib_python.client import AiClientBuilder
        client = await (
            AiClientBuilder()
            .model("openai/gpt-4o")
            .api_key("sk-test")
            .production_ready()
            .build()
        )

        # Fail multiple times
        for _ in range(3):
            with pytest.raises(TransportError):
                await client.chat().messages([Message.user("Test")]).execute()

        # Circuit should be open, request should fail immediately
        with pytest.raises(TransportError):
            await client.chat().messages([Message.user("Test")]).execute()
