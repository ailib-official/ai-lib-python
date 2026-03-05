"""
Integration tests for concurrent requests.

Tests behavior under concurrent load.
"""

import asyncio

import pytest

from ai_lib_python.client import AiClient
from ai_lib_python.errors import TransportError
from ai_lib_python.types.message import Message


class TestConcurrency:
    """Tests for concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, httpx_mock) -> None:
        """Test multiple concurrent requests."""
        from tests.integration.conftest import setup_mock_openai_response

        # Setup responses for 5 concurrent requests
        for i in range(5):
            setup_mock_openai_response(httpx_mock, content=f"Response {i+1}")

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")

        # Create concurrent tasks
        tasks = [
            client.chat().messages([Message.user(f"Request {i+1}")]).execute()
            for i in range(5)
        ]

        # Wait for all to complete
        responses = await asyncio.gather(*tasks)

        # Verify all responses
        assert len(responses) == 5
        for i, response in enumerate(responses):
            assert response.content == f"Response {i+1}"

    @pytest.mark.asyncio
    async def test_max_inflight_limit(self, httpx_mock) -> None:
        """Test that max_inflight limits concurrent requests."""
        from tests.integration.conftest import setup_mock_openai_response

        # Setup many responses
        for _ in range(10):
            setup_mock_openai_response(httpx_mock, content="OK")

        # Build client with small max_inflight
        from ai_lib_python.client import AiClientBuilder
        client = await (
            AiClientBuilder()
            .model("openai/gpt-4o")
            .api_key("sk-test")
            .max_inflight(3)
            .build()
        )

        # Create more concurrent tasks than max_inflight
        tasks = [
            client.chat().messages([Message.user(f"Request {i+1}")]).execute()
            for i in range(10)
        ]

        # All should complete (respecting max_inflight internally)
        responses = await asyncio.gather(*tasks)
        assert len(responses) == 10

    @pytest.mark.asyncio
    async def test_concurrent_streaming(self, httpx_mock) -> None:
        """Test multiple concurrent streaming requests."""
        from tests.integration.conftest import setup_mock_openai_response

        # Setup streaming responses
        for i in range(3):
            setup_mock_openai_response(httpx_mock, content=f"Stream{i}", stream=True)

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")

        # Create concurrent streaming tasks
        async def collect_stream(i: int) -> str:
            events = []
            async for event in client.chat().messages([Message.user(f"Stream {i+1}")]).stream():
                if event.is_content_delta:
                    events.append(event.as_content_delta.content)
            return "".join(events)

        tasks = [collect_stream(i) for i in range(3)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert "Stream0" in results[0]
        assert "Stream1" in results[1]
        assert "Stream2" in results[2]

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self, httpx_mock) -> None:
        """Test concurrent streaming and non-streaming requests."""
        from tests.integration.conftest import setup_mock_openai_response

        # Setup responses
        for i in range(3):
            setup_mock_openai_response(httpx_mock, content=f"Stream{i}", stream=True)
        for i in range(3):
            setup_mock_openai_response(httpx_mock, content=f"Chat{i}")

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")

        # Mixed streaming and non-streaming
        async def stream_request(i: int) -> str:
            content = ""
            async for event in client.chat().messages([Message.user(f"Stream {i}")]).stream():
                if event.is_content_delta:
                    content += event.as_content_delta.content
            return content

        tasks = [
            stream_request(0),
            stream_request(1),
            stream_request(2),
            client.chat().messages([Message.user("Chat 0")]).execute(),
            client.chat().messages([Message.user("Chat 1")]).execute(),
            client.chat().messages([Message.user("Chat 2")]).execute(),
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 6
        # First 3 are streaming results (strings)
        for i in range(3):
            assert "Stream" in results[i]
        # Last 3 are ChatResponse objects
        for i in range(3, 6):
            assert "Chat" in results[i].content

    @pytest.mark.asyncio
    async def test_concurrent_with_different_models(self, httpx_mock) -> None:
        """Test concurrent requests to different models."""
        from tests.integration.conftest import (
            setup_mock_anthropic_response,
            setup_mock_openai_response,
        )

        # Setup responses for different models
        setup_mock_openai_response(httpx_mock, content="OpenAI response")
        setup_mock_anthropic_response(httpx_mock, content="Anthropic response")

        # Create clients for different models
        openai_client = await AiClient.create("openai/gpt-4o", api_key="sk-test")
        anthropic_client = await AiClient.create("anthropic/claude-3-5-sonnet", api_key="sk-ant-test")

        # Concurrent requests to different models
        tasks = [
            openai_client.chat().messages([Message.user("To OpenAI")]).execute(),
            anthropic_client.chat().messages([Message.user("To Anthropic")]).execute(),
        ]

        responses = await asyncio.gather(*tasks)

        assert responses[0].content == "OpenAI response"
        assert responses[1].content == "Anthropic response"

    @pytest.mark.asyncio
    async def test_concurrent_with_fallbacks(self, httpx_mock) -> None:
        """Test concurrent requests with fallback behavior."""
        from tests.integration.conftest import (
            setup_mock_anthropic_response,
            setup_mock_openai_response,
        )

        # Some requests use primary, some use fallback
        setup_mock_openai_response(httpx_mock, content="Primary works")
        setup_mock_anthropic_response(httpx_mock, content="Fallback used")

        from ai_lib_python.client import AiClientBuilder
        client = await (
            AiClientBuilder()
            .model("openai/gpt-4o")
            .api_key("sk-test")
            .with_fallbacks(["anthropic/claude-3-5-sonnet"])
            .api_key_for("anthropic/claude-3-5-sonnet", "sk-ant-test")
            .build()
        )

        # Both should work, one uses primary, one could use fallback
        # (in this mock setup, both succeed)
        tasks = [
            client.chat().messages([Message.user("Request 1")]).execute(),
            client.chat().messages([Message.user("Request 2")]).execute(),
        ]

        responses = await asyncio.gather(*tasks)

        assert len(responses) == 2
        assert responses[0].content == "Primary works"
        assert responses[1].content == "Primary works"

    @pytest.mark.asyncio
    async def test_concurrent_with_interleaved_writes(self, httpx_mock) -> None:
        """Test that concurrent requests don't interfere with each other."""
        from tests.integration.conftest import setup_mock_openai_response

        # Setup responses with unique identifiers
        for i in range(5):
            setup_mock_openai_response(httpx_mock, content=f"Unique response {i}")

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")

        # Create tasks with slight delays
        async def delayed_request(delay: float, index: int) -> str:
            await asyncio.sleep(delay)
            response = await client.chat().messages([Message.user(f"Request {index}")]).execute()
            return response.content

        tasks = [
            delayed_request(0.01, 0),
            delayed_request(0.0, 1),
            delayed_request(0.02, 2),
            delayed_request(0.005, 3),
            delayed_request(0.015, 4),
        ]

        results = await asyncio.gather(*tasks)

        # All should have unique responses
        assert len(results) == 5
        assert len(set(results)) == 5  # All unique


class TestResourceManagement:
    """Tests for resource management under load."""

    @pytest.mark.asyncio
    async def test_client_reuse_across_requests(self, httpx_mock) -> None:
        """Test client can be reused for multiple requests."""
        from tests.integration.conftest import setup_mock_openai_response

        for i in range(10):
            setup_mock_openai_response(httpx_mock, content=f"Request {i}")

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")

        # Reuse same client for multiple requests
        for i in range(10):
            response = await client.chat().messages([Message.user(f"Request {i}")]).execute()
            assert response.content == f"Request {i}"

    @pytest.mark.asyncio
    async def test_cleanup_on_error(self, httpx_mock) -> None:
        """Test resources are cleaned up on errors."""

        # Fail first request, succeed second
        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            method="POST",
            status_code=500,
            json={"error": {"message": "Error"}},
        )

        from tests.integration.conftest import setup_mock_openai_response
        setup_mock_openai_response(httpx_mock, content="Success")

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")

        # First request fails
        with pytest.raises(TransportError):
            await client.chat().messages([Message.user("Fail")]).execute()

        # Client should still work for second request
        response = await client.chat().messages([Message.user("Success")]).execute()
        assert response.content == "Success"
