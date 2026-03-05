"""
Integration tests for basic chat functionality.

Tests end-to-end chat interactions with mocked API responses.
"""

import pytest

from ai_lib_python.client import AiClient
from ai_lib_python.types.message import Message


class TestBasicChat:
    """Tests for basic non-streaming chat."""

    @pytest.mark.asyncio
    async def test_openai_simple_chat(self, httpx_mock) -> None:
        """Test simple OpenAI chat request."""
        from tests.integration.conftest import setup_mock_openai_response

        setup_mock_openai_response(httpx_mock, content="Hello from OpenAI!")

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")
        response = await client.chat().messages([Message.user("Hello")]).execute()

        assert response.content == "Hello from OpenAI!"
        assert not response.has_tool_calls
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_anthropic_simple_chat(self, httpx_mock) -> None:
        """Test simple Anthropic chat request."""
        from tests.integration.conftest import setup_mock_anthropic_response

        setup_mock_anthropic_response(httpx_mock, content="Hello from Anthropic!")

        client = await AiClient.create("anthropic/claude-3-5-sonnet", api_key="sk-ant-test")
        response = await client.chat().messages([Message.user("Hello")]).execute()

        assert response.content == "Hello from Anthropic!"
        assert not response.has_tool_calls

    @pytest.mark.asyncio
    async def test_chat_with_system_message(self, httpx_mock) -> None:
        """Test chat with system message."""
        from tests.integration.conftest import setup_mock_openai_response

        setup_mock_openai_response(httpx_mock, content="I'll help!")

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")
        response = await (
            client.chat()
            .system("You are a helpful assistant")
            .user("Can you help me?")
            .execute()
        )

        assert "help" in response.content.lower()

    @pytest.mark.asyncio
    async def test_chat_conversation_history(self, httpx_mock) -> None:
        """Test multi-turn conversation."""
        from tests.integration.conftest import setup_mock_openai_response

        # Mock multiple responses
        setup_mock_openai_response(httpx_mock, content="Hello!")
        setup_mock_openai_response(httpx_mock, content="How can I help?")
        setup_mock_openai_response(httpx_mock, content="Goodbye!")

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")

        # First turn
        response1 = await client.chat().messages([Message.user("Hi")]).execute()
        assert response1.content == "Hello!"

        # Second turn with history
        response2 = await (
            client.chat()
            .messages([Message.user("Hi"), Message.assistant("Hello!"), Message.user("Help")])
            .execute()
        )
        assert response2.content == "How can I help?"

        # Third turn
        history = [
            Message.user("Hi"),
            Message.assistant("Hello!"),
            Message.user("Help"),
            Message.assistant("How can I help?"),
            Message.user("Bye"),
        ]
        response3 = await client.chat().messages(history).execute()
        assert response3.content == "Goodbye!"

    @pytest.mark.asyncio
    async def test_chat_with_parameters(self, httpx_mock) -> None:
        """Test chat with custom parameters."""
        from tests.integration.conftest import setup_mock_openai_response

        setup_mock_openai_response(httpx_mock, content="Response with params")

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")
        response = await (
            client.chat()
            .messages([Message.user("Test")])
            .temperature(0.5)
            .max_tokens(100)
            .top_p(0.9)
            .execute()
        )

        assert response.content == "Response with params"

    @pytest.mark.asyncio
    async def test_chat_with_stop_sequences(self, httpx_mock) -> None:
        """Test chat with stop sequences."""
        from tests.integration.conftest import setup_mock_openai_response

        setup_mock_openai_response(httpx_mock, content="Short response")

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")
        response = await (
            client.chat()
            .messages([Message.user("Test")])
            .stop(["\n"])
            .execute()
        )

        assert response.content == "Short response"

    @pytest.mark.asyncio
    async def test_chat_response_metadata(self, httpx_mock) -> None:
        """Test chat response includes metadata."""
        from tests.integration.conftest import setup_mock_openai_response

        setup_mock_openai_response(
            httpx_mock,
            content="Test",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")
        response = await client.chat().messages([Message.user("Test")]).execute()

        assert response.prompt_tokens == 10
        assert response.completion_tokens == 5
        assert response.total_tokens == 15
        assert response.model is not None


class TestStreamingChat:
    """Tests for streaming chat functionality."""

    @pytest.mark.asyncio
    async def test_openai_streaming_chat(self, httpx_mock) -> None:
        """Test OpenAI streaming chat."""
        from tests.integration.conftest import setup_mock_openai_response

        content = "Hello"
        setup_mock_openai_response(httpx_mock, content=content, stream=True)

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")

        events = []
        async for event in client.chat().messages([Message.user("Hi")]).stream():
            events.append(event)

        # Verify we got events
        assert len(events) > 0

        # Verify content deltas
        content_deltas = [e.as_content_delta for e in events if e.is_content_delta]
        collected_content = "".join([d.content for d in content_deltas])
        assert collected_content == content

    @pytest.mark.asyncio
    async def test_anthropic_streaming_chat(self, httpx_mock) -> None:
        """Test Anthropic streaming chat."""
        from tests.integration.conftest import setup_mock_anthropic_response

        content = "World"
        setup_mock_anthropic_response(httpx_mock, content=content, stream=True)

        client = await AiClient.create("anthropic/claude-3-5-sonnet", api_key="sk-ant-test")

        events = []
        async for event in client.chat().messages([Message.user("Test")]).stream():
            events.append(event)

        # Verify we got events
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_streaming_with_cancellation(self, httpx_mock) -> None:
        """Test streaming can be cancelled."""
        from tests.integration.conftest import setup_mock_openai_response

        long_content = "x" * 100
        setup_mock_openai_response(httpx_mock, content=long_content, stream=True)

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")

        # Create cancel handle
        stream = client.chat().messages([Message.user("Test")]).stream()
        cancel_handle = stream._cancel_handle

        # Consume a few events then cancel
        count = 0
        async for _event in stream:
            count += 1
            if count >= 10:
                cancel_handle.cancel("User requested")

        # Should have consumed some events
        assert count < len(long_content)
