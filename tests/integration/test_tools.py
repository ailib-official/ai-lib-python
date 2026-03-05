"""
Integration tests for tool calling functionality.

Tests tool use scenarios with mocked API responses.
"""

import pytest

from ai_lib_python.client import AiClient
from ai_lib_python.types.message import Message
from ai_lib_python.types.tool import ToolDefinition


class TestToolCalling:
    """Tests for tool/function calling."""

    @pytest.mark.asyncio
    async def test_openai_tool_call(self, httpx_mock) -> None:
        """Test OpenAI tool call request."""
        from tests.integration.conftest import mock_openai_tool_call_response

        tool = ToolDefinition.from_function(
            name="get_weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )

        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            method="POST",
            json=mock_openai_tool_call_response(tool_name="get_weather", tool_args={"city": "Tokyo"}),
        )

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")
        response = await (
            client.chat()
            .messages([Message.user("What's the weather in Tokyo?")])
            .tools([tool])
            .execute()
        )

        assert response.has_tool_calls
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].function_name == "get_weather"
        assert response.tool_calls[0].arguments == {"city": "Tokyo"}
        assert response.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_anthropic_tool_call(self, httpx_mock) -> None:
        """Test Anthropic tool use request."""
        from tests.integration.conftest import mock_anthropic_tool_call_response

        tool = ToolDefinition.from_function(
            name="get_weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )

        httpx_mock.add_response(
            url="https://api.anthropic.com/v1/messages",
            method="POST",
            json=mock_anthropic_tool_call_response(tool_name="get_weather", tool_input={"city": "Tokyo"}),
        )

        client = await AiClient.create("anthropic/claude-3-5-sonnet", api_key="sk-ant-test")
        response = await (
            client.chat()
            .messages([Message.user("What's the weather in Tokyo?")])
            .tools([tool])
            .execute()
        )

        assert response.has_tool_calls
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].function_name == "get_weather"

    @pytest.mark.asyncio
    async def test_multiple_tools(self, httpx_mock) -> None:
        """Test with multiple tool definitions."""
        from tests.integration.conftest import mock_openai_tool_call_response

        tools = [
            ToolDefinition.from_function(
                name="get_weather",
                description="Get weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
            ToolDefinition.from_function(
                name="get_time",
                description="Get time",
                parameters={
                    "type": "object",
                    "properties": {"timezone": {"type": "string"}},
                    "required": ["timezone"],
                },
            ),
        ]

        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            method="POST",
            json=mock_openai_tool_call_response(tool_name="get_time", tool_args={"timezone": "UTC"}),
        )

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")
        response = await (
            client.chat()
            .messages([Message.user("What time is it in UTC?")])
            .tools(tools)
            .execute()
        )

        assert response.has_tool_calls
        assert response.tool_calls[0].function_name == "get_time"

    @pytest.mark.asyncio
    async def test_tool_choice_auto(self, httpx_mock) -> None:
        """Test tool_choice='auto'."""
        from tests.integration.conftest import (
            setup_mock_openai_response,
        )

        tool = ToolDefinition.from_function(
            name="get_weather",
            description="Get weather",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )

        # Mock that no tool is called (auto decision)
        setup_mock_openai_response(httpx_mock, content="I can answer without tools")

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")
        response = await (
            client.chat()
            .messages([Message.user("Hello")])
            .tools([tool])
            .tool_choice("auto")
            .execute()
        )

        assert not response.has_tool_calls
        assert response.content == "I can answer without tools"

    @pytest.mark.asyncio
    async def test_tool_choice_required(self, httpx_mock) -> None:
        """Test tool_choice='required' forces tool call."""
        from tests.integration.conftest import mock_openai_tool_call_response

        tool = ToolDefinition.from_function(
            name="get_weather",
            description="Get weather",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )

        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            method="POST",
            json=mock_openai_tool_call_response(),
        )

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")
        response = await (
            client.chat()
            .messages([Message.user("Just say hello")])
            .tools([tool])
            .tool_choice("required")
            .execute()
        )

        assert response.has_tool_calls

    @pytest.mark.asyncio
    async def test_tool_response_to_message(self, httpx_mock) -> None:
        """Test converting tool response to message."""
        from tests.integration.conftest import (
            mock_openai_tool_call_response,
        )

        # First call: tool requested
        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            method="POST",
            json=mock_openai_tool_call_response(),
        )

        # Second call: final response
        from tests.integration.conftest import mock_openai_chat_response
        httpx_mock.add_response(
            url="https://api.openai.com/v1/chat/completions",
            method="POST",
            json=mock_openai_chat_response(content="Based on the weather report"),
        )

        client = await AiClient.create("openai/gpt-4o", api_key="sk-test")
        tool = ToolDefinition.from_function(
            name="get_weather",
            description="Get weather",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )

        # First request
        response = await (
            client.chat()
            .messages([Message.user("What's the weather?")])
            .tools([tool])
            .execute()
        )

        assert response.has_tool_calls

        # Convert tool response to message
        from ai_lib_python.types.tool import ToolCall
        tool_call = response.tool_calls[0]
        tool_message = ToolCall.to_message(tool_call, {"result": "25 degrees"})

        # Second request with tool result
        response2 = await (
            client.chat()
            .messages([
                Message.user("What's the weather?"),
                Message(role="assistant", tool_calls=[tool_call]),
                tool_message,
            ])
            .tools([tool])
            .execute()
        )

        assert "weather report" in response2.content.lower()
