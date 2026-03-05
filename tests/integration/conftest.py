"""
Integration test helper utilities.

Shared fixtures and utilities for integration tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:

    import pytest_httpx


def mock_openai_chat_response(
    content: str = "Hello from OpenAI!",
    model: str = "gpt-4o",
    stream: bool = False,
    finish_reason: str = "stop",
    usage: dict | None = None,
) -> dict:
    """Create a mock OpenAI chat response."""
    response = {
        "id": "chatcmpl-123",
        "object": "chat.completion" if not stream else "chat.completion.chunk",
        "created": 1699012345,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
    }

    if usage:
        response["usage"] = usage

    if stream:
        response["choices"][0]["delta"] = response["choices"][0].pop("message")

    return response


def mock_openai_streaming_chunks(content: str, model: str = "gpt-4o") -> list[dict]:
    """Create mock OpenAI streaming chunks."""
    chunks = []
    for _i, char in enumerate(content):
        chunks.append({
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1699012345,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": char},
                    "finish_reason": None,
                }
            ],
        })

    # Final chunk with finish_reason
    chunks.append({
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1699012345,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    })

    return chunks


def mock_anthropic_chat_response(
    content: str = "Hello from Anthropic!",
    model: str = "claude-3-5-sonnet-20241022",
    stream: bool = False,
    stop_reason: str = "end_turn",
    usage: dict | None = None,
) -> dict:
    """Create a mock Anthropic chat response."""
    response = {
        "id": "msg_123",
        "type": "message" if not stream else "content_block_delta",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": content}],
        "stop_reason": stop_reason,
    }

    if usage:
        response["usage"] = usage

    return response


def mock_anthropic_streaming_chunks(content: str, model: str = "claude-3-5-sonnet-20241022") -> list[dict]:
    """Create mock Anthropic streaming chunks."""
    chunks = []

    # Start message
    chunks.append({
        "type": "message_start",
        "message": {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 0},
        },
    })

    # Content block start
    chunks.append({
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })

    # Delta chunks
    for _i, char in enumerate(content):
        chunks.append({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": char},
        })

    # Content block stop
    chunks.append({
        "type": "content_block_stop",
        "index": 0,
    })

    # Message stop
    chunks.append({
        "type": "message_stop",
    })

    return chunks


def setup_mock_openai_response(
    httpx_mock: pytest_httpx.HTTPXMock,
    content: str | None = None,
    model: str = "gpt-4o",
    stream: bool = False,
) -> list:
    """Setup mock OpenAI API response and return chunks."""
    actual_content = content or "Hello from OpenAI!"

    if stream:
        json_body = mock_openai_streaming_chunks(actual_content, model)
    else:
        json_body = mock_openai_chat_response(actual_content, model, stream)

    url = "https://api.openai.com/v1/chat/completions"
    httpx_mock.add_response(
        url=url,
        method="POST",
        json=json_body,
    )

    return json_body if isinstance(json_body, list) else [json_body]


def setup_mock_anthropic_response(
    httpx_mock: pytest_httpx.HTTPXMock,
    content: str | None = None,
    model: str = "claude-3-5-sonnet-20241022",
    stream: bool = False,
) -> list:
    """Setup mock Anthropic API response and return chunks."""
    actual_content = content or "Hello from Anthropic!"

    if stream:
        json_body = mock_anthropic_streaming_chunks(actual_content, model)
    else:
        json_body = mock_anthropic_chat_response(actual_content, model, stream)

    url = "https://api.anthropic.com/v1/messages"
    httpx_mock.add_response(
        url=url,
        method="POST",
        json=json_body,
    )

    return json_body if isinstance(json_body, list) else [json_body]


def mock_openai_tool_call_response(
    tool_name: str = "get_weather",
    tool_args: dict | None = None,
    model: str = "gpt-4o",
) -> dict:
    """Create a mock OpenAI tool call response."""
    if tool_args is None:
        tool_args = {"city": "Tokyo"}

    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1699012345,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": str(tool_args).replace("'", '"'),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def mock_anthropic_tool_call_response(
    tool_name: str = "get_weather",
    tool_input: dict | None = None,
    model: str = "claude-3-5-sonnet-20241022",
) -> dict:
    """Create a mock Anthropic tool use response."""
    if tool_input is None:
        tool_input = {"city": "Tokyo"}

    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": tool_name,
                "input": tool_input,
            }
        ],
        "stop_reason": "tool_use",
    }
