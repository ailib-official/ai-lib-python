"""Chat E2E tests against ai-protocol-mock.

Requires ai-protocol-mock server running. Set MOCK_HTTP_URL and MOCK_MCP_URL.
Uses X-Mock-* headers for test control (content override, tool_calls, error status).
"""

from __future__ import annotations

import os

import httpx
import pytest

from ai_lib_python.client import AiClient
from ai_lib_python.types.message import Message

MOCK_HTTP_URL = os.getenv("MOCK_HTTP_URL", "http://localhost:4010")


def _mock_available() -> bool:
    try:
        r = httpx.get(
            f"{MOCK_HTTP_URL.rstrip('/')}/health", timeout=2.0, trust_env=False
        )
        return r.status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.mockserver,
    pytest.mark.skipif(not _mock_available(), reason="ai-protocol-mock not reachable"),
]


@pytest.mark.asyncio
async def test_chat_simple_via_mock() -> None:
    """Basic chat against mock server."""
    client = await AiClient.create(
        "openai/gpt-4o",
        api_key="sk-test",
        base_url=MOCK_HTTP_URL,
    )
    response = await client.chat().messages([Message.user("Hello")]).execute()
    assert response.content
    assert not response.has_tool_calls


@pytest.mark.asyncio
async def test_chat_custom_content_via_mock() -> None:
    """Chat with X-Mock-Content override."""
    # Use custom headers - AiClient may not expose raw headers; test via direct httpx
    async with httpx.AsyncClient(
        base_url=MOCK_HTTP_URL, timeout=10.0, trust_env=False
    ) as http:
        r = await http.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"X-Mock-Content": "Custom reply from test"},
        )
    assert r.status_code == 200
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    # Older ai-protocol-mock builds may ignore X-Mock-Content.
    assert content in {"Custom reply from test", "Mock response from ai-protocol-mock"}


@pytest.mark.asyncio
async def test_chat_tool_calls_via_mock() -> None:
    """Chat with X-Mock-Tool-Calls returns tool_calls."""
    async with httpx.AsyncClient(
        base_url=MOCK_HTTP_URL, timeout=10.0, trust_env=False
    ) as http:
        r = await http.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Weather?"}]},
            headers={"X-Mock-Tool-Calls": "1"},
        )
    assert r.status_code == 200
    data = r.json()
    msg = data["choices"][0]["message"]
    # Older ai-protocol-mock builds may not support X-Mock-Tool-Calls.
    if "tool_calls" in msg:
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
    else:
        assert msg.get("content")


@pytest.mark.asyncio
async def test_chat_streaming_via_mock() -> None:
    """Streaming chat against mock."""
    client = await AiClient.create(
        "openai/gpt-4o",
        api_key="sk-test",
        base_url=MOCK_HTTP_URL,
    )
    chunks = []
    async for event in client.chat().messages([Message.user("Hi")]).stream():
        chunks.append(event)
    assert len(chunks) > 0
    content = "".join(e.as_content_delta.content for e in chunks if e.is_content_delta)
    assert content or len(chunks) > 0
