"""MCP Bridge end-to-end tests against ai-protocol-mock.

Requires ai-protocol-mock server running (e.g. docker-compose up -d).
Set MOCK_MCP_URL=http://localhost:4010/mcp or use default.
"""

from __future__ import annotations

import os

import httpx
import pytest

from ai_lib_python.mcp import McpTool, McpToolBridge

MOCK_MCP_URL = os.getenv("MOCK_MCP_URL", "http://localhost:4010/mcp")


def _mock_server_available() -> bool:
    """Check if mock server is reachable."""
    try:
        r = httpx.get(MOCK_MCP_URL.replace("/mcp", "/health"), timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.mockserver,
    pytest.mark.skipif(
        not _mock_server_available(),
        reason="ai-protocol-mock server not running (start with docker-compose up -d)",
    ),
]


@pytest.mark.asyncio
async def test_mcp_tools_list_via_mock() -> None:
    """Test tools/list returns valid tool list from mock server."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.post(
            MOCK_MCP_URL,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
    assert r.status_code == 200
    data = r.json()
    assert "result" in data
    assert "tools" in data["result"]
    tools = data["result"]["tools"]
    assert len(tools) > 0
    for t in tools:
        assert "name" in t


@pytest.mark.asyncio
async def test_mcp_tools_call_via_mock() -> None:
    """Test tools/call returns result from mock server."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.post(
            MOCK_MCP_URL,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "read_file", "arguments": {"path": "/tmp/test.txt"}},
            },
        )
    assert r.status_code == 200
    data = r.json()
    assert "result" in data
    assert "content" in data["result"]
    assert data["result"]["content"][0]["type"] == "text"


@pytest.mark.asyncio
async def test_mcp_bridge_roundtrip_with_mock() -> None:
    """Test McpToolBridge with tools from mock server."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.post(
            MOCK_MCP_URL,
            json={"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}},
        )
    assert r.status_code == 200
    tools_data = r.json()["result"]["tools"]
    mcp_tools = [
        McpTool(
            name=t["name"],
            description=t.get("description"),
            input_schema=t.get("inputSchema"),
        )
        for t in tools_data
    ]
    bridge = McpToolBridge("mock")
    protocol_tools = bridge.mcp_tools_to_protocol(mcp_tools)
    assert len(protocol_tools) == len(mcp_tools)
    assert protocol_tools[0]["function"]["name"].startswith("mcp__mock__")
    # Round-trip: protocol call -> mcp invocation
    call = {"name": protocol_tools[0]["function"]["name"], "arguments": {"path": "/tmp/x"}}
    inv = bridge.protocol_call_to_mcp(call)
    assert inv is not None
    assert inv.name == mcp_tools[0].name
