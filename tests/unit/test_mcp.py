"""MCP 工具桥接单元测试。"""

from __future__ import annotations

from ai_lib_python.mcp import (
    McpTool,
    McpToolBridge,
    McpToolResult,
    extract_provider_config,
)


def _sample_mcp_tools() -> list[McpTool]:
    return [
        McpTool(
            name="read_file",
            description="Read a file from disk",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        ),
        McpTool(name="search", description="Search the web", input_schema=None),
        McpTool(name="exec_dangerous", description="Execute shell command"),
    ]


class TestMcpToolBridge:
    def test_mcp_to_protocol(self) -> None:
        bridge = McpToolBridge("fileserver")
        tools = bridge.mcp_tools_to_protocol(_sample_mcp_tools())
        assert len(tools) == 3
        assert tools[0]["function"]["name"] == "mcp__fileserver__read_file"
        assert tools[0]["type"] == "function"

    def test_allow_filter(self) -> None:
        bridge = McpToolBridge("srv", allow_filter={"read_file", "search"})
        tools = bridge.mcp_tools_to_protocol(_sample_mcp_tools())
        assert len(tools) == 2
        names = [t["function"]["name"] for t in tools]
        assert not any("exec_dangerous" in n for n in names)

    def test_deny_filter(self) -> None:
        bridge = McpToolBridge("srv", deny_filter={"exec_dangerous"})
        tools = bridge.mcp_tools_to_protocol(_sample_mcp_tools())
        assert len(tools) == 2

    def test_protocol_call_to_mcp(self) -> None:
        bridge = McpToolBridge("srv")
        call = {"name": "mcp__srv__read_file", "arguments": {"path": "/tmp/test.txt"}}
        invocation = bridge.protocol_call_to_mcp(call)
        assert invocation is not None
        assert invocation.name == "read_file"
        assert invocation.arguments["path"] == "/tmp/test.txt"

    def test_wrong_namespace_returns_none(self) -> None:
        bridge = McpToolBridge("srv")
        call = {"name": "mcp__other__read_file", "arguments": {}}
        assert bridge.protocol_call_to_mcp(call) is None

    def test_mcp_result_to_protocol(self) -> None:
        bridge = McpToolBridge("srv")
        result = McpToolResult(
            content=[{"type": "text", "text": "file contents"}], is_error=False
        )
        proto = bridge.mcp_result_to_protocol("call_123", result)
        assert proto["tool_use_id"] == "call_123"
        assert not proto["is_error"]

    def test_mcp_result_error(self) -> None:
        bridge = McpToolBridge("srv")
        result = McpToolResult(
            content=[{"type": "text", "text": "not found"}], is_error=True
        )
        proto = bridge.mcp_result_to_protocol("call_1", result)
        assert proto["is_error"]
        assert "not found" in proto["content"]["error"]


class TestExtractProviderConfig:
    def test_anthropic_config(self) -> None:
        config = {
            "client": {
                "supported": True,
                "provider_mapping": {
                    "tool_type": "mcp",
                    "beta_header": "mcp-client-2025-11-20",
                    "config_method": "tool_parameter",
                },
            }
        }
        prov = extract_provider_config(config)
        assert prov is not None
        assert prov.tool_type == "mcp"
        assert prov.beta_header == "mcp-client-2025-11-20"

    def test_unsupported_returns_none(self) -> None:
        config = {"client": {"supported": False}}
        assert extract_provider_config(config) is None

    def test_none_returns_none(self) -> None:
        assert extract_provider_config(None) is None
