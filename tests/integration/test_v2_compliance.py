"""V2 协议跨模块集成测试 — 验证 Manifest→Driver→Registry→MCP→CU 全链路。"""

from __future__ import annotations

import pytest

from ai_lib_python.computer_use import (
    ComputerAction,
    ImplementationStyle,
    SafetyPolicy,
    SafetyViolation,
)
from ai_lib_python.computer_use import (
    extract_provider_config as cu_extract,
)
from ai_lib_python.drivers import create_driver
from ai_lib_python.mcp import (
    McpTool,
    McpToolBridge,
    McpToolResult,
)
from ai_lib_python.mcp import (
    extract_provider_config as mcp_extract,
)
from ai_lib_python.multimodal import (
    Modality,
    MultimodalCapabilities,
    validate_content_modalities,
)
from ai_lib_python.protocol.v2.capabilities import CapabilitiesV2, Capability
from ai_lib_python.protocol.v2.manifest import ApiStyle, EndpointV2, ManifestV2
from ai_lib_python.registry import CapabilityRegistry


class TestFullChainOpenAI:
    """End-to-end: parse V2 manifest → driver → registry → MCP → CU → multimodal."""

    def test_full_chain(self) -> None:
        manifest = ManifestV2(
            id="openai",
            protocol_version="2.0",
            name="OpenAI",
            endpoints=EndpointV2(base_url="https://api.openai.com/v1", chat="/chat/completions"),
            capabilities=CapabilitiesV2(
                required=[Capability.TEXT, Capability.STREAMING, Capability.TOOLS, Capability.MCP_CLIENT],
                optional=[Capability.VISION, Capability.COMPUTER_USE],
            ),
        )

        # API style detection
        assert manifest.detect_api_style() == ApiStyle.OPENAI_COMPATIBLE

        # Driver creation
        driver = create_driver(manifest.detect_api_style(), "openai", manifest.capabilities.required)
        assert Capability.MCP_CLIENT in driver.supported_capabilities()

        # Registry — check that core capabilities are active
        registry = CapabilityRegistry.from_capabilities(manifest.capabilities)
        report = registry.status_report()
        assert "active" in report[Capability.TEXT].value
        assert "active" in report[Capability.STREAMING].value

        # MCP bridge
        bridge = McpToolBridge("filesystem")
        tools = bridge.mcp_tools_to_protocol([
            McpTool(name="read_file", description="Read file", input_schema={"type": "object"}),
        ])
        assert tools[0]["function"]["name"] == "mcp__filesystem__read_file"

        # Round-trip
        call = {"name": "mcp__filesystem__read_file", "arguments": {"path": "/tmp/test"}}
        invocation = bridge.protocol_call_to_mcp(call)
        assert invocation is not None
        assert invocation.name == "read_file"


class TestFullChainAnthropic:
    """End-to-end: Anthropic — different API style, MCP beta header."""

    def test_full_chain(self) -> None:
        manifest = ManifestV2(
            id="anthropic",
            protocol_version="2.0",
            endpoints=EndpointV2(base_url="https://api.anthropic.com/v1", chat="/messages"),
            capabilities=CapabilitiesV2(
                required=[Capability.TEXT, Capability.STREAMING, Capability.COMPUTER_USE],
            ),
        )
        assert manifest.detect_api_style() == ApiStyle.ANTHROPIC_MESSAGES

        driver = create_driver(
            ApiStyle.ANTHROPIC_MESSAGES,
            "anthropic",
            [Capability.TEXT, Capability.STREAMING, Capability.COMPUTER_USE],
        )
        assert Capability.TEXT in driver.supported_capabilities()

        # MCP config with beta header
        mcp_config = {
            "client": {
                "supported": True,
                "transports": ["sse"],
                "provider_mapping": {"beta_header": "mcp-client-2025-11-20"},
            }
        }
        prov = mcp_extract(mcp_config)
        assert prov is not None
        assert prov.beta_header == "mcp-client-2025-11-20"

        # CU config
        cu_config = {
            "supported": True,
            "implementation": "screen_based",
            "provider_mapping": {
                "tool_type": "computer_20251124",
                "beta_header": "computer-use-2025-11-24",
            },
        }
        cu_prov = cu_extract(cu_config)
        assert cu_prov is not None
        assert cu_prov.tool_type == "computer_20251124"
        assert cu_prov.implementation == ImplementationStyle.SCREEN_BASED


class TestFullChainGemini:
    """End-to-end: Gemini — tool_based CU, SDK config."""

    def test_full_chain(self) -> None:
        manifest = ManifestV2(
            id="google",
            protocol_version="2.0",
            endpoints=EndpointV2(
                base_url="https://generativelanguage.googleapis.com/v1beta",
                chat="/models/{model}:generateContent",
            ),
            capabilities=CapabilitiesV2(
                required=[Capability.TEXT, Capability.STREAMING, Capability.COMPUTER_USE],
                optional=[Capability.VISION, Capability.VIDEO],
            ),
        )
        assert manifest.detect_api_style() == ApiStyle.GEMINI_GENERATE

        cu_config = {
            "supported": True,
            "implementation": "tool_based",
            "provider_mapping": {"tool_type": "computer_use", "config_method": "sdk_config"},
        }
        cu_prov = cu_extract(cu_config)
        assert cu_prov is not None
        assert cu_prov.implementation == ImplementationStyle.TOOL_BASED

        # Multimodal: video support
        mm_config = {
            "input": {
                "vision": {"supported": True, "formats": ["jpeg", "png"], "encoding_methods": ["url"]},
                "video": {"supported": True, "formats": ["mp4", "mov"]},
            },
            "output": {"text": True},
        }
        caps = MultimodalCapabilities.from_config(mm_config)
        assert caps.supports_input(Modality.VIDEO)
        assert caps.validate_video_format("mp4")


class TestMcpBridgeRoundtrip:
    def test_roundtrip(self) -> None:
        bridge = McpToolBridge("testserver")
        tools = bridge.mcp_tools_to_protocol([
            McpTool(name="calculate", description="Math", input_schema={"type": "object"}),
        ])
        assert "mcp__testserver__calculate" in tools[0]["function"]["name"]

        call = {"name": "mcp__testserver__calculate", "arguments": {"expr": "2+2"}}
        inv = bridge.protocol_call_to_mcp(call)
        assert inv is not None
        assert inv.name == "calculate"

        result = McpToolResult(content=[{"type": "text", "text": "4"}])
        proto = bridge.mcp_result_to_protocol("call_1", result)
        assert not proto["is_error"]


class TestCuSafetyEnforcement:
    def test_max_actions_and_domain(self) -> None:
        policy = SafetyPolicy(
            max_actions_per_turn=3,
            domain_allowlist={"example.com"},
        )
        action = ComputerAction.screenshot()
        policy.validate_action(action, 0)
        policy.validate_action(action, 2)
        with pytest.raises(SafetyViolation, match="Max actions"):
            policy.validate_action(action, 3)

        ok_nav = ComputerAction.browser_navigate("https://example.com/page")
        policy.validate_action(ok_nav)

        with pytest.raises(SafetyViolation, match="not in the allowlist"):
            policy.validate_action(ComputerAction.browser_navigate("https://evil.com"))


class TestMultimodalValidation:
    def test_reject_unsupported(self) -> None:
        caps = MultimodalCapabilities.from_config({
            "input": {
                "vision": {"supported": True, "formats": ["jpeg"], "encoding_methods": ["url"]},
            },
            "output": {"text": True},
        })
        valid = [{"type": "text", "text": "Hello"}, {"type": "image", "source": {}}]
        assert validate_content_modalities(valid, caps) == []

        invalid = [{"type": "video", "source": {}}]
        unsupported = validate_content_modalities(invalid, caps)
        assert Modality.VIDEO in unsupported
