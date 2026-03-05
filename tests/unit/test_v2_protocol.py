"""V2 协议模块单元测试。"""

from __future__ import annotations

from ai_lib_python.protocol.v2.capabilities import (
    CapabilitiesV2,
    Capability,
    LegacyCapabilities,
)
from ai_lib_python.protocol.v2.manifest import ApiStyle, ManifestV2


class TestCapability:
    def test_extra_name(self) -> None:
        assert Capability.TEXT.extra_name is None
        assert Capability.MCP_CLIENT.extra_name == "mcp"
        assert Capability.COMPUTER_USE.extra_name == "computer_use"

    def test_is_extra_gated(self) -> None:
        assert not Capability.STREAMING.is_extra_gated
        assert Capability.AUDIO.is_extra_gated

    def test_module_path(self) -> None:
        assert Capability.TEXT.module_path == "core"
        assert Capability.MCP_CLIENT.module_path == "mcp.client"


class TestCapabilitiesV2:
    def test_structured(self) -> None:
        caps = CapabilitiesV2(
            required=[Capability.TEXT, Capability.STREAMING],
            optional=[Capability.VISION, Capability.MCP_CLIENT],
        )
        assert caps.has_capability(Capability.TEXT)
        assert caps.has_capability(Capability.MCP_CLIENT)
        assert not caps.has_capability(Capability.COMPUTER_USE)

    def test_required_only(self) -> None:
        caps = CapabilitiesV2(
            required=[Capability.TEXT, Capability.STREAMING],
            optional=[Capability.VISION],
        )
        required = caps.required_capabilities()
        assert Capability.TEXT in required
        assert Capability.VISION not in required

    def test_all_capabilities(self) -> None:
        caps = CapabilitiesV2(
            required=[Capability.TEXT],
            optional=[Capability.VISION, Capability.TOOLS],
        )
        all_caps = caps.all_capabilities()
        assert len(all_caps) == 3
        assert Capability.TEXT in all_caps
        assert Capability.VISION in all_caps

    def test_from_legacy(self) -> None:
        legacy = LegacyCapabilities(streaming=True, tools=True, vision=True)
        v2 = CapabilitiesV2.from_legacy(legacy)
        assert Capability.TEXT in v2.required
        assert Capability.STREAMING in v2.required
        assert Capability.TOOLS in v2.optional
        assert Capability.VISION in v2.optional

    def test_from_raw_v2_format(self) -> None:
        data = {
            "required": ["text", "streaming"],
            "optional": ["vision"],
            "feature_flags": {"structured_output": True},
        }
        caps = CapabilitiesV2.from_raw(data)
        assert caps.has_capability(Capability.TEXT)
        assert caps.feature_flags.structured_output

    def test_from_raw_v1_format(self) -> None:
        data = {"streaming": True, "tools": True, "vision": False}
        caps = CapabilitiesV2.from_raw(data)
        assert Capability.STREAMING in caps.required
        assert Capability.TOOLS in caps.optional
        assert not caps.has_capability(Capability.VISION)


class TestManifestV2:
    def test_basic_manifest(self) -> None:
        m = ManifestV2(
            id="openai",
            name="OpenAI",
            protocol_version="2.0",
            api_style=ApiStyle.OPENAI_COMPATIBLE,
        )
        assert m.is_v2
        assert m.protocol_semver == (2, 0, 0)

    def test_v1_manifest(self) -> None:
        m = ManifestV2(id="test", protocol_version="1.5")
        assert not m.is_v2
        assert m.protocol_semver == (1, 5, 0)

    def test_detect_api_style(self) -> None:
        m = ManifestV2(id="anthropic", name="Anthropic")
        assert m.detect_api_style() == ApiStyle.ANTHROPIC_MESSAGES

        m2 = ManifestV2(id="google-gemini", name="Gemini")
        assert m2.detect_api_style() == ApiStyle.GEMINI_GENERATE

    def test_mcp_supported(self) -> None:
        from ai_lib_python.protocol.v2.manifest import McpConfig

        m = ManifestV2(id="test", mcp=McpConfig(supported=True))
        assert m.mcp_client_supported()

        m2 = ManifestV2(id="test2")
        assert not m2.mcp_client_supported()

    def test_has_capability(self) -> None:
        m = ManifestV2(
            id="test",
            capabilities=CapabilitiesV2(
                required=[Capability.TEXT],
                optional=[Capability.VISION],
            ),
        )
        assert m.has_capability(Capability.TEXT)
        assert m.has_capability("vision")
        assert not m.has_capability(Capability.MCP_CLIENT)
