"""Registry 单元测试。"""

from __future__ import annotations

from ai_lib_python.protocol.v2.capabilities import CapabilitiesV2, Capability
from ai_lib_python.registry import CapabilityRegistry, CapabilityStatus


class TestCapabilityRegistry:
    def test_from_capabilities(self) -> None:
        caps = CapabilitiesV2(
            required=[Capability.TEXT, Capability.STREAMING],
            optional=[Capability.VISION, Capability.TOOLS],
        )
        reg = CapabilityRegistry.from_capabilities(caps)
        assert reg.is_active(Capability.TEXT)
        assert reg.is_active(Capability.STREAMING)
        assert reg.is_active(Capability.TOOLS)

    def test_validate_requirements_pass(self) -> None:
        caps = CapabilitiesV2(required=[Capability.TEXT, Capability.STREAMING])
        reg = CapabilityRegistry.from_capabilities(caps)
        reg.validate_requirements()  # should not raise

    def test_active_capabilities(self) -> None:
        caps = CapabilitiesV2(
            required=[Capability.TEXT],
            optional=[Capability.VISION, Capability.MCP_CLIENT],
        )
        reg = CapabilityRegistry.from_capabilities(caps)
        active = reg.active_capabilities()
        assert Capability.TEXT in active

    def test_status_report(self) -> None:
        caps = CapabilitiesV2(
            required=[Capability.TEXT],
            optional=[Capability.VISION],
        )
        reg = CapabilityRegistry.from_capabilities(caps)
        report = reg.status_report()
        assert report[Capability.TEXT] == CapabilityStatus.ACTIVE_REQUIRED

    def test_undeclared_cap_not_active(self) -> None:
        caps = CapabilitiesV2(required=[Capability.TEXT])
        reg = CapabilityRegistry.from_capabilities(caps)
        # Even though STREAMING is always available, it's not declared
        assert not reg.is_active(Capability.IMAGE_GENERATION)
